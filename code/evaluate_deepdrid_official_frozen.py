#!/usr/bin/env python3
"""One-shot frozen evaluation on the official DeepDRiD evaluation split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from infer_and_evaluate_eyeq_protocol import make_disease_model, make_quality_model
from optimize_protocol_aligned_gate import (
    aggregate_to_eye,
    apply_calibrator,
    bootstrap_patient_clusters,
    bootstrap_summary,
    entropy,
    fit_calibrator,
    fit_gate,
    severity_table,
    strategy_metrics,
)
from train_deepdrid_protocol_model import prepare_fundus_cache


QUALITY_COLUMNS = ("q_illumination", "q_clarity", "q_contrast", "q_overall")
FROZEN_CALIBRATION = "beta"
FROZEN_GATE_C = 0.01
FROZEN_PROTECTION_LAMBDA = 0.2


class EvaluationDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), str(row["image_id"])


def build_evaluation_manifest(root):
    root = Path(root)
    disease = pd.read_excel(root / "Challenge1_labels.xlsx")
    quality = pd.read_excel(root / "Challenge2_labels.xlsx")
    disease.columns = [str(column).strip() for column in disease.columns]
    quality.columns = [str(column).strip() for column in quality.columns]
    frame = disease.merge(quality, on="image_id", how="inner", validate="one_to_one")
    frame["image_id"] = frame["image_id"].astype(str)
    frame["patient_id"] = frame["image_id"].str.split("_").str[0].astype(int)
    side_view = frame["image_id"].str.split("_").str[1]
    frame["eye"] = side_view.str[0].map({"l": "left", "r": "right"})
    frame["view"] = side_view.str[1:].astype(int)
    frame["dr_grade"] = frame["DR_Levels"].astype(int)
    frame["referable_dr"] = (frame["dr_grade"] >= 2).astype(int)
    frame["y_true"] = frame["referable_dr"]
    frame["human_overall_quality"] = frame["Overall quality"].astype(int)
    frame["image_path"] = frame.apply(
        lambda row: str(root / "Images" / str(row["patient_id"]) / f"{row['image_id']}.jpg"),
        axis=1,
    )
    missing = frame.loc[~frame["image_path"].map(lambda value: Path(value).exists()), "image_path"]
    if len(missing):
        raise FileNotFoundError(f"Missing {len(missing)} official evaluation images")
    if len(frame) != 400 or frame["patient_id"].nunique() != 100:
        raise ValueError("Official evaluation manifest does not contain the expected 400 images/100 patients")
    return frame


def load_training_eyes(train_oof_path, quality_path, manifest_path):
    quality = pd.read_csv(quality_path)
    quality["image_id"] = quality["image_name"].astype(str).str.replace(".jpg", "", regex=False)
    manifest = pd.read_csv(manifest_path)[["image_id", "eye", "view"]]
    train = pd.read_csv(train_oof_path)
    train = train.merge(quality[["image_id", *QUALITY_COLUMNS]], on="image_id", how="inner", validate="one_to_one")
    train = train.merge(manifest, on="image_id", how="left", validate="one_to_one")
    return aggregate_to_eye(train)


@torch.inference_mode()
def infer_both_models(frame, disease_checkpoint, quality_checkpoint, device, image_size, batch_size, workers):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        EvaluationDataset(frame, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    disease_model = make_disease_model(disease_checkpoint).to(device).eval()
    quality_model = make_quality_model(quality_checkpoint).to(device).eval()
    amp_enabled = device.type == "cuda"
    rows = []
    for images, image_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            disease_probability = torch.sigmoid(disease_model(images).squeeze(1)).float().cpu().numpy()
            quality_probability = torch.sigmoid(quality_model(images)).float().cpu().numpy()
        for image_id, disease_score, quality_scores in zip(image_ids, disease_probability, quality_probability):
            row = {"image_id": image_id, "disease_prob": float(disease_score)}
            for column, score in zip(QUALITY_COLUMNS, quality_scores):
                row[column] = float(score)
            rows.append(row)
    return pd.DataFrame(rows)


def fit_frozen_training_gate(train_eye):
    y = train_eye["y_true"].to_numpy(dtype=int)
    raw = train_eye["disease_prob_raw"].to_numpy(dtype=float)
    calibrator = fit_calibrator(FROZEN_CALIBRATION, raw, y)
    calibrated = apply_calibrator(FROZEN_CALIBRATION, calibrator, raw)
    errors = ((calibrated >= 0.5).astype(int) != y).astype(int)
    features = np.column_stack((train_eye["q_overall_risk"], entropy(calibrated)))
    gate = fit_gate(features, errors, FROZEN_GATE_C)
    return calibrator, gate, {
        "training_eyes": int(len(train_eye)),
        "training_patients": int(train_eye["patient_id"].nunique()),
        "training_error_events": int(errors.sum()),
    }


def apply_frozen_gate(evaluation_eye, calibrator, gate):
    result = evaluation_eye.copy().reset_index(drop=True)
    raw = result["disease_prob_raw"].to_numpy(dtype=float)
    calibrated = apply_calibrator(FROZEN_CALIBRATION, calibrator, raw)
    result["disease_prob_cal"] = calibrated
    result["y_pred"] = (calibrated >= 0.5).astype(int)
    result["base_error"] = (result["y_pred"] != result["y_true"]).astype(int)
    result["confidence_risk"] = entropy(calibrated)
    result["quality_risk"] = result["q_overall_risk"]
    features = np.column_stack((result["q_overall_risk"], result["confidence_risk"]))
    result["joint_risk"] = gate.predict_proba(features)[:, 1]
    result["severity_protected_joint_risk"] = (
        result["joint_risk"] - FROZEN_PROTECTION_LAMBDA * result["disease_prob_cal"]
    )
    result["random_risk"] = np.random.default_rng(20260817).random(len(result))
    return result


def row_from_summary(summary, metric):
    row = summary[summary["metric"] == metric].iloc[0]
    return {
        "observed": float(row["observed"]),
        "ci95": [float(row["ci95_low"]), float(row["ci95_high"])],
    }


def point_grade_gap(severity, strategy, low_grade=0, high_grade=4):
    selected = severity[
        (severity["strategy"] == strategy) & (severity["group_variable"] == "dr_grade")
    ]
    coverage = {float(row.group): float(row.coverage) for row in selected.itertuples()}
    return coverage.get(float(low_grade), np.nan) - coverage.get(float(high_grade), np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        default="research_mshf/data/external/DeepDRiD/extracted/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/Online-Challenge1&2-Evaluation",
    )
    parser.add_argument(
        "--train-oof", default="research_mshf/outputs/deepdrid_oof_convnext_pretrained/train_oof_predictions.csv"
    )
    parser.add_argument(
        "--train-quality", default="research_mshf/outputs/deepdrid_quality_convnext_pretrained/deepdrid_quality_predictions.csv"
    )
    parser.add_argument(
        "--train-manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv"
    )
    parser.add_argument(
        "--disease-checkpoint", default="research_mshf/outputs/deepdrid_dr_convnext_pretrained/final_convnext_tiny.pt"
    )
    parser.add_argument(
        "--quality-checkpoint", default="research_mshf/outputs/mshf_quality_convnext_pretrained/final_mshf_quality_convnext_tiny.pt"
    )
    parser.add_argument(
        "--cache-dir", default="research_mshf/outputs/cache/deepdrid_official_evaluation_fundus_448"
    )
    parser.add_argument(
        "--output-dir", default="research_mshf/outputs/deepdrid_official_evaluation_frozen"
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--cache-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = build_evaluation_manifest(args.evaluation_root)
    overlap_manifest = pd.read_csv(args.train_manifest)
    overlap = {
        split: {
            "images": int(len(set(manifest["image_id"]) & set(group["image_id"]))),
            "patients": int(len(set(manifest["patient_id"]) & set(group["patient_id"]))),
        }
        for split, group in overlap_manifest.groupby("split")
    }
    if any(item["images"] or item["patients"] for item in overlap.values()):
        raise ValueError(f"Official evaluation overlaps existing development data: {overlap}")
    manifest = prepare_fundus_cache(manifest, args.cache_dir, args.cache_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = infer_both_models(
        manifest,
        args.disease_checkpoint,
        args.quality_checkpoint,
        device,
        args.image_size,
        args.batch_size,
        args.workers,
    )
    predictions.to_csv(output_directory / "official_evaluation_image_model_outputs.csv", index=False)

    image_frame = manifest.merge(predictions, on="image_id", how="inner", validate="one_to_one")
    image_frame["fold"] = -1
    evaluation_eye = aggregate_to_eye(image_frame)
    train_eye = load_training_eyes(args.train_oof, args.train_quality, args.train_manifest)
    calibrator, gate, training_counts = fit_frozen_training_gate(train_eye)
    evaluation = apply_frozen_gate(evaluation_eye, calibrator, gate)
    metrics = strategy_metrics(evaluation)
    severity = severity_table(evaluation)
    bootstrap = bootstrap_patient_clusters(evaluation, args.bootstrap_replicates, args.seed)
    bootstrap_stats = bootstrap_summary(evaluation, bootstrap)

    evaluation.to_csv(output_directory / "official_evaluation_eye_predictions_and_risks.csv", index=False)
    metrics.to_csv(output_directory / "official_evaluation_strategy_metrics.csv", index=False)
    severity.to_csv(output_directory / "official_evaluation_severity_coverage_80.csv", index=False)
    bootstrap.to_csv(output_directory / "official_evaluation_patient_bootstrap_replicates.csv", index=False)
    bootstrap_stats.to_csv(output_directory / "official_evaluation_patient_bootstrap_summary.csv", index=False)
    manifest.to_csv(output_directory / "official_evaluation_manifest.csv", index=False)

    y_image = image_frame["y_true"].to_numpy(dtype=int)
    p_image = image_frame["disease_prob"].to_numpy(dtype=float)
    human_quality = image_frame["human_overall_quality"].to_numpy(dtype=int)
    delta_aurc = row_from_summary(bootstrap_stats, "delta_joint_minus_confidence_aurc_60_100")
    delta_error = row_from_summary(bootstrap_stats, "delta_joint_minus_confidence_error_rate_80")
    delta_fnr = row_from_summary(bootstrap_stats, "delta_joint_minus_confidence_fnr_80")
    joint_grade_gap_row = bootstrap_stats[
        bootstrap_stats["metric"] == "joint_grade0_minus_grade4_gap_80"
    ].iloc[0]
    quality_grade_gap_row = bootstrap_stats[
        bootstrap_stats["metric"] == "quality_grade0_minus_grade4_gap_80"
    ].iloc[0]
    summary = {
        "confirmatory_dataset": "DeepDRiD Online Challenge 1&2 Evaluation",
        "evaluation_seen_before_lock": False,
        "development_overlap": overlap,
        "n_images": int(len(image_frame)),
        "n_eyes": int(len(evaluation)),
        "n_patients": int(evaluation["patient_id"].nunique()),
        "frozen_configuration": {
            "calibration": FROZEN_CALIBRATION,
            "gate_c": FROZEN_GATE_C,
            "features": ["q_overall_risk", "calibrated_entropy"],
            "eye_fusion": "mean disease logit",
            "quality_aggregation": "worst-view q_overall risk",
            "aurc_coverage_range": [0.60, 1.00],
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "training_gate_counts": training_counts,
        "base_disease_image_metrics": {
            "auroc": float(roc_auc_score(y_image, p_image)),
            "auprc": float(average_precision_score(y_image, p_image)),
            "brier": float(brier_score_loss(y_image, p_image)),
        },
        "quality_transfer_image_auroc": float(
            roc_auc_score(human_quality, image_frame["q_overall"])
        ),
        "primary_delta_joint_minus_confidence_aurc_60_100": delta_aurc,
        "secondary_delta_error_rate_80": delta_error,
        "secondary_delta_fnr_80": delta_fnr,
        "joint_grade0_minus_grade4_gap_80": {
            "observed": float(point_grade_gap(severity, "joint_primary")),
            "ci95": [float(joint_grade_gap_row["ci95_low"]), float(joint_grade_gap_row["ci95_high"])],
        },
        "quality_only_grade0_minus_grade4_gap_80": {
            "observed": float(point_grade_gap(severity, "quality_only")),
            "ci95": [float(quality_grade_gap_row["ci95_low"]), float(quality_grade_gap_row["ci95_high"])],
        },
        "h1_primary_supported": bool(delta_aurc["ci95"][1] < 0),
        "h2_error_rate_supported": bool(delta_error["ci95"][1] < 0),
        "h2_fnr_supported": bool(delta_fnr["ci95"][1] <= 0),
    }
    (output_directory / "official_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_directory / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen EyeQ inference and external quality-informed gate evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from optimize_protocol_aligned_gate import (
    apply_calibrator,
    bootstrap_patient_clusters,
    bootstrap_summary,
    entropy,
    fit_calibrator,
    fit_gate,
    load_eye_data,
    nested_oof_selection,
    severity_table,
    strategy_metrics,
)
from train_deepdrid_protocol_model import FundusBorderCrop


QUALITY_DIMENSIONS = ("illumination", "clarity", "contrast", "overall")


class EyeQDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), str(row["image_name"])


def make_disease_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(checkpoint["model"])
    return model


def make_quality_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(QUALITY_DIMENSIONS))
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def infer_models(frame, disease_model, quality_model, device, image_size, batch_size, workers):
    transform = transforms.Compose(
        [
            FundusBorderCrop(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        EyeQDataset(frame, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    disease_model.eval().to(device)
    quality_model.eval().to(device)
    amp_enabled = device.type == "cuda"
    rows = []
    for batch_index, (images, image_names) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            disease_probability = torch.sigmoid(disease_model(images).squeeze(1)).float().cpu().numpy()
            quality_probability = torch.sigmoid(quality_model(images)).float().cpu().numpy()
        for image_name, disease_score, quality_scores in zip(
            image_names, disease_probability, quality_probability
        ):
            row = {"image_name": image_name, "disease_prob_raw": float(disease_score)}
            for dimension, score in zip(QUALITY_DIMENSIONS, quality_scores):
                row[f"q_{dimension}"] = float(score)
            rows.append(row)
        if batch_index % 100 == 0:
            print(json.dumps({"inferred": min(batch_index * batch_size, len(frame)), "total": len(frame)}), flush=True)
    return pd.DataFrame(rows)


def fit_deepdrid_gate(args):
    train_eye, _ = load_eye_data(
        args.deepdrid_train_oof,
        args.deepdrid_validation,
        args.deepdrid_quality,
        args.deepdrid_manifest,
    )
    _, _, calibration, gate_c, protection = nested_oof_selection(train_eye)
    y_train = train_eye["y_true"].to_numpy(dtype=int)
    raw_train = train_eye["disease_prob_raw"].to_numpy(dtype=float)
    calibrator = fit_calibrator(calibration, raw_train, y_train)
    calibrated_train = apply_calibrator(calibration, calibrator, raw_train)
    train_errors = ((calibrated_train >= 0.5).astype(int) != y_train).astype(int)
    train_features = np.column_stack(
        (train_eye["q_overall_risk"].to_numpy(dtype=float), entropy(calibrated_train))
    )
    gate = fit_gate(train_features, train_errors, gate_c)
    return calibration, gate_c, protection, calibrator, gate, train_eye


def apply_frozen_gate(external, calibration, protection, calibrator, gate):
    result = external.copy().reset_index(drop=True)
    result["disease_prob_cal"] = apply_calibrator(
        calibration, calibrator, result["disease_prob_raw"].to_numpy(dtype=float)
    )
    result["y_pred"] = (result["disease_prob_cal"] >= 0.5).astype(int)
    result["base_error"] = (result["y_pred"] != result["y_true"]).astype(int)
    result["q_overall_risk"] = 1.0 - result["q_overall"]
    result["quality_risk"] = result["q_overall_risk"]
    result["confidence_risk"] = entropy(result["disease_prob_cal"])
    gate_features = np.column_stack((result["q_overall_risk"], result["confidence_risk"]))
    result["joint_risk"] = gate.predict_proba(gate_features)[:, 1]
    result["severity_protected_joint_risk"] = result["joint_risk"] - protection * result["disease_prob_cal"]
    result["random_risk"] = np.random.default_rng(20260817).random(len(result))
    return result


def quality_transfer_metrics(frame):
    good = (frame["quality_label"] == "Good").astype(int)
    reject = (frame["quality_label"] == "Reject").astype(int)
    ordinal_badness = frame["quality_label"].map({"Good": 0, "Usable": 1, "Reject": 2})
    return {
        "n": int(len(frame)),
        "quality_label_counts": frame["quality_label"].value_counts().to_dict(),
        "q_overall_auroc_good_vs_other": float(roc_auc_score(good, frame["q_overall"])),
        "q_overall_auroc_reject_vs_other_using_risk": float(roc_auc_score(reject, 1.0 - frame["q_overall"])),
        "q_overall_spearman_with_ordinal_badness": float(
            spearmanr(frame["q_overall"], ordinal_badness, nan_policy="omit").statistic
        ),
    }


def disease_metrics(frame):
    y = frame["y_true"].to_numpy(dtype=int)
    raw = frame["disease_prob_raw"].to_numpy(dtype=float)
    calibrated = frame["disease_prob_cal"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "positive_rate": float(np.mean(y)),
        "raw_auroc": float(roc_auc_score(y, raw)),
        "raw_auprc": float(average_precision_score(y, raw)),
        "raw_brier": float(brier_score_loss(y, raw)),
        "calibrated_brier": float(brier_score_loss(y, calibrated)),
        "calibrated_error_rate_05": float(np.mean((calibrated >= 0.5).astype(int) != y)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eyeq-labels", default="research_mshf/outputs/eyeq_external_validation/eyeq_predictions_with_labels.csv"
    )
    parser.add_argument(
        "--disease-checkpoint",
        default="research_mshf/outputs/deepdrid_dr_convnext_pretrained/final_convnext_tiny.pt",
    )
    parser.add_argument(
        "--quality-checkpoint",
        default="research_mshf/outputs/mshf_quality_convnext_pretrained/final_mshf_quality_convnext_tiny.pt",
    )
    parser.add_argument(
        "--deepdrid-train-oof",
        default="research_mshf/outputs/deepdrid_oof_convnext_pretrained/train_oof_predictions.csv",
    )
    parser.add_argument(
        "--deepdrid-validation",
        default="research_mshf/outputs/deepdrid_dr_convnext_pretrained/validation_predictions.csv",
    )
    parser.add_argument(
        "--deepdrid-quality",
        default="research_mshf/outputs/deepdrid_quality_convnext_pretrained/deepdrid_quality_predictions.csv",
    )
    parser.add_argument(
        "--deepdrid-manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv"
    )
    parser.add_argument(
        "--output-dir", default="research_mshf/outputs/eyeq_protocol_external_convnext"
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.eyeq_labels)[
        ["split", "image_name", "quality", "quality_label", "dr_grade", "referable_dr", "image_path"]
    ].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "eyeq_images": int(len(labels)),
            }
        ),
        flush=True,
    )
    model_output_cache = output_directory / "eyeq_frozen_model_outputs.csv"
    if model_output_cache.exists():
        predictions = pd.read_csv(model_output_cache)
        if len(predictions) != len(labels):
            raise ValueError("Cached EyeQ model-output row count does not match the label manifest")
        print(json.dumps({"reused_model_output_cache": str(model_output_cache)}), flush=True)
    else:
        predictions = infer_models(
            labels,
            make_disease_model(args.disease_checkpoint),
            make_quality_model(args.quality_checkpoint),
            device,
            args.image_size,
            args.batch_size,
            args.workers,
        )
        predictions.to_csv(model_output_cache, index=False)
        print(json.dumps({"saved_model_output_cache": str(model_output_cache)}), flush=True)
    external = labels.merge(predictions, on="image_name", how="inner", validate="one_to_one")
    external["patient_id"] = external["image_name"].str.replace(
        r"_(left|right)\.[^.]+$", "", regex=True
    )
    external["eye"] = external["image_name"].str.extract(r"_(left|right)\.", expand=False)
    external["eye_id"] = external["image_name"]
    external["y_true"] = external["referable_dr"].astype(int)

    calibration, gate_c, protection, calibrator, gate, train_eye = fit_deepdrid_gate(args)
    external = apply_frozen_gate(external, calibration, protection, calibrator, gate)
    metrics = strategy_metrics(external)
    severity = severity_table(external)
    bootstrap = bootstrap_patient_clusters(external, args.bootstrap_replicates, args.seed)
    bootstrap_stats = bootstrap_summary(external, bootstrap)
    quality_metrics = quality_transfer_metrics(external)
    base_metrics = disease_metrics(external)

    external.to_csv(output_directory / "eyeq_frozen_predictions_and_risks.csv", index=False)
    metrics.to_csv(output_directory / "eyeq_strategy_metrics.csv", index=False)
    severity.to_csv(output_directory / "eyeq_severity_coverage_80.csv", index=False)
    bootstrap.to_csv(output_directory / "eyeq_patient_cluster_bootstrap_replicates.csv", index=False)
    bootstrap_stats.to_csv(output_directory / "eyeq_patient_cluster_bootstrap_summary.csv", index=False)
    delta = bootstrap_stats[
        bootstrap_stats["metric"] == "delta_joint_minus_confidence_aurc_60_100"
    ].iloc[0]
    summary = {
        "external_dataset": "EyeQ",
        "target_domain_retuning": False,
        "calibration_fitted_on": "DeepDRiD training OOF",
        "gate_fitted_on": "DeepDRiD training OOF",
        "selected_calibration": calibration,
        "selected_gate_c": gate_c,
        "selected_protection_lambda_sensitivity_only": protection,
        "deepdrid_gate_training_eyes": int(len(train_eye)),
        "quality_transfer": quality_metrics,
        "disease_model": base_metrics,
        "delta_joint_minus_confidence_aurc_60_100": float(delta["observed"]),
        "delta_ci95": [float(delta["ci95_low"]), float(delta["ci95_high"])],
        "external_direction_supported": bool(delta["observed"] < 0),
        "external_statistically_conclusive": bool(delta["ci95_high"] < 0),
    }
    (output_directory / "eyeq_external_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_directory / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

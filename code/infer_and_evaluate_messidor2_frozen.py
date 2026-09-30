#!/usr/bin/env python3
"""Final one-shot frozen external replication on Messidor-2."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from evaluate_deepdrid_official_frozen import (
    FROZEN_CALIBRATION,
    FROZEN_GATE_C,
    FROZEN_PROTECTION_LAMBDA,
    fit_frozen_training_gate,
    load_training_eyes,
    point_grade_gap,
    row_from_summary,
)
from infer_and_evaluate_eyeq_protocol import (
    apply_frozen_gate,
    infer_models,
    make_disease_model,
    make_quality_model,
)
from optimize_protocol_aligned_gate import (
    bootstrap_patient_clusters,
    bootstrap_summary,
    severity_table,
    strategy_metrics,
)


def build_manifest(image_root, labels_path):
    paths = list(Path(image_root).rglob("*.jpg")) + list(Path(image_root).rglob("*.png"))
    by_stem = {path.stem: path for path in paths}
    if len(by_stem) != len(paths):
        raise ValueError("Duplicate Messidor-2 image stems")
    labels = pd.read_csv(labels_path)
    required = {"image_id", "adjudicated_dr_grade", "adjudicated_gradable"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing Messidor-2 columns: {required - set(labels.columns)}")
    labels = labels[labels["adjudicated_dr_grade"].notna()].copy()
    labels["stem"] = labels["image_id"].astype(str).str.rsplit(".", n=1).str[0]
    labels = labels[labels["stem"].isin(by_stem)].copy()
    if len(labels) != 1744:
        raise ValueError(f"Expected 1,744 graded matched images, found {len(labels)}")
    labels["dr_grade"] = labels["adjudicated_dr_grade"].astype(int)
    labels["image_path"] = labels["stem"].map(lambda value: str(by_stem[value]))
    labels["image_name"] = labels["stem"].map(lambda value: by_stem[value].name)
    labels["patient_id"] = labels["stem"]
    labels["eye_id"] = labels["stem"]
    labels["referable_dr"] = (labels["dr_grade"] >= 2).astype(int)
    labels["y_true"] = labels["referable_dr"]
    return labels[
        [
            "image_name",
            "image_path",
            "patient_id",
            "eye_id",
            "dr_grade",
            "referable_dr",
            "y_true",
            "adjudicated_gradable",
        ]
    ].sort_values("image_name").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="research_mshf/data/external/Messidor2_complete")
    parser.add_argument(
        "--labels", default="research_mshf/data/external/Messidor2/messidor_data.csv"
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
        "--output-dir", default="research_mshf/outputs/messidor2_frozen_external_replication"
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.image_root, args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "images": int(len(manifest)),
                "grade_counts": manifest["dr_grade"].value_counts().sort_index().to_dict(),
            }
        ),
        flush=True,
    )

    cache = output / "messidor2_frozen_model_outputs.csv"
    if cache.exists():
        predictions = pd.read_csv(cache)
        if len(predictions) != len(manifest):
            raise ValueError("Cached Messidor-2 output count does not match the manifest")
        print(json.dumps({"reused_model_output_cache": str(cache)}), flush=True)
    else:
        predictions = infer_models(
            manifest,
            make_disease_model(args.disease_checkpoint),
            make_quality_model(args.quality_checkpoint),
            device,
            args.image_size,
            args.batch_size,
            args.workers,
        )
        predictions.to_csv(cache, index=False)
        print(json.dumps({"saved_model_output_cache": str(cache)}), flush=True)

    external = manifest.merge(predictions, on="image_name", how="inner", validate="one_to_one")
    train_eye = load_training_eyes(args.train_oof, args.train_quality, args.train_manifest)
    calibrator, gate, training_counts = fit_frozen_training_gate(train_eye)
    external = apply_frozen_gate(
        external,
        FROZEN_CALIBRATION,
        FROZEN_PROTECTION_LAMBDA,
        calibrator,
        gate,
    )
    metrics = strategy_metrics(external)
    severity = severity_table(external)
    bootstrap = bootstrap_patient_clusters(external, args.bootstrap_replicates, args.seed)
    bootstrap_stats = bootstrap_summary(external, bootstrap)
    external.to_csv(output / "messidor2_frozen_predictions_and_risks.csv", index=False)
    metrics.to_csv(output / "messidor2_strategy_metrics.csv", index=False)
    severity.to_csv(output / "messidor2_severity_coverage_80.csv", index=False)
    bootstrap.to_csv(output / "messidor2_image_bootstrap_replicates.csv", index=False)
    bootstrap_stats.to_csv(output / "messidor2_image_bootstrap_summary.csv", index=False)
    manifest.to_csv(output / "messidor2_manifest.csv", index=False)

    y = external["y_true"].to_numpy(dtype=int)
    raw = external["disease_prob_raw"].to_numpy(dtype=float)
    calibrated = external["disease_prob_cal"].to_numpy(dtype=float)
    delta_aurc = row_from_summary(bootstrap_stats, "delta_joint_minus_confidence_aurc_60_100")
    delta_error = row_from_summary(bootstrap_stats, "delta_joint_minus_confidence_error_rate_80")
    delta_fnr = row_from_summary(bootstrap_stats, "delta_joint_minus_confidence_fnr_80")
    joint_gap = bootstrap_stats[
        bootstrap_stats["metric"] == "joint_grade0_minus_grade4_gap_80"
    ].iloc[0]
    quality_gap = bootstrap_stats[
        bootstrap_stats["metric"] == "quality_grade0_minus_grade4_gap_80"
    ].iloc[0]
    summary = {
        "external_dataset": "Messidor-2 preprocessed derivative with adjudicated grades",
        "target_domain_retuning": False,
        "analysis_unit": "image; verified patient identifiers unavailable",
        "n_images": int(len(external)),
        "grade_counts": {str(k): int(v) for k, v in external["dr_grade"].value_counts().sort_index().items()},
        "frozen_configuration": {
            "calibration": FROZEN_CALIBRATION,
            "gate_c": FROZEN_GATE_C,
            "features": ["q_overall_risk", "calibrated_entropy"],
            "aurc_coverage_range": [0.60, 1.00],
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "training_gate_counts": training_counts,
        "base_disease_metrics": {
            "raw_auroc": float(roc_auc_score(y, raw)),
            "raw_auprc": float(average_precision_score(y, raw)),
            "raw_brier": float(brier_score_loss(y, raw)),
            "calibrated_brier": float(brier_score_loss(y, calibrated)),
            "calibrated_error_rate": float(np.mean((calibrated >= 0.5).astype(int) != y)),
        },
        "primary_delta_joint_minus_confidence_aurc_60_100": delta_aurc,
        "secondary_delta_error_rate_80": delta_error,
        "secondary_delta_fnr_80": delta_fnr,
        "joint_grade0_minus_grade4_gap_80": {
            "observed": float(point_grade_gap(severity, "joint_primary")),
            "ci95": [float(joint_gap["ci95_low"]), float(joint_gap["ci95_high"])],
        },
        "quality_only_grade0_minus_grade4_gap_80": {
            "observed": float(point_grade_gap(severity, "quality_only")),
            "ci95": [float(quality_gap["ci95_low"]), float(quality_gap["ci95_high"])],
        },
        "external_direction_supported": bool(delta_aurc["observed"] < 0),
        "external_statistically_conclusive": bool(delta_aurc["ci95"][1] < 0),
    }
    (output / "messidor2_external_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

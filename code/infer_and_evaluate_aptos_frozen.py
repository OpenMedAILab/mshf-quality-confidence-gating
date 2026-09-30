#!/usr/bin/env python3
"""One-shot frozen external replication on the public APTOS 224px mirror."""

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


GRADE_BY_FOLDER = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}


def build_manifest(root):
    root = Path(root)
    rows = []
    for folder, grade in GRADE_BY_FOLDER.items():
        matches = sorted(root.rglob(f"{folder}/*.png"))
        for path in matches:
            rows.append(
                {
                    "image_name": path.name,
                    "image_path": str(path),
                    "patient_id": path.stem,
                    "eye_id": path.stem,
                    "dr_grade": grade,
                    "referable_dr": int(grade >= 2),
                    "y_true": int(grade >= 2),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No APTOS images found under {root}")
    if frame["image_name"].duplicated().any():
        raise ValueError("APTOS image names are not unique")
    if len(frame) != 3662:
        raise ValueError(f"Expected 3,662 APTOS images, found {len(frame)}")
    return frame.sort_values("image_name").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aptos-root", default="research_mshf/data/external/APTOS_224")
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
        "--output-dir", default="research_mshf/outputs/aptos_frozen_external_replication"
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.aptos_root)
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

    cache = output / "aptos_frozen_model_outputs.csv"
    if cache.exists():
        predictions = pd.read_csv(cache)
        if len(predictions) != len(manifest):
            raise ValueError("Cached APTOS output count does not match the manifest")
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
    external.to_csv(output / "aptos_frozen_predictions_and_risks.csv", index=False)
    metrics.to_csv(output / "aptos_strategy_metrics.csv", index=False)
    severity.to_csv(output / "aptos_severity_coverage_80.csv", index=False)
    bootstrap.to_csv(output / "aptos_image_bootstrap_replicates.csv", index=False)
    bootstrap_stats.to_csv(output / "aptos_image_bootstrap_summary.csv", index=False)
    manifest.to_csv(output / "aptos_manifest.csv", index=False)

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
        "external_dataset": "APTOS 2019 public 224px Gaussian-filtered derivative",
        "target_domain_retuning": False,
        "analysis_unit": "image; patient identifiers unavailable",
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
    (output / "aptos_external_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

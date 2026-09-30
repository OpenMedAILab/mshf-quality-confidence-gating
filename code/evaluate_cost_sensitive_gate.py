#!/usr/bin/env python3
"""Exploratory OOF-trained cost-sensitive gate sensitivity analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optimize_protocol_aligned_gate import (
    apply_calibrator,
    coverage_by_group,
    entropy,
    fit_calibrator,
    full_aurc,
    load_eye_data,
    nested_oof_selection,
    operating_metrics,
    restricted_aurc,
)


STRATEGIES = {
    "confidence_only": "confidence_risk",
    "fn_weighted_5_to_1": "fn_weighted_risk",
    "false_negative_only": "fn_only_risk",
}


def fit_logistic_gate(features, targets, regularization_c, sample_weights=None):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
        ),
    )
    fit_parameters = {}
    if sample_weights is not None:
        fit_parameters["logisticregression__sample_weight"] = sample_weights
    model.fit(features, targets, **fit_parameters)
    return model


def train_cost_gates(train_eye, calibration, gate_c):
    y = train_eye["y_true"].to_numpy(dtype=int)
    raw = train_eye["disease_prob_raw"].to_numpy(dtype=float)
    calibrator = fit_calibrator(calibration, raw, y)
    calibrated = apply_calibrator(calibration, calibrator, raw)
    predicted = (calibrated >= 0.5).astype(int)
    errors = (predicted != y).astype(int)
    false_negatives = ((predicted == 0) & (y == 1)).astype(int)
    features = np.column_stack((train_eye["q_overall_risk"], entropy(calibrated)))
    weights = np.where(false_negatives == 1, 5.0, 1.0)
    weighted_gate = fit_logistic_gate(features, errors, gate_c, weights)
    fn_only_gate = fit_logistic_gate(features, false_negatives, gate_c)
    counts = {
        "training_eyes": int(len(train_eye)),
        "all_errors": int(errors.sum()),
        "false_negatives": int(false_negatives.sum()),
        "false_positives": int(errors.sum() - false_negatives.sum()),
        "false_negative_weight": 5.0,
    }
    return calibrator, weighted_gate, fn_only_gate, counts


def apply_gates(frame, calibration, calibrator, weighted_gate, fn_only_gate):
    result = frame.copy().reset_index(drop=True)
    calibrated = apply_calibrator(
        calibration, calibrator, result["disease_prob_raw"].to_numpy(dtype=float)
    )
    result["disease_prob_cal"] = calibrated
    result["y_pred"] = (calibrated >= 0.5).astype(int)
    result["base_error"] = (result["y_pred"] != result["y_true"]).astype(int)
    result["confidence_risk"] = entropy(calibrated)
    features = np.column_stack((result["q_overall_risk"], result["confidence_risk"]))
    result["fn_weighted_risk"] = weighted_gate.predict_proba(features)[:, 1]
    result["fn_only_risk"] = fn_only_gate.predict_proba(features)[:, 1]
    return result


def make_eyeq_frame(labels_path, model_outputs_path):
    labels = pd.read_csv(labels_path)[
        ["split", "image_name", "quality_label", "dr_grade", "referable_dr"]
    ]
    outputs = pd.read_csv(model_outputs_path)
    frame = labels.merge(outputs, on="image_name", how="inner", validate="one_to_one")
    frame["patient_id"] = frame["image_name"].str.replace(
        r"_(left|right)\.[^.]+$", "", regex=True
    )
    frame["eye"] = frame["image_name"].str.extract(r"_(left|right)\.", expand=False)
    frame["eye_id"] = frame["image_name"]
    frame["y_true"] = frame["referable_dr"].astype(int)
    frame["q_overall_risk"] = 1.0 - frame["q_overall"]
    return frame


def point_metrics(frame):
    errors = frame["base_error"].to_numpy(dtype=int)
    rows = []
    for strategy, risk_column in STRATEGIES.items():
        risk = frame[risk_column].to_numpy(dtype=float)
        row = {
            "strategy": strategy,
            "n": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "error_events": int(errors.sum()),
            "aurc_60_100": restricted_aurc(errors, risk),
            "aurc_full": full_aurc(errors, risk),
            "error_detection_auroc": float(roc_auc_score(errors, risk)),
            "error_detection_auprc": float(average_precision_score(errors, risk)),
        }
        for coverage in (0.70, 0.80, 0.90):
            for key, value in operating_metrics(frame, risk_column, coverage).items():
                row[f"{key}_{int(coverage * 100)}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def group_gap(frame, risk_column, group_column, low_group, high_group):
    values = {
        float(item["group"]): item["coverage"]
        for item in coverage_by_group(frame, risk_column, group_column, 0.80)
    }
    return values.get(float(low_group), np.nan) - values.get(float(high_group), np.nan)


def cluster_bootstrap(frame, replicates, seed):
    patient_ids = frame["patient_id"].drop_duplicates().to_numpy()
    patient_rows = {
        patient: frame.index[frame["patient_id"] == patient].to_numpy() for patient in patient_ids
    }
    rng = np.random.default_rng(seed)
    output = []
    for replicate in range(replicates):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        positions = np.concatenate([patient_rows[patient] for patient in sampled])
        boot = frame.iloc[positions].reset_index(drop=True)
        row = {"replicate": replicate}
        for strategy, risk_column in STRATEGIES.items():
            row[f"{strategy}_aurc_60_100"] = restricted_aurc(
                boot["base_error"], boot[risk_column]
            )
            operating = operating_metrics(boot, risk_column, 0.80)
            row[f"{strategy}_error_rate_80"] = operating["accepted_error_rate"]
            row[f"{strategy}_fnr_80"] = operating["false_negative_rate"]
            row[f"{strategy}_grade0_minus_grade4_gap_80"] = group_gap(
                boot, risk_column, "dr_grade", 0, 4
            )
            row[f"{strategy}_nonref_minus_ref_gap_80"] = group_gap(
                boot, risk_column, "referable_dr", 0, 1
            )
        for strategy in ("fn_weighted_5_to_1", "false_negative_only"):
            row[f"delta_{strategy}_minus_confidence_aurc_60_100"] = (
                row[f"{strategy}_aurc_60_100"] - row["confidence_only_aurc_60_100"]
            )
            row[f"delta_{strategy}_minus_confidence_error_rate_80"] = (
                row[f"{strategy}_error_rate_80"] - row["confidence_only_error_rate_80"]
            )
            row[f"delta_{strategy}_minus_confidence_fnr_80"] = (
                row[f"{strategy}_fnr_80"] - row["confidence_only_fnr_80"]
            )
        output.append(row)
    return pd.DataFrame(output)


def summarize_bootstrap(frame, point, bootstrap):
    point_by_strategy = point.set_index("strategy")
    observed = {}
    for strategy in STRATEGIES:
        observed[f"{strategy}_aurc_60_100"] = point_by_strategy.loc[strategy, "aurc_60_100"]
        observed[f"{strategy}_error_rate_80"] = point_by_strategy.loc[
            strategy, "accepted_error_rate_80"
        ]
        observed[f"{strategy}_fnr_80"] = point_by_strategy.loc[strategy, "false_negative_rate_80"]
    for strategy in ("fn_weighted_5_to_1", "false_negative_only"):
        for metric in ("aurc_60_100", "error_rate_80", "fnr_80"):
            observed[f"delta_{strategy}_minus_confidence_{metric}"] = (
                observed[f"{strategy}_{metric}"] - observed[f"confidence_only_{metric}"]
            )

    rows = []
    for column in bootstrap.columns:
        if column == "replicate":
            continue
        values = bootstrap[column].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": column,
                "observed": observed.get(column, np.nan),
                "bootstrap_mean": float(np.mean(values)) if len(values) else np.nan,
                "ci95_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci95_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "valid_replicates": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def extract_variant_result(summary, variant):
    metrics = {}
    for short in ("aurc_60_100", "error_rate_80", "fnr_80"):
        name = f"delta_{variant}_minus_confidence_{short}"
        row = summary[summary["metric"] == name].iloc[0]
        metrics[short] = {
            "delta": float(row["observed"]),
            "ci95": [float(row["ci95_low"]), float(row["ci95_high"])],
        }
    metrics["observed_pareto_favorable"] = bool(
        metrics["error_rate_80"]["delta"] < 0 and metrics["fnr_80"]["delta"] < 0
    )
    metrics["statistically_pareto_favorable"] = bool(
        metrics["error_rate_80"]["ci95"][1] < 0 and metrics["fnr_80"]["ci95"][1] < 0
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-oof",
        default="research_mshf/outputs/deepdrid_oof_convnext_pretrained/train_oof_predictions.csv",
    )
    parser.add_argument(
        "--validation",
        default="research_mshf/outputs/deepdrid_dr_convnext_pretrained/validation_predictions.csv",
    )
    parser.add_argument(
        "--quality",
        default="research_mshf/outputs/deepdrid_quality_convnext_pretrained/deepdrid_quality_predictions.csv",
    )
    parser.add_argument(
        "--manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv"
    )
    parser.add_argument(
        "--eyeq-labels", default="research_mshf/outputs/eyeq_external_validation/eyeq_predictions_with_labels.csv"
    )
    parser.add_argument(
        "--eyeq-model-outputs",
        default="research_mshf/outputs/eyeq_protocol_external_convnext/eyeq_frozen_model_outputs.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="research_mshf/outputs/optimization_gate_variants/cost_sensitive_gate",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    train_eye, validation_eye = load_eye_data(
        args.train_oof, args.validation, args.quality, args.manifest
    )
    _, _, calibration, gate_c, _ = nested_oof_selection(train_eye)
    calibrator, weighted_gate, fn_only_gate, counts = train_cost_gates(
        train_eye, calibration, gate_c
    )
    deepdrid = apply_gates(
        validation_eye, calibration, calibrator, weighted_gate, fn_only_gate
    )
    eyeq = apply_gates(
        make_eyeq_frame(args.eyeq_labels, args.eyeq_model_outputs),
        calibration,
        calibrator,
        weighted_gate,
        fn_only_gate,
    )

    result = {
        "exploratory_only": True,
        "false_negative_weight": 5.0,
        "features": ["q_overall_risk", "calibrated_entropy"],
        "calibration": calibration,
        "gate_c": gate_c,
        "training_counts": counts,
        "datasets": {},
    }
    for dataset_name, frame in (("deepdrid_validation", deepdrid), ("eyeq_external", eyeq)):
        point = point_metrics(frame)
        bootstrap = cluster_bootstrap(frame, args.bootstrap_replicates, args.seed)
        summary = summarize_bootstrap(frame, point, bootstrap)
        point.to_csv(output_directory / f"{dataset_name}_strategy_metrics.csv", index=False)
        bootstrap.to_csv(output_directory / f"{dataset_name}_bootstrap_replicates.csv", index=False)
        summary.to_csv(output_directory / f"{dataset_name}_bootstrap_summary.csv", index=False)
        result["datasets"][dataset_name] = {
            "fn_weighted_5_to_1": extract_variant_result(summary, "fn_weighted_5_to_1"),
            "false_negative_only": extract_variant_result(summary, "false_negative_only"),
        }
    (output_directory / "cost_sensitive_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Protocol-aligned eye-level evaluation of quality-informed error gating.

Model and hyperparameter selection use DeepDRiD training OOF predictions only.
The validation split is evaluated after selection and is never used to choose a
calibrator, gate regularization strength, or severity-protection parameter.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CALIBRATORS = ("raw", "platt_logit", "platt_prob", "beta", "temperature", "isotonic")
GATE_C_GRID = (0.01, 0.1, 1.0, 10.0)
PROTECTION_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
QUALITY_COLUMNS = ("q_illumination", "q_clarity", "q_contrast", "q_overall")


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(values):
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def sigmoid(values):
    z = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def entropy(values):
    p = clip_prob(values)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def calibrator_features(name, values):
    p = clip_prob(values)
    if name == "platt_logit":
        return logit(p).reshape(-1, 1)
    if name == "platt_prob":
        return p.reshape(-1, 1)
    if name == "beta":
        return np.column_stack((np.log(p), -np.log1p(-p)))
    raise ValueError(f"No feature transform for calibrator {name}")


def fit_calibrator(name, probabilities, labels):
    p = clip_prob(probabilities)
    y = np.asarray(labels, dtype=int)
    if name == "raw":
        return None
    if name in {"platt_logit", "platt_prob", "beta"}:
        model = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
        model.fit(calibrator_features(name, p), y)
        return model
    if name == "temperature":
        logits = logit(p)

        def objective(log_temperature):
            calibrated = sigmoid(logits / np.exp(log_temperature))
            return -np.mean(y * np.log(clip_prob(calibrated)) + (1 - y) * np.log(clip_prob(1 - calibrated)))

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        return float(np.exp(result.x))
    if name == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        model.fit(p, y)
        return model
    raise ValueError(f"Unknown calibrator {name}")


def apply_calibrator(name, model, probabilities):
    p = clip_prob(probabilities)
    if name == "raw":
        return p
    if name in {"platt_logit", "platt_prob", "beta"}:
        return model.predict_proba(calibrator_features(name, p))[:, 1]
    if name == "temperature":
        return sigmoid(logit(p) / model)
    if name == "isotonic":
        return clip_prob(model.predict(p))
    raise ValueError(f"Unknown calibrator {name}")


def restricted_aurc(errors, risks, minimum_coverage=0.60):
    err = np.asarray(errors, dtype=float)
    score = np.asarray(risks, dtype=float)
    order = np.argsort(score, kind="mergesort")
    cumulative = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    coverage = np.arange(1, len(err) + 1) / len(err)
    keep = coverage >= minimum_coverage
    x = coverage[keep]
    y = cumulative[keep]
    if x[0] > minimum_coverage:
        x = np.insert(x, 0, minimum_coverage)
        y = np.insert(y, 0, y[0])
    return float(np.trapezoid(y, x) / (1.0 - minimum_coverage))


def full_aurc(errors, risks):
    err = np.asarray(errors, dtype=float)
    order = np.argsort(np.asarray(risks, dtype=float), kind="mergesort")
    cumulative = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    coverage = np.arange(1, len(err) + 1) / len(err)
    return float(np.trapezoid(cumulative, coverage))


def accepted_indices(risks, coverage):
    n_keep = max(1, int(np.floor(len(risks) * coverage)))
    return np.argsort(np.asarray(risks, dtype=float), kind="mergesort")[:n_keep]


def operating_metrics(frame, risk_column, coverage=0.80):
    accepted = frame.iloc[accepted_indices(frame[risk_column].to_numpy(), coverage)]
    y = accepted["y_true"].to_numpy(dtype=int)
    pred = accepted["y_pred"].to_numpy(dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "accepted_n": int(len(accepted)),
        "accepted_error_rate": float(np.mean(pred != y)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "false_negative_rate": float(fn / (tp + fn)) if tp + fn else np.nan,
        "ppv": float(tp / (tp + fp)) if tp + fp else np.nan,
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
    }


def coverage_by_group(frame, risk_column, group_column, coverage=0.80):
    accepted = np.zeros(len(frame), dtype=bool)
    accepted[accepted_indices(frame[risk_column].to_numpy(), coverage)] = True
    values = []
    for group_value, positions in frame.groupby(group_column, sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        values.append(
            {
                "group_variable": group_column,
                "group": group_value,
                "n": int(len(positions)),
                "accepted_n": int(accepted[positions].sum()),
                "coverage": float(accepted[positions].mean()),
            }
        )
    return values


def aggregate_to_eye(frame):
    work = frame.copy()
    for q in QUALITY_COLUMNS:
        work[f"{q}_risk"] = 1.0 - work[q].astype(float)
    work["raw_logit"] = logit(work["disease_prob"])
    grouped = work.groupby(["patient_id", "eye"], sort=True, observed=True)
    inconsistent = grouped["y_true"].nunique().max()
    if inconsistent > 1:
        raise ValueError("Inconsistent binary labels between views of the same eye")
    eye = grouped.agg(
        y_true=("y_true", "first"),
        raw_logit=("raw_logit", "mean"),
        dr_grade=("dr_grade", "max"),
        referable_dr=("referable_dr", "max"),
        fold=("fold", "first"),
        n_views=("image_id", "size"),
        **{f"{q}_risk": (f"{q}_risk", "max") for q in QUALITY_COLUMNS},
    ).reset_index()
    eye["eye_id"] = eye["patient_id"].astype(str) + "_" + eye["eye"].astype(str)
    eye["disease_prob_raw"] = sigmoid(eye["raw_logit"])
    return eye


def load_eye_data(train_path, validation_path, quality_path, manifest_path):
    quality = pd.read_csv(quality_path)
    quality["image_id"] = quality["image_name"].astype(str).str.replace(".jpg", "", regex=False)
    quality = quality[["image_id", *QUALITY_COLUMNS]]
    manifest = pd.read_csv(manifest_path)
    manifest_core = manifest[["image_id", "patient_id", "eye", "view", "dr_grade", "referable_dr"]]

    train = pd.read_csv(train_path)
    train = train.merge(quality, on="image_id", how="inner", validate="one_to_one")
    train = train.merge(manifest_core[["image_id", "eye", "view"]], on="image_id", how="left", validate="one_to_one")

    validation = pd.read_csv(validation_path)
    validation = validation.merge(quality, on="image_id", how="inner", validate="one_to_one")
    validation = validation.merge(manifest_core, on="image_id", how="left", validate="one_to_one")
    validation["fold"] = -1

    if train["eye"].isna().any() or validation["eye"].isna().any():
        raise ValueError("Some image IDs could not be mapped to an eye")
    return aggregate_to_eye(train), aggregate_to_eye(validation)


def fit_gate(features, errors, regularization_c):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=regularization_c, class_weight="balanced", max_iter=5000, solver="lbfgs"),
    )
    model.fit(features, errors)
    return model


def nested_oof_selection(train_eye):
    y = train_eye["y_true"].to_numpy(dtype=int)
    raw = train_eye["disease_prob_raw"].to_numpy(dtype=float)
    q = train_eye["q_overall_risk"].to_numpy(dtype=float)
    groups = train_eye["patient_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    rows = []
    cache = {}

    for calibration in CALIBRATORS:
        calibrated_cv = np.full(len(train_eye), np.nan)
        joint_by_c = {c: np.full(len(train_eye), np.nan) for c in GATE_C_GRID}
        for fit_index, holdout_index in splitter.split(train_eye, y, groups):
            calibrator = fit_calibrator(calibration, raw[fit_index], y[fit_index])
            p_fit = apply_calibrator(calibration, calibrator, raw[fit_index])
            p_holdout = apply_calibrator(calibration, calibrator, raw[holdout_index])
            calibrated_cv[holdout_index] = p_holdout
            fit_errors = ((p_fit >= 0.5).astype(int) != y[fit_index]).astype(int)
            fit_features = np.column_stack((q[fit_index], entropy(p_fit)))
            holdout_features = np.column_stack((q[holdout_index], entropy(p_holdout)))
            for c_value in GATE_C_GRID:
                gate = fit_gate(fit_features, fit_errors, c_value)
                joint_by_c[c_value][holdout_index] = gate.predict_proba(holdout_features)[:, 1]

        cv_errors = ((calibrated_cv >= 0.5).astype(int) != y).astype(int)
        confidence_risk = entropy(calibrated_cv)
        cv_brier = float(brier_score_loss(y, calibrated_cv))
        for c_value in GATE_C_GRID:
            joint_risk = joint_by_c[c_value]
            row = {
                "calibration": calibration,
                "gate_c": c_value,
                "n_eyes": int(len(train_eye)),
                "n_patients": int(train_eye["patient_id"].nunique()),
                "error_events": int(cv_errors.sum()),
                "cv_brier": cv_brier,
                "cv_confidence_aurc_60_100": restricted_aurc(cv_errors, confidence_risk),
                "cv_joint_aurc_60_100": restricted_aurc(cv_errors, joint_risk),
            }
            row["cv_delta_joint_minus_confidence"] = row["cv_joint_aurc_60_100"] - row["cv_confidence_aurc_60_100"]
            rows.append(row)
            cache[(calibration, c_value)] = (calibrated_cv.copy(), joint_risk.copy(), cv_errors.copy())

    results = pd.DataFrame(rows)
    calibration_scores = results.groupby("calibration", as_index=False)["cv_brier"].first()
    selected_calibration = calibration_scores.sort_values(["cv_brier", "calibration"]).iloc[0]["calibration"]
    selected_row = (
        results[results["calibration"] == selected_calibration]
        .sort_values(["cv_joint_aurc_60_100", "gate_c"])
        .iloc[0]
    )
    selected_c = float(selected_row["gate_c"])
    calibrated_cv, joint_cv, cv_errors = cache[(selected_calibration, selected_c)]

    protection_rows = []
    for protection in PROTECTION_GRID:
        protected = joint_cv - protection * calibrated_cv
        temp = train_eye.copy().reset_index(drop=True)
        temp["protected_risk"] = protected
        grade_cov = {float(item["group"]): item["coverage"] for item in coverage_by_group(temp, "protected_risk", "dr_grade")}
        ref_cov = {float(item["group"]): item["coverage"] for item in coverage_by_group(temp, "protected_risk", "referable_dr")}
        grade_gap = grade_cov.get(0.0, np.nan) - grade_cov.get(4.0, np.nan)
        ref_gap = ref_cov.get(0.0, np.nan) - ref_cov.get(1.0, np.nan)
        protection_rows.append(
            {
                "protection_lambda": protection,
                "cv_aurc_60_100": restricted_aurc(cv_errors, protected),
                "grade0_minus_grade4_coverage_gap_80": grade_gap,
                "nonreferable_minus_referable_coverage_gap_80": ref_gap,
                "meets_safety_constraint": bool(grade_gap <= 0.05 and ref_gap <= 0.05),
            }
        )
    protection_results = pd.DataFrame(protection_rows)
    eligible = protection_results[protection_results["meets_safety_constraint"]]
    if len(eligible):
        selected_protection = float(eligible.sort_values(["cv_aurc_60_100", "protection_lambda"]).iloc[0]["protection_lambda"])
    else:
        protection_results["constraint_violation"] = (
            np.maximum(protection_results["grade0_minus_grade4_coverage_gap_80"] - 0.05, 0.0)
            + np.maximum(protection_results["nonreferable_minus_referable_coverage_gap_80"] - 0.05, 0.0)
        )
        selected_protection = float(
            protection_results.sort_values(["constraint_violation", "cv_aurc_60_100", "protection_lambda"]).iloc[0]["protection_lambda"]
        )
    return results, protection_results, selected_calibration, selected_c, selected_protection


def fit_and_apply_selected(train_eye, validation_eye, calibration, gate_c, protection_lambda):
    y_train = train_eye["y_true"].to_numpy(dtype=int)
    raw_train = train_eye["disease_prob_raw"].to_numpy(dtype=float)
    raw_validation = validation_eye["disease_prob_raw"].to_numpy(dtype=float)
    calibrator = fit_calibrator(calibration, raw_train, y_train)
    p_train = apply_calibrator(calibration, calibrator, raw_train)
    p_validation = apply_calibrator(calibration, calibrator, raw_validation)
    train_errors = ((p_train >= 0.5).astype(int) != y_train).astype(int)
    train_features = np.column_stack((train_eye["q_overall_risk"].to_numpy(), entropy(p_train)))
    validation_features = np.column_stack((validation_eye["q_overall_risk"].to_numpy(), entropy(p_validation)))
    gate = fit_gate(train_features, train_errors, gate_c)

    result = validation_eye.copy().reset_index(drop=True)
    result["disease_prob_cal"] = p_validation
    result["y_pred"] = (p_validation >= 0.5).astype(int)
    result["base_error"] = (result["y_pred"] != result["y_true"]).astype(int)
    result["confidence_risk"] = entropy(p_validation)
    result["quality_risk"] = result["q_overall_risk"]
    result["joint_risk"] = gate.predict_proba(validation_features)[:, 1]
    result["severity_protected_joint_risk"] = result["joint_risk"] - protection_lambda * p_validation
    rng = np.random.default_rng(20260817)
    result["random_risk"] = rng.random(len(result))
    return result


def strategy_metrics(frame):
    strategies = {
        "random": "random_risk",
        "quality_only": "quality_risk",
        "confidence_only": "confidence_risk",
        "joint_primary": "joint_risk",
        "joint_severity_protected_sensitivity": "severity_protected_joint_risk",
    }
    rows = []
    errors = frame["base_error"].to_numpy(dtype=int)
    for strategy, risk_column in strategies.items():
        risk = frame[risk_column].to_numpy(dtype=float)
        row = {
            "strategy": strategy,
            "risk_column": risk_column,
            "n_eyes": int(len(frame)),
            "n_patients": int(frame["patient_id"].nunique()),
            "error_events": int(errors.sum()),
            "aurc_60_100": restricted_aurc(errors, risk),
            "aurc_full": full_aurc(errors, risk),
            "error_detection_auroc": float(roc_auc_score(errors, risk)),
            "error_detection_auprc": float(average_precision_score(errors, risk)),
        }
        for coverage in (0.70, 0.80, 0.90):
            metrics = operating_metrics(frame, risk_column, coverage)
            for key, value in metrics.items():
                row[f"{key}_{int(coverage * 100)}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def severity_table(frame):
    strategies = {
        "quality_only": "quality_risk",
        "confidence_only": "confidence_risk",
        "joint_primary": "joint_risk",
        "joint_severity_protected_sensitivity": "severity_protected_joint_risk",
    }
    rows = []
    for strategy, risk_column in strategies.items():
        for group_column in ("dr_grade", "referable_dr"):
            for item in coverage_by_group(frame, risk_column, group_column):
                item.update({"strategy": strategy, "coverage_target": 0.80})
                rows.append(item)
    return pd.DataFrame(rows)


def bootstrap_patient_clusters(frame, replicates, seed):
    patient_ids = frame["patient_id"].drop_duplicates().to_numpy()
    patient_rows = {patient: frame.index[frame["patient_id"] == patient].to_numpy() for patient in patient_ids}
    rng = np.random.default_rng(seed)
    output = []
    for replicate in range(replicates):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        positions = np.concatenate([patient_rows[patient] for patient in sampled])
        boot = frame.iloc[positions].reset_index(drop=True)
        row = {"replicate": replicate}
        for short, risk_column in (
            ("confidence", "confidence_risk"),
            ("joint", "joint_risk"),
            ("quality", "quality_risk"),
            ("protected", "severity_protected_joint_risk"),
        ):
            row[f"{short}_aurc_60_100"] = restricted_aurc(boot["base_error"], boot[risk_column])
            operating = operating_metrics(boot, risk_column, 0.80)
            row[f"{short}_error_rate_80"] = operating["accepted_error_rate"]
            row[f"{short}_fnr_80"] = operating["false_negative_rate"]
            grade_cov = {float(item["group"]): item["coverage"] for item in coverage_by_group(boot, risk_column, "dr_grade")}
            ref_cov = {float(item["group"]): item["coverage"] for item in coverage_by_group(boot, risk_column, "referable_dr")}
            row[f"{short}_grade0_minus_grade4_gap_80"] = grade_cov.get(0.0, np.nan) - grade_cov.get(4.0, np.nan)
            row[f"{short}_nonref_minus_ref_gap_80"] = ref_cov.get(0.0, np.nan) - ref_cov.get(1.0, np.nan)
        row["delta_joint_minus_confidence_aurc_60_100"] = row["joint_aurc_60_100"] - row["confidence_aurc_60_100"]
        row["delta_joint_minus_confidence_error_rate_80"] = row["joint_error_rate_80"] - row["confidence_error_rate_80"]
        row["delta_joint_minus_confidence_fnr_80"] = row["joint_fnr_80"] - row["confidence_fnr_80"]
        output.append(row)
    return pd.DataFrame(output)


def bootstrap_summary(frame, bootstrap):
    observed = {}
    for short, risk_column in (
        ("confidence", "confidence_risk"),
        ("joint", "joint_risk"),
        ("quality", "quality_risk"),
        ("protected", "severity_protected_joint_risk"),
    ):
        observed[f"{short}_aurc_60_100"] = restricted_aurc(frame["base_error"], frame[risk_column])
        operating = operating_metrics(frame, risk_column, 0.80)
        observed[f"{short}_error_rate_80"] = operating["accepted_error_rate"]
        observed[f"{short}_fnr_80"] = operating["false_negative_rate"]
    observed["delta_joint_minus_confidence_aurc_60_100"] = observed["joint_aurc_60_100"] - observed["confidence_aurc_60_100"]
    observed["delta_joint_minus_confidence_error_rate_80"] = observed["joint_error_rate_80"] - observed["confidence_error_rate_80"]
    observed["delta_joint_minus_confidence_fnr_80"] = observed["joint_fnr_80"] - observed["confidence_fnr_80"]

    rows = []
    for column in bootstrap.columns:
        if column == "replicate":
            continue
        values = bootstrap[column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            rows.append(
                {
                    "metric": column,
                    "observed": observed.get(column, np.nan),
                    "bootstrap_mean": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "valid_replicates": 0,
                }
            )
            continue
        rows.append(
            {
                "metric": column,
                "observed": observed.get(column, np.nan),
                "bootstrap_mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "valid_replicates": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-oof", default="research_mshf/outputs/deepdrid_oof/train_oof_predictions.csv")
    parser.add_argument("--validation", default="research_mshf/outputs/deepdrid_dr/validation_predictions.csv")
    parser.add_argument("--quality", default="research_mshf/outputs/deepdrid_quality/deepdrid_quality_predictions.csv")
    parser.add_argument("--manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv")
    parser.add_argument("--output-dir", default="research_mshf/outputs/optimization_gate_variants/protocol_aligned_eye_level")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_eye, validation_eye = load_eye_data(args.train_oof, args.validation, args.quality, args.manifest)
    cv_results, protection_results, calibration, gate_c, protection = nested_oof_selection(train_eye)
    validation = fit_and_apply_selected(train_eye, validation_eye, calibration, gate_c, protection)
    validation_metrics = strategy_metrics(validation)
    severity = severity_table(validation)
    bootstrap = bootstrap_patient_clusters(validation, args.bootstrap_replicates, args.seed)
    bootstrap_stats = bootstrap_summary(validation, bootstrap)

    cv_results.to_csv(output_dir / "nested_oof_selection.csv", index=False)
    protection_results.to_csv(output_dir / "oof_severity_protection_selection.csv", index=False)
    validation.to_csv(output_dir / "validation_eye_predictions_and_risks.csv", index=False)
    validation_metrics.to_csv(output_dir / "validation_strategy_metrics.csv", index=False)
    severity.to_csv(output_dir / "validation_severity_coverage_80.csv", index=False)
    bootstrap.to_csv(output_dir / "patient_cluster_bootstrap_replicates.csv", index=False)
    bootstrap_stats.to_csv(output_dir / "patient_cluster_bootstrap_summary.csv", index=False)

    delta_row = bootstrap_stats[bootstrap_stats["metric"] == "delta_joint_minus_confidence_aurc_60_100"].iloc[0]
    selection = {
        "analysis_unit": "eye",
        "view_fusion": "mean raw disease logit",
        "quality_aggregation": "worst-view q_overall risk",
        "primary_coverage_range": [0.60, 1.00],
        "calibrator_selection_criterion": "minimum nested patient-grouped OOF Brier score",
        "selected_calibration": calibration,
        "selected_gate_c": gate_c,
        "selected_severity_protection_lambda_sensitivity_only": protection,
        "train_eyes": int(len(train_eye)),
        "train_patients": int(train_eye["patient_id"].nunique()),
        "validation_eyes": int(len(validation)),
        "validation_patients": int(validation["patient_id"].nunique()),
        "validation_error_events": int(validation["base_error"].sum()),
        "primary_delta_joint_minus_confidence_aurc_60_100": float(delta_row["observed"]),
        "primary_delta_ci95": [float(delta_row["ci95_low"]), float(delta_row["ci95_high"])],
        "primary_direction_supported": bool(delta_row["observed"] < 0),
        "primary_statistically_conclusive": bool(delta_row["ci95_high"] < 0),
        "confirmatory_warning": "DeepDRiD validation has been inspected in prior iterations; an untouched evaluation set is still required for a confirmatory claim.",
    }
    (output_dir / "selection_and_primary_result.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

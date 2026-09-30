#!/usr/bin/env python3
"""Correct the protocol gate to use all four MSHF quality dimensions.

All selection in this script is restricted to DeepDRiD training OOF disease
predictions. No validation, official-evaluation, or EyeQ labels are loaded.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optimize_protocol_aligned_gate import (
    apply_calibrator,
    entropy,
    fit_calibrator,
    logit,
    restricted_aurc,
)


QUALITY_COLUMNS = ("q_illumination", "q_clarity", "q_contrast", "q_overall")
FEATURE_SETS = {
    "overall": ("entropy", "q_overall_risk"),
    "multidimensional": (
        "entropy",
        "q_illumination_risk",
        "q_clarity_risk",
        "q_contrast_risk",
        "q_overall_risk",
    ),
    "multidimensional_view": (
        "entropy",
        "q_illumination_risk",
        "q_clarity_risk",
        "q_contrast_risk",
        "q_overall_risk",
        "disease_logit_range",
        "q_overall_view_range",
    ),
}
C_GRID = (0.001, 0.01, 0.1, 1.0)


def fit_gate(x, errors, c_value):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=20260817,
        ),
    )
    model.fit(x, errors)
    return model


def load_training_eyes(oof_path, quality_path, manifest_path):
    oof = pd.read_csv(oof_path)
    quality = pd.read_csv(quality_path)
    quality["image_id"] = quality["image_name"].astype(str).str.replace(".jpg", "", regex=False)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[["image_id", "eye", "view"]]

    image = oof.merge(
        quality[["image_id", *QUALITY_COLUMNS]], on="image_id", how="inner", validate="one_to_one"
    ).merge(manifest, on="image_id", how="left", validate="one_to_one")
    if len(image) != len(oof) or image["eye"].isna().any():
        raise ValueError("Incomplete OOF-quality-manifest merge")
    image["raw_logit"] = logit(image["disease_prob"].to_numpy())
    for column in QUALITY_COLUMNS:
        image[f"{column}_risk"] = 1.0 - image[column].astype(float)

    rows = []
    for (patient_id, eye_name), group in image.groupby(["patient_id", "eye"], sort=True):
        if group["y_true"].nunique() != 1:
            raise ValueError(f"Inconsistent eye labels for {patient_id}/{eye_name}")
        row = {
            "patient_id": patient_id,
            "eye": eye_name,
            "eye_id": f"{patient_id}_{eye_name}",
            "y_true": int(group["y_true"].iloc[0]),
            "dr_grade": int(group["dr_grade"].max()),
            "referable_dr": int(group["referable_dr"].max()),
            "fold": int(group["fold"].iloc[0]),
            "n_views": int(len(group)),
            "raw_logit": float(group["raw_logit"].mean()),
            "disease_logit_range": float(group["raw_logit"].max() - group["raw_logit"].min()),
            "q_overall_view_range": float(
                group["q_overall_risk"].max() - group["q_overall_risk"].min()
            ),
        }
        for column in QUALITY_COLUMNS:
            row[f"{column}_risk"] = float(group[f"{column}_risk"].max())
        rows.append(row)
    eyes = pd.DataFrame(rows)
    eyes["disease_prob_raw"] = 1.0 / (1.0 + np.exp(-np.clip(eyes["raw_logit"], -40, 40)))
    return eyes


def select_c_nested(frame, feature_columns):
    y = frame["y_true"].to_numpy(dtype=int)
    raw = frame["disease_prob_raw"].to_numpy(dtype=float)
    groups = frame["patient_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    risks = {c_value: np.full(len(frame), np.nan) for c_value in C_GRID}
    errors = np.full(len(frame), -1, dtype=int)

    for fit_index, holdout_index in splitter.split(frame, y, groups):
        calibrator = fit_calibrator("beta", raw[fit_index], y[fit_index])
        p_fit = apply_calibrator("beta", calibrator, raw[fit_index])
        p_holdout = apply_calibrator("beta", calibrator, raw[holdout_index])
        fit_errors = ((p_fit >= 0.5).astype(int) != y[fit_index]).astype(int)
        errors[holdout_index] = ((p_holdout >= 0.5).astype(int) != y[holdout_index]).astype(int)

        fit_frame = frame.iloc[fit_index].copy()
        holdout_frame = frame.iloc[holdout_index].copy()
        fit_frame["entropy"] = entropy(p_fit)
        holdout_frame["entropy"] = entropy(p_holdout)
        for c_value in C_GRID:
            gate = fit_gate(fit_frame[list(feature_columns)], fit_errors, c_value)
            risks[c_value][holdout_index] = gate.predict_proba(
                holdout_frame[list(feature_columns)]
            )[:, 1]

    scores = {c_value: restricted_aurc(errors, risk) for c_value, risk in risks.items()}
    selected_c = min(scores, key=lambda value: (scores[value], value))
    return selected_c, scores


def outer_nested_predictions(eyes):
    y = eyes["y_true"].to_numpy(dtype=int)
    raw = eyes["disease_prob_raw"].to_numpy(dtype=float)
    groups = eyes["patient_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    output = eyes.copy()
    output["calibrated_probability"] = np.nan
    output["error"] = -1
    output["risk_confidence"] = np.nan
    for name in FEATURE_SETS:
        output[f"risk_{name}"] = np.nan
    selected_rows = []

    for outer_fold, (fit_index, holdout_index) in enumerate(splitter.split(eyes, y, groups), start=1):
        fit_frame = eyes.iloc[fit_index].reset_index(drop=True)
        holdout_frame = eyes.iloc[holdout_index].copy()
        calibrator = fit_calibrator(
            "beta",
            fit_frame["disease_prob_raw"].to_numpy(),
            fit_frame["y_true"].to_numpy(dtype=int),
        )
        p_fit = apply_calibrator("beta", calibrator, fit_frame["disease_prob_raw"].to_numpy())
        p_holdout = apply_calibrator(
            "beta", calibrator, holdout_frame["disease_prob_raw"].to_numpy()
        )
        fit_errors = (
            (p_fit >= 0.5).astype(int) != fit_frame["y_true"].to_numpy(dtype=int)
        ).astype(int)
        holdout_errors = (
            (p_holdout >= 0.5).astype(int) != holdout_frame["y_true"].to_numpy(dtype=int)
        ).astype(int)
        fit_frame["entropy"] = entropy(p_fit)
        holdout_frame["entropy"] = entropy(p_holdout)

        output.loc[output.index[holdout_index], "calibrated_probability"] = p_holdout
        output.loc[output.index[holdout_index], "error"] = holdout_errors
        output.loc[output.index[holdout_index], "risk_confidence"] = entropy(p_holdout)

        for name, columns in FEATURE_SETS.items():
            selected_c, inner_scores = select_c_nested(fit_frame, columns)
            gate = fit_gate(fit_frame[list(columns)], fit_errors, selected_c)
            risk = gate.predict_proba(holdout_frame[list(columns)])[:, 1]
            output.loc[output.index[holdout_index], f"risk_{name}"] = risk
            selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": name,
                    "selected_c": selected_c,
                    **{f"inner_aurc_c_{c_value}": score for c_value, score in inner_scores.items()},
                }
            )
    output["error"] = output["error"].astype(int)
    return output, pd.DataFrame(selected_rows)


def cluster_bootstrap(frame, replicates, seed):
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].unique()
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = []
        for draw, patient in enumerate(sampled):
            patient_indices = frame.index[frame["patient_id"] == patient].tolist()
            indices.extend(patient_indices)
        boot = frame.loc[indices]
        confidence = restricted_aurc(boot["error"], boot["risk_confidence"])
        overall = restricted_aurc(boot["error"], boot["risk_overall"])
        multidimensional = restricted_aurc(boot["error"], boot["risk_multidimensional"])
        rows.append(
            {
                "replicate": replicate,
                "confidence_aurc": confidence,
                "overall_aurc": overall,
                "multidimensional_aurc": multidimensional,
                "delta_multidimensional_minus_confidence": multidimensional - confidence,
                "delta_multidimensional_minus_overall": multidimensional - overall,
            }
        )
    return pd.DataFrame(rows)


def h3_joint_test(eyes):
    calibrator = fit_calibrator(
        "beta", eyes["disease_prob_raw"].to_numpy(), eyes["y_true"].to_numpy(dtype=int)
    )
    p = apply_calibrator("beta", calibrator, eyes["disease_prob_raw"].to_numpy())
    errors = ((p >= 0.5).astype(int) != eyes["y_true"].to_numpy(dtype=int)).astype(int)
    columns = [f"{column}_risk" for column in QUALITY_COLUMNS]
    x_raw = eyes[columns].copy()
    x_raw.insert(0, "entropy", entropy(p))
    x = (x_raw - x_raw.mean()) / x_raw.std(ddof=0)
    x = sm.add_constant(x, has_constant="add")
    model = sm.GLM(errors, x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": eyes["patient_id"].to_numpy()}
    )
    restriction = np.zeros((4, len(model.params)))
    parameter_names = list(model.params.index)
    for row, column in enumerate(columns):
        restriction[row, parameter_names.index(column)] = 1.0
    joint = model.wald_test(restriction, scalar=True)
    estimates = {}
    for column in ["entropy", *columns]:
        coefficient = float(model.params[column])
        standard_error = float(model.bse[column])
        estimates[column] = {
            "odds_ratio_per_sd": float(np.exp(coefficient)),
            "ci95": [
                float(np.exp(coefficient - 1.96 * standard_error)),
                float(np.exp(coefficient + 1.96 * standard_error)),
            ],
            "p_value": float(model.pvalues[column]),
        }
    return {
        "n_eyes": int(len(eyes)),
        "error_events": int(errors.sum()),
        "joint_quality_wald_statistic": float(joint.statistic),
        "joint_quality_df": 4,
        "joint_quality_p_value": float(joint.pvalue),
        "coefficients": estimates,
    }


def fit_frozen_model(eyes, selected_rows):
    y = eyes["y_true"].to_numpy(dtype=int)
    raw = eyes["disease_prob_raw"].to_numpy(dtype=float)
    calibrator = fit_calibrator("beta", raw, y)
    p = apply_calibrator("beta", calibrator, raw)
    errors = ((p >= 0.5).astype(int) != y).astype(int)
    train = eyes.copy()
    train["entropy"] = entropy(p)
    models = {}
    selected_c = {}
    for name, columns in FEATURE_SETS.items():
        choices = selected_rows.loc[selected_rows["model"] == name, "selected_c"].tolist()
        c_value = sorted(Counter(choices).items(), key=lambda item: (-item[1], item[0]))[0][0]
        models[name] = fit_gate(train[list(columns)], errors, c_value)
        selected_c[name] = c_value
    return {
        "calibration": "beta",
        "calibrator": calibrator,
        "feature_sets": FEATURE_SETS,
        "selected_c": selected_c,
        "models": models,
        "view_fusion": "mean_raw_logit",
        "quality_aggregation": "worst_view_per_dimension",
        "seed": 20260817,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof", default="research_mshf/outputs/deepdrid_oof_convnext_pretrained/train_oof_predictions.csv"
    )
    parser.add_argument(
        "--quality",
        default="research_mshf/outputs/deepdrid_quality_convnext_pretrained/deepdrid_quality_predictions.csv",
    )
    parser.add_argument(
        "--manifest", default="research_mshf/outputs/manifests/deepdrid_regular_manifest.csv"
    )
    parser.add_argument(
        "--output", default="research_mshf/outputs/optimization_gate_variants/multidimensional_protocol_correction"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eyes = load_training_eyes(args.oof, args.quality, args.manifest)
    nested, selections = outer_nested_predictions(eyes)
    bootstrap = cluster_bootstrap(nested, args.bootstrap, args.seed)
    h3 = h3_joint_test(eyes)

    risk_columns = {
        "confidence": "risk_confidence",
        "overall": "risk_overall",
        "multidimensional": "risk_multidimensional",
        "multidimensional_view": "risk_multidimensional_view",
    }
    metrics = {}
    for name, column in risk_columns.items():
        metrics[name] = {
            "oof_aurc_60_100": restricted_aurc(nested["error"], nested[column]),
        }
    metrics["overall"]["delta_vs_confidence"] = (
        metrics["overall"]["oof_aurc_60_100"] - metrics["confidence"]["oof_aurc_60_100"]
    )
    for name in ("multidimensional", "multidimensional_view"):
        metrics[name]["delta_vs_confidence"] = (
            metrics[name]["oof_aurc_60_100"] - metrics["confidence"]["oof_aurc_60_100"]
        )
        metrics[name]["delta_vs_overall"] = (
            metrics[name]["oof_aurc_60_100"] - metrics["overall"]["oof_aurc_60_100"]
        )

    summary = {
        "development_only": True,
        "n_eyes": int(len(nested)),
        "n_patients": int(nested["patient_id"].nunique()),
        "error_events": int(nested["error"].sum()),
        "metrics": metrics,
        "bootstrap_ci95": {
            column: [float(bootstrap[column].quantile(0.025)), float(bootstrap[column].quantile(0.975))]
            for column in (
                "delta_multidimensional_minus_confidence",
                "delta_multidimensional_minus_overall",
            )
        },
        "advance_multidimensional": bool(
            metrics["multidimensional"]["oof_aurc_60_100"]
            < min(
                metrics["confidence"]["oof_aurc_60_100"],
                metrics["overall"]["oof_aurc_60_100"],
            )
        ),
        "h3_joint_quality": h3,
        "integrity_note": "No holdout or EyeQ labels were loaded by this script.",
    }
    frozen = fit_frozen_model(eyes, selections)

    nested.to_csv(output / "nested_outer_oof_predictions.csv", index=False)
    selections.to_csv(output / "nested_regularization_selection.csv", index=False)
    bootstrap.to_csv(output / "patient_cluster_bootstrap.csv", index=False)
    (output / "development_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(frozen, output / "frozen_multidimensional_gate.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

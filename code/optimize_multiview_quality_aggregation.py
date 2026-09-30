#!/usr/bin/env python3
"""Nested-OOF comparison of eye-level quality aggregation rules."""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from optimize_multidimensional_gate import fit_gate
from optimize_protocol_aligned_gate import (
    apply_calibrator,
    entropy,
    fit_calibrator,
    logit,
    restricted_aurc,
)


C_GRID = (0.001, 0.01, 0.1, 1.0)
FEATURE_SETS = {
    "worst": ("entropy", "q_worst"),
    "mean": ("entropy", "q_mean"),
    "both_poor": ("entropy", "q_both_poor"),
    "worst_best": ("entropy", "q_worst", "q_both_poor"),
    "decisive": ("entropy", "q_decisive"),
    "uncertainty_weighted": ("entropy", "q_uncertainty_weighted"),
}


def load_eyes(oof_path, quality_path, manifest_path):
    oof = pd.read_csv(oof_path)
    quality = pd.read_csv(quality_path)
    quality["image_id"] = quality["image_name"].astype(str).str.replace(".jpg", "", regex=False)
    manifest = pd.read_csv(manifest_path)[["image_id", "eye", "view"]]
    image = oof.merge(
        quality[["image_id", "q_overall"]], on="image_id", how="inner", validate="one_to_one"
    ).merge(manifest, on="image_id", how="left", validate="one_to_one")
    if len(image) != len(oof) or image["eye"].isna().any():
        raise ValueError("Incomplete OOF-quality-manifest merge")
    image["raw_logit"] = logit(image["disease_prob"].to_numpy())
    image["view_entropy"] = entropy(image["disease_prob"].to_numpy())
    image["q_risk"] = 1.0 - image["q_overall"].astype(float)

    rows = []
    for (patient_id, eye_name), group in image.groupby(["patient_id", "eye"], sort=True):
        if group["y_true"].nunique() != 1:
            raise ValueError(f"Inconsistent eye labels for {patient_id}/{eye_name}")
        q = group["q_risk"].to_numpy(dtype=float)
        logits = group["raw_logit"].to_numpy(dtype=float)
        view_entropy = group["view_entropy"].to_numpy(dtype=float)
        weights = view_entropy + 1e-6
        decisive_index = int(np.argmax(np.abs(logits)))
        mean_logit = float(logits.mean())
        rows.append(
            {
                "patient_id": patient_id,
                "eye": eye_name,
                "eye_id": f"{patient_id}_{eye_name}",
                "y_true": int(group["y_true"].iloc[0]),
                "dr_grade": int(group["dr_grade"].max()),
                "referable_dr": int(group["referable_dr"].max()),
                "fold": int(group["fold"].iloc[0]),
                "n_views": int(len(group)),
                "disease_prob_raw": float(1.0 / (1.0 + np.exp(-np.clip(mean_logit, -40, 40)))),
                "q_worst": float(q.max()),
                "q_mean": float(q.mean()),
                "q_both_poor": float(q.min()),
                "q_decisive": float(q[decisive_index]),
                "q_uncertainty_weighted": float(np.average(q, weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def select_c(frame, feature_columns):
    y = frame["y_true"].to_numpy(dtype=int)
    raw = frame["disease_prob_raw"].to_numpy(dtype=float)
    groups = frame["patient_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    errors = np.full(len(frame), -1, dtype=int)
    risks = {c: np.full(len(frame), np.nan) for c in C_GRID}
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
        for c in C_GRID:
            gate = fit_gate(fit_frame[list(feature_columns)], fit_errors, c)
            risks[c][holdout_index] = gate.predict_proba(
                holdout_frame[list(feature_columns)]
            )[:, 1]
    scores = {c: restricted_aurc(errors, risk) for c, risk in risks.items()}
    selected = min(scores, key=lambda c: (scores[c], c))
    return selected, scores


def nested_outer(eyes):
    y = eyes["y_true"].to_numpy(dtype=int)
    raw = eyes["disease_prob_raw"].to_numpy(dtype=float)
    groups = eyes["patient_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    result = eyes.copy()
    result["error"] = -1
    result["calibrated_probability"] = np.nan
    result["risk_confidence"] = np.nan
    for name in FEATURE_SETS:
        result[f"risk_{name}"] = np.nan
    selection_rows = []

    for outer_fold, (fit_index, holdout_index) in enumerate(splitter.split(eyes, y, groups), start=1):
        fit_frame = eyes.iloc[fit_index].reset_index(drop=True)
        holdout_frame = eyes.iloc[holdout_index].copy()
        calibrator = fit_calibrator(
            "beta", fit_frame["disease_prob_raw"], fit_frame["y_true"].astype(int)
        )
        p_fit = apply_calibrator("beta", calibrator, fit_frame["disease_prob_raw"])
        p_holdout = apply_calibrator("beta", calibrator, holdout_frame["disease_prob_raw"])
        fit_errors = (
            (p_fit >= 0.5).astype(int) != fit_frame["y_true"].to_numpy(dtype=int)
        ).astype(int)
        holdout_errors = (
            (p_holdout >= 0.5).astype(int) != holdout_frame["y_true"].to_numpy(dtype=int)
        ).astype(int)
        fit_frame["entropy"] = entropy(p_fit)
        holdout_frame["entropy"] = entropy(p_holdout)
        index = result.index[holdout_index]
        result.loc[index, "error"] = holdout_errors
        result.loc[index, "calibrated_probability"] = p_holdout
        result.loc[index, "risk_confidence"] = entropy(p_holdout)

        for name, columns in FEATURE_SETS.items():
            selected_c, scores = select_c(fit_frame, columns)
            gate = fit_gate(fit_frame[list(columns)], fit_errors, selected_c)
            result.loc[index, f"risk_{name}"] = gate.predict_proba(
                holdout_frame[list(columns)]
            )[:, 1]
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "aggregation": name,
                    "selected_c": selected_c,
                    **{f"inner_aurc_c_{c}": score for c, score in scores.items()},
                }
            )
    result["error"] = result["error"].astype(int)
    return result, pd.DataFrame(selection_rows)


def bootstrap_differences(frame, replicates, seed):
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].unique()
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = []
        for patient in sampled:
            indices.extend(frame.index[frame["patient_id"] == patient].tolist())
        boot = frame.loc[indices]
        confidence = restricted_aurc(boot["error"], boot["risk_confidence"])
        row = {"replicate": replicate, "confidence_aurc": confidence}
        for name in FEATURE_SETS:
            value = restricted_aurc(boot["error"], boot[f"risk_{name}"])
            row[f"{name}_aurc"] = value
            row[f"delta_{name}_minus_confidence"] = value - confidence
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final(eyes, selections, aggregation):
    y = eyes["y_true"].to_numpy(dtype=int)
    calibrator = fit_calibrator("beta", eyes["disease_prob_raw"], y)
    p = apply_calibrator("beta", calibrator, eyes["disease_prob_raw"])
    errors = ((p >= 0.5).astype(int) != y).astype(int)
    train = eyes.copy()
    train["entropy"] = entropy(p)
    choices = selections.loc[
        selections["aggregation"] == aggregation, "selected_c"
    ].tolist()
    c_value = sorted(Counter(choices).items(), key=lambda item: (-item[1], item[0]))[0][0]
    columns = FEATURE_SETS[aggregation]
    gate = fit_gate(train[list(columns)], errors, c_value)
    return {
        "calibration": "beta",
        "calibrator": calibrator,
        "aggregation": aggregation,
        "features": columns,
        "selected_c": c_value,
        "gate": gate,
        "view_fusion": "mean_raw_logit",
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
        "--output", default="research_mshf/outputs/optimization_gate_variants/multiview_aggregation"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eyes = load_eyes(args.oof, args.quality, args.manifest)
    nested, selections = nested_outer(eyes)
    bootstrap = bootstrap_differences(nested, args.bootstrap, args.seed)

    confidence = restricted_aurc(nested["error"], nested["risk_confidence"])
    metrics = {
        name: {
            "oof_aurc_60_100": restricted_aurc(nested["error"], nested[f"risk_{name}"])
        }
        for name in FEATURE_SETS
    }
    for name in metrics:
        metrics[name]["delta_vs_confidence"] = metrics[name]["oof_aurc_60_100"] - confidence
        column = f"delta_{name}_minus_confidence"
        metrics[name]["delta_ci95"] = [
            float(bootstrap[column].quantile(0.025)),
            float(bootstrap[column].quantile(0.975)),
        ]
    best = min(metrics, key=lambda name: metrics[name]["oof_aurc_60_100"])
    advance = bool(metrics[best]["delta_ci95"][1] < 0)
    summary = {
        "development_only": True,
        "n_eyes": int(len(nested)),
        "n_patients": int(nested["patient_id"].nunique()),
        "error_events": int(nested["error"].sum()),
        "confidence_aurc_60_100": confidence,
        "aggregations": metrics,
        "best_aggregation": best,
        "advance": advance,
        "integrity_note": "No holdout or external labels were loaded.",
    }
    nested.to_csv(output / "nested_outer_oof_predictions.csv", index=False)
    selections.to_csv(output / "nested_regularization_selection.csv", index=False)
    bootstrap.to_csv(output / "patient_cluster_bootstrap.csv", index=False)
    (output / "development_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if advance:
        joblib.dump(fit_final(eyes, selections, best), output / "frozen_best_aggregation_gate.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

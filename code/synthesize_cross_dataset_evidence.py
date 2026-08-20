#!/usr/bin/env python3
"""Pre-specified synthesis of four frozen quality-gate evaluations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research_mshf/outputs")
DATASETS = {
    "DeepDRiD official evaluation": {
        "bootstrap": ROOT / "deepdrid_official_evaluation_frozen/official_evaluation_patient_bootstrap_replicates.csv",
        "summary": ROOT / "deepdrid_official_evaluation_frozen/official_evaluation_patient_bootstrap_summary.csv",
        "n": 200,
        "unit": "eye; patient-cluster bootstrap",
    },
    "EyeQ": {
        "bootstrap": ROOT / "eyeq_protocol_external_convnext/eyeq_patient_cluster_bootstrap_replicates.csv",
        "summary": ROOT / "eyeq_protocol_external_convnext/eyeq_patient_cluster_bootstrap_summary.csv",
        "n": 28792,
        "unit": "image; patient-cluster bootstrap",
    },
    "APTOS": {
        "bootstrap": ROOT / "aptos_frozen_external_replication/aptos_image_bootstrap_replicates.csv",
        "summary": ROOT / "aptos_frozen_external_replication/aptos_image_bootstrap_summary.csv",
        "n": 3662,
        "unit": "image bootstrap",
    },
    "Messidor-2": {
        "bootstrap": ROOT / "messidor2_frozen_external_replication/messidor2_image_bootstrap_replicates.csv",
        "summary": ROOT / "messidor2_frozen_external_replication/messidor2_image_bootstrap_summary.csv",
        "n": 1744,
        "unit": "image bootstrap",
    },
}


def summary_metric(frame, metric):
    row = frame.loc[frame["metric"] == metric].iloc[0]
    return {
        "observed": float(row["observed"]),
        "ci95_low": float(row["ci95_low"]),
        "ci95_high": float(row["ci95_high"]),
    }


def main():
    output = ROOT / "cross_dataset_frozen_synthesis"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260817)
    draws = 10000
    delta_column = "delta_joint_minus_confidence_aurc_60_100"
    dataset_rows = []
    sampled_distributions = []
    variances = []
    effects = []

    for name, files in DATASETS.items():
        bootstrap = pd.read_csv(files["bootstrap"])
        summary = pd.read_csv(files["summary"])
        delta = summary_metric(summary, delta_column)
        quality_gap = summary_metric(summary, "quality_grade0_minus_grade4_gap_80")
        joint_gap = summary_metric(summary, "joint_grade0_minus_grade4_gap_80")
        distribution = bootstrap[delta_column].dropna().to_numpy(dtype=float)
        sampled = rng.choice(distribution, size=draws, replace=True)
        variance = float(np.var(distribution, ddof=1))
        effects.append(delta["observed"])
        variances.append(variance)
        sampled_distributions.append(sampled)
        dataset_rows.append(
            {
                "dataset": name,
                "n_analysis_units": files["n"],
                "uncertainty_unit": files["unit"],
                "delta_aurc": delta["observed"],
                "delta_aurc_ci95_low": delta["ci95_low"],
                "delta_aurc_ci95_high": delta["ci95_high"],
                "negative_direction": delta["observed"] < 0,
                "individually_conclusive": delta["ci95_high"] < 0,
                "quality_grade0_minus_grade4_gap": quality_gap["observed"],
                "quality_gap_ci95_low": quality_gap["ci95_low"],
                "quality_gap_ci95_high": quality_gap["ci95_high"],
                "joint_grade0_minus_grade4_gap": joint_gap["observed"],
                "joint_gap_ci95_low": joint_gap["ci95_low"],
                "joint_gap_ci95_high": joint_gap["ci95_high"],
            }
        )

    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    sampled_matrix = np.vstack(sampled_distributions)
    equal_draws = sampled_matrix.mean(axis=0)
    equal_observed = float(effects.mean())
    inverse_variance_weights = 1.0 / variances
    inverse_variance_weights /= inverse_variance_weights.sum()
    fixed_observed = float(np.sum(inverse_variance_weights * effects))
    fixed_draws = np.sum(inverse_variance_weights[:, None] * sampled_matrix, axis=0)

    q_statistic = float(np.sum((effects - fixed_observed) ** 2 / variances))
    df = len(effects) - 1
    i2 = float(max(0.0, (q_statistic - df) / q_statistic)) if q_statistic > 0 else 0.0
    c_value = float(inverse_variance_weights.sum())
    raw_weights = 1.0 / variances
    c_dl = float(raw_weights.sum() - (raw_weights**2).sum() / raw_weights.sum())
    tau2 = float(max(0.0, (q_statistic - df) / c_dl)) if c_dl > 0 else 0.0
    random_weights = 1.0 / (variances + tau2)
    random_weights /= random_weights.sum()
    random_observed = float(np.sum(random_weights * effects))
    random_draws = np.sum(random_weights[:, None] * sampled_matrix, axis=0)

    synthesis = {
        "date": "2026-08-17",
        "datasets_included": list(DATASETS),
        "datasets_negative_direction": int(np.sum(effects < 0)),
        "datasets_individually_conclusive": int(
            sum(row["individually_conclusive"] for row in dataset_rows)
        ),
        "equal_dataset_weighted_primary": {
            "delta_aurc": equal_observed,
            "ci95": [float(np.quantile(equal_draws, 0.025)), float(np.quantile(equal_draws, 0.975))],
            "conclusive": bool(np.quantile(equal_draws, 0.975) < 0),
        },
        "inverse_variance_fixed_sensitivity": {
            "delta_aurc": fixed_observed,
            "ci95": [float(np.quantile(fixed_draws, 0.025)), float(np.quantile(fixed_draws, 0.975))],
            "weights": {
                name: float(weight) for name, weight in zip(DATASETS, inverse_variance_weights)
            },
            "conclusive": bool(np.quantile(fixed_draws, 0.975) < 0),
        },
        "random_effects_sensitivity": {
            "delta_aurc": random_observed,
            "ci95": [float(np.quantile(random_draws, 0.025)), float(np.quantile(random_draws, 0.975))],
            "tau_squared": tau2,
            "conclusive": bool(np.quantile(random_draws, 0.975) < 0),
        },
        "heterogeneity": {"q": q_statistic, "df": df, "i_squared": i2},
        "interpretation_boundary": (
            "The synthesis supports a small average ranking increment and consistent direction, "
            "but does not convert the DeepDRiD primary endpoint into a significant result or "
            "establish universal operating-point/FNR improvement."
        ),
    }
    dataset_frame = pd.DataFrame(dataset_rows)
    draw_frame = pd.DataFrame(
        {
            "draw": np.arange(draws),
            "equal_weight_delta_aurc": equal_draws,
            "fixed_effect_delta_aurc": fixed_draws,
            "random_effect_delta_aurc": random_draws,
        }
    )
    dataset_frame.to_csv(output / "dataset_level_evidence.csv", index=False)
    draw_frame.to_csv(output / "synthesis_bootstrap_draws.csv", index=False)
    (output / "cross_dataset_synthesis.json").write_text(
        json.dumps(synthesis, indent=2), encoding="utf-8"
    )
    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()

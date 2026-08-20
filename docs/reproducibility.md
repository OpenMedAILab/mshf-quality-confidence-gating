# Reproducibility Notes

## Minimal Route

1. Obtain raw datasets from the original providers listed in `docs/data_sources.md`.
2. Set `DATA_ROOT` for local dataset locations matching `configs/study_config.yaml`, or pass explicit path arguments to each script.
3. Audit MSHF labels and prepare manifests:

```bash
python code/audit_mshf.py
python code/prepare_deepdrid_manifest.py
python code/prepare_deepdrid_regular.py
```

4. Train fixed-schedule ConvNeXt-Tiny models:

```bash
python code/train_mshf_protocol_quality.py
python code/train_deepdrid_protocol_model.py
```

5. Run frozen evaluations:

```bash
python code/evaluate_deepdrid_official_frozen.py
python code/infer_and_evaluate_eyeq_protocol.py
python code/infer_and_evaluate_aptos_frozen.py
python code/infer_and_evaluate_messidor2_frozen.py
```

6. Synthesize dataset-level evidence:

```bash
python code/synthesize_cross_dataset_evidence.py
```

## Fixed Parameters Observed in Code and Metadata

- Quality model: ImageNet-pretrained ConvNeXt-Tiny, four sigmoid heads.
- Disease model: ImageNet-pretrained ConvNeXt-Tiny for referable DR.
- Input size: 384.
- Preprocessed cache size: 448 in local runs.
- Epochs: 8 in final protocol training scripts.
- DeepDRiD cross-fitting: 5 patient-grouped folds.
- Calibration candidates: raw, logit-Platt, probability-Platt, beta, temperature, isotonic.
- Frozen calibration used in final summaries: beta.
- Gate feature set: worst-view overall-quality risk plus calibrated entropy.
- Gate model: standardized ridge logistic regression.
- Final gate regularization in frozen summaries: `C=0.01`.
- Bootstrap replicates in final frozen run metadata: 2,000.

## Seed Audit

Observed evidence:

- `configs/study_config.yaml` records `study.seed: 20260817`.
- `code/train_mshf_protocol_quality.py` default seed is `20260817`.
- `code/train_deepdrid_protocol_model.py` default seed is `20260817`.
- `code/evaluate_deepdrid_official_frozen.py`, `code/infer_and_evaluate_eyeq_protocol.py`, `code/infer_and_evaluate_aptos_frozen.py`, and `code/infer_and_evaluate_messidor2_frozen.py` default to seed `20260817` with 2,000 bootstrap replicates.
- Final local run metadata under `research_mshf/outputs/*/run_config.json` records seed `20260817` for the MSHF quality model, DeepDRiD disease model, DeepDRiD official evaluation, EyeQ, APTOS, and Messidor-2 frozen evaluations.
- An earlier local configuration and an earlier non-final EyeQ severity-bias summary recorded `20260816`; this was an early configuration legacy value and is not used for the final reported results.

Final repository wording:

> Final ConvNeXt protocol training, DeepDRiD patient-grouped cross-fitting, and frozen evaluations with 2,000 bootstrap replicates used seed 20260817. The value 20260816 was an early configuration legacy value and is not used for the final reported results.

## Reproducibility Limits

This repository does not include raw images or model checkpoints. End-to-end reruns require local dataset access and sufficient GPU resources. The included bootstrap tables, table workbooks, and figures are intended to preserve the reported non-image result snapshot.

# MSHF Quality-Confidence Gating

This repository contains public-release code and aggregate non-image study outputs for:

**Beyond Confidence Under Domain Shift: Quality-Informed Error Gating and Severity-Aware Retention Across Four Fundus Datasets**

The study evaluates whether MSHF-derived fundus image quality adds error-ranking value beyond calibrated model confidence for referable diabetic retinopathy selective prediction under domain shift. The analysis uses MSHF for image-quality supervision, DeepDRiD for disease-model and gate development plus official evaluation, and EyeQ/EyePACS, APTOS 2019, and Messidor-2 for frozen external evaluations.

## Data Boundary

This public version contains code, aggregate bootstrap results, final manuscript tables, and final figures.

It does not contain raw fundus images, model weights, image-level manifests, or image-level prediction/risk files. Image-level manifest and prediction/risk files are derived from third-party datasets and may include image names, row identifiers, or dataset-relative paths; they are therefore not released here and must be handled according to the original data providers' access terms.

## Repository Contents

- `code/`: scripts for auditing, manifest preparation, ConvNeXt-Tiny training, calibration, gate evaluation, frozen external validation, severity retention analysis, and cross-dataset synthesis.
- `configs/study_config.yaml`: study configuration template using `DATA_ROOT` placeholders and the final reported seed.
- `derived_outputs/bootstrap/`: numeric bootstrap replicate and summary tables.
- `docs/derived_file_field_inventory.md`: field-level inventory for non-released image-level derived files, retained as governance documentation.
- `tables/`: manuscript table workbooks. Prefer final filenames and exclude `before_*` versions from public release.
- `figures/`: manuscript figures. Prefer final filenames and exclude `before_*` versions from public release.
- `ENVIRONMENT_INVENTORY.txt` and `requirements.txt`: environment summary and pinned major dependencies from the server environment.

## Data Sources

Use the original provider pages only. This staging copy intentionally contains no raw image datasets.

- MSHF: Scientific Data article and Figshare data citation, https://doi.org/10.1038/s41597-023-02188-x and https://doi.org/10.6084/m9.figshare.21507564.v1
- DeepDRiD: Zenodo record, https://zenodo.org/records/8248825
- EyeQ labels: official EyeQ repository, https://github.com/HzFu/EyeQ
- EyePACS images: obtain through the original EyePACS/Kaggle access route used by the EyeQ release terms.
- APTOS 2019: Kaggle competition page, https://www.kaggle.com/competitions/aptos2019-blindness-detection
- Messidor-2: ADCIS page, https://www.adcis.net/en/third-party/messidor2/

## Installation

The environment inventory was generated on the analysis server with Python 3.14.6. A minimal pinned dependency file is provided:

```bash
python -m pip install -r requirements.txt
```

GPU training used PyTorch and torchvision CUDA builds available on the server. If installing elsewhere, use the PyTorch installation command appropriate for the target CUDA version.

## Reproduction Order

Set `DATA_ROOT` to a local directory containing datasets obtained from the original providers, or pass explicit path arguments to each script. Run from the repository root:

```bash
python code/audit_mshf.py
python code/prepare_deepdrid_manifest.py
python code/prepare_deepdrid_regular.py
python code/train_mshf_protocol_quality.py
python code/train_deepdrid_protocol_model.py
python code/evaluate_deepdrid_official_frozen.py
python code/infer_and_evaluate_eyeq_protocol.py
python code/infer_and_evaluate_aptos_frozen.py
python code/infer_and_evaluate_messidor2_frozen.py
python code/synthesize_cross_dataset_evidence.py
```

The scripts expose path, batch-size, worker, seed, and output-directory arguments. Do not commit raw images, caches, checkpoints, credentials, downloaded archives, image-level manifests, or image-level prediction/risk files.

## Fixed Parameters Observed Locally

- Backbone: ImageNet-pretrained ConvNeXt-Tiny for both quality and referable-DR models.
- Quality heads: illumination, clarity, contrast, and overall.
- Input size: 384 pixels for model input; 448-pixel preprocessed cache in local runs.
- Training schedule: 8 epochs in final ConvNeXt protocol scripts.
- DeepDRiD disease cross-fitting: 5 patient-grouped folds.
- Calibration candidates: raw, logit-Platt, probability-Platt, beta, temperature, isotonic.
- Frozen calibration: beta calibration fitted on DeepDRiD training OOF predictions.
- Gate: standardized worst-view overall-quality risk plus calibrated entropy, ridge logistic regression.
- Gate regularization: `C=0.01` in final frozen summaries.
- Bootstrap: 2,000 paired bootstrap replicates in final frozen run metadata.

## Seed Note

Final ConvNeXt training, DeepDRiD patient-grouped cross-fitting, and frozen evaluations with 2,000 bootstrap replicates used seed `20260817`, according to final script defaults and final run metadata. A previous configuration value, `20260816`, was an early configuration legacy value and is not used for the final reported results.

## Citation

Please cite the manuscript when available, and cite the original dataset providers according to their own instructions. Do not cite this repository as a source of raw images.

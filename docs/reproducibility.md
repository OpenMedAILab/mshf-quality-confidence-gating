# Reproducibility guide

## Environment and included result snapshot

Install a compatible PyTorch/torchvision pair for your CPU or GPU from the [official selector](https://pytorch.org/get-started/locally/), then run `python -m pip install -r requirements.txt` from the repository root. Versions recorded on the analysis server are in [`analysis_environment.txt`](analysis_environment.txt). The included aggregate bootstrap files and tables preserve the reported result snapshot; they do not substitute for images, checkpoints, or image-level predictions.

Run `python examples/smoke_metrics.py` to check the local metrics implementation on synthetic data. Run `python code/synthesize_cross_dataset_evidence.py --input-dir results/bootstrap --output-dir outputs/synthesis` to recompute the exploratory equal-dataset synthesis from the released aggregate files.

## Full image-level workflow

1. Obtain MSHF, DeepDRiD, EyeQ/EyePACS, APTOS 2019, and Messidor-2 through the providers listed in [`data_sources.md`](data_sources.md).
2. Prepare DeepDRiD training and evaluation manifests with `code/prepare_deepdrid_regular.py` and `code/prepare_deepdrid_manifest.py`. Audit MSHF labels with `code/audit_mshf.py`.
3. Train the MSHF-derived quality model with `code/train_mshf_protocol_quality.py` and the DeepDRiD disease model with `code/train_deepdrid_protocol_model.py`. Supply explicit manifest, cache, and output paths. Training produces the checkpoints and out-of-fold predictions needed downstream.
4. Fit the gate on DeepDRiD development predictions using `code/optimize_protocol_aligned_gate.py`. Keep the official evaluation set separate from gate selection.
5. Run `code/evaluate_deepdrid_official_frozen.py` for the primary evaluation. Run `code/infer_and_evaluate_eyeq_protocol.py`, `code/infer_and_evaluate_aptos_frozen.py`, and `code/infer_and_evaluate_messidor2_frozen.py` for external exploratory evaluations. Each accepts explicit paths for inputs and output directory.
6. Use `code/severity_coverage_audit.py` for grade-specific retention. The supplementary scripts `code/optimize_multidimensional_gate.py`, `code/optimize_multiview_quality_aggregation.py`, and `code/evaluate_cost_sensitive_gate.py` examine alternative gate designs.

The scripts retain some defaults from the analysis workspace under `research_mshf/outputs/`. In a fresh clone, **pass explicit path options** rather than relying on those defaults. Run each script with `--help` for its supported options.

## Recorded settings and limits

The final protocol used ImageNet-pretrained ConvNeXt-Tiny models, 384-pixel input, five patient-grouped DeepDRiD cross-fitting folds, and seed `20260817`. Frozen evaluation summaries used 2,000 bootstrap replicates. See [`study_config.yaml`](../configs/study_config.yaml) and the scripts for further parameters. Exact end-to-end reproduction requires source data, checkpoint-generating training runs, adequate computing resources, and the same data preprocessing and label choices.

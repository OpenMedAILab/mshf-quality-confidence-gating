# Analysis-to-code map

| Study component | Entry point | Released evidence |
| --- | --- | --- |
| Quality and disease training | `code/audit_mshf.py`, `code/prepare_deepdrid_regular.py`, `code/train_mshf_protocol_quality.py`, `code/train_deepdrid_protocol_model.py` | Model-performance and LOSO tables in `tables/` |
| Gate selection | `code/optimize_protocol_aligned_gate.py` | Selection settings in code and configuration |
| Primary DeepDRiD evaluation | `code/evaluate_deepdrid_official_frozen.py` | `results/bootstrap/official_evaluation_patient_bootstrap_*`; DeepDRiD tables |
| Exploratory external evaluations | `code/infer_and_evaluate_eyeq_protocol.py`, `code/infer_and_evaluate_aptos_frozen.py`, `code/infer_and_evaluate_messidor2_frozen.py` | Dataset bootstrap files and cross-dataset tables |
| Grade-specific retention | `code/severity_coverage_audit.py` and frozen evaluation scripts | Severity and safety tables; `assets/severity_retention.png` |
| Exploratory synthesis | `code/synthesize_cross_dataset_evidence.py` | `results/bootstrap/synthesis_bootstrap_draws.csv` |
| Alternative-gate sensitivity analyses | `code/optimize_multidimensional_gate.py`, `code/optimize_multiview_quality_aggregation.py`, `code/evaluate_cost_sensitive_gate.py` | Recompute from image-level inputs after obtaining source data |

Aggregate bootstrap files cannot regenerate image-level predictions or training checkpoints.

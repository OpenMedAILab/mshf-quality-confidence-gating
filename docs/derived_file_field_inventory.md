# Derived File Field Inventory

Fields marked `ID/PATH` contain image names, paths, patient or eye identifiers, or dataset-specific image identifiers. These fields should be reviewed before public release.

## Manifests

- `manifests/aptos_manifest.csv`: `image_name` ID/PATH, `image_path` ID/PATH, `patient_id` ID/PATH, `eye_id` ID/PATH, `dr_grade`, `referable_dr`, `y_true`.
- `manifests/deepdrid_manifest.csv`: `image_name` ID/PATH, `stem` ID/PATH, `image_path` ID/PATH, `relative_path` ID/PATH, `label_image_id` ID/PATH, `dr_grade`, `label_stem` ID/PATH, `referable_dr`.
- `manifests/deepdrid_regular_manifest.csv`: `split`, `patient_id` ID/PATH, `image_id` ID/PATH, `image_name` ID/PATH, `eye`, `view`, `image_path` ID/PATH, `exists`, `dr_grade`, `referable_dr`, `patient_dr_grade`, `deepdrid_overall_quality`, `deepdrid_clarity`, `deepdrid_field_definition`, `deepdrid_artifact`.
- `manifests/eyeq_image_manifest.csv`: `split`, `image_name` ID/PATH, `quality`, `quality_label`, `dr_grade`, `referable_dr`, `image_path` ID/PATH, `image_exists`.
- `manifests/eyeq_label_manifest.csv`: `split`, `image_name` ID/PATH, `quality`, `quality_label`, `dr_grade`, `referable_dr`.
- `manifests/messidor2_manifest.csv`: `image_name` ID/PATH, `image_path` ID/PATH, `patient_id` ID/PATH, `eye_id` ID/PATH, `dr_grade`, `referable_dr`, `y_true`, `adjudicated_gradable`.
- `manifests/mshf_manifest.csv`: `image_name` ID/PATH, `source_from_name`, `relative_path` ID/PATH, `bytes`, `source_from_label`, rating-count and quality-label columns, `sheet`, `source_from_official_name`, `has_image`, `has_all_majority_labels`, `source`.
- `manifests/official_evaluation_manifest.csv`: `image_id` ID/PATH, `DR_Levels`, `Overall quality`, `Artifact`, `Clarity`, `Field definition`, `patient_id` ID/PATH, `eye`, `view`, `dr_grade`, `referable_dr`, `y_true`, `human_overall_quality`, `image_path` ID/PATH, `original_image_path` ID/PATH.

## Predictions and Risks

- `predictions_and_risks/aptos_frozen_predictions_and_risks.csv`: `image_name` ID/PATH, `image_path` ID/PATH, `patient_id` ID/PATH, `eye_id` ID/PATH, labels, disease probabilities, quality scores, predictions, error flag, quality/confidence/joint risk scores, random risk.
- `predictions_and_risks/eyeq_frozen_predictions_and_risks.csv`: `split`, `image_name` ID/PATH, EyeQ quality fields, labels, `image_path` ID/PATH, disease probabilities, quality scores, `patient_id` ID/PATH, `eye`, `eye_id` ID/PATH, predictions, error flag, quality/confidence/joint risk scores, random risk.
- `predictions_and_risks/messidor2_frozen_predictions_and_risks.csv`: `image_name` ID/PATH, `image_path` ID/PATH, `patient_id` ID/PATH, `eye_id` ID/PATH, labels, `adjudicated_gradable`, disease probabilities, quality scores, predictions, error flag, quality/confidence/joint risk scores, random risk.
- `predictions_and_risks/official_evaluation_eye_predictions_and_risks.csv`: `patient_id` ID/PATH, `eye`, labels, raw logit, fold, view count, quality-risk scores, `eye_id` ID/PATH, disease probabilities, predictions, error flag, confidence/quality/joint risk scores, random risk.

## Public Sanitized Version Proposal

To produce a public sanitized derivative after license confirmation:

1. Remove all `image_path`, `original_image_path`, and other `*_path` columns.
2. Replace `image_name`, `image_id`, `patient_id`, and `eye_id` with salted deterministic hashes where row linkage is needed.
3. Keep analytic fields required to reproduce paper tables and figures: dataset name, split if needed, DR grade, referable label, quality label or quality dimensions, model probabilities, calibrated probabilities, prediction, base error flag, risk scores, coverage or bootstrap metrics.
4. Preserve row order only if needed for deterministic reproduction; otherwise add a release row index.
5. Publish the sanitization script and a schema, but keep the hash salt private unless re-identification through original filenames is explicitly allowed.

The aggregate bootstrap files in `derived_outputs/bootstrap/` do not contain row-level image or patient identifiers and are lower-risk release candidates.

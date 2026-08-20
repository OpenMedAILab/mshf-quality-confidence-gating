# Public Release Boundary

## Intended for Public Release

- Analysis and reproduction code in `code/`.
- Study configuration in `configs/`, using release-safe data-root placeholders and seed `20260817`.
- Numeric bootstrap replicate and summary CSV files in `derived_outputs/bootstrap/`.
- Final manuscript tables in `tables/`, excluding historical `before_*` versions.
- Final manuscript figures in `figures/`, excluding historical `before_*` versions.
- Environment files: `requirements.txt` and `ENVIRONMENT_INVENTORY.txt`.
- Documentation in `README.md` and `docs/`.

## Exclude From Public Release

- Raw fundus images from MSHF, DeepDRiD, EyeQ/EyePACS, APTOS 2019, Messidor-2, or any other clinical dataset.
- Preprocessed image caches.
- Dataset archives or sample image archives.
- Model checkpoints and learned weights unless redistribution is explicitly confirmed.
- Credentials, API tokens, passwords, `.env` files, private keys, and local account configuration.
- Python caches and local runtime artifacts.
- Manuscript DOCX files, submission ZIP packages, cover letters, author forms, internal review files, and historical archive folders unless separately approved.

## Conditional Release After Confirmation

The following non-image derived files are not released in this public repository because they include dataset image names, relative paths, or patient/eye-level identifiers:

- `derived_outputs/manifests/*.csv`
- `derived_outputs/predictions_and_risks/*.csv`

Before any future public upload, confirm that source dataset terms permit redistribution of these derived records. This repository publishes only code, aggregate bootstrap results, final tables, final figures, and release documentation.

## Recommended Sanitization for Conditional CSV Release

- Remove local or dataset-root paths from all `*_path` fields.
- Replace `image_name`, `image_id`, `patient_id`, and `eye_id` with stable salted hashes if row linkage is required.
- Retain non-identifying analytic columns needed for table and figure reproduction, such as labels, predictions, calibrated probabilities, quality scores, risk scores, and bootstrap metrics.
- Keep the salt private if linkage back to original dataset records is not intended.
- Document any transformation in a release note.

## Current Staging Status

The public release excludes raw images, checkpoints, image-level manifests, and image-level prediction/risk files.

# Data Sources

This release candidate does not include raw images. Users must obtain datasets from the original providers and follow their access terms, citation requirements, and redistribution restrictions.

## MSHF

- Primary article: Jin et al., "MSHF: A Multi-Source Heterogeneous Fundus (MSHF) Dataset for Image Quality Assessment", Scientific Data, 2023.
- Article DOI: https://doi.org/10.1038/s41597-023-02188-x
- Figshare data citation in the article: https://doi.org/10.6084/m9.figshare.21507564.v1
- Local role in this study: four-dimensional image-quality supervision for illumination, clarity, contrast, and overall quality.

## DeepDRiD

- Zenodo record: https://zenodo.org/records/8248825
- Related repository link from Zenodo: https://github.com/deepdrdoc/DeepDRiD/tree/v1.1
- Local role in this study: referable-DR disease-model training, patient-grouped OOF gate development, and official frozen evaluation.

## EyeQ and EyePACS

- EyeQ repository: https://github.com/HzFu/EyeQ
- Local role in this study: frozen external validation using EyeQ quality labels and EyePACS-derived fundus images.
- Boundary: EyeQ labels and EyePACS images have separate provenance and access terms. Do not redistribute EyePACS images from this repository.

## APTOS 2019

- Kaggle competition: https://www.kaggle.com/competitions/aptos2019-blindness-detection
- Local role in this study: frozen external referable-DR evaluation.
- Boundary: follow Kaggle competition and dataset terms. This repository should not redistribute raw APTOS images.

## Messidor-2

- ADCIS page: https://www.adcis.net/en/third-party/messidor2/
- Local role in this study: frozen external referable-DR evaluation using a locally available grading file.
- Boundary: ADCIS states dataset use and redistribution limits on its provider page. Do not redistribute raw Messidor-2 images from this repository.

## Derived Labels and Manifests

The derived manifest and prediction CSV files in `derived_outputs/` may contain dataset image filenames and patient/eye-level identifiers. These are not raw images, but public redistribution still requires author and data-governance confirmation.

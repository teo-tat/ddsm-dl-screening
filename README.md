# Deep Learning Breast Cancer Detection

> University of London — AI & Machine Learning Graduate Diploma — Final Project  
> CM3070 Computer Science Final Project | Template 3.2

---

## Overview

This project investigates whether deep learning-assisted X-ray mammography can improve the accuracy of breast cancer screening. Convolutional neural networks (CNNs) are trained on the **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography) to perform binary classification of mammograms as malignant or benign.

The project follows the **Chollet (2018) deep learning workflow** and benchmarks results against the architectures surveyed in **Wang (2024)**.

---

## Status

EDA complete — Chollet Stages 1–3 done. Data pipeline implemented.

---

## Project Structure

```
ddsm-dl-screening/
├── data/                     ← CBIS-DDSM files — see data/README.md
├── notebooks/
│   └── 01_eda.ipynb          ← EDA (Chollet Stages 1–3)
│
├── src/
│   ├── config.py             ← all hyperparameters, paths, and seeds
│   └── data_loader.py        ← CBIS-DDSM pipeline: loading, patient splits, augmentation, tf.data
└── outputs/
    └── figures/eda/          ← EDA plots
```

---

## EDA Key Findings

| Property | Value |
|---|---|
| Training rows | 1,318 (1,231 unique images) |
| Test rows | 378 (361 unique images) |
| Unique train patients | 691 |
| Unique test patients | 201 |
| Patient overlap (train/test) | 0 |
| Class imbalance ratio | 1.07:1 (benign:malignant) |
| Bit depth | 16-bit uniform |
| MONOCHROME1 images | 0 |
| Median resolution | ~5,281 × 3,000 px |
| Train split (after val carve-out) | 1,058 images / 587 patients |
| Val split | 173 images / 104 patients |

**BI-RADS clinical baseline — binding minimum targets for all model phases:**

| Metric | Value |
|---|---|
| AUC-ROC | 0.820 |
| AUC-PR | 0.724 |
| Sensitivity (BI-RADS ≥ 4) | 0.931 |

**Normalisation** (exact, from all 1,231 training images):  
`mean = 0.2108`, `std = 0.2638` (÷ 65,535 → [0, 1])

---

## Dataset

**CBIS-DDSM** — available through The Cancer Imaging Archive (TCIA):  
https://www.cancerimagingarchive.net/collection/cbis-ddsm/

See `data/README.md` for download instructions and directory structure.

---

## References

- Wang, L. (2024). Mammography with deep learning for breast cancer detection. *Frontiers in Oncology*, 14, 1281922. https://doi.org/10.3389/fonc.2024.1281922
- Chollet, F. (2018). *Deep Learning with Python.* Manning.
- Lee, R. et al. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177. https://doi.org/10.1038/sdata.2017.177
- McKinney, S. M. et al. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577, 89–94.
- Rodriguez-Ruiz, A. et al. (2019). Detection of breast cancer with mammography: effect of an artificial intelligence support system. *JNCI*, 111(9), 916–922.
- American College of Radiology (2013). *ACR BI-RADS Atlas* (5th ed.).
- NHS England (2022). *Breast Screening Programme, England 2019–20.* NHS Digital.

---

## License

MIT

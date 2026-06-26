# Deep Learning Breast Cancer Detection

> University of London — AI & Machine Learning Graduate Diploma — Final Project  
> CM3070 Computer Science Final Project | Template 3.2

---

## Overview

This project investigates whether deep learning-assisted X-ray mammography can improve the accuracy of breast cancer screening. Convolutional neural networks (CNNs) are trained on the **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography) to perform binary classification of mammograms as malignant or benign.

The project follows the **Chollet (2018) deep learning workflow** and benchmarks results against the architectures surveyed in **Wang (2024)**, with **Shen et al. (2019)** as the primary benchmark anchor (patient-level split, AUC 0.88 single / 0.91 ensemble).

---

## Status

EDA complete — Chollet Stages 1–3 done. Data pipeline implemented. Baseline CNN (Stage 5) next.

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

## Data Pipeline Design

Lee et al.'s (2017) official CBIS-DDSM train/test split is **discarded**. Both CSVs are pooled, deduplicated (label = max across abnormalities), and split once into a fresh **70/15/15 train/val/test** partition via `StratifiedGroupKFold`, grouped by patient ID and stratified by binary label. This follows Shen et al. (2019), who used their own patient-level split rather than the official one.

---

## EDA Key Findings

| Property | Value |
|---|---|
| Total pooled unique images | 1,592 |
| Train / Val / Test images | 1,134 / 230 / 228 |
| Train / Val / Test patients | 634 / 129 / 129 |
| Patient overlap (all 3 pairs) | 0 — no leakage |
| Class imbalance ratio (train) | 1.13:1 (benign:malignant) — near-balanced |
| Naive baseline accuracy | 53.1% |
| Bit depth | 16-bit uniform (all 1,592 images) |
| MONOCHROME1 images | 0 / 1,592 |
| Median resolution | ~5,281 × 3,000 px |

**BI-RADS clinical baseline (test set, n=228) — binding CNN minimum targets:**

| Metric | Value |
|---|---|
| AUC-ROC | 0.813 |
| AUC-PR | 0.745 |
| Sensitivity (BI-RADS ≥ 4) | 0.897 |

**Normalisation** (exact, from all 1,134 training images):  
`mean = 0.2128`, `std = 0.2651` (÷ 65,535 -> [0, 1])  
Applied post-augmentation in scratch mode; ImageNet stats used for transfer learning.

---

## Dataset

**CBIS-DDSM** — available through The Cancer Imaging Archive (TCIA):  
https://www.cancerimagingarchive.net/collection/cbis-ddsm/

See `data/README.md` for download instructions and directory structure.

---

## References

- Wang, L. (2024). Mammography with deep learning for breast cancer detection. *Frontiers in Oncology*, 14, 1281922. https://doi.org/10.3389/fonc.2024.1281922
- Chollet, F. (2018). *Deep Learning with Python.* Manning.
- Shen, L. et al. (2019). Deep learning to improve breast cancer detection on screening mammography. *Scientific Reports*, 9, 12495. https://doi.org/10.1038/s41598-019-48995-4
- Lee, R. et al. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177. https://doi.org/10.1038/sdata.2017.177
- McKinney, S. M. et al. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577, 89–94.
- American College of Radiology (2013). *ACR BI-RADS Atlas* (5th ed.).
- NHS England (2022). *Breast Screening Programme, England 2019–20.* NHS Digital.

---

## Reproducibility

- Seed: 28 (fixed in `config.py`, propagated to NumPy, Python, TensorFlow)
- All hyperparameters in `config.py` - no magic numbers in notebooks or scripts
- Public dataset + open architectures + pinned dependencies

---

## License

MIT
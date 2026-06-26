"""config.py — Single source of truth for all hyperparameters, paths, and constants."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import tensorflow as tf

SEED: int = 28

def set_seeds() -> None:
    """Fix all random seeds. Call at the top of every training script."""
    import random
    import os
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = _ROOT / "data" / "raw"
TRAIN_CSV: Path = DATA_DIR / "mass_case_description_train_set.csv"
TEST_CSV: Path = DATA_DIR / "mass_case_description_test_set.csv"
TRAIN_IMG_DIR: Path = DATA_DIR / "mass_train" / "images"
TEST_IMG_DIR: Path = DATA_DIR / "mass_test"  / "images"

OUTPUTS_DIR: Path = _ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
WEIGHTS_DIR: Path = OUTPUTS_DIR / "weights"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"

TARGET_SIZE: tuple[int, int] = (224, 224)

# Dataset split
# TRAIN_CSV and TEST_CSV are POOLED in data_loader.build_datasets, then split
# ONCE into a patient-level, label-stratified ~70/15/15 partition via
# StratifiedGroupKFold. Lee et al.'s official train/test boundary is
# discarded post-pool — matches Shen et al. (2019)'s self-constructed
# 85:15 patient-level split. Fractions below are targets; quote the
# achieved counts from _log_split_summary's printed output in the report.
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
BATCH_SIZE: int = 32
MAX_EPOCHS: int = 100
INITIAL_LR: float = 1e-3
FINETUNE_LR: float = 1e-5

EARLY_STOPPING_PATIENCE: int = 10

# Normalisation
# "scratch" mode: divisor-scale to [0,1], augment (still [0,1]), THEN
# standardise with global_mean_01/global_std_01 — see
# data_loader._make_standardize_fn for why standardisation must come after
# augmentation, not before.
# "imagenet" mode: divisor-scale to [0,1], replicate to 3 channels,
# standardise with imagenet_mean/imagenet_std — BEFORE augmentation, since
# only geometric (not pixel-value) augmentation runs in this mode.
#
# global_mean_01/global_std_01 computed on the real pooled train partition
# (n=1134, post-StratifiedGroupKFold) — EDA notebook, Section 10a.
NORM_STATS: dict = {
    "method":          "divide_by_max_16bit",
    "divisor":         65_535,
    "global_mean_01":  0.2128,
    "global_std_01":   0.2651,
    "imagenet_mean":   (0.485, 0.456, 0.406),
    "imagenet_std":    (0.229, 0.224, 0.225),
}

# Augmentation
# Both modes: flip, rotation, zoom-IN only (never out). "scratch" only:
# brightness jitter — skipped in "imagenet" mode. Report should say
# "brightness", not "contrast" (not implemented).
AUG_MAX_ROTATION_DEG: float = 10.0
AUG_BRIGHTNESS_DELTA: float = 0.2
AUG_ZOOM_FACTOR: float = 0.10

# BI-RADS baselines from the test set.
BIRADS_AUC_BASELINE: float = 0.813
BIRADS_SENSITIVITY_BASELINE: float = 0.897
BIRADS_AUCPR_BASELINE: float = 0.745
CLINICAL_SENSITIVITY_MIN: float = 0.80

# Wang (2024) Table 5 classification project target:
PROJECT_AUC_TARGET: float = 0.90
PROJECT_SENSITIVITY_TARGET: float = 0.85
PROJECT_SPECIFICITY_TARGET: float = 0.80

BOOTSTRAP_ITERATIONS: int = 1_000
BOOTSTRAP_CI_LEVEL: float = 0.95

CLASS_WEIGHT_STRATEGY: str = "none"

LABEL_MAP: dict[str, int] = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

GRADCAM_ALPHA: float = 0.4


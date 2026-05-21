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

#  Dataset split 
# Val is carved from the training CSV; test CSV is the pre-defined CBIS-DDSM hold-out.
VAL_FRACTION: float = 0.15

BATCH_SIZE: int = 32
MAX_EPOCHS: int = 100
INITIAL_LR: float = 1e-3
FINETUNE_LR: float = 1e-5

EARLY_STOPPING_PATIENCE: int = 10

# Normalisation
# Statistics computed from the training CSV. Val images are included (minor
# limitation — shift is below float32 rounding noise in practice).
NORM_STATS: dict = {
    "method":          "divide_by_max_16bit",
    "divisor":         65_535,
    "global_mean_01":  0.2108,
    "global_std_01":   0.2638,
    "imagenet_mean":   (0.485, 0.456, 0.406),
    "imagenet_std":    (0.229, 0.224, 0.225),
}

# Augmentation
AUG_MAX_ROTATION_DEG: float = 10.0
AUG_BRIGHTNESS_DELTA: float = 0.2
AUG_ZOOM_FACTOR: float = 0.10

# BI-RADS baselines from the test set.
BIRADS_AUC_BASELINE: float = 0.820
BIRADS_SENSITIVITY_BASELINE: float = 0.931
BIRADS_AUCPR_BASELINE: float = 0.724
CLINICAL_SENSITIVITY_MIN: float = 0.80

# Wang (2024) Table 3 project target:
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


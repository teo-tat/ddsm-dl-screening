"""Single source of truth for the project's hyperparameters, paths and constants."""

from __future__ import annotations
from pathlib import Path
import numpy as np

SEED: int = 28


def set_seeds(seed: int = SEED) -> None:
    """Fix all random seeds. Call at the top of every training script.
    TensorFlow is imported here so `config` stays importable without it."""
    import random
    import os
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = _ROOT / "data" / "raw"
TRAIN_CSV: Path = DATA_DIR / "mass_case_description_train_set.csv"
TEST_CSV: Path = DATA_DIR / "mass_case_description_test_set.csv"
TRAIN_IMG_DIR: Path = DATA_DIR / "mass_train" / "images"
TEST_IMG_DIR: Path = DATA_DIR / "mass_test" / "images"

# Per-lesion ROI folders. Each holds the mask and the cropped patch; the loader tells
# them apart by size.
TRAIN_ROI_DIR: Path = DATA_DIR / "mass_train_roi" / "images"
TEST_ROI_DIR: Path = DATA_DIR / "mass_test_roi" / "images"

OUTPUTS_DIR: Path = _ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
WEIGHTS_DIR: Path = OUTPUTS_DIR / "weights"

# Weights from CUDA runs land here, apart from WEIGHTS_DIR.
CUDA_ROOT: Path = OUTPUTS_DIR / "_cuda"

# Session folders that hold weights one level down. Per-run folders directly under
# CUDA_ROOT are searched as well.
CUDA_WEIGHTS_DIRS: tuple[Path, ...] = (
    CUDA_ROOT / "s8_weights",
    CUDA_ROOT / "s9_fusion" / "weights",
    CUDA_ROOT / "s0917" / "weights",  # the training session of the reported classifiers
)

WEIGHTS_VARIANTS: tuple[str, ...] = ("best", "finetune_best", "final")

# Suffix naming the training session of the reported classifiers; it goes before the seed suffix.
LEDGER_SUFFIX: str = "_cuda0917"

# Suffix naming the training session of the two context-fusion models.
CONTEXT_FUSION_SUFFIX: str = "_cuda0918"


class WeightsRefused(FileNotFoundError):
    """No single CUDA-ledger file can be named for a tag. A FileNotFoundError subclass,
    so callers that stop on a missing-weights error stop on this one too."""


def _cuda_search_dirs() -> tuple[Path, ...]:
    """The named session dirs, then every per-run dir directly under CUDA_ROOT."""
    per_run = (
        tuple(sorted(p for p in CUDA_ROOT.iterdir() if p.is_dir())) if CUDA_ROOT.is_dir() else ()
    )
    seen, out = set(), []
    for d in (*CUDA_WEIGHTS_DIRS, *per_run):
        if d not in seen:
            seen.add(d)
            out.append(d)
    return tuple(out)


def resolve_weights(
    tag: str, *, variants: tuple[str, ...] = WEIGHTS_VARIANTS, allow_metal: bool = False
) -> Path:
    """Path to `tag`'s weights, refusing rather than guessing: a laptop file with a CUDA
    twin, or two differing files for one tag, raises WeightsRefused."""
    import hashlib
    import re

    cuda_dirs = _cuda_search_dirs()
    found: list[Path] = []
    for d in (WEIGHTS_DIR, *cuda_dirs):
        for v in variants:
            c = d / f"{tag}_{v}.weights.h5"
            if c.exists():
                found.append(c)
                break  # one variant per directory, in preference order
    if not found:
        raise FileNotFoundError(
            f"No weights for tag {tag!r}. Searched WEIGHTS_DIR and {len(cuda_dirs)} CUDA "
            f"dir(s) for variants {list(variants)}:\n  {WEIGHTS_DIR}\n"
            + "\n".join(f"  {d}" for d in cuda_dirs)
        )

    cuda_found = [p for p in found if WEIGHTS_DIR not in p.parents]
    if len(found) > 1:

        def _md5(p: Path) -> str:
            h = hashlib.md5()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            return h.hexdigest()

        if len({(p.stat().st_size, _md5(p)) for p in found}) > 1:
            raise WeightsRefused(
                f"Tag {tag!r} names {len(found)} DIFFERENT weight files; refusing to pick one:\n"
                + "\n".join(f"  {p}" for p in found)
            )
        return (cuda_found or found)[0]  # identical copies: prefer the CUDA one

    only = found[0]
    # A laptop file must not stand in for a CUDA run of the same condition.
    if WEIGHTS_DIR in only.parents and not allow_metal:
        twin = re.compile(
            rf"^{re.escape(tag)}(_s\d+)?_cuda\d*(_s\d+)?_"
            rf"(?:{'|'.join(map(re.escape, variants))})\.weights\.h5$"
        )
        twins = sorted(
            p for d in cuda_dirs if d.is_dir() for p in d.iterdir() if twin.match(p.name)
        )
        if twins:
            raise WeightsRefused(
                f"Tag {tag!r} resolves only to the laptop file {only}, trained on "
                f"tensorflow-metal, but CUDA run(s) of the same condition "
                f"exist:\n"
                + "\n".join(f"  {p}" for p in twins)
                + "\nAsk for the CUDA tag explicitly, or pass allow_metal=True for a "
                "historical reading."
            )
    return only


LOGS_DIR: Path = OUTPUTS_DIR / "logs"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"

TARGET_SIZE: tuple[int, int] = (224, 224)

# The two case files are pooled and split once by patient, stratified on the label,
# into about 70/15/15.
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
BATCH_SIZE: int = 32
MAX_EPOCHS: int = 100
INITIAL_LR: float = 1e-3
REGULARISED_LR: float = 5e-4  # the regularised CNN trains at a lower rate

# Intensity scaling

# How raw 16-bit pixels are mapped to [0, 1]: fixed_max, percentile, minmax or clahe.
# Whole images use fixed_max, which keeps brightness differences between images.
INTENSITY_NORM: str = "fixed_max"

# Crops use percentile windowing: each is stretched between its own percentiles.
CROP_INTENSITY_NORM: str = "percentile"

# Crops are cut from the mammogram around the mask's box plus a margin. The dataset's
# own cropped patches are not used.
CROP_SIZING: str = "mask_margin"
PERCENTILE_LOW: float = 1.0
PERCENTILE_HIGH: float = 99.0

# CLAHE clip limit, on skimage's [0, 1] scale. The mean and SD are approximate.
CLAHE_CLIP_LIMIT: float = 0.02
CLAHE_NORM_MEAN: float = 0.45
CLAHE_NORM_STD: float = 0.30

# Mean and SD used to standardise percentile crops for the scratch CNNs. stores.py norms
# measures them on zero-padded squares, not on the stretched crops the pipeline produces, so
# standardised inputs are not exactly zero-mean and unit-variance; every split uses the same
# values.
PERCENTILE_NORM_MEAN: float = 0.4430  # crop training partition at margin 0.20
PERCENTILE_NORM_STD: float = 0.3049

# Placeholder values, not measured; no reported model uses the minmax mode.
MINMAX_NORM_MEAN: float = 0.50
MINMAX_NORM_STD: float = 0.25

# Smallest share of its bounding box that a mask's foreground may fill. Every training and
# validation mask fills at least 0.40, so less means stray foreground stretched the box.
MASK_MIN_BOX_FILL: float = 0.20

# A sample that fails to decode would enter a batch as a black tile with a real label.
# A split with more than this share of failures halts the run.
MAX_LOAD_FAILURE_RATE: float = 0.005

CROP_MARGIN_FRAC: float = 0.20  # fraction of the box side, added on every side

# Mean and SD used to standardise fixed_max crops for the scratch CNNs, measured by stores.py
# norms on the dataset's own tight crops padded to squares; no reported model uses them.
CROP_NORM_MEAN: float = 0.6393  # crop training partition
CROP_NORM_STD: float = 0.2798
FINETUNE_LR: float = 1e-5

EARLY_STOPPING_PATIENCE: int = 10
REGULARISED_PATIENCE: int = 15  # the regularised CNN peaks later

# Lighter regularisation for the regularised CNN on crops. The builder's defaults
# (l2=1e-4, dropout=0.5) underfit there.
REGULARISED_CROP_L2: float = 3e-5
REGULARISED_CROP_HEAD_DROPOUT: float = 0.3

# Order in both modes: scale to [0, 1], replicate to three channels (pretrained models
# only), augment, standardise. Standardising last keeps fill borders black.
NORM_STATS: dict = {
    "method": "divide_by_max_16bit",
    "divisor": 65_535,
    # scratch-mode whole images only; not computed on the current training partition,
    # and no reported model uses it
    "global_mean_01": 0.2128,
    "global_std_01": 0.2651,
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
}

# Whole-image augmentation: flip, rotation and zoom-in, plus brightness jitter for the
# scratch CNNs.
AUG_MAX_ROTATION_DEG: float = 10.0
AUG_BRIGHTNESS_DELTA: float = 0.2
AUG_ZOOM_FACTOR: float = 0.10
# Crops are brighter than whole images, so they take less brightness jitter.
CROP_AUG_BRIGHTNESS_DELTA: float = 0.15

# Crop augmentation: flip, rotation with reflect fill and zoom in and out, plus mild
# contrast and gamma jitter for the scratch CNNs. No vertical flip or shear.
CROP_AUG_MAX_ROTATION_DEG: float = 90.0
CROP_AUG_ZOOM_RANGE: tuple[float, float] = (-0.15, 0.15)  # (zoom-in, zoom-out)
CROP_AUG_CONTRAST_DELTA: float = 0.15  # random_contrast in [1-δ, 1+δ]
CROP_AUG_GAMMA_RANGE: tuple[float, float] = (0.80, 1.20)

# Crops are stretched to the input, not padded to a square first.
CROP_PAD_SQUARE: bool = False

# Specialist baseline

# The radiologist's BI-RADS assessment, read as an ordered score. Assessment 0 means
# incomplete imaging (ACR, 2013), not low suspicion, so those images are left out.
BIRADS: dict = {
    "variant": "drop_zero",
    "threshold": 4.0,
    "auc_roc": 0.809,
    "auc_roc_ci": (0.747, 0.872),
    "auc_pr": 0.770,
    "sens": 0.940,
    "spec": 0.427,
    "n_test_comparison": 203,
    "n_test_full": 221,
    # Images with a category from 1 to 5. Every comparison with a model is paired on them.
    "comparison_set": "artifacts/birads_comparison_set.csv",
    "partition_source": "lesion-level patient partition (data_loader.patient_partition)",
}

# Malignancy rates (NHS England, 2026: 2,150,000 screened; 73,451 recalled; 19,291
# cancers). Predictive values and net benefit are re-weighted to these.
PREVALENCE: dict = {
    "cbis_test": 0.4796,
    "nhs_assessment": 0.2626,  # 19,291 / 73,451
    "nhs_screening": 0.0090,  # 19,291 / 2,150,000
}

# An image's score is the highest probability among its lesions.
IMAGE_SCORE_AGGREGATION: str = "max"

# Targets from the literature (Wang, 2024, Table 5), measured on other studies' splits.
LITERATURE_AUC_REFERENCE: float = 0.90
LITERATURE_SENSITIVITY_REFERENCE: float = 0.85
LITERATURE_SPECIFICITY_REFERENCE: float = 0.80

BOOTSTRAP_ITERATIONS: int = 1_000
BOOTSTRAP_CI_LEVEL: float = 0.95

# The operating point is the highest validation threshold that still reaches this
# sensitivity. It is carried to test unchanged.
TARGET_SENSITIVITY: float = 0.90

# The grouping unit for the split and for the cluster bootstrap.
GROUP_COL: str = "patient_id"

CLASS_WEIGHT_STRATEGY: str = "balanced"

LABEL_MAP: dict[str, int] = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}


# ROI mask overlay: the mask is dilated by this many pixels, at input resolution, before
# tissue outside it is zeroed.
ROI_MASK_DILATION_PX: int = 25

# Localiser

# The canvas is a letterbox that keeps the mammogram's shape. A square input would be
# about 40% padding.
LOC_INPUT_H: int = 1024
LOC_INPUT_W: int = 576

# Upper clip percentile of the localiser's inputs, equal to the one its caches use.
LOC_PERCENTILE_HIGH: float = 99.0
LOC_DEPTH: int = 4  # four halvings of resolution
LOC_BASE_FILTERS: int = 32
LOC_BATCH_SIZE: int = 4

# Foreground is well under 1% of a mammogram, so cross-entropy alone collapses to all
# background. The loss adds soft Dice.
LOC_DICE_BCE_WEIGHTS: tuple[float, float] = (0.5, 0.5)

# Post-processing of a predicted mask: threshold it, keep one component, and treat a
# component under LOC_MIN_COMPONENT_PX as no detection.
LOC_MASK_THRESHOLD: float = 0.10
LOC_KEEP_LARGEST_COMPONENT: bool = True
LOC_MIN_COMPONENT_PX: int = 64
# localise_eval labels a localiser as passing its gate at this mean box IoU
LOC_GATE_BOX_IOU: float = 0.30

# localise_eval also keeps the top components by summed probability and matches them
# to the image's lesions, for mammograms with several lesions.
LOC_TOPK: int = 3

# Training targets merge all the masks of a mammogram, so a second lesion is never
# labelled background. Evaluation stays per lesion.
LOC_TENSORS_MERGED_DIR = OUTPUTS_DIR / "localiser_tensors_merged"

LOC_LR: float = 1e-3  # read by the provenance notebooks
LOC_EPOCHS: int = 120

# VGG16 encoder, trained on patches that oversample the lesion.
LOC_ENCODER: str = "vgg16"  # lighter in memory than ResNet50
LOC_PATCH_SIZE: int = 512
LOC_PATCH_BATCH_SIZE: int = 8
LOC_PATCHES_PER_IMAGE: int = 2
LOC_LESION_CENTRED_FRAC: float = 0.5  # share of an image's patches centred on a lesion
LOC_PATCH_JITTER_FRAC: float = 0.25  # centre jitter, as a fraction of the patch size
# Stronger regularisation and augmentation for the patch localiser.
LOC_R4C_DECODER_DROPOUT: float = 0.3  # after every decoder block
LOC_R4C_DECODER_L2: float = 1e-4  # on decoder conv kernels
LOC_R4C_LESION_CENTRED_FRAC: float = 0.30  # per patch, not an exact share
LOC_R4C_ELASTIC_ALPHA: tuple[float, float] = (3.0, 6.0)  # mean absolute displacement, px
LOC_R4C_ELASTIC_SIGMA: tuple[float, float] = (32.0, 64.0)  # field smoothing, px
LOC_R4C_GAMMA: tuple[float, float] = (0.7, 1.4)
LOC_R4C_BRIGHTNESS: float = 0.15  # additive, as a fraction of the [0, 1] range
LOC_R4C_CONTRAST: float = 0.15  # multiplicative, about the patch mean
LOC_R4C_NOISE_SD: float = 0.01  # Gaussian noise on the [0, 1] image
LOC_ENC_LR: float = 1e-4  # pretrained encoder
LOC_DEC_LR: float = 1e-3  # new decoder
LOC_WARMUP_EPOCHS: int = 5  # linear warm-up, then cosine
LOC_LR_MIN: float = 1e-6  # cosine floor at LOC_EPOCHS
LOC_PATIENCE: int = 20  # read by the provenance notebooks
LOC_START_FROM_EPOCH: int = 20  # read by the provenance notebooks
LOC_RLROP_PATIENCE: int = 8  # read by the provenance notebooks

# Differences in single-seed validation AUC below this are within seed-to-seed noise.
VAL_AUC_NOISE_FLOOR: float = 0.02

# Localiser post-processing, swept on cached probability maps: selected on the training
# partition and gated on validation by box IoU.
LOC_BUNDLE_DIR = OUTPUTS_DIR / "results" / "localiser_bundle"
LOC_BUNDLE_RUNS: tuple[str, ...] = ("run1", "run3")
LOC_BUNDLE_THRESHOLD_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
LOC_BUNDLE_MIN_PX_GRID: tuple[int, ...] = (16, 32, 64, 128, 256)
LOC_BUNDLE_RULES: tuple[str, ...] = ("largest", "sum", "peak")  # which component to keep
LOC_BUNDLE_BREAST_OPENING_PX: int = 7  # the opening localise_eval uses for the breast box

# The Grad-CAM map is binarised at this fraction of its own maximum (Zhou et al., 2016).
CAM_THRESHOLD: float = 0.3  # chosen on validation
CAM_THRESHOLD_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
CAM_MIN_COMPONENT_PX: int = 16  # on the 224 x 224 input
CAM_SWEEP_TOL: float = 0.005  # plateau width when choosing the threshold

# Context fusion pairs a lesion's crop with a second crop of it at this wider margin.
FUSION_CONTEXT_MARGIN: float = 1.00

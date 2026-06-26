"""
data_loader.py — CBIS-DDSM mass-case data pipeline.

Responsibilities:
  - Load, deduplicate, and POOL the train/test CSVs (Lee et al.'s official
    train/test boundary is discarded post-pool)
  - Assign binary labels (malignant=1, benign=0)
  - Patient-level, label-stratified ~70/15/15 split via StratifiedGroupKFold,
    enforced by a hard assertion against cross-partition patient leakage
  - Pad-to-square + resize + normalise each DICOM
  - Return tf.data.Dataset objects for train, val, and test splits
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold 
from src import config

_LABEL_MAP: dict[str, int] = config.LABEL_MAP

#  ImageNet normalisation constants
_IMAGENET_MEAN = np.array(config.NORM_STATS["imagenet_mean"], dtype=np.float32)
_IMAGENET_STD = np.array(config.NORM_STATS["imagenet_std"],  dtype=np.float32)

# Augmentation layers created once at module level.
_AUG_ROTATION = tf.keras.layers.RandomRotation(
    factor=config.AUG_MAX_ROTATION_DEG / 360.0,
    fill_mode="constant",
    fill_value=0.0,
)
_AUG_ZOOM = tf.keras.layers.RandomZoom(
    height_factor=(-config.AUG_ZOOM_FACTOR, 0.0),
    fill_mode="constant",
    fill_value=0.0,
)

def build_datasets(
    train_csv: str | Path,
    test_csv: str | Path,
    train_img_dir: str | Path,
    test_img_dir: str | Path,
    *,
    target_size: tuple[int, int] = config.TARGET_SIZE,
    val_fraction: float = config.VAL_FRACTION,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    normalisation: Literal["scratch", "imagenet"] = "scratch",
    augment: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame]:
    """
    Build train, validation, and test tf.data.Dataset objects.

    Returns:
        (train_ds, val_ds, test_ds, test_frame) — test_frame is retained for
        evaluation in evaluate.py.
    """
    _set_seeds(seed)

    train_csv = Path(train_csv)
    test_csv = Path(test_csv)
    train_img_dir = Path(train_img_dir)
    test_img_dir = Path(test_img_dir)

    train_frame = _prepare_frame(train_csv)
    test_frame_csv = _prepare_frame(test_csv)
    pool_frame = _pool_frames(train_frame, test_frame_csv)

    train_split, val_split, test_split = _three_way_split(
        pool_frame, test_fraction=config.TEST_FRACTION,
        val_fraction=val_fraction, seed=seed,
    )

    lookup = _build_combined_lookup(train_img_dir, test_img_dir)

    train_ds = _make_dataset(train_split, lookup, target_size, batch_size, seed,
                             normalisation=normalisation, augment=augment, shuffle=True)
    val_ds   = _make_dataset(val_split, lookup, target_size, batch_size, seed,
                             normalisation=normalisation, augment=False, shuffle=False)
    test_ds  = _make_dataset(test_split, lookup, target_size, batch_size, seed,
                             normalisation=normalisation, augment=False, shuffle=False)

    _log_split_summary(train_split, val_split, test_split)
    return train_ds, val_ds, test_ds, test_split


# Internal helpers

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _prepare_frame(csv_path: Path) -> pd.DataFrame:
    """Load CSV, assign binary labels, deduplicate on image path."""
    df = pd.read_csv(csv_path)
    required = {"patient_id", "pathology", "image file path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    df["label"] = df["pathology"].map(_LABEL_MAP)
    unexpected = df["label"].isna().sum()
    if unexpected:
        bad = df[df["label"].isna()]["pathology"].unique()
        raise ValueError(f"{unexpected} unmapped pathology values in {csv_path.name}: {bad}")
    df["label"] = df["label"].astype(int)

    deduped = (
        df.groupby("image file path", sort=False)
        .agg(
            patient_id=("patient_id", "first"),
            label=("label", "max"),
        )
        .reset_index()
    )
    n_dropped = len(df) - len(deduped)
    if n_dropped:
        print(f"[data_loader] {csv_path.name}: dropped {n_dropped} duplicate rows "
              f"({len(df)} -> {len(deduped)} unique images)")

    return deduped


def _pool_frames(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Pool both deduplicated mass frames and re-dedup defensively on image path."""
    pool = pd.concat([a, b], ignore_index=True)
    return (
        pool.groupby("image file path", sort=False)
        .agg(patient_id=("patient_id", "first"), label=("label", "max"))
        .reset_index()
    )


def _sgkf_holdout(df: pd.DataFrame, frac: float, seed: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a ~`frac` patient-grouped, label-stratified holdout off `df`."""
    n_splits = max(2, round(1.0 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    return (df.iloc[keep_idx].reset_index(drop=True),
            df.iloc[hold_idx].reset_index(drop=True))


def _three_way_split(df: pd.DataFrame, *, test_fraction: float,
                     val_fraction: float, seed: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-level, label-stratified 70/15/15 split over the pooled mass cases."""
    trainval, test_split = _sgkf_holdout(df, test_fraction, seed)
    val_rel = val_fraction / (1.0 - test_fraction)   # val as a fraction of train+val
    train_split, val_split = _sgkf_holdout(trainval, val_rel, seed)
    _assert_disjoint(train_split, val_split, test_split)
    return train_split, val_split, test_split


def _assert_disjoint(train_split: pd.DataFrame, val_split: pd.DataFrame,
                     test_split: pd.DataFrame) -> None:
    """Halt if patients are shared across ANY pair of partitions."""
    p_tr = set(train_split["patient_id"])
    p_va = set(val_split["patient_id"])
    p_te = set(test_split["patient_id"])
    tv, tt, vt = len(p_tr & p_va), len(p_tr & p_te), len(p_va & p_te)
    if tv or tt or vt:
        raise RuntimeError(
            f"Patient leakage across partitions — train∩val={tv}, "
            f"train∩test={tt}, val∩test={vt}"
        )


def _build_combined_lookup(train_img_dir: Path, test_img_dir: Path) -> dict[Path, Path]:
    """Merge train/test directory lookups; raises on any relative-path collision
    rather than silently letting one overwrite the other."""
    train_lookup = _build_dir_lookup(train_img_dir)
    test_lookup = _build_dir_lookup(test_img_dir)
    collisions = set(train_lookup) & set(test_lookup)
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} relative path(s) exist in both directories — "
            f"example: {next(iter(collisions))}"
        )
    return {**train_lookup, **test_lookup}

def _build_dir_lookup(img_dir: Path) -> dict[Path, Path]:
    """
    Map each parent directory (relative to img_dir) to a .dcm file path.

    The CSV uses 000000.dcm filenames; TCIA downloads use UUID filenames.
    Matching on the parent directory (which is stable) resolves this mismatch.
    """
    return {
        p.parent.relative_to(img_dir): p
        for p in img_dir.glob("**/*.dcm")
    }


def _load_image(
    csv_path_str: str,
    lookup: dict[Path, Path],
    target_size: tuple[int, int],
    normalisation: Literal["scratch", "imagenet"],
) -> np.ndarray | None:
    """Returns float32 (H, W, C) array, or None if the file is not found."""
    parent_dir = Path(csv_path_str).parent
    dcm_path = lookup.get(parent_dir)
    if dcm_path is None:
        return None

    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        print(f"[data_loader] WARNING: could not read {dcm_path}: {e}")
        return None

    interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if interp == "MONOCHROME1":
        print(f"[data_loader] WARNING: MONOCHROME1 detected in {dcm_path.name} — inverting")
        arr = arr.max() - arr

    # Fixed-divisor scaling to [0,1] — a constant, not a per-partition
    # statistic. global_mean_01/global_std_01 standardisation for "scratch"
    # mode is applied LATER, post-augmentation (_make_standardize_fn) — not
    # here, since brightness jitter needs [0,1] values to operate on.
    arr = arr / float(config.NORM_STATS["divisor"])
    arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape
    sq = max(h, w)
    padded = np.zeros((sq, sq), dtype=np.float32)
    padded[:h, :w] = arr

    target_h, target_w = target_size
    resized = tf.image.resize(padded[..., np.newaxis], [target_h, target_w]).numpy()

    if normalisation == "imagenet":
        resized = np.repeat(resized, 3, axis=-1)
        resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD

    return resized.astype(np.float32)


def _make_dataset(
    df: pd.DataFrame,
    lookup: dict[Path, Path],
    target_size: tuple[int, int],
    batch_size: int,
    seed: int,
    *,
    normalisation: Literal["scratch", "imagenet"],
    augment: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    Build a batched tf.data.Dataset from a deduplicated DataFrame.

    Images that cannot be loaded are silently skipped (logged to stdout).
    """
    images, labels = [], []
    skipped = 0

    for _, row in df.iterrows():
        img = _load_image(
            row["image file path"],
            lookup,
            target_size,
            normalisation,
        )
        if img is None:
            skipped += 1
            continue
        images.append(img)
        labels.append(int(row["label"]))

    if skipped:
        print(f"[data_loader] WARNING: {skipped} images could not be loaded and were skipped")

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((images_arr, labels_arr))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels_arr), seed=seed, reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(_make_augment_fn(normalisation), num_parallel_calls=tf.data.AUTOTUNE)

    # Always applied — val/test need standardising too, just not augmenting.
    ds = ds.map(_make_standardize_fn(normalisation), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def _make_standardize_fn(normalisation: Literal["scratch", "imagenet"]):
    """Standardise with train-set stats. "imagenet" mode is a no-op here —
    it's already standardised in _load_image, before augmentation."""
    if normalisation != "scratch":
        return lambda image, label: (image, label)
    mean = config.NORM_STATS["global_mean_01"]
    std = config.NORM_STATS["global_std_01"]
    def _fn(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return (image - mean) / std, label
    return _fn

def _make_augment_fn(
    normalisation: Literal["scratch", "imagenet"],
):
    """
    Return an augmentation function appropriate for the given normalisation mode.

    Brightness and [0,1] clipping are only safe in "scratch" mode where pixels live in
    [0, 1]. In "imagenet" mode values are zero-centred, so clipping would corrupt the
    training signal. Brightness is skipped for "imagenet"; apply it pre-normalisation
    if needed.
    """
    def _fn(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_flip_left_right(image)
        image = _AUG_ROTATION(image[tf.newaxis], training=True)[0]
        image = _AUG_ZOOM(image[tf.newaxis], training=True)[0]
        if normalisation == "scratch":
            image = tf.image.random_brightness(image, max_delta=config.AUG_BRIGHTNESS_DELTA)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    return _fn


def _log_split_summary(train_split, val_split, test_split):
    p_tr, p_va, p_te = (set(d["patient_id"]) for d in (train_split, val_split, test_split))
    print("\n--- Split Summary ---")
    for a, b, na, nb in [(p_tr, p_va, "train", "val"),
                         (p_tr, p_te, "train", "test"),
                         (p_va, p_te, "val", "test")]:
        n = len(a & b)
        print(f"  {na}/{nb} patient-level leakage check: {'PASS' if not n else f'FAIL ({n})'}")
    n_tr, n_va, n_te = len(train_split), len(val_split), len(test_split)
    n_total = n_tr + n_va + n_te
    print(f"  Images — train: {n_tr} ({n_tr/n_total:.1%}) | "
          f"val: {n_va} ({n_va/n_total:.1%}) | test: {n_te} ({n_te/n_total:.1%})\n")
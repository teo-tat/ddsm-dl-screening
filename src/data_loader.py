"""
data_loader.py — CBIS-DDSM mass-case data pipeline.

Responsibilities:
  - Load and deduplicate the train/test CSVs
  - Assign binary labels (malignant=1, benign=0)
  - Patient-level stratified train/validation split (GroupShuffleSplit)
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
from sklearn.model_selection import GroupShuffleSplit
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
    test_frame = _prepare_frame(test_csv)

    train_split, val_split = _patient_split(train_frame, val_fraction, seed)

    train_lookup = _build_dir_lookup(train_img_dir)
    test_lookup = _build_dir_lookup(test_img_dir)

    train_ds = _make_dataset(
        train_split, train_lookup, target_size, batch_size, seed,
        normalisation=normalisation, augment=augment, shuffle=True,
    )
    val_ds = _make_dataset(
        val_split, train_lookup, target_size, batch_size, seed,
        normalisation=normalisation, augment=False, shuffle=False,
    )
    test_ds = _make_dataset(
        test_frame, test_lookup, target_size, batch_size, seed,
        normalisation=normalisation, augment=False, shuffle=False,
    )

    _log_split_summary(train_split, val_split, test_frame)
    return train_ds, val_ds, test_ds, test_frame


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


def _patient_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level split; GroupShuffleSplit does not support stratify so label
    balance across subsets is approximate."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))

    train_split = df.iloc[train_idx].reset_index(drop=True)
    val_split = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_split["patient_id"]) & set(val_split["patient_id"])
    if overlap:
        raise RuntimeError(
            f"Patient leakage in train/val split: {len(overlap)} patients overlap"
        )

    return train_split, val_split


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

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


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


def _log_split_summary(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    print("\n[data_loader] Split summary")
    print(f"  Train : {len(train_split):4d} images  "
          f"(malignant: {train_split['label'].sum()}, "
          f"benign: {(train_split['label'] == 0).sum()})")
    print(f"  Val   : {len(val_split):4d} images  "
          f"(malignant: {val_split['label'].sum()}, "
          f"benign: {(val_split['label'] == 0).sum()})")
    print(f"  Test  : {len(test_frame):4d} images  "
          f"(malignant: {test_frame['label'].sum()}, "
          f"benign: {(test_frame['label'] == 0).sum()})")

    overlap = set(train_split["patient_id"]) & set(val_split["patient_id"])
    status = "PASS" if not overlap else f"FAIL ({len(overlap)} patients overlap)"
    print(f"  Train/val patient-level leakage check: {status}")

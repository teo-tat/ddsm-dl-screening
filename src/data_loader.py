"""CBIS-DDSM mass-case data pipeline: pool and deduplicate the train/test CSVs,
label them (malignant=1, benign=0), split patients 70/15/15 with label
stratification, and decode each DICOM into the train/val/test tf.data pipelines."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Literal, TYPE_CHECKING
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from . import config

if TYPE_CHECKING:  # annotation only
    from .boxes import BoxProvider

_LABEL_MAP: dict[str, int] = config.LABEL_MAP

# ImageNet normalisation constants
_IMAGENET_MEAN = np.array(config.NORM_STATS["imagenet_mean"], dtype=np.float32)
_IMAGENET_STD = np.array(config.NORM_STATS["imagenet_std"], dtype=np.float32)

# Whole-image augmentation: constant black fill, zoom-in only.
_AUG_ROTATION = tf.keras.layers.RandomRotation(
    factor=config.AUG_MAX_ROTATION_DEG / 360.0, fill_mode="constant", fill_value=0.0
)
_AUG_ZOOM = tf.keras.layers.RandomZoom(
    height_factor=(-config.AUG_ZOOM_FACTOR, 0.0), fill_mode="constant", fill_value=0.0
)
# Crop augmentation: reflect fill leaves no black wedge, and the symmetric zoom
# never clips the lesion out of frame.
_AUG_ROTATION_CROP = tf.keras.layers.RandomRotation(
    factor=config.CROP_AUG_MAX_ROTATION_DEG / 360.0, fill_mode="reflect"
)
_AUG_ZOOM_CROP = tf.keras.layers.RandomZoom(
    height_factor=config.CROP_AUG_ZOOM_RANGE, fill_mode="reflect"
)

import threading

_LOAD_STATS = {"attempted": 0, "failed": 0, "fallback": 0, "errors": []}
_LOAD_LOCK = threading.Lock()


def _record_load(ok: bool, path: str = "", err: str = "", fallback: bool = False) -> None:
    """Thread-safe tally of per-sample decode outcomes; `fallback` counts samples
    that fell back to the whole mammogram because the localiser produced no box."""
    with _LOAD_LOCK:
        _LOAD_STATS["attempted"] += 1
        if fallback:
            _LOAD_STATS["fallback"] += 1
        if not ok:
            _LOAD_STATS["failed"] += 1
            if len(_LOAD_STATS["errors"]) < 20:
                _LOAD_STATS["errors"].append(f"{path}: {err}")


def reset_load_stats() -> None:
    with _LOAD_LOCK:
        _LOAD_STATS.update(attempted=0, failed=0, fallback=0, errors=[])


def assert_load_health(tolerance: float = config.MAX_LOAD_FAILURE_RATE) -> dict:
    """Halt if too many samples decoded to a black tile. Decoding is lazy, so call
    this after a full pass over a dataset, not straight after build_datasets."""
    with _LOAD_LOCK:
        stats = dict(_LOAD_STATS)
    n, f = stats["attempted"], stats["failed"]
    rate = (f / n) if n else 0.0
    print(f"[data_loader] decode health: {f}/{n} failed ({rate:.3%})")
    fb = stats.get("fallback", 0)
    print(f"[data_loader] whole-image fallbacks: {fb}/{n} " f"({(fb / n if n else 0):.2%})")
    if rate > tolerance:
        raise RuntimeError(
            f"Decode failure rate {rate:.3%} exceeds {tolerance:.3%}. These "
            f"samples became black tiles with real labels. First failures:\n  "
            + "\n  ".join(stats["errors"])
        )
    return stats


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
    clahe: bool = False,
    source: Literal["full", "crop"] = "full",
    train_roi_dir: str | Path | None = None,
    test_roi_dir: str | Path | None = None,
    pad_square: bool | None = None,
    intensity_norm: str | None = None,
    crop_sizing: str = config.CROP_SIZING,
    crop_margin: float = config.CROP_MARGIN_FRAC,
    mask_overlay: bool = False,
    offline_aug_k: int = 1,
    box_provider: "BoxProvider | None" = None,
    shuffle_seed: int | None = None,
    shuffle_train: bool = True,
    standardize: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame]:
    """Build the train, validation and test pipelines and return them with the test
    and train frames; without the test-pass token the test split is refused."""
    config.set_seeds(seed)
    # None decides by source: whole mammograms are far from square and need
    # padding, crops do not. An explicit True/False still wins.
    if pad_square is None:
        pad_square = config.CROP_PAD_SQUARE if source == "crop" else True

    train_csv = Path(train_csv)
    test_csv = Path(test_csv)

    path_col = "cropped image file path" if source == "crop" else "image file path"

    train_frame = _prepare_frame(train_csv, path_col)
    test_frame_csv = _prepare_frame(test_csv, path_col)
    pool_frame = _pool_frames(train_frame, test_frame_csv, path_col)

    train_split, val_split, test_split = _three_way_split(
        pool_frame, test_fraction=config.TEST_FRACTION, val_fraction=val_fraction, seed=seed
    )

    if clahe:
        strategy = "clahe"
    elif intensity_norm is not None:
        strategy = intensity_norm  # explicit caller override wins
    else:
        strategy = config.CROP_INTENSITY_NORM if source == "crop" else config.INTENSITY_NORM

    if source == "crop":
        if train_roi_dir is None or test_roi_dir is None:
            raise ValueError("source='crop' requires train_roi_dir and test_roi_dir")
        lookup = _build_combined_roi_lookup(Path(train_roi_dir), Path(test_roi_dir))
        # Both modes crop from the full mammogram, so both need the name-keyed
        # full-image lookup; only mask_margin also needs the masks.
        if crop_sizing in ("mask_margin", "box_provider"):
            mask_lookup = (
                _build_combined_roi_lookup(Path(train_roi_dir), Path(test_roi_dir), pick="mask")
                if crop_sizing == "mask_margin"
                else None
            )
            _full = _build_combined_lookup(Path(train_img_dir), Path(test_img_dir))
            full_lookup = {Path(k).parts[0]: v for k, v in _full.items()}
        else:
            mask_lookup = full_lookup = None
    else:
        lookup = _build_combined_lookup(Path(train_img_dir), Path(test_img_dir))
        mask_lookup = full_lookup = None

    common = dict(
        normalisation=normalisation,
        intensity_norm=strategy,
        source=source,
        path_col=path_col,
        pad_square=pad_square,
        crop_sizing=crop_sizing,
        crop_margin=crop_margin,
        mask_lookup=mask_lookup,
        full_lookup=full_lookup,
        mask_overlay=mask_overlay,
        offline_aug_k=offline_aug_k,
        box_provider=box_provider,
        standardize=standardize,
    )
    ds_seed = seed if shuffle_seed is None else shuffle_seed
    train_ds = _make_dataset(
        train_split,
        lookup,
        target_size,
        batch_size,
        ds_seed,
        augment=augment,
        shuffle=shuffle_train,
        **common,
    )
    val_ds = _make_dataset(
        val_split, lookup, target_size, batch_size, seed, augment=False, shuffle=False, **common
    )
    _log_split_summary(train_split, val_split, test_split)
    # The test split is built only for the authorised pass; otherwise its rows are
    # never walked and the placeholder returned in its place refuses every use.
    from .test_pass_guard import authorised

    if not authorised():
        refused = _RefusedSplit("data_loader.build_datasets() test split")
        return train_ds, val_ds, refused, refused, train_split
    test_ds = _make_dataset(
        test_split, lookup, target_size, batch_size, seed, augment=False, shuffle=False, **common
    )
    return train_ds, val_ds, test_ds, test_split, train_split


class _GuardedSplits(dict):
    """The split dict with the test partition behind the freeze token: reading
    `frames["test"]` raises unless this process is the authorised single pass."""

    def __getitem__(self, key):
        if key == "test":
            from .test_pass_guard import require_test_pass

            require_test_pass("data_loader.split_frames()['test']")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "test":
            from .test_pass_guard import require_test_pass

            require_test_pass("data_loader.split_frames().get('test')")
        return super().get(key, default)

    # items() and values() hand over the test frame; __iter__ makes dict(frames)
    # fetch values through __getitem__ rather than copying them directly.
    def items(self):
        from .test_pass_guard import require_test_pass

        require_test_pass("data_loader.split_frames().items()")
        return super().items()

    def values(self):
        from .test_pass_guard import require_test_pass

        require_test_pass("data_loader.split_frames().values()")
        return super().values()

    def __iter__(self):
        return super().__iter__()

    def copy(self):
        return _GuardedSplits({k: self[k] for k in self})


class _RefusedSplit:
    """Stands in for the test split when the freeze token is absent; any use of it
    raises the guard's PermissionError."""

    def __init__(self, caller: str):
        object.__setattr__(self, "_caller", caller)

    def _refuse(self, *args, **kwargs):
        from .test_pass_guard import require_test_pass

        require_test_pass(object.__getattribute__(self, "_caller"))
        raise PermissionError(object.__getattribute__(self, "_caller"))

    def __getattr__(self, name):
        return self._refuse()

    __iter__ = __len__ = __getitem__ = __bool__ = _refuse


def split_frames(
    source: Literal["full", "crop"] = "full",
    seed: int = config.SEED,
    val_fraction: float = config.VAL_FRACTION,
) -> dict[str, pd.DataFrame]:
    """The three partition frames, exactly as build_datasets derives them, for
    callers that need the metadata without building tf.data pipelines."""
    path_col = "cropped image file path" if source == "crop" else "image file path"
    pool = _pool_frames(
        _prepare_frame(config.TRAIN_CSV, path_col),
        _prepare_frame(config.TEST_CSV, path_col),
        path_col,
    )
    tr, va, te = _three_way_split(
        pool, test_fraction=config.TEST_FRACTION, val_fraction=val_fraction, seed=seed
    )
    return _GuardedSplits(train=tr, val=va, test=te)


# Internal helpers

# Deduplication policy: rows collapse only for source="full", where one mammogram
# can carry several abnormalities. max keeps malignant, min the least conspicuous.
_DEDUP_POLICY: dict[str, tuple[str, ...]] = {
    "first": (
        "patient_id",
        "breast_density",
        "breast density",
        "left or right breast",
        "image view",
        "abnormality id",
        "abnormality type",
        "mass shape",
        "mass margins",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
    ),
    "max": ("label", "pathology", "assessment"),
    "min": ("subtlety",),
}

# Columns that must not vary within a group; a violation means the group key
# is wrong, not that the value needs picking.
_CONSTANT_PER_GROUP = (
    "patient_id",
    "breast_density",
    "breast density",
    "left or right breast",
    "image view",
)


def _dedup_frame(df: pd.DataFrame, path_col: str, *, origin: str = "") -> pd.DataFrame:
    """Collapse rows sharing ``path_col``, adding n_abnormalities and labels_mixed
    (the merged rows disagreed on label). Idempotent, so pooling can reapply it."""
    if "label" not in df.columns:
        raise ValueError("_dedup_frame requires a 'label' column")
    if "n_abnormalities" not in df.columns:
        df = df.assign(n_abnormalities=1)
    if "labels_mixed" not in df.columns:
        df = df.assign(labels_mixed=False)

    agg_map: dict[str, str] = {}
    for how, cols in _DEDUP_POLICY.items():
        for col in cols:
            if col in df.columns and col != path_col:
                agg_map[col] = how
    agg_map["n_abnormalities"] = "sum"
    agg_map["labels_mixed"] = "max"

    grouped = df.groupby(path_col, sort=False)

    for col in _CONSTANT_PER_GROUP:
        if col in df.columns and col != path_col:
            n_bad = int((grouped[col].nunique(dropna=False) > 1).sum())
            if n_bad:
                print(
                    f"[data_loader] WARNING{origin}: '{col}' varies within "
                    f"{n_bad} '{path_col}' group(s) - keeping the first value"
                )

    label_nunique = grouped["label"].nunique()
    out = grouped.agg(agg_map).reset_index()
    out["labels_mixed"] = out["labels_mixed"] | (out[path_col].map(label_nunique) > 1)

    # Stable join keys for evaluate.py, the BI-RADS comparison set and the ROI
    # lookups, which key on the lesion folder name, not the CSV path.
    if "cropped image file path" in out.columns:
        out["lesion_id"] = out["cropped image file path"].map(
            lambda s: Path(s).parts[0] if isinstance(s, str) else None
        )
    if "image file path" in out.columns:
        out["image_id"] = out["image file path"].map(
            lambda s: Path(s).parts[0] if isinstance(s, str) else None
        )

    return out


def _prepare_frame(csv_path: Path, path_col: str = "image file path") -> pd.DataFrame:
    """Load CSV, assign binary labels, deduplicate on ``path_col``, which sets the
    unit: one row per mammogram, or one per lesion for the cropped path."""
    df = pd.read_csv(csv_path)
    required = {"patient_id", "pathology", path_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    df["label"] = df["pathology"].map(_LABEL_MAP)
    unexpected = df["label"].isna().sum()
    if unexpected:
        bad = df[df["label"].isna()]["pathology"].unique()
        raise ValueError(f"{unexpected} unmapped pathology values in {csv_path.name}: {bad}")
    df["label"] = df["label"].astype(int)

    n_raw = len(df)
    deduped = _dedup_frame(df, path_col, origin=f" [{csv_path.name}]")
    n_dropped = n_raw - len(deduped)
    if n_dropped:
        n_mixed = int(deduped["labels_mixed"].sum())
        print(
            f"[data_loader] {csv_path.name}: merged {n_dropped} duplicate rows "
            f"({n_raw} -> {len(deduped)} unique on '{path_col}'), "
            f"{n_mixed} of which disagreed on label"
        )

    return deduped


def _pool_frames(
    a: pd.DataFrame, b: pd.DataFrame, path_col: str = "image file path"
) -> pd.DataFrame:
    """Pool both deduplicated mass frames and re-dedup defensively on ``path_col``."""
    return _dedup_frame(pd.concat([a, b], ignore_index=True), path_col, origin=" [pooled]")


def _sgkf_holdout(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a ~`frac` patient-grouped, label-stratified holdout off `df`."""
    n_splits = max(2, round(1.0 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    return (df.iloc[keep_idx].reset_index(drop=True), df.iloc[hold_idx].reset_index(drop=True))


_PARTITION_CACHE: dict[tuple, dict[str, str]] = {}


def _sgkf_three_way(
    df: pd.DataFrame, *, test_fraction: float, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw patient-grouped, label-stratified 70/15/15 split of ONE frame."""
    trainval, test_split = _sgkf_holdout(df, test_fraction, seed)
    val_rel = val_fraction / (1.0 - test_fraction)  # val as a fraction of train+val
    train_split, val_split = _sgkf_holdout(trainval, val_rel, seed)
    _assert_disjoint(train_split, val_split, test_split)
    return train_split, val_split, test_split


def patient_partition(
    seed: int = config.SEED,
    test_fraction: float = config.TEST_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> dict[str, str]:
    """patient_id -> 'train' | 'val' | 'test', defined once for the whole project:
    every row set inherits this partition instead of splitting itself."""
    key = (seed, test_fraction, val_fraction)
    if key not in _PARTITION_CACHE:
        col = "cropped image file path"
        pool = _pool_frames(
            _prepare_frame(config.TRAIN_CSV, col), _prepare_frame(config.TEST_CSV, col), col
        )
        part: dict[str, str] = {}
        for name, fr in zip(
            ("train", "val", "test"),
            _sgkf_three_way(
                pool, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
            ),
        ):
            for p in fr["patient_id"].unique():
                part[str(p)] = name
        _PARTITION_CACHE[key] = part
    return _PARTITION_CACHE[key]


def _three_way_split(
    df: pd.DataFrame, *, test_fraction: float, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition any pooled frame by the project-wide patient partition, keeping
    each frame's own row order."""
    part = patient_partition(seed, test_fraction, val_fraction)
    missing = set(df["patient_id"].astype(str)) - set(part)
    if missing:
        raise RuntimeError(
            f"{len(missing)} patients in this frame are absent from "
            f"the lesion-level partition, e.g. {sorted(missing)[:5]}"
        )
    lab = df["patient_id"].astype(str).map(part)
    splits = tuple(df[lab == n].reset_index(drop=True) for n in ("train", "val", "test"))
    _assert_disjoint(*splits)
    return splits


def _assert_disjoint(
    train_split: pd.DataFrame, val_split: pd.DataFrame, test_split: pd.DataFrame
) -> None:
    """Halt if a patient appears in more than one partition."""
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
    """Map each parent directory (relative to img_dir) to a .dcm file path. The CSV
    uses 000000.dcm names while downloads use UUIDs, so the parent is the only key."""
    return {p.parent.relative_to(img_dir): p for p in img_dir.glob("**/*.dcm")}


def _build_combined_roi_lookup(
    train_roi_dir: Path, test_roi_dir: Path, pick: str = "crop"
) -> dict[str, Path]:
    """Merge train/test ROI lookups, keyed by lesion folder name: pick="crop" gives
    the cropped-lesion DICOM, pick="mask" the full-mammogram-sized binary mask."""
    train_lookup = _build_roi_crop_lookup(train_roi_dir, pick=pick)
    test_lookup = _build_roi_crop_lookup(test_roi_dir, pick=pick)
    collisions = set(train_lookup) & set(test_lookup)
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} lesion name(s) exist in both ROI directories — "
            f"example: {next(iter(collisions))}"
        )
    return {**train_lookup, **test_lookup}


def _build_roi_crop_lookup(roi_img_dir: Path, pick: str = "crop") -> dict[str, Path]:
    """Map each lesion folder name to its ROI crop or mask DICOM. Filenames are
    unreliable, so the two are told apart by header size: the mask is the larger."""
    lookup: dict[str, Path] = {}
    for lesion_dir in sorted(roi_img_dir.iterdir()):
        if not lesion_dir.is_dir():
            continue
        dcms = list(lesion_dir.glob("**/*.dcm"))
        if not dcms:
            continue
        if len(dcms) == 1:
            if pick == "crop":
                lookup[lesion_dir.name] = dcms[0]
            continue
        sized: list[tuple[int, Path]] = []
        for d in dcms:
            try:
                hdr = pydicom.dcmread(d, stop_before_pixels=True, force=True)
                area = int(hdr.Rows) * int(hdr.Columns)
            except Exception:
                area = np.iinfo(np.int64).max  # unreadable header → never pick as crop
            sized.append((area, d))
        chooser = max if pick == "mask" else min
        lookup[lesion_dir.name] = chooser(sized, key=lambda t: t[0])[1]
    return lookup


def _normalise_intensity(
    arr: np.ndarray, strategy: str = "fixed_max", percentile_high: float | None = None
) -> np.ndarray:
    """Map raw DICOM pixel values to [0,1] by the named strategy, applied after any
    MONOCHROME1 inversion and before resize."""
    arr = arr.astype(np.float32)
    if strategy == "fixed_max":
        out = arr / float(config.NORM_STATS["divisor"])
    elif strategy == "percentile":
        # percentile_high overrides the configured ceiling here only, so the
        # localiser decode can differ from the classifier path sharing this code.
        hi_pct = config.PERCENTILE_HIGH if percentile_high is None else percentile_high
        lo, hi = np.percentile(arr, [config.PERCENTILE_LOW, hi_pct])
        if hi <= lo:  # degenerate (near-constant) image
            lo, hi = float(arr.min()), float(arr.max())
        out = (arr - lo) / (hi - lo + 1e-8)
    elif strategy == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        out = (arr - lo) / (hi - lo + 1e-8)
    elif strategy == "clahe":
        from skimage.exposure import equalize_adapthist

        base = np.clip(arr / float(config.NORM_STATS["divisor"]), 0.0, 1.0)
        out = equalize_adapthist(base, clip_limit=config.CLAHE_CLIP_LIMIT)
    else:
        raise ValueError(f"Unknown intensity_norm strategy: {strategy!r}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _scratch_norm_stats(strategy: str, source: str) -> tuple[float, float]:
    """(mean, std) for scratch-mode standardisation, from the per-strategy constants in
    config; imagenet mode ignores these and uses the ImageNet channel statistics."""
    if strategy == "clahe":
        return config.CLAHE_NORM_MEAN, config.CLAHE_NORM_STD
    if strategy == "percentile":
        return config.PERCENTILE_NORM_MEAN, config.PERCENTILE_NORM_STD
    if strategy == "minmax":
        return config.MINMAX_NORM_MEAN, config.MINMAX_NORM_STD
    # fixed_max
    if source == "crop":
        return config.CROP_NORM_MEAN, config.CROP_NORM_STD
    return config.NORM_STATS["global_mean_01"], config.NORM_STATS["global_std_01"]


def _pad_and_resize(
    arr: np.ndarray, target_size: tuple[int, int], pad_square: bool = True
) -> np.ndarray:
    """Resize a single-channel 2D array to `target_size`; pad_square=True zero-pads
    to a square canvas first, preserving the aspect ratio of whole mammograms."""
    if pad_square:
        h, w = arr.shape
        sq = max(h, w)
        canvas = np.zeros((sq, sq), dtype=np.float32)
        canvas[:h, :w] = arr
        arr = canvas

    target_h, target_w = target_size
    with tf.device("/CPU:0"):
        resized = tf.image.resize(
            arr[..., np.newaxis], [target_h, target_w], antialias=True
        ).numpy()
    return resized.astype(np.float32)


def _read_full(path_str: str) -> np.ndarray:
    """Decode a DICOM to float32, applying MONOCHROME1 inversion. The localiser
    reuses this read path, so masks and images stay registered against each other."""
    ds_dcm = pydicom.dcmread(path_str, force=True)
    arr = ds_dcm.pixel_array.astype(np.float32)
    if getattr(ds_dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


_SHAPE_CACHE: dict[str, tuple[int, int]] = {}


def _image_shape(path) -> tuple[int, int]:
    """(Rows, Columns) from the DICOM header, cached; no pixels are read."""
    key = str(path)
    if key not in _SHAPE_CACHE:
        hdr = pydicom.dcmread(key, stop_before_pixels=True, force=True)
        _SHAPE_CACHE[key] = (int(hdr.Rows), int(hdr.Columns))
    return _SHAPE_CACHE[key]


def _resolve_lookup_key(path_str: str, source: str):
    """Map a CSV path to its lookup key: the parent directory in full mode, the
    lesion folder name in crop mode, matching how each lookup is built."""
    if source == "crop":
        return Path(path_str).parts[0]
    return Path(path_str).parent


class MaskGeometryError(ValueError):
    """A ROI mask could not yield a usable bounding box. Not caught per sample: a
    bad box halts the run instead of becoming a black tile or a silent fallback."""


def _derive_bbox(mask: np.ndarray, full_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Foreground bounding box of `mask` in `full_shape` coordinates; None for an
    empty mask, leaving the fallback to the caller."""
    m = np.asarray(mask)
    thresh = (m.max() * 0.5) if m.max() > 0 else 0
    binary = m > thresh
    n_fg = int(binary.sum())
    if n_fg == 0:
        return None

    rows_any, cols_any = np.any(binary, axis=1), np.any(binary, axis=0)
    y0, y1 = (int(v) for v in np.flatnonzero(rows_any)[[0, -1]])
    x0, x1 = (int(v) for v in np.flatnonzero(cols_any)[[0, -1]])

    fill = n_fg / ((y1 - y0 + 1) * (x1 - x0 + 1))
    if fill < config.MASK_MIN_BOX_FILL:
        raise MaskGeometryError(
            f"Mask bbox fill {fill:.3f} < {config.MASK_MIN_BOX_FILL}: the "
            f"foreground is disconnected and the box spans unrelated tissue."
        )

    sy = sx = 1.0
    if binary.shape != tuple(full_shape):  # 78 masks sit at 0.87 of scale
        sy = full_shape[0] / binary.shape[0]
        sx = full_shape[1] / binary.shape[1]
        y0, y1 = int(y0 * sy), int(y1 * sy)
        x0, x1 = int(x0 * sx), int(x1 * sx)

    if not (0 <= y0 <= y1 < full_shape[0] and 0 <= x0 <= x1 < full_shape[1]):
        raise MaskGeometryError(
            f"Derived box ({y0},{y1},{x0},{x1}) falls outside image {full_shape}"
        )
    return y0, y1, x0, x1


def _crop_image_and_mask(
    full: np.ndarray, mask: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop mammogram and mask to the same ROI region + margin, or return
    (full, None) for an empty mask."""
    box = _derive_bbox(mask, full.shape)
    if box is None:
        return full, None
    y0, y1, x0, x1 = box

    m = np.asarray(mask)
    sy = full.shape[0] / m.shape[0]
    sx = full.shape[1] / m.shape[1]

    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    my, mx = int(h * margin), int(w * margin)
    y0c, y1c = max(0, y0 - my), min(full.shape[0], y1 + 1 + my)
    x0c, x1c = max(0, x0 - mx), min(full.shape[1], x1 + 1 + mx)

    crop_img = full[y0c:y1c, x0c:x1c]
    y0m, y1m = max(0, int(y0c / sy)), min(m.shape[0], int(y1c / sy))
    x0m, x1m = max(0, int(x0c / sx)), min(m.shape[1], int(x1c / sx))
    return crop_img, m[y0m:y1m, x0m:x1m]


def _apply_dilated_mask_overlay(
    image: np.ndarray, mask_crop: np.ndarray, target_size: tuple[int, int], dilation_px: int
) -> np.ndarray:
    """Zero out tissue beyond a dilated ROI mask boundary, leaving the lesion plus
    its peritumoral margin; `dilation_px` is the expansion radius at target_size."""
    from scipy.ndimage import binary_dilation

    h, w = target_size
    with tf.device("/CPU:0"):
        mask_f = tf.image.resize(
            mask_crop[..., np.newaxis].astype(np.float32), [h, w], antialias=True
        ).numpy()[..., 0]

    thresh = float(mask_f.max()) * 0.5 if mask_f.max() > 0 else 0.0
    binary = mask_f > thresh

    if binary.any() and dilation_px > 0:
        side = dilation_px * 2 + 1
        binary = binary_dilation(binary, structure=np.ones((side, side), dtype=bool))

    return (image * binary[..., np.newaxis].astype(np.float32)).astype(np.float32)


def _crop_to_mask_bbox(full: np.ndarray, mask: np.ndarray, margin: float) -> np.ndarray:
    """Crop `full` to the mask's foreground bbox expanded by `margin`, a fraction of
    bbox size per side that recovers the peritumoral margin the CBIS crop omits."""
    box = _derive_bbox(mask, full.shape)
    if box is None:
        return full
    y0, y1, x0, x1 = box
    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    my, mx = int(h * margin), int(w * margin)
    y0, y1 = max(0, y0 - my), min(full.shape[0], y1 + 1 + my)
    x0, x1 = max(0, x0 - mx), min(full.shape[1], x1 + 1 + mx)
    return full[y0:y1, x0:x1]


def _crop_box(full: np.ndarray, box: tuple[int, int, int, int], margin: float) -> np.ndarray:
    """Crop to a precomputed box + margin, with the slice bounds of
    _crop_to_mask_bbox, so equal oracle and predicted boxes give equal geometry."""
    y0, y1, x0, x1 = box
    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    my, mx = int(h * margin), int(w * margin)
    y0, y1 = max(0, y0 - my), min(full.shape[0], y1 + 1 + my)
    x0, x1 = max(0, x0 - mx), min(full.shape[1], x1 + 1 + mx)
    return full[y0:y1, x0:x1]


def _make_dataset(
    df: pd.DataFrame,
    lookup: dict,
    target_size: tuple[int, int],
    batch_size: int,
    seed: int,
    *,
    normalisation: Literal["scratch", "imagenet"],
    augment: bool,
    shuffle: bool,
    intensity_norm: str = "fixed_max",
    source: Literal["full", "crop"] = "full",
    path_col: str = "image file path",
    pad_square: bool = True,
    crop_sizing: str = "provided_tight",
    crop_margin: float = config.CROP_MARGIN_FRAC,
    mask_lookup: dict | None = None,
    full_lookup: dict | None = None,
    mask_overlay: bool = False,
    offline_aug_k: int = 1,
    box_provider: "BoxProvider | None" = None,
    standardize: bool = True,
) -> tf.data.Dataset:
    """Build a batched tf.data.Dataset from a deduplicated DataFrame; decode and
    resize run through tf.numpy_function and are cached after the first epoch."""
    use_mask_margin = source == "crop" and crop_sizing == "mask_margin"
    use_provider = source == "crop" and crop_sizing == "box_provider"
    if use_provider and box_provider is None:
        raise ValueError("crop_sizing='box_provider' requires box_provider=...")

    valid_primary: list[str] = []  # crop/full DICOM, or full mammogram
    valid_mask: list[str] = []  # mask DICOM (mask_margin only; "" otherwise)
    valid_boxes: list[tuple[int, int, int, int]] = []  # provider only
    valid_labels: list[int] = []
    skipped = 0
    for _, row in df.iterrows():
        if use_mask_margin:
            lesion_name = Path(row[path_col]).parts[0]
            image_name = re.sub(r"_\d+$", "", lesion_name)  # drop abnormality id
            primary = (full_lookup or {}).get(image_name)
            mask = (mask_lookup or {}).get(lesion_name)
            if primary is None or mask is None:
                skipped += 1
                continue
            valid_mask.append(str(mask))
            valid_boxes.append((-1, -1, -1, -1))
        elif use_provider:
            lesion_name = Path(row[path_col]).parts[0]
            image_name = re.sub(r"_\d+$", "", lesion_name)
            primary = (full_lookup or {}).get(image_name)
            if primary is None:
                skipped += 1
                continue
            box = box_provider.get(lesion_name, _image_shape(primary))
            # (-1,-1,-1,-1) means no detection: the sample falls back to the whole
            # mammogram and is counted rather than silently substituted.
            valid_boxes.append(box if box is not None else (-1, -1, -1, -1))
            valid_mask.append("")
        else:
            primary = lookup.get(_resolve_lookup_key(row[path_col], source))
            if primary is None:
                skipped += 1
                continue
            valid_mask.append("")
            valid_boxes.append((-1, -1, -1, -1))
        valid_primary.append(str(primary))
        valid_labels.append(int(row["label"]))

    if skipped:
        rate = skipped / len(df)
        msg = (
            f"{skipped}/{len(df)} rows ({rate:.2%}) could not be matched to a "
            f"file. Under crop_sizing='mask_margin' this usually means a "
            f"lesion folder holds one DICOM instead of two."
        )
        if rate > config.MAX_LOAD_FAILURE_RATE:
            raise RuntimeError(msg)
        print(f"[data_loader] WARNING: {msg}")
    if not valid_primary:
        raise ValueError("No valid images found to build dataset!")

    # Offline augmentation repeats each sample offline_aug_k times under fixed 90°
    # rotations, with horizontal flips when offline_aug_k is 6 or 8; it is
    # training-only, so val and test are never affected.
    valid_rots: list[int] = []
    if augment and offline_aug_k > 1:
        ex_primary, ex_mask, ex_boxes, ex_labels, ex_rots = [], [], [], [], []
        for p, m, bx, lbl in zip(valid_primary, valid_mask, valid_boxes, valid_labels):
            for k in range(offline_aug_k):
                ex_primary.append(p)
                ex_mask.append(m)
                ex_boxes.append(bx)
                ex_labels.append(lbl)
                ex_rots.append(k)
        valid_primary, valid_mask, valid_boxes, valid_labels, valid_rots = (
            ex_primary,
            ex_mask,
            ex_boxes,
            ex_labels,
            ex_rots,
        )
        print(
            f"[data_loader] Offline aug ×{offline_aug_k}: "
            f"expanded training set from {len(ex_primary)//offline_aug_k} "
            f"→ {len(valid_primary)} cached samples"
        )
    else:
        valid_rots = [0] * len(valid_primary)

    channels = 3 if normalisation == "imagenet" else 1

    def _finalise(arr: np.ndarray) -> np.ndarray:
        """Intensity-normalise → pad/resize → (replicate to 3ch for imagenet)."""
        arr = _normalise_intensity(arr, intensity_norm)
        resized = _pad_and_resize(arr, target_size, pad_square=pad_square)
        if normalisation == "imagenet":
            resized = np.repeat(resized, 3, axis=-1)
        return resized.astype(np.float32)

    def _decode_aug(aug_idx: int) -> tuple[int, bool]:
        """Decode a flat augmentation index into (rot90_k, do_hflip); 6 or 8 give
        k // 2 rotations in each of two flip states. Both operations are exact."""
        if offline_aug_k in (6, 8):
            num_rots = offline_aug_k // 2
            rot_k = aug_idx % num_rots
            do_flip = aug_idx >= num_rots
        else:
            rot_k = aug_idx  # 0..offline_aug_k-1, each a 90° multiple
            do_flip = False
        return rot_k, do_flip

    def _apply_aug(arr: np.ndarray, aug_idx: int) -> np.ndarray:
        """Apply the decoded offline rotation + optional h-flip."""
        rot_k, do_flip = _decode_aug(aug_idx)
        if rot_k:
            arr = np.ascontiguousarray(np.rot90(arr, k=rot_k, axes=(0, 1)))
        if do_flip:
            arr = np.ascontiguousarray(arr[:, ::-1, :])
        return arr

    def _py_provided(path_bytes: bytes, rot_k_t) -> np.ndarray:
        path_str = path_bytes.decode("utf-8")
        try:
            result = _apply_aug(_finalise(_read_full(path_str)), int(rot_k_t))
            _record_load(True)
            return result
        except MaskGeometryError:
            raise
        except Exception as e:
            _record_load(False, path_str, f"{type(e).__name__}: {e}")
            return np.zeros((*target_size, channels), dtype=np.float32)

    def _py_box_crop(path_bytes: bytes, box, rot_k_t) -> np.ndarray:
        """Crop the full mammogram to a box supplied by the BoxProvider."""
        path_str = path_bytes.decode("utf-8")
        try:
            full = _read_full(path_str)
            b = np.asarray(box).astype(int)
            if b[0] < 0:  # localiser produced no box
                # The whole-mammogram fallback pads to square, so a 1.8:1 image is
                # not squashed; crops keep the crop path's no-padding rule.
                _record_load(True, fallback=True)
                arr = _normalise_intensity(full, intensity_norm)
                resized = _pad_and_resize(arr, target_size, pad_square=True)
                if normalisation == "imagenet":
                    resized = np.repeat(resized, 3, axis=-1)
                return _apply_aug(resized.astype(np.float32), int(rot_k_t))
            crop = _crop_box(full, (int(b[0]), int(b[1]), int(b[2]), int(b[3])), crop_margin)
            _record_load(True)
            return _apply_aug(_finalise(crop), int(rot_k_t))
        except Exception as e:
            _record_load(False, path_str, f"{type(e).__name__}: {e}")
            return np.zeros((*target_size, channels), dtype=np.float32)

    def _py_mask_margin(full_bytes: bytes, mask_bytes: bytes, rot_k_t) -> np.ndarray:
        try:
            full = _read_full(full_bytes.decode("utf-8"))
            ds_mask = pydicom.dcmread(mask_bytes.decode("utf-8"), force=True)
            mask = ds_mask.pixel_array
            # CBIS-DDSM masks can be MONOCHROME1; without inversion the threshold
            # finds the background bbox and crops everything except the lesion.
            if getattr(ds_mask, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                mask = mask.max() - mask
            if mask_overlay:
                crop_img, crop_mask = _crop_image_and_mask(full, mask, crop_margin)
            else:
                crop_img, crop_mask = _crop_to_mask_bbox(full, mask, crop_margin), None
            result = _finalise(crop_img)
            if mask_overlay and crop_mask is not None:
                result = _apply_dilated_mask_overlay(
                    result, crop_mask, target_size, config.ROI_MASK_DILATION_PX
                )
            out = _apply_aug(result, int(rot_k_t))
            _record_load(True)
            return out
        except MaskGeometryError:
            raise
        except Exception as e:
            _record_load(False, mask_bytes.decode("utf-8"), f"{type(e).__name__}: {e}")
            return np.zeros((*target_size, channels), dtype=np.float32)

    if use_provider:

        def _wrap(p, b, rot_k, label):
            img = tf.numpy_function(_py_box_crop, [p, b, rot_k], tf.float32)
            img.set_shape((target_size[0], target_size[1], channels))
            return img, label

        ds = tf.data.Dataset.from_tensor_slices(
            (valid_primary, np.asarray(valid_boxes, dtype=np.int64), valid_rots, valid_labels)
        )
        ds = ds.map(_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    elif use_mask_margin:

        def _wrap(p, mp, rot_k, label):
            img = tf.numpy_function(_py_mask_margin, [p, mp, rot_k], tf.float32)
            img.set_shape((target_size[0], target_size[1], channels))
            return img, label

        ds = tf.data.Dataset.from_tensor_slices(
            (valid_primary, valid_mask, valid_rots, valid_labels)
        )
        ds = ds.map(_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    else:

        def _wrap(p, rot_k, label):
            img = tf.numpy_function(_py_provided, [p, rot_k], tf.float32)
            img.set_shape((target_size[0], target_size[1], channels))
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((valid_primary, valid_rots, valid_labels))
        ds = ds.map(_wrap, num_parallel_calls=tf.data.AUTOTUNE)

    # Parallel decode is the expensive step, so it's what gets cached.
    ds = ds.cache()

    # Shuffle after cache: shuffling before it would freeze epoch one's order
    # into the cache and disable reshuffling.
    if shuffle:
        ds = ds.shuffle(buffer_size=len(valid_labels), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(
            _make_batch_augment_fn(normalisation, source=source),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if standardize:
        ds = ds.map(
            _make_batch_standardize_fn(normalisation, intensity_norm=intensity_norm, source=source),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _make_batch_standardize_fn(
    normalisation: Literal["scratch", "imagenet"],
    intensity_norm: str = "fixed_max",
    source: Literal["full", "crop"] = "full",
):
    """Standardise after augmentation, which both modes perform in [0,1], using
    the config statistics in scratch mode and ImageNet ones otherwise."""
    if normalisation == "scratch":
        mean, std = _scratch_norm_stats(intensity_norm, source)

        def _fn(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return (images - mean) / std, labels

        return _fn
    else:
        mean = tf.constant(config.NORM_STATS["imagenet_mean"], dtype=tf.float32)
        std = tf.constant(config.NORM_STATS["imagenet_std"], dtype=tf.float32)

        def _fn(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return (images - mean) / std, labels

        return _fn


def _make_batch_augment_fn(
    normalisation: Literal["scratch", "imagenet"], source: Literal["full", "crop"] = "full"
):
    """Return a batch-level augmentation function over (B, H, W, C) tensors in [0,1].
    Photometric jitter runs only in scratch mode."""
    is_crop = source == "crop"
    brightness_delta = config.CROP_AUG_BRIGHTNESS_DELTA if is_crop else config.AUG_BRIGHTNESS_DELTA
    rotation = _AUG_ROTATION_CROP if is_crop else _AUG_ROTATION
    zoom = _AUG_ZOOM_CROP if is_crop else _AUG_ZOOM

    def _fn(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Left and right breasts are mirror images, so a horizontal flip is valid.
        images = tf.image.random_flip_left_right(images)
        images = rotation(images, training=True)
        images = zoom(images, training=True)
        if normalisation == "scratch":
            if is_crop:
                # Detector and exposure variation; clip before gamma, which needs ≥0.
                images = tf.image.random_contrast(
                    images,
                    1.0 - config.CROP_AUG_CONTRAST_DELTA,
                    1.0 + config.CROP_AUG_CONTRAST_DELTA,
                )
                images = tf.clip_by_value(images, 0.0, 1.0)
                gamma = tf.random.uniform([], *config.CROP_AUG_GAMMA_RANGE)
                images = tf.image.adjust_gamma(images, gamma=gamma)
            images = tf.image.random_brightness(images, max_delta=brightness_delta)
            images = tf.clip_by_value(images, 0.0, 1.0)
        return images, labels

    return _fn


def compute_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    """Balanced class weights, w_c = n_samples / (n_classes * n_c), ready to pass
    to model.fit(class_weight=...)."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_dict: dict[int, float] = dict(zip(classes.tolist(), weights.tolist()))
    counts = {int(c): int((labels == c).sum()) for c in classes}
    print(f"[data_loader] Class counts (train): {counts}")
    print(f"[data_loader] Class weights:        {weight_dict}")
    return weight_dict


def _log_split_summary(train_split, val_split, test_split):
    p_tr, p_va, p_te = (set(d["patient_id"]) for d in (train_split, val_split, test_split))
    print("\n--- Split Summary ---")
    for a, b, na, nb in [
        (p_tr, p_va, "train", "val"),
        (p_tr, p_te, "train", "test"),
        (p_va, p_te, "val", "test"),
    ]:
        n = len(a & b)
        print(f"  {na}/{nb} patient-level leakage check: {'PASS' if not n else f'FAIL ({n})'}")
    n_tr, n_va, n_te = len(train_split), len(val_split), len(test_split)
    n_total = n_tr + n_va + n_te
    print(
        f"  Rows — train: {n_tr} ({n_tr/n_total:.1%}) | "
        f"val: {n_va} ({n_va/n_total:.1%}) | test: {n_te} ({n_te/n_total:.1%})\n"
    )

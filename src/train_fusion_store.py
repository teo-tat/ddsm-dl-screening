"""Train a fusion model from an exported pair store. The store holds the decoded,
un-standardised crop of each stream in pair order; standardisation, the swapped
ordering, augmentation and the two-stage schedule are applied here as src.fusion
applies them.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from . import config, data_loader, train
from .models import build_fusion, build_model, unfreeze_for_finetuning


def _standardise(x: np.ndarray) -> np.ndarray:
    """Single channel in [0,1] -> 3 channels, ImageNet-standardised: the same
    transform data_loader._finalise and the batch standardiser apply in that mode.
    """
    mean = np.asarray(config.NORM_STATS["imagenet_mean"], np.float32)
    std = np.asarray(config.NORM_STATS["imagenet_std"], np.float32)
    return ((np.repeat(x, 3, axis=-1) - mean) / std).astype(np.float32)


def load_store(store: Path, split: str, limit: int | None = None):
    """Return (images_a, images_b or None, labels, meta) for one split."""
    xa = np.ascontiguousarray(
        np.load(store / f"{split}_images_a.npy", mmap_mode="r"), dtype=np.float32
    )
    pb = store / f"{split}_images_b.npy"
    xb = np.ascontiguousarray(np.load(pb, mmap_mode="r"), dtype=np.float32) if pb.exists() else None
    meta = pd.read_csv(store / f"{split}_meta.csv")
    if limit is not None:
        xa, meta = xa[:limit], meta.iloc[:limit]
        xb = None if xb is None else xb[:limit]
    if len(xa) != len(meta) or (xb is not None and len(xb) != len(xa)):
        raise ValueError(f"{split}: store is misaligned")
    return xa, xb, meta["label"].to_numpy(np.int64), meta


def make_dataset(
    xa: np.ndarray,
    xb: np.ndarray | None,
    y: np.ndarray,
    *,
    augment: bool,
    shuffle: bool,
    batch_size: int,
    both_orderings: bool,
) -> tf.data.Dataset:
    """The post-cache half of fusion.build_pair_dataset / build_single_dataset."""
    a = _standardise(xa)
    if xb is None:  # control: single stream
        ds = tf.data.Dataset.from_tensor_slices((a, y))
        n = len(y)
        if shuffle:
            ds = ds.shuffle(buffer_size=n, seed=config.SEED, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if augment:
            ds = ds.map(
                data_loader._make_batch_augment_fn("imagenet", source="crop"),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds.prefetch(tf.data.AUTOTUNE)

    b = _standardise(xb)
    ds = tf.data.Dataset.from_tensor_slices(((a, b), y))
    n = len(y)
    if both_orderings:
        ds = ds.concatenate(ds.map(lambda x, t: ((x[1], x[0]), t)))
        n *= 2
    if shuffle:
        ds = ds.shuffle(buffer_size=n, seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = data_loader._make_batch_augment_fn("imagenet", source="crop")

        def _aug(x, t):
            xa_, _ = aug(x[0], t)
            xb_, _ = aug(x[1], t)  # second call -> independent draw
            return (xa_, xb_), t

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _swap(ds: tf.data.Dataset) -> tf.data.Dataset:
    """((a, b), y) -> ((b, a), y), as fusion.swap_pairs."""
    return ds.map(lambda x, y: ((x[1], x[0]), y))


def score(
    model: keras.Model, ds: tf.data.Dataset, symmetric: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Predictions through the eager call, swap-averaged in the symmetric mode:
    `model.predict` returns the pre-activation values on a Metal GPU.
    """

    def collect(d):
        ys, ps = [], []
        for x, t in d:
            ps.append(np.asarray(model(x, training=False)).ravel())
            ys.append(np.asarray(t).ravel())
        return np.concatenate(ys).astype(int), np.concatenate(ps).astype(float)

    y, p = collect(ds)
    if symmetric:
        y2, p2 = collect(_swap(ds))
        if not np.array_equal(y, y2):
            raise RuntimeError("swapped ordering changed the label order")
        p = (p + p2) / 2.0
    return y, p


def main(
    store: str | Path,
    *,
    arch: str = "vgg16",
    seed: int | None = None,
    epochs: int = config.MAX_EPOCHS,
    platform_suffix: str = "_cuda",
    tag_suffix: str = "",
    out_dir: str | Path | None = None,
    limit: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    backup_dir: str | Path | None = None,
    eager_val_auc: bool | None = None,
) -> dict:
    """Train one fusion tag from its store and report its exact validation AUC."""
    store = Path(store)
    manifest_in = json.loads((store / "manifest.json").read_text())
    pair = manifest_in["pair"]
    paired = bool(manifest_in["paired"])
    symmetric = pair == "view"
    batch_size = batch_size or config.BATCH_SIZE

    if out_dir is not None:
        config.WEIGHTS_DIR = config.LOGS_DIR = config.RESULTS_DIR = Path(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    train._ensure_dirs()

    # tag_suffix sits between the store's fusion_tag and the seed, the order
    # src.fusion composes them in, so src.fusion resolves the same name.
    tag = (
        f"{manifest_in['fusion_tag']}{tag_suffix}"
        f"{train._run_suffix('', ablate=None, lr=None, seed=seed)}"
        f"{platform_suffix}"
    )
    config.set_seeds(config.SEED)
    train._reseed(seed)

    xa_t, xb_t, y_t, meta_t = load_store(store, "train", limit)
    xa_v, xb_v, y_v, _ = load_store(store, "val", limit)
    tr = make_dataset(
        xa_t, xb_t, y_t, augment=True, shuffle=True, batch_size=batch_size, both_orderings=symmetric
    )
    va = make_dataset(
        xa_v, xb_v, y_v, augment=False, shuffle=False, batch_size=batch_size, both_orderings=False
    )

    class_weight = (
        data_loader.compute_class_weights(meta_t)
        if config.CLASS_WEIGHT_STRATEGY == "balanced"
        else None
    )
    shape = (config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)
    model = (
        build_fusion(arch, input_shape=shape) if paired else build_model(arch, input_shape=shape)
    )

    def _cbs(stage: str):
        kw = {"stage": stage}
        if patience is not None:
            kw["patience"] = patience
        cbs = train.maybe_eager_val_callback(va, eager_val_auc) + train._get_callbacks(tag, **kw)
        if backup_dir:
            d = Path(backup_dir) / stage
            d.mkdir(parents=True, exist_ok=True)
            cbs.append(
                keras.callbacks.BackupAndRestore(str(d), save_freq="epoch", delete_checkpoint=False)
            )
        return cbs

    t0 = time.time()
    print(
        f"[train_fusion_store] {tag}: pair={pair} paired={paired} symmetric={symmetric} "
        f"rows={len(y_t)} batch={batch_size} epochs={epochs}"
    )

    print(f"\n--- Stage 1: frozen backbone ({tag}) ---")
    train._compile(model, lr=config.INITIAL_LR)
    h1 = model.fit(
        tr,
        validation_data=va,
        epochs=epochs,
        callbacks=_cbs("frozen"),
        class_weight=class_weight,
        verbose=2,
    )
    train._save_history(h1, config.RESULTS_DIR / f"{tag}_frozen_history.json")
    best_frozen = config.WEIGHTS_DIR / f"{tag}_frozen_best.weights.h5"
    if best_frozen.exists():
        model.load_weights(str(best_frozen))

    print(f"\n--- Stage 2: fine-tuning ({tag}) ---")
    unfreeze_for_finetuning(model, backbone_name=arch)
    train._compile(model, lr=config.FINETUNE_LR)
    h2 = model.fit(
        tr,
        validation_data=va,
        epochs=epochs,
        callbacks=_cbs("finetune"),
        class_weight=class_weight,
        verbose=2,
    )
    train._save_history(h2, config.RESULTS_DIR / f"{tag}_finetune_history.json")
    best_ft = config.WEIGHTS_DIR / f"{tag}_finetune_best.weights.h5"
    if best_ft.exists():
        model.load_weights(str(best_ft))
    model.save_weights(str(config.WEIGHTS_DIR / f"{tag}_best.weights.h5"))

    y, p = score(model, va, symmetric)
    auc = float(roc_auc_score(y, p))
    va_auc = np.asarray(h2.history["val_auc"], float)
    out = {
        "tag": tag,
        "arch": arch,
        "pair": pair,
        "paired": paired,
        "symmetric": symmetric,
        "seed": seed,
        "patience": patience,
        "batch_size": batch_size,
        "store": str(store),
        "exact_val_auc": round(auc, 6),
        "n_val": int(len(y)),
        "history_best_val_auc": float(va_auc.max()),
        "best_epoch": int(va_auc.argmax()) + 1,
        "stage_epochs": {"frozen": int(len(h1.history["val_auc"])), "finetune": int(len(va_auc))},
        "tf_version": tf.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "platform": platform.platform(),
        "wall_time_s": round(time.time() - t0, 1),
        # Card, driver and effective seed, so the session can be verified later.
        **train._device_record(seed),
    }
    (config.RESULTS_DIR / f"{tag}_manifest.json").write_text(json.dumps(out, indent=2))
    print(
        f"[train_fusion_store] {tag}: exact val AUC {auc:.4f} "
        f"(history {va_auc.max():.4f}) in {out['wall_time_s']:.0f}s"
    )
    print(f"-> {config.RESULTS_DIR / f'{tag}_manifest.json'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--store", required=True)
    ap.add_argument("--arch", default="vgg16")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--platform-suffix", default="_cuda")
    ap.add_argument(
        "--tag-suffix",
        default="",
        help="appended to the store's fusion_tag before the seed, e.g. "
        "config.LEDGER_SUFFIX (pass --platform-suffix '' with it). Default: none.",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N pairs")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--eager-val-auc", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag "
        "that no longer exists fails in a second.",
    )
    a = ap.parse_args()
    if a.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        raise SystemExit(0)
    kw = dict(
        arch=a.arch,
        seed=a.seed,
        epochs=a.epochs,
        platform_suffix=a.platform_suffix,
        tag_suffix=a.tag_suffix,
        out_dir=a.out_dir,
        limit=a.limit,
        patience=a.patience,
        batch_size=a.batch_size,
        backup_dir=a.backup_dir,
        eager_val_auc={"auto": None, "on": True, "off": False}[a.eager_val_auc],
    )
    if a.cpu:
        with tf.device("/CPU:0"):
            main(a.store, **kw)
    else:
        main(a.store, **kw)

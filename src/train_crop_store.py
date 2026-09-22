"""Train any registry classifier from an exported store. The store holds the decoded,
un-standardised crops in frame order; the seeded reshuffle, augmentation and
standardisation stay here, so a crop-store run matches one from the images. Every store,
the whole-image one included, gets the crop-path augmentation and standardisation.
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

from . import config, data_loader, train
from .models import (
    MODEL_REGISTRY,
    _SCRATCH_MODELS,
    build_model,
    get_normalisation_mode,
    unfreeze_for_finetuning,
)


def load_store(store: Path, split: str, limit: int | None = None):
    x = np.load(store / f"{split}_images.npy", mmap_mode="r")
    meta = pd.read_csv(store / f"{split}_meta.csv")
    if limit is not None:
        x, meta = x[:limit], meta.iloc[:limit]
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(x) != len(meta):
        raise ValueError(f"{split}: {len(x)} images vs {len(meta)} meta rows")
    return x, meta


def make_datasets(
    store: Path,
    normalisation: str,
    intensity_norm: str,
    *,
    augment: bool,
    batch_size: int,
    shuffle_seed: int,
    limit: int | None = None,
):
    """(train_ds, val_ds, train_meta): the post-cache half of data_loader._make_dataset,
    with source="crop" for every store, so a whole-image store gets the crop transforms."""
    xt, mt = load_store(store, "train", limit)
    xv, mv = load_store(store, "val", limit)
    if normalisation == "imagenet":  # what _finalise does for 3-channel models
        xt, xv = np.repeat(xt, 3, axis=-1), np.repeat(xv, 3, axis=-1)
    yt, yv = mt["label"].to_numpy(np.int64), mv["label"].to_numpy(np.int64)

    tr = tf.data.Dataset.from_tensor_slices((xt, yt))
    tr = tr.shuffle(buffer_size=len(yt), seed=shuffle_seed, reshuffle_each_iteration=True)
    tr = tr.batch(batch_size)
    if augment:
        tr = tr.map(
            data_loader._make_batch_augment_fn(normalisation, source="crop"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    std_fn = data_loader._make_batch_standardize_fn(
        normalisation, intensity_norm=intensity_norm, source="crop"
    )
    tr = tr.map(std_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    va = (
        tf.data.Dataset.from_tensor_slices((xv, yv))
        .batch(batch_size)
        .map(std_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tr, va, mt


def _redirect_outputs(out_dir: Path) -> None:
    """Point every output directory at one folder: train's callback stacks write to
    config.{WEIGHTS,LOGS,RESULTS}_DIR, which a run off the repo has to redirect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config.WEIGHTS_DIR = config.LOGS_DIR = config.RESULTS_DIR = out_dir


def main(
    store: str | Path,
    *,
    model: str = "regularised",
    ablate: str | None = None,
    lr: float | None = None,
    seed: int | None = None,
    epochs: int = config.MAX_EPOCHS,
    tag_suffix: str = "_m020",
    platform_suffix: str = "_kg",
    tag_base: str | None = None,
    out_dir: str | Path | None = None,
    limit: int | None = None,
    eager_val_auc: bool | None = None,
    backup_dir: str | Path | None = None,
    patience: int | None = None,
    amp: bool = False,
    batch_size: int = config.BATCH_SIZE,
) -> dict:
    store = Path(store)
    manifest_in = json.loads((store / "manifest.json").read_text())
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}")
    if ablate and model != "regularised":
        raise ValueError("--ablate applies to the regularised model only")
    if ablate and ablate not in train.ABLATIONS:
        raise ValueError(f"unknown ablation {ablate!r}; choose from {train.ABLATIONS}")

    if out_dir is not None:
        _redirect_outputs(Path(out_dir))
    train._ensure_dirs()

    # Same tag rule as train.train_baseline / train_transfer, plus the platform suffix;
    # tag_base replaces the `{model}_crop` stem for a whole-image run.
    base = f"{model}_crop" if tag_base is None else tag_base
    tag = f"{base}{train._run_suffix(tag_suffix, ablate=ablate, lr=lr, seed=seed)}{platform_suffix}"
    norm = get_normalisation_mode(model)
    augment = model != "scaled" and ablate != "aug"  # the crop-path rule in train_baseline
    if lr is None:
        lr = config.REGULARISED_LR if model == "regularised" else config.INITIAL_LR

    # Seeding as train.py: partition seed first (also resets the global RNGs),
    # then the replicate seed for init / augmentation / shuffle.
    config.set_seeds(config.SEED)
    shuffle_seed = config.SEED if seed is None else seed
    train._reseed(seed)

    tr, va, train_meta = make_datasets(
        store,
        norm,
        manifest_in["intensity_norm"],
        augment=augment,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        limit=limit,
    )
    class_weight = (
        data_loader.compute_class_weights(train_meta)
        if config.CLASS_WEIGHT_STRATEGY == "balanced"
        else None
    )

    if amp:
        # Mixed precision changes the numerics, so it is off by default; the output is
        # cast back to float32, since a float16 sigmoid loses resolution in the loss.
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[train_crop_store] mixed_float16 enabled (output cast back to float32)")

    ts = tuple(manifest_in["target_size"])
    m = build_model(
        model,
        input_shape=(ts[0], ts[1], MODEL_REGISTRY[model]["channels"]),
        **train._regularised_kwargs(model, "crop", ablate),
    )
    if amp:
        m = tf.keras.Model(
            m.inputs,
            tf.keras.layers.Activation("linear", dtype="float32", name="output_float32")(m.output),
            name=m.name,
        )
    print(
        f"[train_crop_store] {tag}: lr={lr:g} augment={augment} ablate={ablate} "
        f"store={store.name} rows={len(train_meta)} epochs={epochs}"
    )

    # The eager validation metric goes first in every stack: it rewrites
    # logs["val_auc"] before the checkpoint and early-stopping callbacks read it.
    def _backup(stage: str) -> list:
        """BackupAndRestore callbacks, one directory per stage: the snapshot is taken at
        every epoch boundary, so an interrupted run costs at most the epoch in flight.
        """
        if backup_dir is None:
            return []
        d = Path(backup_dir) / (stage or "single")
        d.mkdir(parents=True, exist_ok=True)
        return [
            tf.keras.callbacks.BackupAndRestore(str(d), save_freq="epoch", delete_checkpoint=False)
        ]

    def _cbs(*args, **kw):
        stage = kw.get("stage", "")
        # patience=None keeps train._get_callbacks' own default.
        if patience is not None:
            kw.setdefault("patience", patience)
        return (
            train.maybe_eager_val_callback(va, eager_val_auc)
            + train._get_callbacks(*args, **kw)
            + _backup(stage)
        )

    t0 = time.time()
    if model in _SCRATCH_MODELS:
        # Single stage, exactly as train.train_baseline.
        train._compile(m, lr=lr)
        if model == "scaled":
            cbs = (
                train.maybe_eager_val_callback(va, eager_val_auc)
                + train._get_overfit_callbacks(tag)
                + _backup("")
            )
        elif model == "regularised":
            cbs = _cbs(tag, patience=patience or config.REGULARISED_PATIENCE)
        else:
            cbs = _cbs(tag)
        hist = m.fit(
            tr,
            validation_data=va,
            epochs=epochs,
            callbacks=cbs,
            class_weight=class_weight,
            verbose=2,
        )
        train._save_history(hist, config.RESULTS_DIR / f"{tag}_history.json")
        va_auc = np.asarray(hist.history["val_auc"], float)
        stage_epochs = {"single": int(len(va_auc))}
    else:
        # Two stages, exactly as train.train_transfer: the head on a frozen backbone at
        # INITIAL_LR (or --lr), then the upper backbone unfrozen at FINETUNE_LR.
        print(f"\n--- Stage 1: frozen backbone ({tag}) ---")
        train._compile(m, lr=config.INITIAL_LR if lr is None else lr)
        h1 = m.fit(
            tr,
            validation_data=va,
            epochs=epochs,
            callbacks=_cbs(tag, stage="frozen"),
            class_weight=class_weight,
            verbose=2,
        )
        train._save_history(h1, config.RESULTS_DIR / f"{tag}_frozen_history.json")
        best_frozen = config.WEIGHTS_DIR / f"{tag}_frozen_best.weights.h5"
        if best_frozen.exists():
            m.load_weights(str(best_frozen))
            print(f"[train_crop_store] Loaded best frozen weights from {best_frozen}")

        print(f"\n--- Stage 2: fine-tuning ({tag}) ---")
        unfreeze_for_finetuning(m, backbone_name=model)
        train._compile(m, lr=config.FINETUNE_LR)
        h2 = m.fit(
            tr,
            validation_data=va,
            epochs=epochs,
            callbacks=_cbs(tag, stage="finetune"),
            class_weight=class_weight,
            verbose=2,
        )
        train._save_history(h2, config.RESULTS_DIR / f"{tag}_finetune_history.json")
        # Carry the store's boxes folder into the run's manifest, so a later pass can
        # check the weights it scores were trained on the boxes it crops with.
        _sm = json.loads((Path(store) / "manifest.json").read_text())
        train._save_train_manifest(
            tag,
            crop_margin=float(_sm.get("margin", config.CROP_MARGIN_FRAC)),
            seed=seed,
            source="crop",
            box_source=_sm.get("box_source"),
            boxes_dir=_sm.get("boxes_dir"),
            extra={"trained_from_store": str(store), "store_files": _sm.get("files")},
        )
        merged = train._merge_histories(
            {k: [float(v) for v in vs] for k, vs in h1.history.items()},
            {k: [float(v) for v in vs] for k, vs in h2.history.items()},
        )
        (config.RESULTS_DIR / f"{tag}_full_history.json").write_text(json.dumps(merged, indent=2))

        best_ft = config.WEIGHTS_DIR / f"{tag}_finetune_best.weights.h5"
        if best_ft.exists():
            m.load_weights(str(best_ft))
            print(f"[train_crop_store] Loaded best fine-tuned weights from {best_ft}")
        # Canonical name, the one evaluate.py resolves.
        m.save_weights(str(config.WEIGHTS_DIR / f"{tag}_best.weights.h5"))
        hist = h2
        va_auc = np.asarray(h2.history["val_auc"], float)
        stage_epochs = {"frozen": int(len(h1.history["val_auc"])), "finetune": int(len(va_auc))}

    m.save_weights(str(config.WEIGHTS_DIR / f"{tag}_final.weights.h5"))
    best = int(va_auc.argmax())
    out = {
        "tag": tag,
        "model": model,
        "ablate": ablate,
        "lr": lr,
        "seed": seed,
        "augment": augment,
        "epochs_run": int(len(va_auc)),
        "epochs_cap": epochs,
        "stage_epochs": stage_epochs,
        "two_stage": model not in _SCRATCH_MODELS,
        "patience": patience,
        "amp": amp,
        "batch_size": batch_size,
        "target_size": list(ts),
        "val_auc_eager_best": (
            float(np.asarray(hist.history["val_auc_eager"], float).max())
            if "val_auc_eager" in hist.history
            else None
        ),
        "best_epoch": best + 1,
        "val_auc_best": float(va_auc[best]),
        "train_auc_at_best": float(hist.history["auc"][best]),
        "store": str(store),
        "store_files": manifest_in.get("files"),
        "store_manifest": {k: manifest_in[k] for k in ("margin", "mask_overlay", "intensity_norm")},
        "tf_version": tf.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "platform": platform.platform(),
        "wall_time_s": round(time.time() - t0, 1),
        # Card, driver and effective seed, so the session can be verified later.
        **train._device_record(seed),
    }
    (config.RESULTS_DIR / f"{tag}_manifest.json").write_text(json.dumps(out, indent=2))
    print(
        f"[train_crop_store] {tag}: best val_auc {out['val_auc_best']:.4f} at epoch "
        f"{out['best_epoch']}/{out['epochs_run']} ({out['wall_time_s']:.0f} s)"
    )
    print(f"-> {config.RESULTS_DIR / f'{tag}_manifest.json'}")
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--store", required=True)
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    ap.add_argument("--model", default="regularised", choices=list(MODEL_REGISTRY))
    ap.add_argument("--ablate", default=None, choices=list(train.ABLATIONS))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--tag-suffix", default="_m020")
    ap.add_argument(
        "--tag-base",
        default=None,
        help="replace the {model}_crop stem of the tag, e.g. with vgg16 for the "
        "whole-image store. Default: none.",
    )
    ap.add_argument("--platform-suffix", default="_kg")
    ap.add_argument("--out-dir", default=None, help="default: the repo's outputs/ folders")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N rows per split")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="early-stopping patience; default keeps config "
        "(10, or 15 for the regularised model)",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="mixed_float16. Changes numerics, so do not use it for a "
        "run being compared like-for-like against an fp32 result.",
    )
    ap.add_argument(
        "--backup-dir",
        default=None,
        help="BackupAndRestore directory; an interrupted run resumes from "
        "the last epoch boundary. Put it on the network volume.",
    )
    ap.add_argument(
        "--eager-val-auc",
        choices=["auto", "on", "off"],
        default="auto",
        help="Recompute val_auc through the eager path each epoch and select "
        "on it. 'auto' enables it on a Metal GPU, where the compiled "
        "graph miscomputes ReLU, and not on CUDA.",
    )
    a = ap.parse_args()
    if a.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    kw = dict(
        model=a.model,
        ablate=a.ablate,
        lr=a.lr,
        seed=a.seed,
        epochs=a.epochs,
        tag_suffix=a.tag_suffix,
        platform_suffix=a.platform_suffix,
        tag_base=a.tag_base,
        out_dir=a.out_dir,
        limit=a.limit,
        batch_size=a.batch_size,
        backup_dir=a.backup_dir,
        patience=a.patience,
        amp=a.amp,
        eager_val_auc={"auto": None, "on": True, "off": False}[a.eager_val_auc],
    )
    if a.cpu:
        with tf.device("/CPU:0"):
            main(a.store, **kw)
    else:
        main(a.store, **kw)


if __name__ == "__main__":
    _cli()

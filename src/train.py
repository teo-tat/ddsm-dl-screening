"""Train CBIS-DDSM mass classifiers: scratch CNNs in a single stage (three with
--indomain), pretrained backbones in two (the head on a frozen backbone, then a
fine-tune at a lower learning rate). Every mode selects the epoch with the highest
validation AUC."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import config
from . import data_loader
from .models import (
    MODEL_REGISTRY,
    _SCRATCH_MODELS,
    build_model,
    freeze_scratch_backbone,
    get_normalisation_mode,
    unfreeze_for_finetuning,
    unfreeze_scratch_backbone,
)


def _ensure_dirs() -> None:
    for d in (
        config.OUTPUTS_DIR,
        config.FIGURES_DIR,
        config.WEIGHTS_DIR,
        config.LOGS_DIR,
        config.RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _compile(model: keras.Model, lr: float, optimizer: str = "adam") -> None:
    """Compile with the chosen optimiser, binary cross-entropy and an AUC metric."""
    if optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )


def _get_overfit_callbacks(model_name: str) -> list[keras.callbacks.Callback]:
    """Best-val_auc checkpoint and a CSV log only: no early stopping and no LR
    reduction, so an over-capacity model is left to overfit."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.WEIGHTS_DIR / f"{model_name}_best.weights.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(config.LOGS_DIR / f"{model_name}_training.csv"), append=False
        ),
    ]


def _get_callbacks(
    model_name: str, stage: str = "", patience: int = config.EARLY_STOPPING_PATIENCE
) -> list[keras.callbacks.Callback]:
    """Early stopping, best checkpoint, LR-on-plateau and a CSV log, the first
    three on val_auc (max), so the weights kept are the best-validation epoch."""
    tag = f"{model_name}_{stage}" if stage else model_name

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.WEIGHTS_DIR / f"{tag}_best.weights.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=10, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(str(config.LOGS_DIR / f"{tag}_training.csv"), append=False),
    ]
    return callbacks


class EagerValAUC(keras.callbacks.Callback):
    """Recompute val_auc each epoch through the eager call path, because the compiled
    graph on tensorflow-metal miscomputes ReLU. Must come first: it overwrites val_auc."""

    def __init__(self, val_ds: tf.data.Dataset, replace: bool = True, verbose: bool = True) -> None:
        super().__init__()
        self.val_ds = val_ds
        self.replace = replace
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Score validation eagerly and write the result into the epoch logs."""
        if logs is None:
            return
        from sklearn.metrics import roc_auc_score

        y_parts, p_parts = [], []
        for images, labels in self.val_ds:
            probs = self.model(images, training=False)
            y_parts.append(np.asarray(labels).ravel())
            p_parts.append(np.asarray(probs).ravel())
        y = np.concatenate(y_parts)
        p = np.concatenate(p_parts)
        if len(np.unique(y)) < 2:  # one class only: AUC undefined
            return
        auc = float(roc_auc_score(y, p))

        compiled = logs.get("val_auc")
        logs["val_auc_eager"] = auc
        if self.replace:
            logs["val_auc"] = auc
        if self.verbose:
            delta = "" if compiled is None else f" (compiled {compiled:.4f})"
            print(f"[train] epoch {epoch + 1}: val_auc_eager {auc:.4f}{delta}")


def _metal_gpu_present() -> bool:
    try:
        import platform

        return platform.system() == "Darwin" and bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


def maybe_eager_val_callback(
    val_ds: tf.data.Dataset, enabled: bool | None = None
) -> list[keras.callbacks.Callback]:
    """EagerValAUC in a list, or an empty list, so callers can always prepend it.
    enabled=None uses it whenever a Metal GPU is present, where the metric is wrong."""
    use = _metal_gpu_present() if enabled is None else bool(enabled)
    if not use:
        return []
    print(
        "[train] Metal GPU detected: val_auc will be recomputed through the "
        "eager path and used for selection (see EagerValAUC)."
    )
    return [EagerValAUC(val_ds)]


def _boxes_md5(boxes_dir: str | Path | None) -> dict[str, str] | None:
    """md5 of the box tables a run cropped with, keyed by split. None means no
    folder was named; a named folder with no readable table records why instead."""
    if not boxes_dir:
        return None
    import hashlib

    out = {}
    for split in ("train", "val"):
        f = Path(boxes_dir) / f"localiser_boxes_{split}.csv"
        if f.exists():
            out[split] = hashlib.md5(f.read_bytes()).hexdigest()
    if out:
        return out
    import platform
    import sys as _sys

    why = (
        f"boxes_dir {boxes_dir} was named but no localiser_boxes_(train|val).csv was "
        f"readable on {platform.node()}; md5 NOT computed. Backfill it on a machine "
        f"that holds the folder before the test pass reads this manifest."
    )
    print(f"[train] WARNING: {why}", file=_sys.stderr, flush=True)
    return {"unavailable": why}


def _save_train_manifest(
    tag: str,
    *,
    crop_margin: float,
    seed: int | None,
    source: str,
    box_provider=None,
    box_source: str | None = None,
    boxes_dir: str | Path | None = None,
    extra: dict | None = None,
) -> None:
    """Record the boxes a run trained on, not just its tag: weights load by tag
    and crops come from a separately named folder, so a tag match proves nothing."""
    # Either a live BoxProvider, or an explicit folder when training from an
    # exported store, which names the folder in its own manifest.
    if boxes_dir is None:
        boxes_dir = getattr(box_provider, "boxes_dir", None)
    if box_source is None:
        box_source = getattr(box_provider, "source", None)
    entry = {
        "tag": tag,
        "source": source,
        "crop_margin": float(crop_margin),
        "seed": int(seed) if seed is not None else config.SEED,
        "box_source": box_source,
        "boxes_dir": str(boxes_dir) if boxes_dir else None,
        "boxes_md5": _boxes_md5(boxes_dir),
    }
    if extra:
        entry.update(extra)
    path = config.RESULTS_DIR / f"{tag}_train_manifest.json"
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"[train] Train manifest saved → {path}")


def _save_history(history: keras.callbacks.History, path: Path) -> None:
    """Persist training history as JSON (floats, not numpy types)."""
    serialisable = {}
    for key, vals in history.history.items():
        serialisable[key] = [float(v) for v in vals]
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[train] History saved → {path}")


def _merge_histories(h1: dict, h2: dict) -> dict:
    merged = {}
    all_keys = set(h1) | set(h2)
    for k in sorted(all_keys):
        merged[k] = h1.get(k, []) + h2.get(k, [])
    return merged


def _build_datasets(
    normalisation: str,
    augment: bool = True,
    clahe: bool = False,
    source: str = "full",
    target_size: tuple[int, int] | None = None,
    pad_square: bool | None = None,
    batch_size: int | None = None,
    intensity_norm: str | None = None,
    crop_sizing: str = config.CROP_SIZING,
    crop_margin: float = config.CROP_MARGIN_FRAC,
    mask_overlay: bool = False,
    offline_aug_k: int = 1,
    box_provider=None,
    shuffle_seed: int | None = None,
):
    """Returns (train_ds, val_ds, test_ds, test_split, train_split), where source
    selects whole mammograms ("full") or per-lesion ROI crops ("crop")."""
    return data_loader.build_datasets(
        train_csv=config.TRAIN_CSV,
        test_csv=config.TEST_CSV,
        train_img_dir=config.TRAIN_IMG_DIR,
        test_img_dir=config.TEST_IMG_DIR,
        train_roi_dir=config.TRAIN_ROI_DIR,
        test_roi_dir=config.TEST_ROI_DIR,
        target_size=target_size or config.TARGET_SIZE,
        val_fraction=config.VAL_FRACTION,
        batch_size=batch_size or config.BATCH_SIZE,
        seed=config.SEED,
        normalisation=normalisation,
        augment=augment,
        clahe=clahe,
        source=source,
        pad_square=pad_square,
        intensity_norm=intensity_norm,
        crop_sizing=crop_sizing,
        crop_margin=crop_margin,
        mask_overlay=mask_overlay,
        offline_aug_k=offline_aug_k,
        box_provider=box_provider,
        shuffle_seed=shuffle_seed,
    )


# Each ablation turns exactly one knob of the regularised CNN off and leaves
# every other setting, the LR and the callbacks included, as that run has them.
ABLATIONS: tuple[str, ...] = ("dropout", "l2", "se", "aug")


def _fmt_lr(lr: float) -> str:
    """5e-4 -> '5e-4', 2e-3 -> '2e-3', 1e-5 -> '1e-5' (tag-friendly)."""
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def _run_suffix(tag_suffix: str, *, ablate: str | None, lr: float | None, seed: int | None) -> str:
    """Tag suffix spelling out a run's deviations from the defaults, in a fixed
    order (ablation, LR, seed) so one run always gets one tag."""
    if ablate:
        tag_suffix += f"_no_{ablate}"
    if lr is not None:
        tag_suffix += f"_lr{_fmt_lr(lr)}"
    if seed is not None and seed != config.SEED:
        tag_suffix += f"_s{seed}"
    return tag_suffix


def _device_record(seed: int | None) -> dict:
    """The card, driver and effective seed for a run manifest, where seed=None
    means config.SEED. A field that cannot be read is None, never guessed."""
    import subprocess

    import tensorflow as tf

    rec = {
        "gpu_name": None,
        "gpu_compute_capability": None,
        "gpu_driver": None,
        "seed_effective": config.SEED if seed is None else int(seed),
    }
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            det = tf.config.experimental.get_device_details(gpus[0])
            rec["gpu_name"] = det.get("device_name")
            cc = det.get("compute_capability")
            rec["gpu_compute_capability"] = ".".join(map(str, cc)) if cc else None
        except Exception:
            pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if q.returncode == 0 and q.stdout.strip():
            name, _, drv = q.stdout.strip().splitlines()[0].partition(",")
            rec["gpu_name"] = rec["gpu_name"] or name.strip()
            rec["gpu_driver"] = drv.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return rec


def _reseed(seed: int | None) -> None:
    """Re-apply a replicate seed after build_datasets, which calls
    config.set_seeds(config.SEED) to fix the partition and resets the RNGs."""
    if seed is not None and seed != config.SEED:
        config.set_seeds(seed)
        print(
            f"[train] Replicate seed {seed} applied to initialisation, "
            f"augmentation and shuffle (partition unchanged, seed {config.SEED})"
        )


def _regularised_kwargs(name: str, source: str, ablate: str | None = None) -> dict:
    """Lighter regularisation for the regularised CNN on the crop path, which
    underfits under the heavier whole-image defaults. Empty for every other model."""
    if name != "regularised":
        return {}
    kw = (
        dict(l2_reg=config.REGULARISED_CROP_L2, head_dropout=config.REGULARISED_CROP_HEAD_DROPOUT)
        if source == "crop"
        else {}
    )
    if ablate == "dropout":
        kw["head_dropout"] = 0.0
    elif ablate == "l2":
        kw["l2_reg"] = 0.0
    elif ablate == "se":
        kw["use_se"] = False
    return kw


def train_baseline(
    name: str = "baseline",
    source: str = "full",
    *,
    clahe: bool = False,
    target_size: tuple[int, int] | None = None,
    pad_square: bool | None = None,
    batch_size: int | None = None,
    intensity_norm: str | None = None,
    crop_sizing: str = config.CROP_SIZING,
    crop_margin: float = config.CROP_MARGIN_FRAC,
    tag_suffix: str = "",
    optimizer: str = "adam",
    mask_overlay: bool = False,
    offline_aug_k: int = 1,
    box_provider=None,
    ablate: str | None = None,
    lr: float | None = None,
    eager_val_auc: bool | None = None,
    seed: int | None = None,
) -> keras.Model:
    """Train a from-scratch CNN in a single stage on whole mammograms
    (source="full") or per-lesion ROI crops (source="crop", tagged "{name}_crop")."""
    if ablate and name != "regularised":
        raise ValueError(f"--ablate applies to the regularised model only, not '{name}'")
    if ablate and ablate not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{ablate}'; choose from {ABLATIONS}")
    tag = f"{name}_crop" if source == "crop" else name
    tag = f"{tag}{_run_suffix(tag_suffix, ablate=ablate, lr=lr, seed=seed)}"
    print(f"\n{'='*60}")
    print(f"  Training: {tag}  (from scratch, source={source})")
    print(f"{'='*60}\n")

    norm = get_normalisation_mode(name)
    # Augment on the crop path only, and never for the scaled model: augmentation
    # is regularisation, and that model exists to show overfitting.
    augment = source == "crop" and name != "scaled" and ablate != "aug"

    ts = target_size or config.TARGET_SIZE
    train_ds, val_ds, _, _, train_split = _build_datasets(
        norm,
        augment=augment,
        clahe=clahe,
        source=source,
        target_size=target_size,
        pad_square=pad_square,
        batch_size=batch_size,
        intensity_norm=intensity_norm,
        crop_sizing=crop_sizing,
        crop_margin=crop_margin,
        mask_overlay=mask_overlay,
        offline_aug_k=offline_aug_k,
        box_provider=box_provider,
        shuffle_seed=seed,
    )
    _reseed(seed)

    class_weight = (
        data_loader.compute_class_weights(train_split)
        if config.CLASS_WEIGHT_STRATEGY == "balanced"
        else None
    )

    model = build_model(
        name,
        input_shape=(ts[0], ts[1], MODEL_REGISTRY[name]["channels"]),
        **_regularised_kwargs(name, source, ablate),
    )
    # The regularised model starts at a lower LR: 1e-3 produces val_loss spikes
    # on a heavily regularised 1.3M-parameter network.
    if lr is None:
        lr = config.REGULARISED_LR if name == "regularised" else config.INITIAL_LR
    print(f"[train] {tag}: lr={lr:g} augment={augment} ablate={ablate}")
    _compile(model, lr=lr, optimizer=optimizer)
    model.summary()

    # The scaled model takes the overfit callbacks so the whole divergence curve
    # is recorded; the regularised model peaks late and gets extra patience.
    if name == "scaled":
        cbs = _get_overfit_callbacks(tag)
    elif name == "regularised":
        cbs = _get_callbacks(tag, patience=config.REGULARISED_PATIENCE)
    else:
        cbs = _get_callbacks(tag)
    # Must come first: it rewrites logs["val_auc"] before the checkpoint and
    # early-stopping callbacks read it.
    cbs = maybe_eager_val_callback(val_ds, eager_val_auc) + cbs

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=2,
    )

    _save_history(history, config.RESULTS_DIR / f"{tag}_history.json")
    _save_train_manifest(
        tag, box_provider=box_provider, crop_margin=crop_margin, seed=seed, source=source
    )

    # The last epoch's weights as well; the best are already on disk.
    final_path = config.WEIGHTS_DIR / f"{tag}_final.weights.h5"
    model.save_weights(str(final_path))
    print(f"[train] Final weights → {final_path}")

    return model


def train_indomain(name: str = "regularised") -> keras.Model:
    """In-domain transfer (Shen et al., 2019): pretrain the conv body on ROI crops,
    then freeze and fine-tune it on whole mammograms. Stage A is reused if present."""
    print(f"\n{'='*60}")
    print(f"  In-domain transfer: {name}  (crop pretrain → whole-image)")
    print(f"{'='*60}\n")

    norm = get_normalisation_mode(name)
    # Crop-tuned regularisation: the body is pretrained on crops and carries
    # those settings through to the whole-image stages.
    model = build_model(name, **_regularised_kwargs(name, "crop"))

    # Stage A: pretrain conv body on ROI crops
    crop_weights = config.WEIGHTS_DIR / f"{name}_crop_best.weights.h5"
    if crop_weights.exists():
        _compile(model, lr=config.REGULARISED_LR)
        model.load_weights(str(crop_weights))
        print(f"[train] Stage A skipped — loaded crop-pretrained weights from {crop_weights}")
    else:
        print(f"\n--- Stage A: crop pretraining ({name}) ---")
        train_ds, val_ds, _, _, train_split = _build_datasets(
            norm, augment=True, clahe=False, source="crop"
        )
        class_weight = (
            data_loader.compute_class_weights(train_split)
            if config.CLASS_WEIGHT_STRATEGY == "balanced"
            else None
        )
        _compile(model, lr=config.REGULARISED_LR)
        model.summary()
        hA = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.MAX_EPOCHS,
            callbacks=_get_callbacks(f"{name}_crop", patience=config.REGULARISED_PATIENCE),
            class_weight=class_weight,
            verbose=2,
        )
        _save_history(hA, config.RESULTS_DIR / f"{name}_crop_history.json")
        if crop_weights.exists():
            model.load_weights(str(crop_weights))
            print(f"[train] Loaded best crop weights from {crop_weights}")

    # Stage B: transfer to whole mammograms
    train_ds, val_ds, _, _, train_split = _build_datasets(
        norm, augment=True, clahe=False, source="full"
    )
    class_weight = (
        data_loader.compute_class_weights(train_split)
        if config.CLASS_WEIGHT_STRATEGY == "balanced"
        else None
    )

    # Stage B-1: freeze the crop-pretrained conv body and train the head, which keeps
    # its crop weights as the starting point.
    print(f"\n--- Stage B-1: frozen conv body, train head ({name}) ---")
    freeze_scratch_backbone(model)
    _compile(model, lr=config.INITIAL_LR)
    model.summary()
    hB1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=_get_callbacks(f"{name}_indomain", stage="frozen"),
        class_weight=class_weight,
        verbose=2,
    )
    _save_history(hB1, config.RESULTS_DIR / f"{name}_indomain_frozen_history.json")
    best_frozen = config.WEIGHTS_DIR / f"{name}_indomain_frozen_best.weights.h5"
    if best_frozen.exists():
        model.load_weights(str(best_frozen))
        print(f"[train] Loaded best frozen weights from {best_frozen}")

    # Stage B-2: unfreeze the whole network, fine-tune at low LR.
    print(f"\n--- Stage B-2: full fine-tune ({name}) ---")
    unfreeze_scratch_backbone(model)
    _compile(model, lr=config.FINETUNE_LR)
    model.summary()
    hB2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=_get_callbacks(f"{name}_indomain", stage="finetune"),
        class_weight=class_weight,
        verbose=2,
    )
    _save_history(hB2, config.RESULTS_DIR / f"{name}_indomain_finetune_history.json")
    best_ft = config.WEIGHTS_DIR / f"{name}_indomain_finetune_best.weights.h5"
    if best_ft.exists():
        model.load_weights(str(best_ft))
        print(f"[train] Loaded best fine-tuned weights from {best_ft}")

    # Canonical weights for evaluate.py (--model regularised --weights ...).
    canonical = config.WEIGHTS_DIR / f"{name}_indomain_best.weights.h5"
    model.save_weights(str(canonical))
    print(f"[train] Canonical in-domain weights → {canonical}")

    merged = _merge_histories(
        {k: [float(v) for v in vs] for k, vs in hB1.history.items()},
        {k: [float(v) for v in vs] for k, vs in hB2.history.items()},
    )
    with open(config.RESULTS_DIR / f"{name}_indomain_full_history.json", "w") as f:
        json.dump(merged, f, indent=2)

    return model


def train_transfer(
    name: str,
    source: str = "full",
    *,
    clahe: bool = False,
    target_size: tuple[int, int] | None = None,
    pad_square: bool | None = None,
    batch_size: int | None = None,
    intensity_norm: str | None = None,
    crop_sizing: str = config.CROP_SIZING,
    crop_margin: float = config.CROP_MARGIN_FRAC,
    tag_suffix: str = "",
    optimizer: str = "adam",
    mask_overlay: bool = False,
    offline_aug_k: int = 1,
    box_provider=None,
    lr: float | None = None,
    eager_val_auc: bool | None = None,
    seed: int | None = None,
) -> keras.Model:
    """Two-stage transfer: the head with the backbone frozen, then the upper
    backbone fine-tuned at FINETUNE_LR; source="crop" tags "{name}_crop"."""
    tag = f"{name}_crop" if source == "crop" else name
    tag = f"{tag}{_run_suffix(tag_suffix, ablate=None, lr=lr, seed=seed)}"
    print(f"\n{'='*60}")
    print(f"  Training: {tag}  (transfer learning, 2-stage, source={source})")
    print(f"{'='*60}\n")

    norm = get_normalisation_mode(name)
    train_ds, val_ds, test_ds, test_split, train_split = _build_datasets(
        norm,
        source=source,
        clahe=clahe,
        target_size=target_size,
        pad_square=pad_square,
        batch_size=batch_size,
        intensity_norm=intensity_norm,
        crop_sizing=crop_sizing,
        crop_margin=crop_margin,
        mask_overlay=mask_overlay,
        offline_aug_k=offline_aug_k,
        box_provider=box_provider,
        shuffle_seed=seed,
    )
    _reseed(seed)

    class_weight = (
        data_loader.compute_class_weights(train_split)
        if config.CLASS_WEIGHT_STRATEGY == "balanced"
        else None
    )

    # Build the backbone at the (possibly overridden) input resolution.
    channels = MODEL_REGISTRY[name]["channels"]

    ts = target_size or config.TARGET_SIZE
    model = build_model(name, input_shape=(ts[0], ts[1], channels))

    # Stage 1: frozen backbone
    print(f"\n--- Stage 1: frozen backbone ({tag}) ---")
    _compile(model, lr=config.INITIAL_LR if lr is None else lr, optimizer=optimizer)
    model.summary()

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=(
            maybe_eager_val_callback(val_ds, eager_val_auc) + _get_callbacks(tag, stage="frozen")
        ),
        class_weight=class_weight,
        verbose=2,
    )
    _save_history(h1, config.RESULTS_DIR / f"{tag}_frozen_history.json")

    # Fine-tune from the best stage-1 weights, not from the last epoch's.
    best_frozen = config.WEIGHTS_DIR / f"{tag}_frozen_best.weights.h5"
    if best_frozen.exists():
        model.load_weights(str(best_frozen))
        print(f"[train] Loaded best frozen weights from {best_frozen}")

    # Stage 2: fine-tune upper layers
    print(f"\n--- Stage 2: fine-tuning ({tag}) ---")
    unfreeze_for_finetuning(model, backbone_name=name)
    _compile(model, lr=config.FINETUNE_LR, optimizer=optimizer)
    model.summary()

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=(
            maybe_eager_val_callback(val_ds, eager_val_auc) + _get_callbacks(tag, stage="finetune")
        ),
        class_weight=class_weight,
        verbose=2,
    )
    _save_history(h2, config.RESULTS_DIR / f"{tag}_finetune_history.json")
    _save_train_manifest(
        tag, box_provider=box_provider, crop_margin=crop_margin, seed=seed, source=source
    )

    merged = _merge_histories(
        {k: [float(v) for v in vs] for k, vs in h1.history.items()},
        {k: [float(v) for v in vs] for k, vs in h2.history.items()},
    )
    with open(config.RESULTS_DIR / f"{tag}_full_history.json", "w") as f:
        json.dump(merged, f, indent=2)

    best_ft = config.WEIGHTS_DIR / f"{tag}_finetune_best.weights.h5"
    if best_ft.exists():
        model.load_weights(str(best_ft))
        print(f"[train] Loaded best fine-tuned weights from {best_ft}")

    # A copy under the tag evaluate.py loads by.
    canonical = config.WEIGHTS_DIR / f"{tag}_best.weights.h5"
    model.save_weights(str(canonical))
    print(f"[train] Canonical weights → {canonical}")

    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CBIS-DDSM mass classifiers.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", type=str, choices=list(MODEL_REGISTRY), help="Which model to train."
    )
    group.add_argument("--all", action="store_true", help="Train every model in sequence.")
    p.add_argument(
        "--source",
        choices=["full", "crop"],
        default="full",
        help="Image source: whole mammogram or ROI crop.",
    )
    p.add_argument(
        "--indomain",
        action="store_true",
        help="In-domain transfer: pretrain on ROI crops, fine-tune on whole "
        "images (scratch models only).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override input resolution (e.g. 320, 384). Default config.TARGET_SIZE.",
    )
    p.add_argument(
        "--clahe", action="store_true", help="Apply CLAHE contrast enhancement before resize."
    )
    p.add_argument(
        "--pad-square",
        dest="pad_square",
        action="store_true",
        help="Force square zero-padding before resize.",
    )
    p.add_argument(
        "--no-pad-square",
        dest="pad_square",
        action="store_false",
        help="Force no padding (resize directly to target).",
    )
    p.set_defaults(pad_square=None)  # None = decide by source (config.CROP_PAD_SQUARE for crops)
    p.add_argument(
        "--intensity-norm",
        choices=["fixed_max", "percentile", "minmax", "clahe"],
        default=None,
        help="Intensity-normalisation strategy. Default: decide by source "
        "(config.INTENSITY_NORM for whole images, config.CROP_INTENSITY_NORM for crops).",
    )
    p.add_argument(
        "--crop-sizing",
        choices=["provided_tight", "mask_margin", "box_provider"],
        default=config.CROP_SIZING,
        help="Crop derivation: the provided CBIS crop, the full mammogram re-cropped to the "
        "ROI mask bbox + margin, or a --box-source box + margin (box_provider).",
    )
    p.add_argument(
        "--box-source",
        choices=["oracle", "unet", "cam", "jitter", "shuffled", "whole"],
        default=None,
        help="Ladder rung: crop from this box source (implies --crop-sizing "
        "box_provider). Boxes are read from artifacts/, or from --boxes-dir for unet "
        "and jitter; masks are not re-read.",
    )
    p.add_argument(
        "--crop-margin",
        type=float,
        default=config.CROP_MARGIN_FRAC,
        help="Bbox margin fraction for mask_margin and box_provider crops (e.g. 0.40).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (lower it if a high-res run runs out of memory).",
    )
    p.add_argument(
        "--tag-suffix",
        type=str,
        default="",
        help="Suffix appended to weight/history filenames so experiment "
        "variants don't overwrite the baseline run (e.g. _hires).",
    )
    p.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimiser: adam (default) or sgd with Nesterov momentum (often "
        "generalises better on small medical datasets).",
    )
    p.add_argument(
        "--mask-overlay",
        action="store_true",
        help="Apply dilated ROI mask overlay to crop inputs (mask_margin source "
        "only): zeros out tissue beyond config.ROI_MASK_DILATION_PX from the "
        "segmented lesion, forcing the model to focus on the lesion + margin.",
    )
    p.add_argument(
        "--offline-aug",
        type=int,
        choices=[1, 2, 4, 6, 8],
        default=1,
        metavar="K",
        help="Offline augmentation multiplier: expand training set K× by caching "
        "exact 90° rotations and optional h-flips before the epoch loop. "
        "4 = 0°/90°/180°/270° (pure rotations); "
        "6 = (0°/90°/180°) × (no-flip/h-flip); "
        "8 = (0°/90°/180°/270°) × (no-flip/h-flip). "
        "Applies whenever the training set is augmented.",
    )
    p.add_argument(
        "--ablate",
        choices=list(ABLATIONS),
        default=None,
        help="Single-variable ablation of the regularised model: turn one of "
        "head dropout, L2, SE blocks or augmentation off. Tag gets _no_<x>.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate (scratch: the single stage; transfer: "
        "stage 1 only). Tag gets _lr<value>, e.g. _lr5e-4.",
    )
    p.add_argument(
        "--boxes-dir",
        type=str,
        default=None,
        help="Candidate boxes folder for --box-source unet|jitter (localiser_boxes_{split}.csv and "
        "localiser_error_model.json of a promoted localiser); default: the frozen artefacts.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Replicate seed for initialisation, augmentation and shuffle order. "
        "The patient partition always stays on config.SEED. Tag gets _s<N>.",
    )
    p.add_argument(
        "--eager-val-auc",
        choices=["auto", "on", "off"],
        default="auto",
        help="Recompute val_auc through the eager path each epoch and select on it "
        "(EagerValAUC). 'auto' enables it whenever a Metal GPU is present, where "
        "the compiled graph miscomputes ReLU; 'off' selects on Keras's compiled metric.",
    )
    p.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    config.set_seeds()
    _ensure_dirs()
    if args.ablate and (args.all or args.model != "regularised"):
        raise SystemExit("--ablate applies to --model regularised only.")

    if args.all:
        names = list(MODEL_REGISTRY)
    else:
        names = [args.model]

    if args.indomain:
        if args.all or names[0] not in _SCRATCH_MODELS:
            raise SystemExit(
                "--indomain applies to a single scratch model "
                f"(one of {sorted(_SCRATCH_MODELS)}), not --all or transfer models."
            )
        train_indomain(names[0])
        print("\n[train] In-domain transfer complete.")
        return

    target_size = (args.image_size, args.image_size) if args.image_size else None
    pad_square = args.pad_square  # None = decide by source
    provider, crop_sizing = None, args.crop_sizing
    if args.box_source:
        from .boxes import provider_for

        provider, crop_sizing = (
            provider_for(args.box_source, boxes_dir=args.boxes_dir, splits=("train", "val")),
            "box_provider",
        )
    overrides = dict(
        clahe=args.clahe,
        target_size=target_size,
        pad_square=pad_square,
        batch_size=args.batch_size,
        intensity_norm=args.intensity_norm,
        crop_sizing=crop_sizing,
        crop_margin=args.crop_margin,
        tag_suffix=args.tag_suffix,
        optimizer=args.optimizer,
        mask_overlay=args.mask_overlay,
        offline_aug_k=args.offline_aug,
        box_provider=provider,
        lr=args.lr,
        seed=args.seed,
        eager_val_auc={"auto": None, "on": True, "off": False}[args.eager_val_auc],
    )

    for name in names:
        if name in _SCRATCH_MODELS:
            train_baseline(name, source=args.source, ablate=args.ablate, **overrides)
        else:
            train_transfer(name, source=args.source, **overrides)

    print("\n[train] All requested models trained.")


if __name__ == "__main__":
    main()

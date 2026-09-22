"""Trains the localiser: a VGG16-encoder U-Net on sampled patches, half binary
cross-entropy and half soft Dice, warm-up then cosine schedules with separate
encoder and decoder learning rates. Validation is on whole images every epoch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage

from . import config
from .localise import (
    _ENCODER_LAYER_PREFIX,
    TwoRateModel,
    WarmupCosine,
    _augment_pair,
    bce_dice_loss,
    build_unet,
    build_unet_pretrained,
    dice_coefficient,
    elastic_photometric_pair,
    encoder_decoder_variables,
    hard_iou,
)

# Elastic and photometric parameters for --r4c, passed as `aug` to PatchSampler.
R4C_AUG: dict = dict(
    alpha=config.LOC_R4C_ELASTIC_ALPHA,
    sigma=config.LOC_R4C_ELASTIC_SIGMA,
    gamma=config.LOC_R4C_GAMMA,
    brightness=config.LOC_R4C_BRIGHTNESS,
    contrast=config.LOC_R4C_CONTRAST,
    noise_sd=config.LOC_R4C_NOISE_SD,
)
_AUG_KEYS = ("elastic_alpha", "elastic_sigma", "gamma", "brightness", "contrast", "mean_disp_px")
from .localise_eval import _otsu_threshold

# Stores


def load_store(
    store_dir: str | Path, split: str, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """(images, masks, meta, image_scale) for one split of a tensor store; images
    keep their stored dtype and are scaled to [0, 1] on the fly."""
    d = Path(store_dir)
    imgs = np.load(d / f"{split}_images.npy", mmap_mode="r")
    msks = np.load(d / f"{split}_masks.npy", mmap_mode="r")
    meta = pd.read_csv(d / f"{split}_meta.csv")
    if limit:
        imgs, msks, meta = (imgs[:limit], msks[:limit], meta.iloc[:limit].reset_index(drop=True))
    imgs = np.ascontiguousarray(imgs)
    msks = (np.ascontiguousarray(msks) > 0).astype(np.uint8)
    scale = 255.0 if imgs.dtype == np.uint8 else 1.0
    return imgs, msks, meta, scale


def breast_boxes(
    imgs: np.ndarray, meta: pd.DataFrame, scale: float, cache: Path | None = None
) -> np.ndarray:
    """Per-image breast foreground box on the canvas, (y0, y1, x0, x1) inclusive:
    Otsu on the content region, a 7x7 opening to sever film edges, largest blob."""
    if cache is not None and cache.exists():
        b = np.load(cache)
        if len(b) == len(imgs):
            return b
    out = np.zeros((len(imgs), 4), np.int32)
    for i in range(len(imgs)):
        r = meta.iloc[i]
        py, px = int(r["pad_y"]), int(r["pad_x"])
        ch = max(1, int(round(int(r["orig_h"]) * float(r["scale"]))))
        cw = max(1, int(round(int(r["orig_w"]) * float(r["scale"]))))
        arr = imgs[i].astype(np.float32) / scale
        content = arr[py : py + ch, px : px + cw]
        tissue = arr > _otsu_threshold(content)
        tissue = ndimage.binary_opening(tissue, structure=np.ones((7, 7)))
        box = (py, py + ch - 1, px, px + cw - 1)
        if tissue.any():
            lab, n = ndimage.label(tissue)
            if n:
                lab = lab == (int(np.bincount(lab.ravel())[1:].argmax()) + 1)
                ys, xs = np.flatnonzero(lab.any(1)), np.flatnonzero(lab.any(0))
                box = (int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1]))
        out[i] = box
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


# Patch sampler

# Patch kinds, stored in column 3 of the epoch plan.
_KIND_ANYWHERE, _KIND_CENTRED, _KIND_BACKGROUND = 0, 1, 2


class PatchSampler:
    """Deterministic per-epoch patch plans: per image, a share of patches centred
    on a jittered lesion pixel and the rest in the breast box, from seed + epoch."""

    def __init__(
        self,
        imgs: np.ndarray,
        msks: np.ndarray,
        breast: np.ndarray,
        *,
        scale: float,
        patch: int = config.LOC_PATCH_SIZE,
        per_image: int = config.LOC_PATCHES_PER_IMAGE,
        centred_frac: float = config.LOC_LESION_CENTRED_FRAC,
        jitter_frac: float = config.LOC_PATCH_JITTER_FRAC,
        seed: int = config.SEED,
        centred_stochastic: bool = False,
        background_frac: float = 0.0,
        aug: dict | None = None,
    ):
        """background_frac is the share of patches drawn with no ground-truth pixel:
        a 512-px patch nearly always overlaps a lesion, so plain tissue is scarce."""
        self.imgs, self.msks, self.breast, self.scale = imgs, msks, breast, scale
        self.patch, self.per_image, self.seed = patch, per_image, seed
        self.centred_frac, self.centred_stochastic = float(centred_frac), bool(centred_stochastic)
        self.background_frac = float(background_frac)
        if not 0.0 <= self.background_frac <= 1.0:
            raise ValueError(f"background_frac must be in [0, 1], got {background_frac}")
        if self.centred_frac + self.background_frac > 1.0 + 1e-9:
            raise ValueError(
                f"centred_frac {centred_frac} + background_frac {background_frac} exceeds 1"
            )
        self.n_centred = int(round(per_image * centred_frac))
        self.aug = dict(aug) if aug else None
        self.jitter = int(round(jitter_frac * patch))
        self.h, self.w = imgs.shape[1], imgs.shape[2]
        if self.patch > self.h or self.patch > self.w:
            raise ValueError(f"patch {patch} larger than canvas {(self.h, self.w)}")
        # Foreground coordinates per image; a lesion-centred patch picks one
        # uniformly, so larger lesions are centred proportionally more often.
        self.fg = [np.argwhere(m > 0) for m in msks]
        self.epoch_stats: dict[int, dict] = {}

    @property
    def patches_per_epoch(self) -> int:
        return len(self.imgs) * self.per_image

    def _clamp(self, y: float, x: float) -> tuple[int, int]:
        y0 = int(min(max(round(y), 0), self.h - self.patch))
        x0 = int(min(max(round(x), 0), self.w - self.patch))
        return y0, x0

    def _centred(self, rng: np.random.Generator, i: int) -> tuple[int, int]:
        fg = self.fg[i]
        if len(fg) == 0:  # no foreground: fall back to anywhere
            return self._anywhere(rng, i)
        cy, cx = fg[rng.integers(len(fg))]
        jy, jx = rng.integers(-self.jitter, self.jitter + 1, size=2)
        return self._clamp(cy + jy - self.patch / 2, cx + jx - self.patch / 2)

    def _anywhere(self, rng: np.random.Generator, i: int) -> tuple[int, int]:
        y0b, y1b, x0b, x1b = (int(v) for v in self.breast[i])

        def _axis(lo: int, hi: int, limit: int) -> int:
            top = hi - self.patch + 1  # last valid origin inside the box
            if top <= lo:  # box narrower than the patch: centre on it
                return int(min(max(round((lo + hi) / 2 - self.patch / 2), 0), limit - self.patch))
            return int(rng.integers(lo, top + 1))

        return _axis(y0b, y1b, self.h), _axis(x0b, x1b, self.w)

    def _background(
        self, rng: np.random.Generator, i: int, tries: int = 24
    ) -> tuple[int, int, bool]:
        """A window inside the breast box holding no ground-truth pixel, by
        rejection sampling. ok is False when none turned up within `tries`."""
        P = self.patch
        y0 = x0 = 0
        for _ in range(tries):
            y0, x0 = self._anywhere(rng, i)
            if not self.msks[i, y0 : y0 + P, x0 : x0 + P].any():
                return y0, x0, True
        return y0, x0, False

    def epoch_plan(self, epoch: int) -> np.ndarray:
        """(n, 4) int array of (image_idx, y0, x0, kind) for one epoch."""
        rng = np.random.default_rng(self.seed + epoch)
        rows = []
        n_bg_wanted = n_bg_ok = 0
        for i in range(len(self.imgs)):
            for k in range(self.per_image):
                if self.background_frac > 0.0:
                    # One categorical per patch, so centred_frac keeps meaning
                    # the share of centred patches whatever background_frac is.
                    u = rng.random()
                    if u < self.centred_frac:
                        kind = _KIND_CENTRED
                    elif u < self.centred_frac + self.background_frac:
                        kind = _KIND_BACKGROUND
                    else:
                        kind = _KIND_ANYWHERE
                elif self.centred_stochastic:
                    kind = _KIND_CENTRED if rng.random() < self.centred_frac else _KIND_ANYWHERE
                else:
                    kind = _KIND_CENTRED if k < self.n_centred else _KIND_ANYWHERE

                if kind == _KIND_CENTRED:
                    y0, x0 = self._centred(rng, i)
                elif kind == _KIND_BACKGROUND:
                    n_bg_wanted += 1
                    y0, x0, ok = self._background(rng, i)
                    n_bg_ok += int(ok)
                    if not ok:
                        kind = _KIND_ANYWHERE  # no empty window: label it honestly
                else:
                    y0, x0 = self._anywhere(rng, i)
                rows.append((i, y0, x0, kind))
        plan = np.asarray(rows, np.int64)
        plan = plan[rng.permutation(len(plan))]
        P = self.patch
        has_fg = np.fromiter(
            (self.msks[i, y : y + P, x : x + P].any() for i, y, x, _ in plan), bool, len(plan)
        )
        self.epoch_stats[epoch] = {
            "lesion_centred_frac": float((plan[:, 3] == _KIND_CENTRED).mean()),
            "background_frac_realised": float((plan[:, 3] == _KIND_BACKGROUND).mean()),
            "background_frac_requested": self.background_frac,
            "background_draws_satisfied": (float(n_bg_ok / n_bg_wanted) if n_bg_wanted else 1.0),
            "fg_patch_frac": float(has_fg.mean()),
        }
        return plan

    def generator(self):
        """Infinite generator over epochs; consumed with steps_per_epoch. With
        `aug`, each patch is transformed by a generator seeded apart from the plan."""
        P = self.patch
        epoch = 0
        while True:
            plan = self.epoch_plan(epoch)
            arng = np.random.default_rng(self.seed + epoch + 10_000) if self.aug else None
            sums = {k: 0.0 for k in _AUG_KEYS}
            n_aug = 0
            for i, y0, x0, _ in plan:
                img = self.imgs[i, y0 : y0 + P, x0 : x0 + P].astype(np.float32) / self.scale
                msk = self.msks[i, y0 : y0 + P, x0 : x0 + P].astype(np.float32)
                if self.aug:
                    img, msk, real = elastic_photometric_pair(img, msk, arng, **self.aug)
                    for k in _AUG_KEYS:
                        sums[k] += real[k]
                    n_aug += 1
                    if n_aug % 64 == 0 or n_aug == len(plan):
                        self.epoch_stats[epoch].update(
                            {f"aug_{k}_mean": sums[k] / n_aug for k in _AUG_KEYS}
                        )
                yield img[..., None], msk[..., None]
            epoch += 1


def patch_dataset(sampler: PatchSampler, batch_size: int) -> tf.data.Dataset:
    P = sampler.patch
    sig = (tf.TensorSpec((P, P, 1), tf.float32), tf.TensorSpec((P, P, 1), tf.float32))
    ds = tf.data.Dataset.from_generator(sampler.generator, output_signature=sig)
    ds = ds.batch(batch_size)
    ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def whole_image_dataset(
    imgs: np.ndarray,
    msks: np.ndarray,
    scale: float,
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """Whole-canvas image/mask pairs, fetched by index so neither array becomes
    a graph constant."""
    h, w = imgs.shape[1], imgs.shape[2]

    def _fetch(i):
        i = int(i)
        return (
            imgs[i].astype(np.float32)[..., None] / scale,
            msks[i].astype(np.float32)[..., None],
        )

    def _wrap(i):
        img, msk = tf.numpy_function(_fetch, [i], [tf.float32, tf.float32])
        img.set_shape((h, w, 1))
        msk.set_shape((h, w, 1))
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(imgs), dtype=np.int64))
    if shuffle:
        ds = ds.shuffle(len(imgs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_wrap, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if augment:
        ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# Callbacks and manifest


class EpochExtras(tf.keras.callbacks.Callback):
    """Inject sampler statistics and both learning rates into the epoch logs.
    Must precede CSVLogger: one logs dict passes through the callbacks in order."""

    def __init__(self, sampler: PatchSampler | None, enc_sched, dec_sched):
        super().__init__()
        self.sampler, self.enc_sched, self.dec_sched = sampler, enc_sched, dec_sched

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if self.sampler is not None:
            logs.update(self.sampler.epoch_stats.get(epoch, {}))
        it = self.model.optimizer.iterations
        logs["lr_encoder"] = float(self.enc_sched(it))
        logs["lr_decoder"] = float(self.dec_sched(it))


def recipe_fields(
    model: tf.keras.Model, encoder: str, merged_dir, epochs: int, encoder_prefix: str | None
) -> dict:
    """Manifest rows describing the recipe, from config and the compiled model; patience
    equals epochs, as there is no EarlyStopping here."""

    # Detected, not assumed, and scoped to encoder_prefix: the TernausNet decoder
    # has BatchNorm of its own, while the VGG16 encoder has none.
    def walk(m):
        """Every layer, descending into nested models."""
        for lyr in getattr(m, "layers", []):
            yield lyr
            yield from walk(lyr)

    enc_bn = encoder_prefix is not None and any(
        isinstance(lyr, tf.keras.layers.BatchNormalization) and lyr.name.startswith(encoder_prefix)
        for lyr in walk(model)
    )
    w_bce, w_dice = config.LOC_DICE_BCE_WEIGHTS
    return {
        "framework": "keras-tf" + ".".join(tf.__version__.split(".")[:2]),
        "decoder": (
            "TernausNet-style U-Net decoder, transposed-conv upsampling, "
            f"base_filters {config.LOC_BASE_FILTERS}, depth 4"
        ),
        "pretrained_weights": (
            "none (scratch U-Net)"
            if encoder == "none"
            else f"ImageNet {encoder.upper()} (Keras applications)"
        ),
        "cache": str(merged_dir),
        "loss": f"{w_bce} * BCE + {w_dice} * soft-Dice",
        "optimizer": type(model.optimizer).__name__,
        "optimizer_params": {
            "weight_decay": float(getattr(model.optimizer, "weight_decay", 0.0) or 0.0)
        },
        "patience": epochs,
        "input_preprocessing": "x255 -> RGB->BGR -> Caffe channel means subtracted",
        "input_channels": 3,
        "encoder_bn": (
            "none (scratch U-Net, no pretrained encoder)"
            if encoder == "none"
            else "trainable" if enc_bn else f"none ({encoder.upper()} has no BatchNorm)"
        ),
        # The rows below describe how maps become boxes, which localise_eval does
        # and this module does not: they record the config values in force here.
        "inference_mode": f"whole-image at {config.LOC_INPUT_H}x{config.LOC_INPUT_W} (no tiling)",
        "flip_tta_at_gate": False,
        "box_rule": "largest surviving connected component",
        "box_threshold": config.LOC_MASK_THRESHOLD,
        "box_min_px": config.LOC_MIN_COMPONENT_PX,
        "recipe_fields_from": (
            "config and the compiled model at training time; the "
            "box and inference rows are the config values in force"
        ),
    }


def write_manifest(out: Path, info: dict, history_csv: Path) -> dict:
    """Manifest summarising the history CSV, written even if training stopped
    early; NaN epochs are recorded rather than dropped."""
    m = dict(info)
    if history_csv.exists():
        h = pd.read_csv(history_csv)
        m["epochs_run"] = int(len(h))
        nan_rows = h.index[h[["loss", "val_loss"]].isna().any(axis=1)].tolist()
        m["first_nan_epoch"] = int(nan_rows[0]) + 1 if nan_rows else None
        ok = h.dropna(subset=["val_hard_iou", "val_loss"])
        if len(ok):
            b = int(ok["val_hard_iou"].idxmax())
            m.update(
                best_epoch=b + 1,
                val_hard_iou_at_best_epoch=float(h.loc[b, "val_hard_iou"]),
                val_dice_at_best_epoch=float(h.loc[b, "val_dice_coefficient"]),
                val_loss_at_best_epoch=float(h.loc[b, "val_loss"]),
                train_hard_iou_at_best_epoch=float(h.loc[b, "hard_iou"]),
                best_valloss_epoch=int(ok["val_loss"].idxmin()) + 1,
                val_hard_iou_at_best_valloss_epoch=float(
                    h.loc[int(ok["val_loss"].idxmin()), "val_hard_iou"]
                ),
                max_val_hard_iou_any_epoch=float(ok["val_hard_iou"].max()),
                max_val_dice_any_epoch=float(ok["val_dice_coefficient"].max()),
                final_train_loss=float(h["loss"].iloc[-1]),
            )
    else:
        m["epochs_run"], m["first_nan_epoch"] = 0, None
    (out / "manifest.json").write_text(json.dumps(m, indent=2, default=float))
    print(f"-> {out / 'manifest.json'}")
    return m


# Main


def main(
    merged_dir: str | Path,
    val_dir: str | Path,
    out_dir: str | Path,
    *,
    encoder: str = config.LOC_ENCODER,
    whole_image: bool = False,
    epochs: int = config.LOC_EPOCHS,
    batch_size: int | None = None,
    patch_size: int = config.LOC_PATCH_SIZE,
    run_name: str | None = None,
    smoke: bool = False,
    seed: int = config.SEED,
    init_weights: str | Path | None = None,
    lr_encoder: float = config.LOC_ENC_LR,
    lr_decoder: float = config.LOC_DEC_LR,
    warmup_epochs: int = config.LOC_WARMUP_EPOCHS,
    decoder_dropout: float = 0.0,
    decoder_l2: float = 0.0,
    centred_frac: float = config.LOC_LESION_CENTRED_FRAC,
    centred_stochastic: bool = False,
    background_frac: float = 0.0,
    aug: dict | None = None,
) -> dict:
    """Train the localiser and return the manifest. seed covers initialisation,
    the patch plan, shuffle and augmentation; the stores are built on config.SEED."""
    config.set_seeds(seed)
    tf.keras.mixed_precision.set_global_policy("float32")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    limit_tr, limit_va = (4, 2) if smoke else (None, None)

    imgs, msks, meta, scale = load_store(merged_dir, "train", limit_tr)
    v_imgs, v_msks, v_meta, v_scale = load_store(val_dir, "val", limit_va)
    # Foreground share of the training masks; the output bias is initialised to
    # its logit, so the network starts at the right base rate.
    prior = float(msks.mean())
    print(
        f"[l4] train store {imgs.shape} {imgs.dtype} (merged, one row per mammogram) | "
        f"val store {v_imgs.shape} {v_imgs.dtype} (per lesion) | output prior {prior:.5f}"
    )

    if whole_image:
        batch = batch_size or config.LOC_BATCH_SIZE
        sampler = None
        train_ds = whole_image_dataset(
            imgs, msks, scale, batch_size=batch, shuffle=True, augment=True, seed=seed
        )
        steps = int(np.ceil(len(imgs) / batch))
        regime = "whole_image"
    else:
        batch = batch_size or config.LOC_PATCH_BATCH_SIZE
        boxes = breast_boxes(imgs, meta, scale, cache=out / "breast_boxes_train.npy")
        sampler = PatchSampler(
            imgs,
            msks,
            boxes,
            scale=scale,
            patch=patch_size,
            seed=seed,
            centred_frac=centred_frac,
            centred_stochastic=centred_stochastic,
            background_frac=background_frac,
            aug=aug,
        )
        train_ds = patch_dataset(sampler, batch)
        steps = int(np.ceil(sampler.patches_per_epoch / batch))
        regime = f"patch{patch_size}"
    if smoke:
        steps, epochs = 2, 1
    val_ds = whole_image_dataset(
        v_imgs, v_msks, v_scale, batch_size=config.LOC_BATCH_SIZE, shuffle=False, augment=False
    )

    total_steps, warm_steps = steps * epochs, steps * warmup_epochs
    enc_sched = WarmupCosine(lr_encoder, config.LOC_LR_MIN, warm_steps, total_steps)
    dec_sched = WarmupCosine(lr_decoder, config.LOC_LR_MIN, warm_steps, total_steps)

    if encoder == "none":  # scratch U-Net: one group, one rate
        base = build_unet(output_prior=prior)
        model = TwoRateModel(base.inputs, base.outputs, name=base.name)
        enc_vars, dec_vars, prefix = [], model.trainable_variables, None
    else:
        base = build_unet_pretrained(
            encoder, output_prior=prior, decoder_dropout=decoder_dropout, decoder_l2=decoder_l2
        )
        model = TwoRateModel(base.inputs, base.outputs, name=base.name)
        enc_vars, dec_vars = encoder_decoder_variables(model, encoder)
        prefix = _ENCODER_LAYER_PREFIX[encoder]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(dec_sched),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, hard_iou],
    )
    model.set_encoder_optimizer(tf.keras.optimizers.Adam(enc_sched), prefix)
    if init_weights:
        model.load_weights(str(init_weights))
        print(f"[l4] initialised from {init_weights}")
    n_params = model.count_params()
    print(
        f"[l4] {model.name}: {n_params:,} params ({sum(int(np.prod(v.shape)) for v in enc_vars):,} "
        f"encoder at {lr_encoder:g}, {sum(int(np.prod(v.shape)) for v in dec_vars):,} decoder at "
        f"{lr_decoder:g}) | regime {regime} | batch {batch} | {steps} steps/epoch x {epochs}"
    )
    assert tf.keras.mixed_precision.global_policy().name == "float32"

    gpus = tf.config.list_physical_devices("GPU")
    device = str(tf.config.experimental.get_device_details(gpus[0])) if gpus else "CPU"
    info = dict(
        run=run_name or out.name,
        device=device,
        encoder=encoder,
        regime=regime,
        train_store="merged (one row per mammogram, union of lesion masks)",
        n_train_images=int(len(imgs)),
        n_val_lesions=int(len(v_imgs)),
        patch_size=None if whole_image else patch_size,
        patches_per_image=None if whole_image else config.LOC_PATCHES_PER_IMAGE,
        lesion_centred_frac=None if whole_image else centred_frac,
        lesion_centred_stochastic=None if whole_image else centred_stochastic,
        background_frac=None if whole_image else background_frac,
        patch_jitter_frac=None if whole_image else config.LOC_PATCH_JITTER_FRAC,
        decoder_dropout=decoder_dropout,
        decoder_l2=decoder_l2,
        elastic_photometric_aug=(dict(aug) if aug else None),
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        lr_min=config.LOC_LR_MIN,
        warmup_epochs=warmup_epochs,
        schedule="linear warm-up then cosine",
        init_weights=(str(init_weights) if init_weights else None),
        batch_size=batch,
        epochs_requested=epochs,
        steps_per_epoch=steps,
        seed=seed,
        input=[config.LOC_INPUT_H, config.LOC_INPUT_W],
        params=int(n_params),
        dice_bce_weights=list(config.LOC_DICE_BCE_WEIGHTS),
        output_prior=prior,
        output_prior_bias=float(np.log(prior / (1 - prior))),
        precision="float32",
        selection_metric="val_hard_iou",
        secondary_checkpoint="val_loss",
        hard_iou_threshold=config.LOC_MASK_THRESHOLD,
        augmentation=(
            "h-flip + rotation (as run 3)"
            if not aug
            else "h-flip + rotation + elastic (joint) + "
            "gamma/brightness/contrast/noise (image only)"
        ),
        smoke=smoke,
    )
    info.update(recipe_fields(model, encoder, merged_dir, epochs, prefix))
    (out / "manifest.json").write_text(json.dumps(info, indent=2, default=float))

    history_csv = out / "history.csv"
    callbacks = [
        EpochExtras(sampler, enc_sched, dec_sched),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"),
            monitor="val_hard_iou",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best_valloss.weights.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            steps_per_epoch=steps,
            callbacks=callbacks,
            verbose=2 if smoke else 1,
        )
    finally:
        info["wall_time_s"] = round(time.time() - t0, 1)
        manifest = write_manifest(out, info, history_csv)
    print(f"[l4] done in {info['wall_time_s']} s -> {out}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--merged-dir", default=str(config.LOC_TENSORS_MERGED_DIR))
    ap.add_argument("--val-dir", default=str(config.OUTPUTS_DIR / "localiser_tensors"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--encoder", choices=["vgg16", "none"], default=config.LOC_ENCODER)
    ap.add_argument(
        "--whole-image",
        action="store_true",
        help="train on whole images at LOC_BATCH_SIZE instead of patches",
    )
    ap.add_argument("--epochs", type=int, default=config.LOC_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--patch-size", type=int, default=config.LOC_PATCH_SIZE)
    ap.add_argument(
        "--smoke", action="store_true", help="4 train images, 2 val images, 1 epoch of 2 steps"
    )
    ap.add_argument("--init-weights", default=None, help="start from these weights")
    ap.add_argument("--lr-encoder", type=float, default=config.LOC_ENC_LR)
    ap.add_argument("--lr-decoder", type=float, default=config.LOC_DEC_LR)
    ap.add_argument("--warmup-epochs", type=int, default=config.LOC_WARMUP_EPOCHS)
    ap.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="replicate seed (init, patch plan, shuffle, augmentation); partition unchanged",
    )
    ap.add_argument(
        "--background-frac",
        type=float,
        default=0.0,
        help="share of patches drawn as pure background (no ground-truth pixel in the "
        "window), Bernoulli per patch",
    )
    ap.add_argument(
        "--r4c",
        action="store_true",
        help="decoder dropout and L2, a Bernoulli lesion-centred fraction, and elastic and "
        "photometric augmentation, from config.LOC_R4C_*; every other setting unchanged",
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    return ap.parse_args(argv)


def r4c_kwargs() -> dict:
    """The --r4c additions, in one place so every launcher passes the same set."""
    return dict(
        decoder_dropout=config.LOC_R4C_DECODER_DROPOUT,
        decoder_l2=config.LOC_R4C_DECODER_L2,
        centred_frac=config.LOC_R4C_LESION_CENTRED_FRAC,
        centred_stochastic=True,
        aug=dict(R4C_AUG),
    )


def cli(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    if a.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    kw = dict(
        encoder=a.encoder,
        whole_image=a.whole_image,
        epochs=a.epochs,
        batch_size=a.batch_size,
        patch_size=a.patch_size,
        run_name=a.run_name,
        smoke=a.smoke,
        seed=a.seed,
        init_weights=a.init_weights,
        lr_encoder=a.lr_encoder,
        lr_decoder=a.lr_decoder,
        warmup_epochs=a.warmup_epochs,
        background_frac=a.background_frac,
    )
    if a.r4c:
        kw.update(r4c_kwargs())
    if a.cpu:
        with tf.device("/CPU:0"):
            main(a.merged_dir, a.val_dir, a.out_dir, **kw)
    else:
        main(a.merged_dir, a.val_dir, a.out_dir, **kw)


if __name__ == "__main__":
    cli()

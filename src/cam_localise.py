"""Weakly-supervised lesion localisation: the Grad-CAM of a whole-image classifier is
binarised, reduced to its largest connected component, and that component's box is
mapped back to mammogram coordinates. No ROI mask shapes a box; the ground-truth boxes
score the boxes and, with --sweep, select the threshold on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config
from .data_loader import (
    _build_combined_lookup,
    _normalise_intensity,
    _pad_and_resize,
    _read_full,
    _scratch_norm_stats,
)
from .localise import box_centroid_hit, box_iou, mask_to_box
from .localise_eval import _add_any_lesion_iou, report
from .models import MODEL_REGISTRY, build_model

ART = Path(__file__).resolve().parent.parent / "artifacts"
CAM_DIR = config.OUTPUTS_DIR / "cam"
CANVAS = config.TARGET_SIZE[0]  # square canvas side the classifier sees

Box = tuple[int, int, int, int]
_DEVICE = "/GPU:0"  # overridden to /CPU:0 by --cpu; set in main()

# Model surgery


def _last_conv_name(m: keras.Model) -> str:
    """Name of the last layer with a spatial (4-D) output that is not a pool."""
    for layer in reversed(m.layers):
        shape = getattr(layer, "output", None)
        shape = getattr(shape, "shape", None)
        if shape is None or len(shape) != 4:
            continue
        if isinstance(layer, (keras.layers.MaxPooling2D, keras.layers.AveragePooling2D)):
            continue
        return layer.name
    raise ValueError(f"No 4-D layer found in {m.name}")


def _split_model(model: keras.Model, layer_name: str | None):
    """Return (forward, layer_name) where forward(x) -> (feature_map, prob). A nested
    backbone's layers are unreachable from the outer graph, so the pass is rebuilt."""
    nested = [l for l in model.layers if isinstance(l, keras.Model)]
    if nested:
        backbone = nested[0]
        name = layer_name or _last_conv_name(backbone)
        feat = keras.Model(backbone.inputs, [backbone.get_layer(name).output, backbone.output])
        head = model.layers[model.layers.index(backbone) + 1 :]

        def forward(x):
            fmap, h = feat(x, training=False)
            for layer in head:
                h = layer(h, training=False)
            return fmap, h

    else:
        name = layer_name or _last_conv_name(model)
        m2 = keras.Model(model.inputs, [model.get_layer(name).output, model.output])

        def forward(x):
            return m2(x, training=False)

    return forward, name


def grad_cam(forward, x: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM for a batch, targeting the predicted class: with one sigmoid output the
    score is p or 1 - p. Returns (cam [B,h,w] in [0,1], prob [B])."""
    with tf.GradientTape() as tape:
        fmap, prob = forward(x)
        p = tf.reshape(prob, [-1])
        score = tf.where(p >= 0.5, p, 1.0 - p)
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # [B, C]
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))
    mx = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = tf.where(mx > 0, cam / tf.maximum(mx, 1e-12), cam)
    return cam.numpy().astype(np.float32), p.numpy().astype(np.float32)


# Input preprocessing: must match the whole-image classifier exactly


def _prep(full: np.ndarray, normalisation: str, intensity_norm: str) -> np.ndarray:
    arr = _normalise_intensity(full, intensity_norm)
    x = _pad_and_resize(arr, config.TARGET_SIZE, pad_square=True)  # [224,224,1]
    if normalisation == "imagenet":
        x = np.repeat(x, 3, axis=-1)
        mean = np.asarray(config.NORM_STATS["imagenet_mean"], np.float32)
        std = np.asarray(config.NORM_STATS["imagenet_std"], np.float32)
    else:
        mean, std = _scratch_norm_stats(intensity_norm, "full")
    return ((x - mean) / std).astype(np.float32)


def compute_cams(
    model_name: str,
    weights: Path,
    split: str,
    *,
    layer: str | None,
    intensity_norm: str,
    batch: int,
    force: bool,
) -> dict:
    """One Grad-CAM per unique image in `split`, cached to outputs/cam/."""
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    cache = CAM_DIR / (
        f"{model_name}_{split}.npz" if layer is None else f"{model_name}_{layer}_{split}.npz"
    )
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        print(f"[cam] loaded cached CAMs {cache} ({len(z['image'])} images)")
        return {k: z[k] for k in z.files} | {"layer": str(z["layer"])}

    rows = pd.read_csv(ART / f"localiser_boxes_{split}.csv")
    images = rows.drop_duplicates("image")[["image", "orig_h", "orig_w"]]
    lookup = {
        Path(k).parts[0]: v
        for k, v in _build_combined_lookup(config.TRAIN_IMG_DIR, config.TEST_IMG_DIR).items()
    }

    normalisation = MODEL_REGISTRY[model_name]["normalisation"]
    with tf.device(_DEVICE):
        model = build_model(model_name)
        model.load_weights(str(weights))
        forward, layer_name = _split_model(model, layer)
    print(f"[cam] {model_name} <- {weights.name}; CAM layer = {layer_name}")

    cams, probs, ids, hs, ws = [], [], [], [], []
    buf, buf_meta = [], []

    def _flush():
        if not buf:
            return
        with tf.device(_DEVICE):
            c, p = grad_cam(forward, tf.constant(np.stack(buf)))
        c = tf.image.resize(c[..., None], [CANVAS, CANVAS], method="bilinear").numpy()[..., 0]
        cams.extend(c.astype(np.float16))
        probs.extend(p)
        for im, h, w in buf_meta:
            ids.append(im)
            hs.append(h)
            ws.append(w)
        buf.clear()
        buf_meta.clear()

    for i, r in enumerate(images.itertuples()):
        path = lookup.get(r.image)
        if path is None:
            print(f"[cam] WARNING: no DICOM for {r.image}; skipped")
            continue
        buf.append(_prep(_read_full(str(path)), normalisation, intensity_norm))
        buf_meta.append((r.image, int(r.orig_h), int(r.orig_w)))
        if len(buf) == batch:
            _flush()
        if (i + 1) % 50 == 0:
            print(f"  {split}: {i + 1}/{len(images)}")
    _flush()

    out = dict(
        cam=np.stack(cams),
        prob=np.asarray(probs, np.float32),
        image=np.asarray(ids),
        orig_h=np.asarray(hs),
        orig_w=np.asarray(ws),
        layer=np.asarray(layer_name),
    )
    np.savez_compressed(cache, **out)
    print(f"[cam] {split}: {len(ids)} images -> {cache}")
    return out | {"layer": layer_name}


# CAM -> box


def cam_to_box(cam: np.ndarray, orig_h: int, orig_w: int, *, tau: float, min_px: int) -> Box | None:
    """Binarise a CAM at `tau` x its maximum over the content region, keep the largest
    component, and scale its box by S / CANVAS, the inverse of _pad_and_resize."""
    # The classifier saw a top-left square canvas whose padding is exact zero, so any
    # activation outside the content region is a border artefact.
    S = max(orig_h, orig_w)
    ch = max(1, int(round(CANVAS * orig_h / S)))
    cw = max(1, int(round(CANVAS * orig_w / S)))
    content = cam[:ch, :cw].astype(np.float32)
    if not np.isfinite(content).any() or content.max() <= 0:
        return None
    binary = np.zeros_like(cam, dtype=np.float32)
    binary[:ch, :cw] = content > tau * float(content.max())
    box = mask_to_box(binary, threshold=0.5, min_px=min_px, keep_largest=True)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    f = S / CANVAS
    return (
        min(orig_h - 1, int(np.floor(y0 * f))),
        min(orig_h - 1, int(np.floor((y1 + 1) * f)) - 1),
        min(orig_w - 1, int(np.floor(x0 * f))),
        min(orig_w - 1, int(np.floor((x1 + 1) * f)) - 1),
    )


def score_split(split: str, cams: dict, *, tau: float, min_px: int) -> pd.DataFrame:
    """Per-lesion rows in the localiser_boxes schema, scored in mammogram space."""
    rows = pd.read_csv(ART / f"localiser_boxes_{split}.csv")
    idx = {im: i for i, im in enumerate(cams["image"])}
    boxes: dict[str, Box | None] = {}
    for im, i in idx.items():
        boxes[im] = cam_to_box(
            cams["cam"][i], int(cams["orig_h"][i]), int(cams["orig_w"][i]), tau=tau, min_px=min_px
        )

    out = []
    for r in rows.itertuples():
        rec = dict(
            split=split,
            lesion_id=r.lesion_id,
            patient_id=r.patient_id,
            label=r.label,
            orig_h=r.orig_h,
            orig_w=r.orig_w,
            image=r.image,
        )
        for c in ("breast_y0", "breast_y1", "breast_x0", "breast_x1"):
            rec[c] = getattr(r, c, np.nan)
        if r.image in idx:
            rec["cam_prob"] = float(cams["prob"][idx[r.image]])
        p = boxes.get(r.image)
        rec["detected"] = p is not None
        has_gt = not pd.isna(getattr(r, "gt_y0", np.nan))
        if not has_gt:
            rec["status"] = "no_gt"
            out.append(rec)
            continue
        g = (int(r.gt_y0), int(r.gt_y1), int(r.gt_x0), int(r.gt_x1))
        rec.update(gt_y0=g[0], gt_y1=g[1], gt_x0=g[2], gt_x1=g[3])
        if p is None:
            rec.update(status="miss", box_iou=0.0, box_iou_mammogram=0.0, centroid_hit=False)
            out.append(rec)
            continue
        iou = box_iou(p, g)
        gh, gw = g[1] - g[0] + 1, g[3] - g[2] + 1
        ph, pw = p[1] - p[0] + 1, p[3] - p[2] + 1
        rec.update(
            status="ok",
            pred_y0=p[0],
            pred_y1=p[1],
            pred_x0=p[2],
            pred_x1=p[3],
            box_iou=iou,
            box_iou_mammogram=iou,
            centroid_hit=box_centroid_hit(p, g),
            dy_frac=(((p[0] + p[1]) - (g[0] + g[1])) / 2) / gh,
            dx_frac=(((p[2] + p[3]) - (g[2] + g[3])) / 2) / gw,
            scale_ratio_h=ph / gh,
            scale_ratio_w=pw / gw,
        )
        out.append(rec)
    return _add_any_lesion_iou(pd.DataFrame(out))


def sweep(cams: dict, *, grid, min_px: int, tol: float) -> tuple[float, pd.DataFrame]:
    """Validation sweep of the max-relative CAM threshold: the middle of the thresholds
    within `tol` of the best box IoU, since the argmax alone would select on noise."""
    recs = []
    for tau in grid:
        df = score_split("val", cams, tau=tau, min_px=min_px)
        scored = df[df["status"] != "no_gt"]
        inc = pd.to_numeric(scored["box_iou"], errors="coerce").fillna(0.0)
        anyv = pd.to_numeric(scored["box_iou_mammogram_any"], errors="coerce").fillna(0.0)
        recs.append(
            dict(
                tau=tau,
                iou_inclusive=float(inc.mean()),
                iou_any_inclusive=float(anyv.mean()),
                detection=float(scored["detected"].mean()),
                at_iou50=float((inc >= 0.5).mean()),
                at_iou30=float((inc >= 0.3).mean()),
                centroid_hit=float(scored["centroid_hit"].fillna(False).astype(bool).mean()),
            )
        )
    tab = pd.DataFrame(recs)
    plateau = tab[tab["iou_inclusive"] >= tab["iou_inclusive"].max() - tol]
    chosen = float(plateau["tau"].iloc[len(plateau) // 2])  # mid-plateau
    print("\n--- CAM threshold sweep (validation, tau x max) ---")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"plateau (within {tol} of max): {list(plateau['tau'])} -> selected tau {chosen:g}")
    return chosen, tab


# Refusals


def _refuse_frozen_records() -> None:
    """Refuses to run while any artifacts/cam_* record this module writes exists."""
    names = ("cam_boxes.csv", "cam_metrics.json", "cam_manifest.json")
    present = [f"artifacts/{n}" for n in names if (ART / n).exists()]
    if present:
        raise SystemExit(
            f"REFUSED: frozen records present: {', '.join(present)}; "
            "this run would overwrite them"
        )


def _refuse_test_split(splits: list[str]) -> None:
    """Refuses a run that includes the test partition unless the freeze token is set."""
    from .test_pass_guard import authorised

    if "test" in splits and not authorised():
        raise SystemExit(
            "REFUSED: --splits includes test; test CAM boxes are produced only before "
            "the freeze and are frozen in artifacts/"
        )


# CLI


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument(
        "--weights",
        default=None,
        help="default outputs/weights/{model}_best.weights.h5, the whole-image run, not _crop.",
    )
    ap.add_argument(
        "--layer", default=None, help="Grad-CAM layer name; default = last conv activation."
    )
    ap.add_argument(
        "--intensity-norm",
        default=config.INTENSITY_NORM,
        help="Must match the classifier's training run.",
    )
    ap.add_argument("--splits", nargs="+", default=["val", "train", "test"])
    ap.add_argument("--sweep", action="store_true", help="Select tau on validation before scoring.")
    ap.add_argument(
        "--tau",
        type=float,
        default=config.CAM_THRESHOLD,
        help="Binarise the CAM at tau x its maximum.",
    )
    ap.add_argument("--min-px", type=int, default=config.CAM_MIN_COMPONENT_PX)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Recompute cached CAMs.")
    ap.add_argument(
        "--cpu", action="store_true", help="Run on the CPU (leave the GPU to a training run)."
    )
    args = ap.parse_args()
    _refuse_frozen_records()
    _refuse_test_split(args.splits)

    global _DEVICE
    if args.cpu:
        _DEVICE = "/CPU:0"
    config.set_seeds()
    weights = Path(args.weights or config.WEIGHTS_DIR / f"{args.model}_best.weights.h5")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    splits = list(args.splits)
    if args.sweep and "val" not in splits:
        splits.insert(0, "val")
    cams = {
        sp: compute_cams(
            args.model,
            weights,
            sp,
            layer=args.layer,
            intensity_norm=args.intensity_norm,
            batch=args.batch,
            force=args.force,
        )
        for sp in splits
    }

    tau, table = args.tau, None
    if args.sweep:
        tau, table = sweep(
            cams["val"],
            grid=config.CAM_THRESHOLD_GRID,
            min_px=args.min_px,
            tol=config.CAM_SWEEP_TOL,
        )

    frames, metrics = [], {}
    for sp in splits:
        df = score_split(sp, cams[sp], tau=tau, min_px=args.min_px)
        frames.append(df)
        metrics[sp] = report(df, sp)

    ART.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(ART / "cam_boxes.csv", index=False)
    (ART / "cam_metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest = dict(
        model=args.model,
        weights=str(weights),
        cam_layer=cams[splits[0]]["layer"],
        cam_target="predicted class",
        threshold_rule="tau x max activation over content region",
        intensity_norm=args.intensity_norm,
        tau=tau,
        min_px=args.min_px,
        threshold_source=(
            "validation sweep, mid-plateau" if args.sweep else "fixed (config.CAM_THRESHOLD)"
        ),
        sweep_table=None if table is None else table.to_dict("records"),
        splits=splits,
    )
    (ART / "cam_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[cam] wrote {ART / 'cam_boxes.csv'}, cam_metrics.json, cam_manifest.json")


if __name__ == "__main__":
    main()

"""Scores a localiser: probability maps turned into boxes, box and mask metrics
per split, and the error model the ladder's jitter rung resamples.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
import tensorflow as tf

from . import config
from .localise import (
    Letterbox,
    mask_to_box,
    topk_components,
    box_iou,
    box_centroid_hit,
    bce_dice_loss,
    dice_coefficient,
    hard_iou,
    build_unet,
    predict_tiled,
)

K = config.LOC_TOPK
_CAND_FIELDS = ("y0", "y1", "x0", "x1", "sum", "area")

TENSORS = config.OUTPUTS_DIR / "localiser_tensors"
OUT = Path(__file__).resolve().parent.parent / "artifacts"


def _load(split: str, tensors_dir: str | Path | None = None, *, images_only: bool = False):
    """Load a split's tensors; tensors_dir selects the normalisation frame a model
    must be scored in. images_only returns msks=None rather than a zero dummy."""
    d = TENSORS if tensors_dir is None else Path(tensors_dir)
    imgs = np.load(d / f"{split}_images.npy")
    mp = d / f"{split}_masks.npy"
    if images_only:
        msks = None
    elif not mp.exists():
        raise FileNotFoundError(
            f"{mp} is absent. Pass images_only=True to score without masks; "
            f"every mask-dependent product is then skipped, not filled in."
        )
    else:
        msks = np.load(mp)
    meta = pd.read_csv(d / f"{split}_meta.csv")
    scale = 255.0 if imgs.dtype == np.uint8 else 1.0
    return imgs, msks, meta, scale


def _otsu_threshold(v: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for a [0,1] array, in numpy (no skimage dependency)."""
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    m1 = m0[-1] - m0
    mu0 = np.where(w0 > 0, m0 / np.maximum(w0, 1.0), 0.0)
    mu1 = np.where(w1 > 0, m1 / np.maximum(w1, 1.0), 0.0)
    return float(centres[int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2))])


def evaluate_split(
    model,
    split: str,
    batch: int = 8,
    tensors_dir: str | Path | None = None,
    tiled: bool = False,
    tile_overlap: int = 64,
) -> pd.DataFrame:
    imgs, msks, meta, scale = _load(split, tensors_dir)
    rows = []
    # Side tables for the multi-lesion pass: per row the canvas GT box and top-k
    # candidates; per image the packed mask, one prediction serving all its rows.
    gt_canvas: dict[int, tuple] = {}
    cands_canvas: dict[int, list] = {}
    packed_pred: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    for start in range(0, len(imgs), batch):
        block = imgs[start : start + batch].astype(np.float32)[..., None] / scale
        if tiled:
            # Predict in LOC_PATCH_SIZE tiles, the scale the network was trained at,
            # and average the overlaps.
            probs = np.stack(
                [
                    predict_tiled(model, block[k, ..., 0], overlap=tile_overlap, batch=batch)
                    for k in range(len(block))
                ]
            )
        else:
            # Eager, never model.predict: tensorflow-metal drops ReLU from the
            # compiled graph. The tiled branch above is eager via predict_tiled.
            probs = np.asarray(model(block, training=False))[..., 0]
        for i, prob in enumerate(probs):
            idx = start + i
            r = meta.iloc[idx]

            # Built before the gt branch: the breast box and the candidate boxes
            # below need it on every row.
            lb = Letterbox(
                scale=float(r["scale"]),
                pad_y=int(r["pad_y"]),
                pad_x=int(r["pad_x"]),
                orig_h=int(r["orig_h"]),
                orig_w=int(r["orig_w"]),
                out_h=int(r["out_h"]),
                out_w=int(r["out_w"]),
            )

            gt = mask_to_box(
                msks[idx].astype(np.float32), threshold=0.5, min_px=1, keep_largest=False
            )
            pred = mask_to_box(prob)  # threshold, min pixels, largest component
            cands = topk_components(prob, K)  # same filter, ranked by summed probability
            cands_canvas[idx] = [c[0] for c in cands]
            image_key = str(r["lesion_id"]).rsplit("_", 1)[0]
            if image_key not in packed_pred:
                packed_pred[image_key] = (np.packbits(prob > config.LOC_MASK_THRESHOLD), prob.shape)

            rec = {
                "split": split,
                "lesion_id": r["lesion_id"],
                "patient_id": r["patient_id"],
                "label": int(r["label"]),
                # Frame dimensions, kept for the record and the breast-area
                # check; the shuffled rung places boxes in the breast box below.
                "orig_h": int(r["orig_h"]),
                "orig_w": int(r["orig_w"]),
                "detected": pred is not None,
            }

            # Otsu on the content region only: digitised film carries base+fog
            # density, and the padding is exactly 0, a third mode Otsu cannot take.
            arr = imgs[idx].astype(np.float32) / scale
            content = arr[lb.pad_y : lb.pad_y + lb.content_h, lb.pad_x : lb.pad_x + lb.content_w]
            tissue = arr > _otsu_threshold(content)
            # A film edge can bridge to the breast, making the largest component
            # span the frame; a 7x7 opening severs the bridge and barely erodes the breast.
            tissue = ndimage.binary_opening(tissue, structure=np.ones((7, 7)))
            if tissue.any():
                lab, n = ndimage.label(tissue)
                if n:
                    lab = lab == (int(np.bincount(lab.ravel())[1:].argmax()) + 1)
                    ys, xs = np.flatnonzero(lab.any(1)), np.flatnonzero(lab.any(0))
                    b = lb.inverse_box((int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])))
                    rec.update(breast_y0=b[0], breast_y1=b[1], breast_x0=b[2], breast_x1=b[3])

            # Mask-level agreement (what the loss optimised).
            pm = prob > config.LOC_MASK_THRESHOLD
            gm = msks[idx] > 0.5
            inter = float((pm & gm).sum())
            rec["mask_iou"] = inter / max(1.0, float((pm | gm).sum()))
            rec["dice"] = 2 * inter / max(1.0, float(pm.sum() + gm.sum()))
            rec["pred_fg_frac"] = float(pm.mean())

            # Candidate boxes in mammogram coordinates: what BoxProvider serves.
            for j in range(K):
                if j < len(cands):
                    cb, csum, carea = cands[j]
                    mb = lb.inverse_box(cb)
                    rec.update(
                        {
                            f"cand{j + 1}_y0": mb[0],
                            f"cand{j + 1}_y1": mb[1],
                            f"cand{j + 1}_x0": mb[2],
                            f"cand{j + 1}_x1": mb[3],
                            f"cand{j + 1}_sum": csum,
                            f"cand{j + 1}_area": carea,
                        }
                    )
                else:
                    rec.update({f"cand{j + 1}_{f}": np.nan for f in _CAND_FIELDS})

            if gt is None:
                rec["status"] = "no_gt"
                rows.append(rec)
                continue

            gt_canvas[idx] = gt
            g = lb.inverse_box(gt)
            rec.update(gt_y0=g[0], gt_y1=g[1], gt_x0=g[2], gt_x1=g[3])

            if pred is None:  # localiser miss - GT still recorded
                rec.update(status="miss", box_iou=0.0, box_iou_mammogram=0.0, centroid_hit=False)
                rows.append(rec)
                continue

            rec["status"] = "ok"

            # Both boxes in mammogram coordinates: what the crop pipeline consumes
            # and what the jitter rung is fitted on; `g` was inverted above.
            p = lb.inverse_box(pred)
            rec.update(pred_y0=p[0], pred_y1=p[1], pred_x0=p[2], pred_x1=p[3])
            rec["box_iou"] = box_iou(gt, pred)  # canvas coords
            rec["box_iou_mammogram"] = box_iou(g, p)  # full resolution
            rec["centroid_hit"] = box_centroid_hit(pred, gt)

            gh, gw = g[1] - g[0] + 1, g[3] - g[2] + 1
            ph, pw = p[1] - p[0] + 1, p[3] - p[2] + 1
            # Error model for the jitter rung: centre offset as a fraction of
            # true box size, and predicted/true size ratio.
            rec["dy_frac"] = (((p[0] + p[1]) - (g[0] + g[1])) / 2) / gh
            rec["dx_frac"] = (((p[2] + p[3]) - (g[2] + g[3])) / 2) / gw
            rec["scale_ratio_h"] = ph / gh
            rec["scale_ratio_w"] = pw / gw
            rows.append(rec)
    df = _add_any_lesion_iou(pd.DataFrame(rows))
    df = _add_matched_topk(df, gt_canvas, cands_canvas, meta)
    return _add_image_dice(df, msks, packed_pred)


def _add_matched_topk(
    df: pd.DataFrame, gt_canvas: dict, cands_canvas: dict, meta: pd.DataFrame
) -> pd.DataFrame:
    """Greedy per-image matching of the top-k candidates to the image's lesions,
    highest IoU first; each serves at most one, and an unmatched lesion is IoU 0."""
    if df.empty or "gt_y0" not in df.columns:
        return df
    df = df.copy()
    rank = np.zeros(len(df), int)
    iou_k = np.zeros(len(df), float)
    boxes = np.full((len(df), 4), np.nan)
    by_image: dict[str, list[int]] = {}
    for i, lid in enumerate(df["lesion_id"].astype(str)):
        by_image.setdefault(lid.rsplit("_", 1)[0], []).append(i)
    for img, idxs in by_image.items():
        cands = cands_canvas.get(idxs[0], [])
        pairs = [
            (box_iou(gt_canvas[i], c), i, j)
            for i in idxs
            if i in gt_canvas
            for j, c in enumerate(cands)
        ]
        pairs = [p for p in pairs if p[0] > 0]
        pairs.sort(key=lambda p: -p[0])
        used_i, used_j = set(), set()
        for v, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            rank[i], iou_k[i] = j + 1, v
            r = df.iloc[i]
            boxes[i] = [
                r[f"cand{j + 1}_y0"],
                r[f"cand{j + 1}_y1"],
                r[f"cand{j + 1}_x0"],
                r[f"cand{j + 1}_x1"],
            ]
    df["k3_rank"] = rank
    df["k3_detected"] = rank > 0
    df["k3_pred_y0"], df["k3_pred_y1"], df["k3_pred_x0"], df["k3_pred_x1"] = boxes.T
    df["box_iou_k3"] = np.where(df["status"] == "no_gt", np.nan, iou_k)
    return df


def _add_image_dice(df: pd.DataFrame, msks: np.ndarray, packed_pred: dict) -> pd.DataFrame:
    """Per-image Dice on the union of the image's lesion masks, repeated on its
    rows; dice_image_detected is NaN when the image has no detection."""
    if df.empty or "gt_y0" not in df.columns:
        return df
    df = df.copy()
    dice_img = np.full(len(df), np.nan)
    by_image: dict[str, list[int]] = {}
    for i, lid in enumerate(df["lesion_id"].astype(str)):
        by_image.setdefault(lid.rsplit("_", 1)[0], []).append(i)
    for img, idxs in by_image.items():
        packed, shape = packed_pred[img]
        pm = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)
        gm = np.zeros(shape, bool)
        for i in idxs:
            gm |= msks[i] > 0.5
        inter = float((pm & gm).sum())
        d = 2 * inter / max(1.0, float(pm.sum() + gm.sum()))
        dice_img[idxs] = d
    df["dice_image"] = dice_img
    df["dice_image_detected"] = np.where(df["detected"], dice_img, np.nan)
    return df


def _add_any_lesion_iou(df: pd.DataFrame) -> pd.DataFrame:
    """box_iou_mammogram_any asks whether a lesion was found, not THE lesion of the row;
    on an image that mixes labels it can credit a row with a lesion of the other label."""
    if df.empty or "gt_y0" not in df.columns:
        return df
    df = df.copy()
    df["image"] = df["lesion_id"].str.replace(r"_\d+$", "", regex=True)

    gts: dict[str, list] = {}
    for r in df.itertuples():
        if not pd.isna(getattr(r, "gt_y0", np.nan)):
            gts.setdefault(r.image, []).append((r.gt_y0, r.gt_y1, r.gt_x0, r.gt_x1))
    df["n_lesions_on_image"] = df["image"].map(lambda im: len(gts.get(im, [])))

    vals = []
    for r in df.itertuples():
        if r.status == "ok":
            vals.append(
                max(box_iou(g, (r.pred_y0, r.pred_y1, r.pred_x0, r.pred_x1)) for g in gts[r.image])
            )
        else:
            vals.append(0.0 if r.status == "miss" else np.nan)
    df["box_iou_mammogram_any"] = vals
    return df


def report(df: pd.DataFrame, split: str) -> dict:
    """Summarise one split. box_iou_inclusive_mean counts a miss as IoU 0 and is
    the gate metric; the conditional mean beside it excludes misses."""
    scored = df[df["status"] != "no_gt"]
    ok = df[df["status"] == "ok"]
    n, n_scored = len(df), len(scored)

    def _mean(frame: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if col not in frame.columns or frame.empty:
            return default
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else default

    iou_inclusive = pd.to_numeric(
        scored.get("box_iou", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)

    s = {
        "split": split,
        "n": n,
        "n_scored": n_scored,
        "n_no_gt": int((df["status"] == "no_gt").sum()),
        "detection_rate": float(scored["detected"].mean()) if n_scored else 0.0,
        "fallback_rate": float((~scored["detected"]).mean()) if n_scored else 0.0,
        "mask_iou_mean": _mean(df, "mask_iou"),
        "dice_mean": _mean(df, "dice"),
        # Miss-inclusive — the gate metric.
        "box_iou_inclusive_mean": float(iou_inclusive.mean()) if n_scored else 0.0,
        "box_iou_inclusive_median": float(iou_inclusive.median()) if n_scored else 0.0,
        # Conditional on the localiser having fired at all.
        "box_iou_mean_given_detected": _mean(ok, "box_iou"),
        "box_iou_mammogram_mean": _mean(ok, "box_iou_mammogram"),
        "detect_at_iou50": float((iou_inclusive >= 0.5).mean()) if n_scored else 0.0,
        "detect_at_iou30": float((iou_inclusive >= 0.3).mean()) if n_scored else 0.0,
        "centroid_hit_rate": (
            float(scored["centroid_hit"].fillna(False).astype(bool).mean())
            if n_scored and "centroid_hit" in scored.columns
            else 0.0
        ),
        "box_iou_any_inclusive_mean": (
            float(
                pd.to_numeric(
                    scored.get("box_iou_mammogram_any", pd.Series(dtype=float)), errors="coerce"
                )
                .fillna(0.0)
                .mean()
            )
            if n_scored
            else 0.0
        ),
        "pct_rows_on_multilesion_images": (
            float((scored["n_lesions_on_image"] > 1).mean())
            if n_scored and "n_lesions_on_image" in scored.columns
            else 0.0
        ),
    }
    # Matched top-k boxes and per-image Dice on the merged mask.
    if "box_iou_k3" in scored.columns and n_scored:
        k3 = pd.to_numeric(scored["box_iou_k3"], errors="coerce").fillna(0.0)
        s["box_iou_matched_k3_inclusive_mean"] = float(k3.mean())
        s["detect_at_iou50_matched_k3"] = float((k3 >= 0.5).mean())
        s["detection_rate_matched_k3"] = float(scored["k3_detected"].mean())
        s["n_rows_matched_box_differs_from_single"] = int(
            (
                (scored["k3_rank"] > 0)
                & scored["detected"]
                & (
                    scored["k3_pred_y0"].ne(scored["pred_y0"])
                    | scored["k3_pred_y1"].ne(scored["pred_y1"])
                    | scored["k3_pred_x0"].ne(scored["pred_x0"])
                    | scored["k3_pred_x1"].ne(scored["pred_x1"])
                )
            ).sum()
            + ((scored["k3_rank"] > 0) & ~scored["detected"]).sum()
            + ((scored["k3_rank"] == 0) & scored["detected"]).sum()
        )
    if "dice_image" in scored.columns and n_scored:
        per_image = scored.drop_duplicates("image")
        s["n_images"] = int(len(per_image))
        s["dice_image_mean_inclusive"] = float(per_image["dice_image"].fillna(0.0).mean())
        det = per_image["dice_image_detected"].dropna()
        s["dice_image_mean_detected"] = float(det.mean()) if len(det) else 0.0
    print(f"\n--- {split} (n={n}, scored={n_scored}, no GT={s['n_no_gt']}) ---")
    for k, v in s.items():
        if k not in ("split", "n", "n_scored", "n_no_gt"):
            print(f"  {k:<30} {v:.4f}")
    return s


def _load_model(
    path: str,
    base_filters: int,
    depth: int,
    encoder: str = "none",
    input_hw: tuple[int, int] | None = None,
):
    """Load the localiser from a .keras archive, an unpacked archive dir or a bare
    weights file; encoder, base_filters and depth describe the SAVED model."""
    p = Path(path)
    if p.is_dir():
        p = p / "model.weights.h5"
    if p.name.endswith(".weights.h5"):
        tf.keras.mixed_precision.set_global_policy("float32")
        if encoder != "none":
            from .localise import build_unet_pretrained

            ih, iw = input_hw or (config.LOC_INPUT_H, config.LOC_INPUT_W)
            model = build_unet_pretrained(encoder, input_h=ih, input_w=iw)
            model.load_weights(str(p))
            print(
                f"[localise_eval] rebuilt {encoder}-encoder U-Net "
                f"({model.count_params():,} params) from {p}"
            )
            return model
        model = (
            build_unet(
                base_filters=base_filters, depth=depth, input_h=input_hw[0], input_w=input_hw[1]
            )
            if input_hw
            else build_unet(base_filters=base_filters, depth=depth)
        )
        model.load_weights(str(p))
        print(
            f"[localise_eval] rebuilt base_filters={base_filters} "
            f"depth={depth} ({model.count_params():,} params) from {p}"
        )
        return model
    return tf.keras.models.load_model(
        str(p),
        compile=False,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coefficient": dice_coefficient,
            "hard_iou": hard_iou,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument(
        "--base-filters",
        type=int,
        default=config.LOC_BASE_FILTERS,
        help="base filter count of the saved model; the default is the config value, "
        "which need not match it",
    )
    ap.add_argument("--depth", type=int, default=config.LOC_DEPTH, help="depth of the saved model")
    ap.add_argument(
        "--encoder",
        choices=["none", "vgg16"],
        default="none",
        help="none = scratch U-Net; vgg16 = U-Net with an ImageNet VGG16 encoder "
        "(--base-filters and --depth ignored)",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["val", "train"],
        help="splits to score; the default leaves out test",
    )
    ap.add_argument(
        "--out-dir",
        default=str(OUT),
        help="where to write localiser_boxes_*.csv, localiser_metrics.json "
        "and localiser_error_model.json. Point it elsewhere for a candidate "
        "model, so the frozen artefacts in artifacts/ survive.",
    )
    ap.add_argument(
        "--tiled",
        action="store_true",
        help="predict in LOC_PATCH_SIZE tiles with --tile-overlap, averaging the "
        "overlaps, instead of one whole-image forward pass. Needed for a 2048x1152 "
        "store, where a whole pass needs four times the activation memory of 1024x576.",
    )
    ap.add_argument("--tile-overlap", type=int, default=64)
    ap.add_argument(
        "--tensors-dir",
        default=str(TENSORS),
        help="tensor store to score from. Score a model from the cache it was "
        "trained on, or it is measured on inputs it never saw.",
    )
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    args = ap.parse_args()
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    if "test" in args.splits:
        raise SystemExit(
            "REFUSED: [localise_eval] WARNING: scoring the TEST split. Only "
            "notebooks/scripts/run_test_pass.sh may do this."
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Tiled inference needs the model declared at the tile size, not the canvas;
    # the network is fully convolutional, so the same weights load either way.
    model = _load_model(
        args.weights,
        args.base_filters,
        args.depth,
        args.encoder,
        input_hw=((config.LOC_PATCH_SIZE, config.LOC_PATCH_SIZE) if args.tiled else None),
    )

    summaries, frames = [], []
    for split in args.splits:
        df = evaluate_split(
            model,
            split,
            tensors_dir=args.tensors_dir,
            tiled=args.tiled,
            tile_overlap=args.tile_overlap,
        )
        df.to_csv(out / f"localiser_boxes_{split}.csv", index=False)
        # Candidate set for BoxProvider(mode="matched"): the k3 columns only.
        k3_cols = [
            "split",
            "lesion_id",
            "image",
            "n_lesions_on_image",
            "status",
            "k3_rank",
            "k3_detected",
            "k3_pred_y0",
            "k3_pred_y1",
            "k3_pred_x0",
            "k3_pred_x1",
            "box_iou_k3",
        ] + [f"cand{j + 1}_{f}" for j in range(K) for f in _CAND_FIELDS]
        df[[c for c in k3_cols if c in df.columns]].to_csv(
            out / f"localiser_boxes_k{K}_{split}.csv", index=False
        )
        s = report(df, split)
        # The frame is part of the measurement: a model scored from a store it
        # was not trained on is comparable with nothing, so record which it was.
        s["tensors_dir"] = str(args.tensors_dir)
        s["tiled_inference"] = bool(args.tiled)
        summaries.append(s)
        frames.append(df)

    allb = pd.concat(frames, ignore_index=True)
    (out / "localiser_metrics.json").write_text(json.dumps(summaries, indent=2))

    # Error model for the jitter rung, fitted on validation: the localiser has seen
    # every training image, so its error there understates the error on new data.
    fit_split = "val"
    tr = allb[allb["split"] == fit_split]
    if tr.empty:
        raise RuntimeError(
            f"No '{fit_split}' rows — the jitter error model cannot be fitted. "
            f"Include '{fit_split}' in --splits."
        )
    ok = tr[tr["status"] == "ok"]
    if ok.empty:
        raise RuntimeError(f"No '{fit_split}' rows with status='ok'.")

    # Empirical, not parametric: the measured error is bimodal, so whole error tuples
    # are resampled.
    err = {
        "fitted_on": fit_split,
        "n_fitted_on": int(len(ok)),
        "n_lesions": int(len(tr)),
        # The fallback fraction, which the jitter rung reproduces: the two rungs
        # differ only in which lesions get a bad box, the U-Net's choice or chance.
        "fallback_rate": float((tr["status"] == "miss").mean()),
        "samples": ok[["dy_frac", "dx_frac", "scale_ratio_h", "scale_ratio_w"]].to_numpy().tolist(),
        # Descriptive only: the parametric fit that was rejected, kept for the record
        # and never used for sampling.
        "gaussian_rejected": {
            c: {"mean": float(ok[c].mean()), "std": float(ok[c].std())}
            for c in ("dy_frac", "dx_frac", "scale_ratio_h", "scale_ratio_w")
        },
    }
    (out / "localiser_error_model.json").write_text(json.dumps(err, indent=2))

    val = next(s for s in summaries if s["split"] == "val")
    gate = val["box_iou_inclusive_mean"] >= config.LOC_GATE_BOX_IOU
    print(
        f"\nGATE: val mean box IoU {val['box_iou_inclusive_mean']:.4f} "
        f"(misses counted as 0; {val['box_iou_mean_given_detected']:.4f} "
        f"conditional on detection, detection rate "
        f"{val['detection_rate']:.3f}) — "
        f"{'PASS, keep the supervised localiser' if gate else 'FAIL, fall back to CAM-only'}"
    )


if __name__ == "__main__":
    main()

"""Validation, box tables and map writing for the PyTorch members. Scoring goes
through the shared `localiser_bundle` functions, so a torch member and a Keras one
are measured by one rule; inference is whole-image by default and tiling opt-in.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from .. import config
from ..geometry import Letterbox, area_downsample, tile_positions
from .dataset import to_encoder_input

_ROOT = Path(__file__).resolve().parents[2]


def bundle():
    """Load `localiser_bundle` as a module: the scorer shared with the Keras members."""
    spec = importlib.util.spec_from_file_location(
        "lb", _ROOT / "notebooks" / "scripts" / "localiser_bundle.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def predict_tiled(
    model,
    img: np.ndarray,
    *,
    patch: int = None,
    overlap: int = 64,
    batch: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """Overlapping-tile probability map; positions come from `geometry.tile_positions`,
    the definition the Keras path also uses, and overlaps are averaged.
    """
    patch = patch or config.LOC_PATCH_SIZE
    h, w = img.shape[:2]
    ys, xs = tile_positions(h, patch, overlap), tile_positions(w, patch, overlap)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    wins = [(y, x) for y in ys for x in xs]
    model.eval()
    for i in range(0, len(wins), batch):
        chunk = wins[i : i + batch]
        blk = np.stack([img[y : y + patch, x : x + patch] for y, x in chunk]).astype(np.float32)
        t = to_encoder_input(torch.from_numpy(blk)[:, None].to(device))
        p = torch.sigmoid(model(t)).cpu().numpy()[:, 0]
        for (y, x), q in zip(chunk, p):
            acc[y : y + patch, x : x + patch] += q
            cnt[y : y + patch, x : x + patch] += 1.0
    return acc / np.maximum(cnt, 1.0)


@torch.no_grad()
def predict_whole(
    model, img: np.ndarray, *, device: str = "cpu", return_logits: bool = False
) -> np.ndarray:
    """Whole-image probability map, the default inference mode: the network is fully
    convolutional, and the Keras reference a member is compared against is scored so.
    """
    model.eval()
    x = to_encoder_input(torch.from_numpy(np.asarray(img, np.float32))[None, None].to(device))
    logits = model(x)
    if return_logits:
        return logits
    return torch.sigmoid(logits).cpu().numpy()[0, 0]


def predict_map(
    model, img, *, tiled: bool = False, overlap: int = 64, batch: int = 4, device: str = "cpu"
) -> np.ndarray:
    """The single inference entry point; `tiled` is opt-in."""
    if tiled:
        return predict_tiled(model, img, overlap=overlap, batch=batch, device=device)
    return predict_whole(model, img, device=device)


def validate(
    model, imgs, msks, *, device="cpu", overlap=64, batch=4, limit=None, tiled: bool = False
):
    """Mask IoU, box IoU, detection rate and loss over the validation images; hard IoU is
    thresholded at `config.LOC_MASK_THRESHOLD` and the box comes from `_select_box`.
    """
    lb = bundle()
    n = len(imgs) if limit is None else min(limit, len(imgs))
    thr = config.LOC_MASK_THRESHOLD
    ious, box_ious, detected = [], [], 0
    losses = []
    for i in range(n):
        img = np.asarray(imgs[i], np.float32)
        prob = predict_map(model, img, tiled=tiled, overlap=overlap, batch=batch, device=device)
        # val_loss from the same forward pass, computed from the probabilities, so it
        # is defined the same way whichever inference path produced them.
        pt = torch.from_numpy(prob).clamp(1e-6, 1 - 1e-6)
        gt_t = torch.from_numpy(np.asarray(msks[i], np.float32) > 0.5).float()
        bce = -(gt_t * pt.log() + (1 - gt_t) * (1 - pt).log()).mean()
        dice = 1 - (2 * (pt * gt_t).sum() + 1e-6) / (pt.sum() + gt_t.sum() + 1e-6)
        losses.append(float(0.5 * bce + 0.5 * dice))
        gm = np.asarray(msks[i]) > 0.5
        pm = prob > thr
        union = float((pm | gm).sum())
        ious.append(float((pm & gm).sum()) / union if union else 0.0)
        gt = lb._gt_box(msks[i])
        if gt is None:
            continue
        # REFERENCE_RULE keeps the largest surviving component, the rule every member's
        # box table is written by, so candidate and reference are selected alike.
        box, _ = lb._select_box(prob, gt=gt, breast=None, **lb.REFERENCE_RULE)
        detected += int(box is not None)
        box_ious.append(lb._iou(gt, box) if box is not None else 0.0)
    return {
        "val_hard_iou": float(np.mean(ious)) if ious else float("nan"),
        "val_box_iou_inclusive": float(np.mean(box_ious)) if box_ious else float("nan"),
        "val_detection_rate": detected / max(len(box_ious), 1),
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "n": n,
    }


def write_maps(
    model,
    cache_dir,
    split: str,
    run: str,
    maps_dir,
    *,
    device="cpu",
    overlap=64,
    batch=4,
    out_hw=(1024, 576),
    limit=None,
    tiled: bool = False,
    weights: str | None = None,
    encoder: str | None = None,
    base_filters: int | None = None,
    depth: int | None = None,
):
    """Probability maps in the layout `localiser_bundle cache` writes: uint8,
    `round(255 * p)`, downsampled to the canvas frame the members are averaged in.
    """
    cache_dir, maps_dir = Path(cache_dir), Path(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    imgs = np.load(cache_dir / f"{split}_images.npy", mmap_mode="r")
    n = len(imgs) if limit is None else min(limit, len(imgs))
    h, w = out_hw
    # Written under a .partial name and renamed once complete, so a map file under
    # its final name is always a finished one.
    final = {k: maps_dir / f"{run}_{split}_{k}.npy" for k in ("plain", "flip")}
    partial = {k: p.with_suffix(".npy.partial") for k, p in final.items()}
    plain = np.lib.format.open_memmap(partial["plain"], mode="w+", dtype=np.uint8, shape=(n, h, w))
    flip = np.lib.format.open_memmap(partial["flip"], mode="w+", dtype=np.uint8, shape=(n, h, w))
    for i in range(n):
        img = np.asarray(imgs[i], np.float32)
        p = predict_map(model, img, tiled=tiled, overlap=overlap, batch=batch, device=device)
        pf = predict_map(
            model,
            np.ascontiguousarray(img[:, ::-1]),
            tiled=tiled,
            overlap=overlap,
            batch=batch,
            device=device,
        )[:, ::-1]
        plain[i] = np.rint(area_downsample(p, h, w) * 255.0).astype(np.uint8)
        flip[i] = np.rint(area_downsample(pf, h, w) * 255.0).astype(np.uint8)
    plain.flush()
    flip.flush()
    for k in final:
        os.replace(partial[k], final[k])
    mpath = maps_dir / f"{run}_manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    man.update(
        {
            "run": run,
            "framework": "pytorch",
            "tensors_dir": str(cache_dir),
            "map_canvas": [h, w],
            "tiled_inference": bool(tiled),
            "tile_overlap": overlap if tiled else None,
            "dtype": "uint8 = round(255 * p)",
            "flip": "horizontal, stored un-flipped",
        }
    )
    # The provenance fields `localiser_bundle apply` reads out of a manifest. Only
    # non-None values are written: base_filters and depth mean nothing for an smp Unet.
    for k, v in (
        ("weights", weights),
        ("encoder", encoder),
        ("base_filters", base_filters),
        ("depth", depth),
    ):
        if v is not None:
            man[k] = v
    man.setdefault("splits", {})[split] = {"n": int(n)}
    mpath.write_text(json.dumps(man, indent=2))
    return {"plain": str(maps_dir / f"{run}_{split}_plain.npy"), "n": int(n)}


def boxes_table(
    model,
    cache_dir,
    split: str,
    *,
    device="cpu",
    overlap=64,
    batch=4,
    limit=None,
    tiled: bool = False,
) -> "object":
    """One row per lesion with the core columns of `localiser_boxes_{split}.csv`, the box
    chosen by `_select_box`; `box_iou` is canvas-frame, the column the gate compares.
    """
    import pandas as pd

    lb = bundle()
    cache_dir = Path(cache_dir)
    imgs = np.load(cache_dir / f"{split}_images.npy", mmap_mode="r")
    msks = np.load(cache_dir / f"{split}_masks.npy", mmap_mode="r")
    meta = pd.read_csv(cache_dir / f"{split}_meta.csv")
    n = len(meta) if limit is None else min(limit, len(meta))
    thr = config.LOC_MASK_THRESHOLD
    rows = []
    for i in range(n):
        r = meta.iloc[i]
        prob = predict_map(
            model,
            np.asarray(imgs[i], np.float32),
            tiled=tiled,
            overlap=overlap,
            batch=batch,
            device=device,
        )
        gt = lb._gt_box(msks[i])
        rec = {
            "split": split,
            "lesion_id": r["lesion_id"],
            "image_id": r.get("image_id"),
            "patient_id": r["patient_id"],
            "label": int(r["label"]),
            "orig_h": int(r["orig_h"]),
            "orig_w": int(r["orig_w"]),
        }
        box, _ = lb._select_box(prob, gt=gt, breast=None, **lb.REFERENCE_RULE)
        rec["detected"] = box is not None
        if gt is None:
            rec.update(status="no_gt", box_iou=np.nan)
        elif box is None:
            rec.update(status="miss", box_iou=0.0)
        else:
            # Mammogram coordinates via the letterbox record the cache carries, so
            # a member at any canvas size lands in the same frame as every other.
            lbx = Letterbox(
                scale=float(r["scale"]),
                pad_y=int(r["pad_y"]),
                pad_x=int(r["pad_x"]),
                orig_h=int(r["orig_h"]),
                orig_w=int(r["orig_w"]),
                out_h=int(r["out_h"]),
                out_w=int(r["out_w"]),
            )
            g, p = lbx.inverse_box(gt), lbx.inverse_box(box)
            rec.update(
                status="ok",
                box_iou=lb._iou(gt, box),
                box_iou_mammogram=lb._iou(g, p),
                gt_y0=g[0],
                gt_y1=g[1],
                gt_x0=g[2],
                gt_x1=g[3],
                pred_y0=p[0],
                pred_y1=p[1],
                pred_x0=p[2],
                pred_x1=p[3],
            )
        rows.append(rec)
    return pd.DataFrame(rows)

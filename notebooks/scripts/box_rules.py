#!/usr/bin/env python
"""box_rules.py — alternative ways of turning localiser output into one box per lesion.

Run from the repo root with ``PYTHONPATH=.``; validation and train only. fusion and hybrid
write a candidate box table per split and relative-threshold one for validation, each
gated like any other localiser candidate; rasterise and scale-tta write their scores only.

  --rule fusion weighted box fusion across members, weights selected on train.
  --rule rasterise each member's box painted back into a map, variant and sigma on train.
  --rule hybrid a base table, with a donor's box only where the base produced none.
  --rule scale-tta scale x flip TTA over the members' cached maps; nothing selected.
  --rule relative-threshold t = k * peak(image), with erosion and flip TTA, selected on train.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

CANVAS = (1024, 576)
DEFAULT_MAPS = config.LOC_BUNDLE_DIR / "maps"
DEFAULT_TENSORS = config.OUTPUTS_DIR / "localiser_tensors"


def bundle():
    """The bundle module beside this one (`_select_box`, `REFERENCE_RULE`, `_gt_box`),
    loaded by path because these tools are scripts in a folder, not a package."""
    spec = importlib.util.spec_from_file_location(
        "localiser_bundle", Path(__file__).resolve().parent / "localiser_bundle.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("localiser_bundle", m)
    spec.loader.exec_module(m)
    return m


def _members(tags: list[str], *, need_path: bool) -> dict:
    """`--tag` entries as {name: path-or-None}: a rule reading member TABLES needs
    `name=path`, one reading their cached MAPS needs the bare name."""
    out = {}
    for t in tags:
        name, _, path = t.partition("=")
        if need_path and not path:
            raise SystemExit(
                f"--rule needs member tables: pass {name}=<path to "
                f"localiser_boxes_{{split}}.csv>, not the bare name {name!r}"
            )
        if not need_path and path:
            raise SystemExit(
                f"--rule reads the members' cached maps: pass the bare name {name!r}, not {t!r}"
            )
        out[name] = path or None
    return out


# fusion — weighted box fusion across members
def _iou(a, b) -> float:
    ih = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    iw = max(0, min(a[3], b[3]) - max(a[2], b[2]) + 1)
    inter = ih * iw
    union = (a[1] - a[0] + 1) * (a[3] - a[2] + 1) + (b[1] - b[0] + 1) * (b[3] - b[2] + 1) - inter
    return inter / union if union else 0.0


def fuse(boxes: list, weights: list, thr: float = 0.3):
    """Weighted box fusion: cluster by IoU, take the heaviest cluster's weighted mean box.
    A member with no box contributes no weight, so it abstains rather than voting."""
    live = [(b, w) for b, w in zip(boxes, weights) if b is not None]
    if not live:
        return None
    clusters: list[list] = []
    for b, w in sorted(live, key=lambda t: -t[1]):
        for c in clusters:
            if _iou(c[0][0], b) >= thr:
                c.append((b, w))
                break
        else:
            clusters.append([(b, w)])
    best = max(clusters, key=lambda c: sum(w for _, w in c))
    tw = sum(w for _, w in best)
    return tuple(int(round(sum(b[k] * w for b, w in best) / tw)) for k in range(4))


def load_member(path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(str(path).replace("{split}", split))
    need = {"lesion_id", "status"}
    if not need <= set(df.columns):
        raise SystemExit(f"{path}: missing {need - set(df.columns)}")
    return df.set_index("lesion_id")


def fused_table(members: dict, split: str, weights: dict, thr: float) -> pd.DataFrame:
    tabs = {k: load_member(v, split) for k, v in members.items()}
    base = tabs[next(iter(tabs))]
    rows = []
    for lid in base.index:
        boxes, ws = [], []
        for k, t in tabs.items():
            if lid not in t.index:
                boxes.append(None)
                ws.append(weights[k])
                continue
            r = t.loc[lid]
            ok = r.get("status") == "ok" and not pd.isna(r.get("pred_y0", np.nan))
            boxes.append(
                (int(r["pred_y0"]), int(r["pred_y1"]), int(r["pred_x0"]), int(r["pred_x1"]))
                if ok
                else None
            )
            ws.append(weights[k])
        b = base.loc[lid]
        rec = {
            "split": split,
            "lesion_id": lid,
            "image_id": b.get("image_id"),
            "patient_id": b["patient_id"],
            "label": int(b["label"]),
        }
        gt = None
        if not pd.isna(b.get("gt_y0", np.nan)):
            gt = (int(b["gt_y0"]), int(b["gt_y1"]), int(b["gt_x0"]), int(b["gt_x1"]))
            rec.update(gt_y0=gt[0], gt_y1=gt[1], gt_x0=gt[2], gt_x1=gt[3])
        f = fuse(boxes, ws, thr)
        if gt is None:
            rec.update(status="no_gt", detected=f is not None, box_iou=np.nan)
        elif f is None:
            rec.update(status="miss", detected=False, box_iou=0.0)
        else:
            rec.update(
                status="ok",
                detected=True,
                pred_y0=f[0],
                pred_y1=f[1],
                pred_x0=f[2],
                pred_x1=f[3],
                box_iou=_iou(gt, f),
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def mean_iou(t: pd.DataFrame) -> float:
    s = t[t["status"] != "no_gt"]["box_iou"].fillna(0.0)
    return float(s.mean()) if len(s) else float("nan")


def rule_fusion(args) -> None:
    members = {k: v for k, v in _members(args.tag, need_path=True).items()}
    names = list(members)
    print(f"[fusion] members: {names}")

    # Weights on TRAIN only, then applied unchanged to validation.
    best, best_v = None, -1.0
    for combo in itertools.product(args.weight_grid, repeat=len(names)):
        w = dict(zip(names, combo))
        v = mean_iou(fused_table(members, "train", w, args.threshold))
        if v > best_v:
            best, best_v = w, v
    print(f"[fusion] weights selected on train: {best} train mean box IoU {best_v:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "members": members,
        "weights": best,
        "iou_threshold": args.threshold,
        "weights_selected_on": "train, by mean per-lesion box IoU (misses 0)",
        "train_mean_box_iou": round(best_v, 6),
    }
    for split in ("train", "val"):
        t = fused_table(members, split, best, args.threshold)
        t.to_csv(out / f"localiser_boxes_{split}.csv", index=False)
        summary[f"{split}_mean_box_iou"] = round(mean_iou(t), 6)
        print(
            f"[fusion] {split}: mean box IoU {summary[f'{split}_mean_box_iou']:.4f} "
            f"({(t.status == 'ok').sum()} scored, {(t.status == 'miss').sum()} misses)"
        )
        print(f"-> {out / f'localiser_boxes_{split}.csv'}")
    (out / "box_fusion.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out / 'box_fusion.json'}")


# rasterise — a box painted back into a map
def raster(box, score, shape, kind, sigma_frac):
    """One box -> one map. `box` is (y0, y1, x0, x1) in canvas coordinates."""
    h, w = shape
    y0, y1, x0, x1 = [int(v) for v in box]
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
    if y1 < y0 or x1 < x0:
        return np.zeros(shape, np.float32)
    if kind == "box":
        m = np.zeros(shape, np.float32)
        m[y0 : y1 + 1, x0 : x1 + 1] = score
        return m
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    sy = max(1.0, sigma_frac * (y1 - y0 + 1) / 2.0)
    sx = max(1.0, sigma_frac * (x1 - x0 + 1) / 2.0)
    # Separable: a 2-D Gaussian is the outer product of two 1-D ones, so this is
    # two vectors rather than a per-pixel loop.
    gy = np.exp(-0.5 * ((np.arange(h) - cy) / sy) ** 2)
    gx = np.exp(-0.5 * ((np.arange(w) - cx) / sx) ** 2)
    return (score * np.outer(gy, gx)).astype(np.float32)


def member_frames(members: dict, split: str) -> dict:
    out = {}
    for name, patt in members.items():
        t = pd.read_csv(patt.format(split=split))
        s = t["cand1_sum"] if "cand1_sum" in t.columns else None
        t["_score"] = 1.0 if s is None else (s.fillna(0) / max(float(s.max()), 1e-9)).clip(0, 1)
        t["_has_score"] = s is not None
        out[name] = t.set_index("lesion_id")
    return out


def score_split(
    members: dict, split: str, kind: str, sigma_frac: float, lb, tensors: Path, limit=None
):
    from src.geometry import Letterbox

    meta = pd.read_csv(tensors / f"{split}_meta.csv")
    msks = np.load(tensors / f"{split}_masks.npy", mmap_mode="r")
    frames = member_frames(members, split)
    n = len(meta) if limit is None else min(limit, len(meta))
    ious = []
    for i in range(n):
        r = meta.iloc[i]
        lid = r["lesion_id"]
        gt = lb._gt_box(msks[i])
        if gt is None:
            continue
        lbx = Letterbox(
            scale=float(r["scale"]),
            pad_y=int(r["pad_y"]),
            pad_x=int(r["pad_x"]),
            orig_h=int(r["orig_h"]),
            orig_w=int(r["orig_w"]),
            out_h=int(r["out_h"]),
            out_w=int(r["out_w"]),
        )
        acc = np.zeros(CANVAS, np.float32)
        used = 0
        for name, t in frames.items():
            if lid not in t.index:
                continue
            row = t.loc[lid]
            if str(row.get("status")) != "ok":
                continue
            mb = (row["pred_y0"], row["pred_y1"], row["pred_x0"], row["pred_x1"])
            if any(pd.isna(v) for v in mb):
                continue
            cb = lbx.forward_box(tuple(int(v) for v in mb))  # mammogram -> canvas
            acc += raster(cb, float(row["_score"]), CANVAS, kind, sigma_frac)
            used += 1
        if used:
            acc /= used
        box = lb._select_box(acc, gt=None, breast=None, **lb.REFERENCE_RULE)[0]
        ious.append(lb._iou(gt, box) if box is not None else 0.0)
    return float(np.mean(ious)) if ious else float("nan"), len(ious)


def rule_rasterise(args) -> None:
    lb = bundle()
    members = _members(args.tag, need_path=True)
    tensors = Path(args.tensors_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[raster] members: {list(members)}")

    results = {}
    # Selection on TRAIN only, over both variants and the declared sigma grid.
    best = (None, None, -1.0)
    for kind in ("box", "gauss"):
        for sf in ([0.0] if kind == "box" else args.sigma_grid):
            tr, ntr = score_split(members, "train", kind, sf, lb, tensors, args.limit)
            results[f"train_{kind}_{sf}"] = tr
            print(f"[raster] train {kind:5} sigma_frac {sf:.2f}: mean box IoU {tr:.4f} (n={ntr})")
            if tr > best[2]:
                best = (kind, sf, tr)
    kind, sf, tr = best
    print(f"[raster] SELECTED ON TRAIN: {kind} sigma_frac {sf} (train {tr:.4f})")
    va, nva = score_split(members, "val", kind, sf, lb, tensors, args.limit)
    print(f"[raster] val mean box IoU {va:.4f} (n={nva})")

    (out / "box_rasterise.json").write_text(
        json.dumps(
            {
                "members": list(members),
                "selected_variant": kind,
                "selected_sigma_frac": sf,
                "train_mean_box_iou": tr,
                "val_mean_box_iou": va,
                "train_grid": results,
                "note": "variant and sigma selected on TRAIN only; validation computed once",
            },
            indent=2,
        )
    )
    print(f"-> {out / 'box_rasterise.json'}")


# hybrid — the base's box, the donor's only where the base has none
def rule_hybrid(args) -> None:
    out = Path(args.out)
    if out.exists() and any(out.glob("localiser_boxes_*.csv")):
        raise SystemExit(f"refusing to overwrite an existing box folder: {out}")
    out.mkdir(parents=True, exist_ok=True)

    summary = {"base": args.base, "donor": args.donor, "splits": {}}
    for split in args.split:
        b = pd.read_csv(Path(args.base) / f"localiser_boxes_{split}.csv")
        d = pd.read_csv(Path(args.donor) / f"localiser_boxes_{split}.csv")
        for f in (b, d):
            if "split" in f.columns:
                f.drop(f.index[f["split"] != split], inplace=True)
        b = b.drop_duplicates("lesion_id").set_index("lesion_id")
        d = d.drop_duplicates("lesion_id").set_index("lesion_id")
        miss = b.index[~b["detected"].astype(bool)]
        usable = miss.intersection(d.index[d["detected"].astype(bool)])
        cols = [
            c
            for c in (
                "detected",
                "pred_y0",
                "pred_y1",
                "pred_x0",
                "pred_x1",
                "box_iou",
                "box_iou_mammogram",
                "status",
            )
            if c in b.columns and c in d.columns
        ]
        merged = b.copy()
        merged.loc[usable, cols] = d.loc[usable, cols]
        merged.reset_index().to_csv(out / f"localiser_boxes_{split}.csv", index=False)
        summary["splits"][split] = {
            "n": int(len(b)),
            "base_misses": int(len(miss)),
            "substituted": int(len(usable)),
            "donor_also_missed": int(len(miss) - len(usable)),
            "base_mean_box_iou": float(b["box_iou"].fillna(0).mean()),
            "hybrid_mean_box_iou": float(merged["box_iou"].fillna(0).mean()),
        }
        s = summary["splits"][split]
        print(
            f"  {split}: {s['n']} lesions, base missed {s['base_misses']}, "
            f"substituted {s['substituted']}, donor also missed {s['donor_also_missed']} "
            f"mean box IoU {s['base_mean_box_iou']:.4f} -> {s['hybrid_mean_box_iou']:.4f}"
        )
    (out / "hybrid_manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out}")


# scale-tta — scale x flip TTA over the cached maps
def rescale(m: np.ndarray, s: float) -> np.ndarray:
    """Zoom about the centre by s, back onto the same canvas."""
    if abs(s - 1.0) < 1e-9:
        return m
    h, w = m.shape
    yy = (np.arange(h) - (h - 1) / 2) / s + (h - 1) / 2
    xx = (np.arange(w) - (w - 1) / 2) / s + (w - 1) / 2
    y0 = np.clip(np.floor(yy).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(yy - y0, 0, 1)[:, None]
    fx = np.clip(xx - x0, 0, 1)[None, :]
    top = m[np.ix_(y0, x0)] * (1 - fx) + m[np.ix_(y0, x1)] * fx
    bot = m[np.ix_(y1, x0)] * (1 - fx) + m[np.ix_(y1, x1)] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def rule_scale_tta(args) -> None:
    lb = bundle()
    runs = list(_members(args.tag, need_path=False))
    maps_dir, tensors = Path(args.maps_dir), Path(args.tensors_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.split:
        msks = np.load(tensors / f"{split}_masks.npy", mmap_mode="r")
        meta = pd.read_csv(tensors / f"{split}_meta.csv")
        n = len(meta) if args.limit is None else min(args.limit, len(meta))

        plain = {r: np.load(maps_dir / f"{r}_{split}_plain.npy", mmap_mode="r") for r in runs}
        flip = {r: np.load(maps_dir / f"{r}_{split}_flip.npy", mmap_mode="r") for r in runs}
        print(
            f"[tta] members {runs} | scales {args.scales} x {{identity, flip}} "
            f"= {len(args.scales) * 2} passes per member"
        )

        base_ious, tta_ious = [], []
        for i in range(n):
            gt = lb._gt_box(msks[i])
            if gt is None:
                continue
            base = np.mean([plain[r][i].astype(np.float32) / 255.0 for r in runs], axis=0)
            acc, k = np.zeros_like(base), 0
            for r in runs:
                for src in (plain, flip):
                    m0 = src[r][i].astype(np.float32) / 255.0
                    for s in args.scales:
                        acc += rescale(m0, s)
                        k += 1
            tta = acc / max(k, 1)
            for arr, dst in ((base, base_ious), (tta, tta_ious)):
                b = lb._select_box(arr, gt=None, breast=None, **lb.REFERENCE_RULE)[0]
                dst.append(lb._iou(gt, b) if b is not None else 0.0)

        b, t = float(np.mean(base_ious)), float(np.mean(tta_ious))
        print(f"[tta] {split}: baseline (plain maps, no TTA) {b:.4f}")
        print(f"[tta] {split}: scale+flip TTA {t:.4f} delta {t - b:+.4f}")
        print(f"[tta] deployment cost: {len(args.scales) * 2}x forward passes per member")
        (out / f"scale_tta_{split}.json").write_text(
            json.dumps(
                {
                    "runs": runs,
                    "scales": args.scales,
                    "passes_per_member": len(args.scales) * 2,
                    "split": split,
                    "n": len(base_ious),
                    "baseline_mean_box_iou": b,
                    "tta_mean_box_iou": t,
                    "delta": t - b,
                },
                indent=2,
            )
        )
        print(f"-> {out / f'scale_tta_{split}.json'}")


# relative-threshold — t = k * peak, swept on train
def rel_box(prob, k, erode, lb, min_px):
    """Largest component after thresholding at k*peak, optionally eroded once."""
    from scipy import ndimage

    peak = float(prob.max())
    if peak <= 0:
        return None
    binary = prob > (k * peak)
    if erode:
        binary = ndimage.binary_erosion(binary, structure=np.ones((3, 3)))
    if not binary.any():
        return None
    lab, n = ndimage.label(binary)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    if sizes.max() < min_px:
        return None
    comp = lab == int(sizes.argmax()) + 1
    ys, xs = np.flatnonzero(comp.any(1)), np.flatnonzero(comp.any(0))
    return int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])


def ensemble_map(maps, runs, i, tta):
    """The member-averaged map for one lesion, with or without flip TTA."""
    acc = None
    for r in runs:
        srcs = (maps[r][0], maps[r][1]) if tta else (maps[r][0],)
        for s in srcs:
            m = s[i].astype(np.float32) / 255.0
            acc = m if acc is None else acc + m
    return acc / (len(runs) * (2 if tta else 1))


def rel_score(split, runs, k, erode, tta, lb, min_px, tensors: Path, maps_dir: Path, limit=None):
    meta = pd.read_csv(tensors / f"{split}_meta.csv")
    msks = np.load(tensors / f"{split}_masks.npy", mmap_mode="r")
    maps = {
        r: (
            np.load(maps_dir / f"{r}_{split}_plain.npy", mmap_mode="r"),
            np.load(maps_dir / f"{r}_{split}_flip.npy", mmap_mode="r"),
        )
        for r in runs
    }
    n = len(meta) if limit is None else min(limit, len(meta))
    rows = []
    for i in range(n):
        gt = lb._gt_box(msks[i])
        r0 = meta.iloc[i]
        rec = {
            "split": split,
            "lesion_id": r0["lesion_id"],
            "patient_id": r0["patient_id"],
            "label": int(r0["label"]),
        }
        if gt is None:
            rec.update(status="no_gt", box_iou=np.nan)
            rows.append(rec)
            continue
        b = rel_box(ensemble_map(maps, runs, i, tta), k, erode, lb, min_px)
        rec.update(
            status="ok" if b is not None else "miss",
            box_iou=(lb._iou(gt, b) if b is not None else 0.0),
        )
        rows.append(rec)
    t = pd.DataFrame(rows)
    m = float(t[t["status"] != "no_gt"]["box_iou"].fillna(0).mean())
    return m, t


def rule_relative_threshold(args) -> None:
    lb = bundle()
    runs = list(_members(args.tag, need_path=False))
    tensors, maps_dir = Path(args.tensors_dir), Path(args.maps_dir)
    min_px = config.LOC_MIN_COMPONENT_PX
    print(
        f"[relthresh] members {runs} (fixed a priori) | k grid {args.k_grid} "
        f"| erosion {{off,on}} | tta {{off,on}} | min_px {min_px}"
    )

    best, grid = None, {}
    for tta in (False, True):
        for erode in (False, True):
            for k in args.k_grid:
                m, _ = rel_score(
                    "train", runs, k, erode, tta, lb, min_px, tensors, maps_dir, args.limit
                )
                grid[f"k={k}_erode={int(erode)}_tta={int(tta)}"] = m
                print(f"[relthresh] train k={k:.2f} erode={int(erode)} tta={int(tta)}: {m:.4f}")
                if best is None or m > best[0]:
                    best = (m, k, erode, tta)
    tr, k, erode, tta = best
    print(f"\n[relthresh] SELECTED ON TRAIN: k={k}, erosion={erode}, tta={tta} (train {tr:.4f})")
    va, table = rel_score("val", runs, k, erode, tta, lb, min_px, tensors, maps_dir, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "localiser_boxes_val.csv", index=False)
    (out / "relthresh.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "selected": {"k": k, "erosion": erode, "tta": tta},
                "train_mean_box_iou": tr,
                "val_mean_box_iou": va,
                "train_grid": grid,
                "note": "selected on train only; validation computed once, on the winner",
            },
            indent=2,
        )
    )
    print(f"[relthresh] val mean box IoU {va:.4f}")
    print(f"-> {out / 'localiser_boxes_val.csv'}")


# CLI
RULES = {
    "fusion": rule_fusion,
    "rasterise": rule_rasterise,
    "hybrid": rule_hybrid,
    "scale-tta": rule_scale_tta,
    "relative-threshold": rule_relative_threshold,
}
# Which flags each rule reads, so an argument that does nothing is refused rather
# than silently ignored.
RULE_ARGS = {
    "fusion": {"tag", "threshold", "weight_grid", "out"},
    "rasterise": {"tag", "tensors_dir", "sigma_grid", "out", "limit"},
    "hybrid": {"base", "donor", "out", "split"},
    "scale-tta": {"tag", "scales", "split", "out", "limit", "maps_dir", "tensors_dir"},
    "relative-threshold": {"tag", "k_grid", "out", "limit", "maps_dir", "tensors_dir"},
}
REQUIRED = {
    "fusion": ("tag",),
    "rasterise": ("tag",),
    "hybrid": ("base", "donor"),
    "scale-tta": ("tag",),
    "relative-threshold": ("tag",),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rule", required=True, choices=list(RULES))
    ap.add_argument("--out", required=True, help="NEW folder for this rule's products")
    ap.add_argument(
        "--tag",
        nargs="+",
        default=None,
        help="the members to combine. A rule that reads cached maps (scale-tta, "
        "relative-threshold) takes bare names; one that reads member tables "
        "(fusion, rasterise) takes name=path, and the path may contain {split}",
    )
    ap.add_argument("--base", default=None, help="hybrid: folder with localiser_boxes_{split}.csv")
    ap.add_argument("--donor", default=None, help="hybrid: the folder whose box fills a base miss")
    ap.add_argument(
        "--split",
        nargs="+",
        default=None,
        choices=["val", "train"],
        help="one or more splits; default val for scale-tta, val and train for hybrid. "
        "The rules that select on train always read both.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="fusion: IoU at which two members' boxes join one cluster",
    )
    ap.add_argument(
        "--weight-grid",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="fusion: per-member weights searched on train",
    )
    ap.add_argument(
        "--sigma-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.50, 0.70, 1.00],
        help="rasterise: Gaussian widths, as a fraction of the box half-size",
    )
    ap.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.9, 1.0, 1.1],
        help="scale-tta: the fixed scale set (declared, not selected)",
    )
    ap.add_argument(
        "--k-grid",
        nargs="+",
        type=float,
        default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        help="relative-threshold: k in t = k * peak, searched on train",
    )
    ap.add_argument("--maps-dir", default=str(DEFAULT_MAPS))
    ap.add_argument("--tensors-dir", default=str(DEFAULT_TENSORS))
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N rows per split")
    args = ap.parse_args(argv)

    given = {
        a.lstrip("-").replace("-", "_")
        for a in (argv if argv is not None else sys.argv[1:])
        if a.startswith("--")
    }
    unused = given - RULE_ARGS[args.rule] - {"rule", "out"}
    if unused:
        raise SystemExit(
            f"--rule {args.rule} does not read "
            f"{sorted('--' + u.replace('_', '-') for u in unused)}; "
            f"it reads {sorted('--' + a.replace('_', '-') for a in RULE_ARGS[args.rule])}"
        )
    for name in REQUIRED[args.rule]:
        if getattr(args, name) is None:
            raise SystemExit(f"--rule {args.rule} needs --{name.replace('_', '-')}")
    if args.split is None:
        args.split = ["val", "train"] if args.rule == "hybrid" else ["val"]
    RULES[args.rule](args)


if __name__ == "__main__":
    main()

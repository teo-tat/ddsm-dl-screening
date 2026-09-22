"""Localiser post-processing bundle: cache maps, sweep configurations, gate a candidate.

Run from the repo root with ``PYTHONPATH=.``. The test partition is read only under
--allow-test, and nothing under ``artifacts/`` is written.

  cache Predict and store each row's 8-bit map, plain and flipped (GPU).
  recheck Recompute the cache-time single-box check from maps already on disk.
  sweep Score the configuration grid, select on train, gate on validation.
  select Re-derive the selection from an existing sweep under a recorded rule.
  candidates The selected map's top components per image, the re-ranker's data.
  metrics Rebuild the selected configuration's tables with the full metric set.
  apply Write the selected boxes for one split into a candidate boxes folder.
  gate Paired patient-cluster bootstrap CI of box IoU against a reference.
  write-torch-maps Maps for a PyTorch member from an existing checkpoint; no training.
  torch-member-boxes Re-derive a PyTorch member's box tables from its checkpoint.
  reexport-selection Export a weighted selection's real box table from the score arrays.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

# src is imported lazily inside the functions that need it: src.localise pulls
# in TensorFlow, and the sweep's worker processes must stay numpy-only.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402 (config has no TensorFlow dependency)

Box = tuple[int, int, int, int]
TENSORS = config.OUTPUTS_DIR / "localiser_tensors"
BUNDLE = config.LOC_BUNDLE_DIR
MAPS = BUNDLE / "maps"
ALLOWED_SPLITS = ("val", "train")
# Which tensor store each ensemble member must be predicted from. The sweep asserts
# it: a member recomputed in another store's normalisation corrupts the ensemble.
MEMBER_TENSORS = {
    "run1": TENSORS,
    "run3": TENSORS,
    "run4": TENSORS,
    "run4b": TENSORS,
    "run4c": TENSORS,
    "l5": config.OUTPUTS_DIR / "localiser_tensors_v2",
    # L6 predicts in cache v3 and its map is stored downsampled to the ensemble
    # frame; the frame asserted here is the one it predicted in.
    "l6": config.OUTPUTS_DIR / "localiser_tensors_v3",
    "l6b": config.OUTPUTS_DIR / "localiser_tensors_v3",
    # PyTorch members on cache v2: the encoder is the intended variable, and the
    # maps are already in the ensemble's frame so they need no downsampling.
    "l7": config.OUTPUTS_DIR / "localiser_tensors_v2",
    "l8": config.OUTPUTS_DIR / "localiser_tensors_v2",
    # PyTorch members on cache v1, the cache run 4 used, through its test-free export.
    "l7v": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "l7a": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "l8a": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "l9": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    # The two seed families, cache v1 on the pod and so resolved through
    # POD_MOUNT_ALIASES. Listed separately because an unlisted member is refused.
    "l7v_s1": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "l7v_s2": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "run4_s1": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "run4_s2": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
}

# Pod mount points and the laptop stores they were uploaded from: a member cached on
# a pod records the path it saw there. Only these known mounts are rewritten.
POD_MOUNT_ALIASES = {
    "/workspace/cache_v1_perlesion": config.OUTPUTS_DIR / "localiser_tensors_v1_notest",
    "/workspace/cache_v2_perlesion": config.OUTPUTS_DIR / "localiser_tensors_v2",
    "/workspace/cache_v2_merged": config.OUTPUTS_DIR / "localiser_tensors_merged_v2",
    "/workspace/cache_v3_perlesion": config.OUTPUTS_DIR / "localiser_tensors_v3",
    "/workspace/cache_v3_merged": config.OUTPUTS_DIR / "localiser_tensors_merged_v3",
}


def _resolve_frame(actual: str) -> tuple[Path, str | None]:
    """The laptop store a recorded tensors_dir names, with the pod mount it came from
    when an alias fired, so the substitution is printed rather than silent."""
    key = str(actual).rstrip("/")
    if key in POD_MOUNT_ALIASES:
        return Path(POD_MOUNT_ALIASES[key]), key
    return Path(actual), None


# The single-box rule every run was scored with; the sweep row that equals it
# must reproduce the frozen numbers, which is the cache's correctness check.
REFERENCE_RULE = dict(
    tta=False,
    breast_filter=False,
    rule="largest",
    threshold=config.LOC_MASK_THRESHOLD,
    min_px=config.LOC_MIN_COMPONENT_PX,
)


def _device_name() -> str:
    """The GPU this process computes on, recorded in the map manifest: a control that
    compares maps across machines needs to know what produced each one."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return "CPU"
        try:
            det = tf.config.experimental.get_device_details(gpus[0])
            return det.get("device_name") or gpus[0].name
        except Exception:
            return gpus[0].name
    except Exception:
        return "unknown"


def _set_bundle_dir(path: str | Path) -> None:
    """Point the bundle and maps folders, read and written, elsewhere (smoke runs use a
    scratch folder, never outputs/results)."""
    global BUNDLE, MAPS
    BUNDLE = Path(path)
    MAPS = BUNDLE / "maps"


def _check_splits(splits, allow_test: bool = False) -> None:
    allowed = ALLOWED_SPLITS + (("test",) if allow_test else ())
    bad = [s for s in splits if s not in allowed]
    if bad:
        raise SystemExit(
            f"refusing splits {bad}: the test partition is frozen (allowed: {ALLOWED_SPLITS})"
        )


def _iou(a: Box, b: Box) -> float:
    """Inclusive-box IoU; the same arithmetic as localise.box_iou, kept local
    so the sweep workers do not import TensorFlow."""
    ih = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    iw = max(0, min(a[3], b[3]) - max(a[2], b[2]) + 1)
    inter = ih * iw
    union = (a[1] - a[0] + 1) * (a[3] - a[2] + 1) + (b[1] - b[0] + 1) * (b[3] - b[2] + 1) - inter
    return inter / union if union else 0.0


def _gt_box(mask: np.ndarray) -> Box | None:
    """Bounding box of the full ground-truth mask (no component filtering, as
    localise_eval scores it)."""
    binary = np.asarray(mask) > 0.5
    if not binary.any():
        return None
    y = np.flatnonzero(binary.any(axis=1))
    x = np.flatnonzero(binary.any(axis=0))
    return int(y[0]), int(y[-1]), int(x[0]), int(x[-1])


# cache


def _mask_canvas(store, split: str) -> tuple[int, int] | None:
    """(h, w) of a store's masks for *split*, or None if the store is not here."""
    p = Path(store) / f"{split}_masks.npy"
    if not p.exists():
        return None
    return tuple(np.load(p, mmap_mode="r").shape[1:3])


def _gt_store_for_canvas(h: int, w: int, split: str, preferred) -> Path | None:
    """The store whose masks lie on the canvas the maps were STORED at: a box taken
    from another frame scores near zero whatever the map contains."""
    # The predicted-from store first, so a member cached in one frame is unaffected.
    seen: list[Path] = []
    for cand in [preferred, TENSORS, *dict.fromkeys(MEMBER_TENSORS.values())]:
        if cand is None:
            continue
        c = Path(cand)
        if c in seen:
            continue
        seen.append(c)
        if _mask_canvas(c, split) == (h, w):
            return c
    return None


def _single_box_check(plain, split: str, tensors_dir, *, limit: int | None = None):
    """Mean single-box IoU of the stored 8-bit maps, in the maps' own frame. A check
    that cannot be run returns NaN and a note saying why, never a number."""
    n = len(plain) if limit is None else min(limit, len(plain))
    h, w = plain.shape[1:3]
    store = _gt_store_for_canvas(h, w, split, tensors_dir)
    if store is None:
        return (float("nan"), None, f"skipped: no store here has {split} masks at {h}x{w}")
    msks = np.load(Path(store) / f"{split}_masks.npy", mmap_mode="r")
    if len(msks) < n:
        return (
            float("nan"),
            str(store),
            (f"skipped: {Path(store).name} has {len(msks)} {split} masks, the maps have {n}"),
        )
    # Row order is assumed by every consumer of these maps; assert it instead,
    # because a store swapped in for its canvas is a different file on disk.
    try:
        a = pd.read_csv(Path(store) / f"{split}_meta.csv")["lesion_id"].head(n).tolist()
        b = pd.read_csv(Path(tensors_dir) / f"{split}_meta.csv")["lesion_id"].head(n).tolist()
        if a != b:
            return (
                float("nan"),
                str(store),
                (
                    "skipped: lesion_id order differs between "
                    f"{Path(store).name} and the predicted-from store"
                ),
            )
    except FileNotFoundError:
        # The predicted-from store may be a pod path that is gone; the canvas match stands.
        print(f"  [skipped] lesion_id order check: no {split}_meta.csv in {store} or {tensors_dir}")
    ious = []
    for i in range(n):
        gt = _gt_box(msks[i])
        if gt is None:
            continue
        box = _select_box(
            plain[i].astype(np.float32) / 255.0, gt=None, breast=None, **REFERENCE_RULE
        )[0]
        ious.append(_iou(gt, box) if box is not None else 0.0)
    return (float(np.mean(ious)) if ious else float("nan")), str(store), "ok"


def recheck(args: argparse.Namespace) -> None:
    """Recompute the cache-time correctness check from maps already on disk; no
    inference, the maps are the input."""
    mpath = MAPS / f"{args.tag}_manifest.json"
    if not mpath.exists():
        raise SystemExit(f"no manifest for {args.tag}: {mpath}")
    manifest = json.loads(mpath.read_text())
    tensors_dir, alias = _resolve_frame(manifest.get("tensors_dir", str(TENSORS)))
    if alias:
        print(f"[recheck] {args.tag}: pod mount {alias} -> {tensors_dir}")
    for split in args.split:
        p = MAPS / f"{args.tag}_{split}_plain.npy"
        if not p.exists():
            print(f"[recheck] {args.tag} {split}: no maps at {p}, skipping")
            continue
        plain = np.load(p, mmap_mode="r")
        mean_iou, store, note = _single_box_check(plain, split, tensors_dir, limit=args.limit)
        was = manifest.get("splits", {}).get(split, {}).get("single_box_mean_iou_8bit")
        print(
            f"[recheck] {args.tag} {split}: single-box mean IoU from 8-bit maps "
            f"{mean_iou:.4f} (was {was if was is None else f'{was:.4f}'}; "
            f"gt from {Path(store).name if store else 'n/a'}; {note})"
        )
        entry = manifest.setdefault("splits", {}).setdefault(split, {})
        if was is not None and note == "ok":
            entry["single_box_mean_iou_8bit_superseded"] = was
            entry["superseded_reason"] = (
                "scored against the predicted-from store's masks, whose canvas differs "
                "from the stored map canvas; see _gt_store_for_canvas"
            )
        entry["single_box_mean_iou_8bit"] = None if np.isnan(mean_iou) else mean_iou
        entry["single_box_check_gt_store"] = store
        entry["single_box_check_note"] = note
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"-> {mpath}")


def cache(args: argparse.Namespace) -> None:
    import tensorflow as tf
    from src.localise_eval import _load_model, _load

    _check_splits(args.split, getattr(args, "allow_test", False))
    MAPS.mkdir(parents=True, exist_ok=True)
    # tf.config.set_visible_devices cannot be called after import; the repo's
    # convention for a CPU run is a device scope around the whole job.
    with tf.device(
        "/CPU:0" if args.cpu else "/GPU:0" if tf.config.list_logical_devices("GPU") else "/CPU:0"
    ):
        _cache_run(args, _load_model, _load)


def _cache_run(args: argparse.Namespace, _load_model, _load) -> None:
    tiled_ = bool(getattr(args, "tiled", False))
    # A tiled member is fed 512-px windows and the localiser declares a fixed input
    # shape, so it must be built at the tile size or it rejects every batch.
    model = _load_model(
        args.weights,
        args.base_filters,
        args.depth,
        args.encoder,
        input_hw=((config.LOC_PATCH_SIZE, config.LOC_PATCH_SIZE) if tiled_ else None),
    )
    mpath = MAPS / f"{args.tag}_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    tensors_dir = str(getattr(args, "tensors_dir", None) or TENSORS)
    tiled = tiled_
    overlap = int(getattr(args, "tile_overlap", 64))
    map_canvas = getattr(args, "canvas", None)
    manifest.update(
        {
            "run": args.tag,
            "weights": str(args.weights),
            "encoder": args.encoder,
            "base_filters": args.base_filters,
            "depth": args.depth,
            "tensors_dir": tensors_dir,
            "tiled_inference": tiled,
            "tile_overlap": overlap if tiled else None,
            "map_canvas": list(map_canvas) if map_canvas else None,
            "dtype": "uint8 = round(255 * p)",
            "flip": "horizontal, stored un-flipped",
            "gpu_name": _device_name(),
            "inference_path": "eager model(x, training=False)",
        }
    )
    manifest.setdefault("splits", {})
    for split in args.split:
        imgs, msks, meta, scale = _load(
            split, tensors_dir, images_only=bool(getattr(args, "images_only", False))
        )
        n = len(imgs) if args.limit is None else min(args.limit, len(imgs))
        # The stored frame is the ensemble's frame, which may be coarser than the
        # frame the member predicts in.
        h, w = tuple(map_canvas) if map_canvas else imgs.shape[1:3]
        # Written under a .partial name and renamed once complete, so a map file under
        # its final name is always a finished one.
        final = {k: MAPS / f"{args.tag}_{split}_{k}.npy" for k in ("plain", "flip")}
        partial = {k: p.with_suffix(".npy.partial") for k, p in final.items()}
        plain = np.lib.format.open_memmap(
            partial["plain"], mode="w+", dtype=np.uint8, shape=(n, h, w)
        )
        flip = np.lib.format.open_memmap(
            partial["flip"], mode="w+", dtype=np.uint8, shape=(n, h, w)
        )
        t0 = time.time()
        for start in range(0, n, args.batch_size):
            # min(..., n) so a --limit below --batch-size does not read a whole batch
            # into a memmap sized to the limit.
            block = (
                imgs[start : min(start + args.batch_size, n)].astype(np.float32)[..., None] / scale
            )
            if tiled:
                from src.localise import predict_tiled

                p = np.stack(
                    [
                        predict_tiled(
                            model, block[k, ..., 0], overlap=overlap, batch=args.batch_size
                        )
                        for k in range(len(block))
                    ]
                )
                # Flip TTA: predict on the mirrored image, then mirror the map back.
                pf = np.stack(
                    [
                        predict_tiled(
                            model, block[k, :, ::-1, 0], overlap=overlap, batch=args.batch_size
                        )
                        for k in range(len(block))
                    ]
                )[:, :, ::-1]
            else:
                # Eager, never model.predict: tensorflow-metal 1.2.0 skips ReLU in the
                # compiled graph, so predict returns pre-activation values.
                p = np.asarray(model(block, training=False))[..., 0]
                pf = np.asarray(model(block[:, :, ::-1], training=False))[..., 0][:, :, ::-1]
            if map_canvas:
                from src.localise import area_downsample

                p = np.stack([area_downsample(q, h, w) for q in p])
                pf = np.stack([area_downsample(q, h, w) for q in pf])
            plain[start : start + len(block)] = np.rint(p * 255.0).astype(np.uint8)
            flip[start : start + len(block)] = np.rint(pf * 255.0).astype(np.uint8)
            if (start // args.batch_size) % 20 == 0:
                print(
                    f"[cache] {args.tag} {split}: {start + len(block)}/{n} rows "
                    f"({time.time() - t0:.0f} s)",
                    flush=True,
                )
        plain.flush()
        flip.flush()
        for k in final:
            os.replace(partial[k], final[k])

        # The single-box rule on the 8-bit maps, scored in the frame the maps were stored
        # in, should reproduce the run's scored mean up to quantisation. It is printed and
        # recorded here; sweep enforces it for the runs with a scored table.
        if getattr(args, "images_only", False):
            # No mask array on this machine, so the check cannot be computed. Named,
            # not filled in: a 0.0 here would read as "the boxes are wrong".
            mean_iou, gt_store, note = (
                float("nan"),
                None,
                "skipped: --images-only, no mask array on this machine",
            )
        else:
            mean_iou, gt_store, note = _single_box_check(
                plain, split, tensors_dir, limit=args.limit
            )
        line = (
            f"[cache] {args.tag} {split}: single-box mean IoU from 8-bit maps "
            f"{mean_iou:.4f} (gt from "
            f"{Path(gt_store).name if gt_store else 'n/a'}; {note})"
        )
        ref = ROOT / "artifacts" / f"localiser_boxes_{split}.csv"
        if args.tag == "run1" and ref.exists() and args.limit is None:
            r = pd.read_csv(ref)
            r = r[r["status"] != "no_gt"]
            line += f"  (frozen run-1 table: {r['box_iou'].fillna(0).mean():.4f})"
        print(line, flush=True)
        manifest["splits"][split] = {
            "n": int(n),
            "single_box_mean_iou_8bit": (None if np.isnan(mean_iou) else mean_iou),
            "single_box_check_gt_store": gt_store,
            "single_box_check_note": note,
            "lesion_ids_head": meta["lesion_id"].head(3).tolist(),
        }
        for f in ("plain", "flip"):
            print(f"-> {MAPS / f'{args.tag}_{split}_{f}.npy'}")
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"-> {mpath}")


# Breast box (canvas coordinates), the same rule as localise_eval


def _breast_boxes(split: str, limit: int | None = None) -> pd.DataFrame:
    """Otsu-on-content tissue mask, opening, largest component, bounding box in CANVAS
    coordinates. Cached per split: it depends on the images only, not on the run."""
    path = BUNDLE / f"breast_boxes_{split}.csv"
    if path.exists() and limit is None:
        return pd.read_csv(path)
    from src.localise_eval import _otsu_threshold
    from src.localise import Letterbox

    imgs = np.load(TENSORS / f"{split}_images.npy", mmap_mode="r")
    meta = pd.read_csv(TENSORS / f"{split}_meta.csv")
    if limit is not None:
        meta = meta.iloc[:limit]
    scale = 255.0 if imgs.dtype == np.uint8 else 1.0
    k = config.LOC_BUNDLE_BREAST_OPENING_PX
    rows = []
    t0 = time.time()
    for i, r in meta.iterrows():
        lb = Letterbox(
            scale=float(r["scale"]),
            pad_y=int(r["pad_y"]),
            pad_x=int(r["pad_x"]),
            orig_h=int(r["orig_h"]),
            orig_w=int(r["orig_w"]),
            out_h=int(r["out_h"]),
            out_w=int(r["out_w"]),
        )
        arr = np.asarray(imgs[i], dtype=np.float32) / scale
        content = arr[lb.pad_y : lb.pad_y + lb.content_h, lb.pad_x : lb.pad_x + lb.content_w]
        tissue = ndimage.binary_opening(arr > _otsu_threshold(content), structure=np.ones((k, k)))
        rec = {"lesion_id": r["lesion_id"], "by0": -1, "by1": -1, "bx0": -1, "bx1": -1}
        if tissue.any():
            lab, n = ndimage.label(tissue)
            if n:
                lab = lab == (int(np.bincount(lab.ravel())[1:].argmax()) + 1)
                ys, xs = np.flatnonzero(lab.any(1)), np.flatnonzero(lab.any(0))
                rec.update(by0=int(ys[0]), by1=int(ys[-1]), bx0=int(xs[0]), bx1=int(xs[-1]))
        rows.append(rec)
        if i % 200 == 0:
            print(f"[breast] {split}: {i}/{len(meta)} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    if limit is None:
        BUNDLE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"-> {path}")
    return df


# The bundle rule and the configuration grid


def _components(prob: np.ndarray, threshold: float):
    """Connected components above threshold: (boxes, sizes, sums, peaks)."""
    lab, n = ndimage.label(prob > threshold)
    if n == 0:
        return [], np.zeros(0, int), np.zeros(0), np.zeros(0)
    idx = np.arange(1, n + 1)
    sizes = np.bincount(lab.ravel())[1:]
    sums = np.asarray(ndimage.sum(prob, lab, index=idx), float)
    peaks = np.asarray(ndimage.maximum(prob, lab, index=idx), float)
    boxes = [
        (int(s[0].start), int(s[0].stop) - 1, int(s[1].start), int(s[1].stop) - 1)
        for s in ndimage.find_objects(lab)
    ]
    return boxes, sizes, sums, peaks


def _apply_breast(prob: np.ndarray, breast: Box | None) -> np.ndarray:
    if breast is None or breast[0] < 0:
        return prob
    out = np.zeros_like(prob)
    out[breast[0] : breast[1] + 1, breast[2] : breast[3] + 1] = prob[
        breast[0] : breast[1] + 1, breast[2] : breast[3] + 1
    ]
    return out


def _select_box(
    prob: np.ndarray,
    *,
    gt,
    breast,
    tta: bool,
    breast_filter: bool,
    rule: str,
    threshold: float,
    min_px: int,
) -> tuple[Box | None, float]:
    """One configuration on one already-averaged map, as (box or None, IoU with gt).
    `tta` is accepted for signature symmetry; averaging happens before the call."""
    p = _apply_breast(prob, breast) if breast_filter else prob
    boxes, sizes, sums, peaks = _components(p, threshold)
    keep = np.flatnonzero(sizes >= min_px)
    if keep.size == 0:
        return None, 0.0
    score = {"largest": sizes, "sum": sums, "peak": peaks}[rule]
    best = keep[int(np.argmax(score[keep]))]
    box = boxes[best]
    return box, (_iou(gt, box) if gt is not None else 0.0)


def _ensembles(runs: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every subset of the runs, singles first, then pairs, ..., then all."""
    from itertools import combinations

    return [c for k in range(1, len(runs) + 1) for c in combinations(runs, k)]


# Weights apply to the FULL ensemble only: carrying them into every subset would
# multiply a grid that is already 420 configurations per subset.
_GROUPS = {
    "keras": ("run1", "run3", "run4"),
    "torch": ("l7v", "l7a", "l8a", "l9"),
    "seeds": ("l7v_s1", "l7v_s2", "run4_s1", "run4_s2"),
}


def _group_of(run: str) -> str:
    for g, members in _GROUPS.items():
        if run in members:
            return g
    return "other"


def _weight_vectors(
    runs: tuple[str, ...], mode: str, train_iou: dict[str, float] | None = None
) -> list[tuple[str, tuple | None]]:
    """(label, weights) for the full ensemble. `none` is the default and returns the
    single equal-weight entry, so grid and arithmetic match the unweighted sweep."""
    from itertools import product

    k = len(runs)
    if mode == "none":
        return [("equal", None)]
    if mode == "ones":  # equivalence probe: equal, via the weighted path
        return [("ones", tuple([1.0] * k))]
    out: list[tuple[str, tuple | None]] = [("equal", None)]
    if mode in ("coarse", "group"):
        grid = (0.5, 1.0, 2.0)
        if mode == "coarse" and k <= 6:  # per-member weights while k <= 6
            for combo in product(grid, repeat=k):
                if all(c == 1.0 for c in combo):
                    continue  # already present as "equal"
                s = sum(combo)
                out.append(
                    ("w:" + ",".join(f"{c:g}" for c in combo), tuple(c * k / s for c in combo))
                )
        else:  # per declared group: mode group, or coarse beyond six members
            groups = sorted({_group_of(r) for r in runs})
            for combo in product(grid, repeat=len(groups)):
                if all(c == 1.0 for c in combo):
                    continue
                gw = dict(zip(groups, combo))
                w = [gw[_group_of(r)] for r in runs]
                s = sum(w)
                out.append(
                    ("g:" + ",".join(f"{g}={gw[g]:g}" for g in groups), tuple(x * k / s for x in w))
                )
    if mode == "proportional" or train_iou is not None:
        if train_iou:
            w = [max(train_iou.get(r, 0.0), 1e-6) for r in runs]
            s = sum(w)
            out.append(("proportional", tuple(x * k / s for x in w)))
    return out


def _grid(runs: tuple[str, ...], wvecs=None) -> list[dict]:
    """Every configuration, in the fixed order the score arrays use."""
    ensembles = _ensembles(runs)
    wvecs = wvecs or [("equal", None)]
    out = []
    for ens in ensembles:
        # Weights apply to the full ensemble only; every subset keeps equal.
        vecs = wvecs if ens == runs else [("equal", None)]
        for wlabel, w in vecs:
            for tta in (False, True):
                for breast_filter in (False, True):
                    for rule in config.LOC_BUNDLE_RULES:
                        for t in config.LOC_BUNDLE_THRESHOLD_GRID:
                            for m in config.LOC_BUNDLE_MIN_PX_GRID:
                                out.append(
                                    dict(
                                        ensemble="+".join(ens),
                                        weights=wlabel,
                                        tta=tta,
                                        breast_filter=breast_filter,
                                        rule=rule,
                                        threshold=float(t),
                                        min_px=int(m),
                                    )
                                )
    return out


# Worker state (spawned processes; numpy/scipy only)
_W: dict = {}


def _init_worker(
    maps_dir: str, split: str, runs: tuple[str, ...], gts: list, breasts: list, wvecs=None
) -> None:
    maps = Path(maps_dir)
    _W["maps"] = {
        r: (
            np.load(maps / f"{r}_{split}_plain.npy", mmap_mode="r"),
            np.load(maps / f"{r}_{split}_flip.npy", mmap_mode="r"),
        )
        for r in runs
    }
    _W["runs"], _W["gts"], _W["breasts"] = runs, gts, breasts
    _W["wvecs"] = wvecs
    _W["grid"] = _grid(runs, wvecs)


def _score_row(i: int) -> tuple[np.ndarray, np.ndarray]:
    """IoU and detection flag of every configuration for row i. Components are
    labelled once per threshold and the rule x min-px choices are look-ups."""
    runs, gts, breasts, grid = _W["runs"], _W["gts"], _W["breasts"], _W["grid"]
    gt, breast = gts[i], breasts[i]
    n = len(grid)
    iou = np.full(n, np.nan, np.float32)
    det = np.zeros(n, np.int8)
    if gt is None:
        return iou, det
    plain, flip = _load_row_maps(_W["maps"], i)
    # Loop nesting MUST match _grid (ensemble, weights, tta, breast, rule, threshold,
    # min_px): the position k is the configuration's identity.
    k = 0
    wvecs = _W.get("wvecs") or [("equal", None)]
    for ens in _ensembles(runs):
        # Weights apply to the full ensemble only; every subset keeps equal.
        for _wlabel, w in (wvecs if ens == runs else [("equal", None)]):
            for tta in (False, True):
                m = _ensemble_map(plain, flip, ens, tta, w)
                for breast_filter in (False, True):
                    p = _apply_breast(m, breast) if breast_filter else m
                    comps = {t: _components(p, t) for t in config.LOC_BUNDLE_THRESHOLD_GRID}
                    ious_t = {t: np.array([_iou(gt, b) for b in comps[t][0]], float) for t in comps}
                    for rule in config.LOC_BUNDLE_RULES:
                        for t in config.LOC_BUNDLE_THRESHOLD_GRID:
                            _, sizes, sums, peaks = comps[t]
                            score = {"largest": sizes, "sum": sums, "peak": peaks}[rule]
                            for mpx in config.LOC_BUNDLE_MIN_PX_GRID:
                                keep = np.flatnonzero(sizes >= mpx)
                                if keep.size:
                                    best = keep[int(np.argmax(score[keep]))]
                                    iou[k], det[k] = ious_t[t][best], 1
                                else:
                                    iou[k] = 0.0
                                k += 1
    assert k == n, f"scored {k} configurations, grid has {n}"
    return iou, det


def _assert_export_matches_selection(
    df: pd.DataFrame, split: str, recorded: float | None, where: str, tol: float = 1e-6
) -> None:
    """The exported table must score what the selection records for it; a dropped
    weight vector is the failure this catches."""
    # Export and sweep agree to float rounding when they compute the same thing, so
    # the tolerance sits far above that noise and far below any real mismatch.
    if recorded is None:
        print(
            f"  [self-check] {split}: no recorded value to compare against "
            f"(the selection records train and val only); this split is unchecked, not passed"
        )
        return
    got = float(df[df["status"] != "no_gt"]["box_iou"].fillna(0).mean())
    d = abs(got - float(recorded))
    if d <= tol:
        print(
            f"  [self-check] {split}: exported {got:.10f} == selected {float(recorded):.10f} "
            f"(|diff| {d:.2e})"
        )
        return
    raise SystemExit(
        f"\nEXPORT DOES NOT MATCH THE SELECTION ({where}, {split}).\n"
        f"  exported table mean box IoU : {got:.10f}\n"
        f"  selection records           : {float(recorded):.10f}\n"
        f"  |difference|                : {d:.3e} (tolerance {tol:g})\n"
        f"  The table written is not the configuration that was selected; a dropped\n"
        f"  weight vector does exactly this. Do not use the table."
    )


def _weights_for_label(runs: tuple[str, ...], label) -> tuple[float, ...] | None:
    """The weight vector a grid row's `weights` label names, or None for equal, found
    by searching `_weight_vectors` so that a label has exactly one definition."""
    if label is None or (isinstance(label, float) and np.isnan(label)) or label in ("", "equal"):
        return None
    for mode in ("group", "coarse", "ones"):
        for lab, w in _weight_vectors(runs, mode):
            if lab == label:
                return w
    raise SystemExit(
        f"weights label {label!r} is not produced by _weight_vectors for {runs}; "
        f"the grid and the label table disagree - stop"
    )


def _ensemble_map(
    plain: dict,
    flip: dict,
    ens: tuple[str, ...],
    tta: bool,
    weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Mean over the ensemble's runs of the plain map, or of the plain/flip average
    when TTA is on. Shared by the sweep and the box rebuild so they cannot drift."""
    maps = [(plain[r] + flip[r]) / 2.0 if tta else plain[r] for r in ens]
    if weights is None:
        return np.mean(maps, axis=0)
    # Scaled to mean 1 and passed through the same np.mean in the same dtype: equal
    # weights stay bit-identical, and np.average would promote float32 to float64.
    w = np.asarray(weights, dtype=np.float32)
    w = w * (np.float32(len(maps)) / w.sum())
    return np.mean([wi * mi for wi, mi in zip(w, maps)], axis=0)


def _load_row_maps(maps: dict, i: int) -> tuple[dict, dict]:
    plain = {r: maps[r][0][i].astype(np.float32) / 255.0 for r in maps}
    flip = {r: maps[r][1][i].astype(np.float32) / 255.0 for r in maps}
    return plain, flip


def _load_split_side_tables(split: str, limit: int | None = None):
    meta = pd.read_csv(TENSORS / f"{split}_meta.csv")
    if limit is not None:
        meta = meta.iloc[:limit]
    msks = np.load(TENSORS / f"{split}_masks.npy", mmap_mode="r")
    gts = [_gt_box(msks[i]) for i in range(len(meta))]
    bb = _breast_boxes(split, limit)
    assert (bb["lesion_id"].values == meta["lesion_id"].values).all(), "breast table misaligned"
    breasts = [
        (int(r.by0), int(r.by1), int(r.bx0), int(r.bx1)) if r.by0 >= 0 else None
        for r in bb.itertuples()
    ]
    return meta, gts, breasts


def _score_split(split: str, runs: tuple[str, ...], workers: int, limit: int | None, wvecs=None):
    meta, gts, breasts = _load_split_side_tables(split, limit)
    n_map = np.load(MAPS / f"{runs[0]}_{split}_plain.npy", mmap_mode="r").shape[0]
    n = len(meta)
    if n_map < n:
        raise SystemExit(f"cache for {split} holds {n_map} rows, store has {n}: re-run cache")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    res = []
    with ctx.Pool(
        workers, initializer=_init_worker, initargs=(str(MAPS), split, runs, gts, breasts, wvecs)
    ) as pool:
        for j, r in enumerate(pool.imap(_score_row, range(n), chunksize=4), 1):
            res.append(r)
            if j % 200 == 0 or j == n:
                print(f"[sweep] {split}: {j}/{n} rows ({time.time() - t0:.0f} s)", flush=True)
    iou = np.stack([r[0] for r in res])
    det = np.stack([r[1] for r in res])
    print(
        f"[sweep] {split}: {n} rows x {iou.shape[1]} configurations in "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    _self_check(split, runs, meta, gts, breasts, iou, wvecs=wvecs)
    return meta.reset_index(drop=True), gts, breasts, iou, det


def _self_check(
    split: str,
    runs: tuple[str, ...],
    meta: pd.DataFrame,
    gts: list,
    breasts: list,
    iou: np.ndarray,
    n_rows: int = 5,
    n_cfgs: int = 12,
    wvecs=None,
) -> None:
    """Rebuilding a configuration row by row must reproduce the sweep's column, so a
    grid position always means the configuration it is labelled with."""
    grid = _grid(runs, wvecs)
    maps = {
        r: (
            np.load(MAPS / f"{r}_{split}_plain.npy", mmap_mode="r"),
            np.load(MAPS / f"{r}_{split}_flip.npy", mmap_mode="r"),
        )
        for r in runs
    }
    rng = np.random.RandomState(config.SEED)
    rows = rng.choice(len(meta), size=min(n_rows, len(meta)), replace=False)
    cfgs = rng.choice(len(grid), size=min(n_cfgs, len(grid)), replace=False)
    for i in rows:
        plain, flip = _load_row_maps(maps, int(i))
        for c in cfgs:
            cfg = grid[c]
            m = _ensemble_map(plain, flip, tuple(cfg["ensemble"].split("+")), cfg["tta"])
            _, v = _select_box(
                m,
                gt=gts[i],
                breast=breasts[i],
                tta=cfg["tta"],
                breast_filter=cfg["breast_filter"],
                rule=cfg["rule"],
                threshold=cfg["threshold"],
                min_px=cfg["min_px"],
            )
            if gts[i] is not None and abs(v - float(iou[i, c])) > 1e-6:
                raise SystemExit(
                    f"self-check failed on {split} row {i} cfg {c} {cfg}: "
                    f"rebuilt {v:.4f} vs sweep {iou[i, c]:.4f}"
                )
    print(f"[sweep] {split}: self-check passed ({len(rows)} rows x {len(cfgs)} configurations)")


def _bundle_boxes(
    split: str,
    runs: tuple[str, ...],
    cfg: dict,
    meta: pd.DataFrame,
    gts: list,
    breasts: list,
    weights: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """The selected configuration's boxes and mask metrics for one split, in the
    localise_eval schema, so the bundle sits in the same table as the single runs."""
    from src.localise import Letterbox, box_centroid_hit, topk_components
    from src.localise_eval import (
        _add_any_lesion_iou,
        _add_image_dice,
        _add_matched_topk,
        _CAND_FIELDS,
    )

    ens = tuple(cfg["ensemble"].split("+"))
    maps = {
        r: (
            np.load(MAPS / f"{r}_{split}_plain.npy", mmap_mode="r"),
            np.load(MAPS / f"{r}_{split}_flip.npy", mmap_mode="r"),
        )
        for r in ens
    }
    msks = np.load(TENSORS / f"{split}_masks.npy", mmap_mode="r")
    rows, packed_pred = [], {}
    gt_canvas, cands_canvas = {}, {}  # for the matched top-k candidate table
    for i, r in meta.iterrows():
        lb = Letterbox(
            scale=float(r["scale"]),
            pad_y=int(r["pad_y"]),
            pad_x=int(r["pad_x"]),
            orig_h=int(r["orig_h"]),
            orig_w=int(r["orig_w"]),
            out_h=int(r["out_h"]),
            out_w=int(r["out_w"]),
        )
        plain, flip = _load_row_maps(maps, i)
        # The weights MUST reach the same _ensemble_map call the sweep used, or
        # the exported table is a different configuration from the selected one.
        m = _ensemble_map(plain, flip, ens, cfg["tta"], weights)
        box, iou = _select_box(
            m,
            gt=gts[i],
            breast=breasts[i],
            tta=cfg["tta"],
            breast_filter=cfg["breast_filter"],
            rule=cfg["rule"],
            threshold=cfg["threshold"],
            min_px=cfg["min_px"],
        )
        rec = {
            "split": split,
            "lesion_id": r["lesion_id"],
            "image_id": r["image_id"],
            "patient_id": r["patient_id"],
            "label": int(r["label"]),
            "orig_h": int(r["orig_h"]),
            "orig_w": int(r["orig_w"]),
            "detected": box is not None,
        }
        # Breast box in mammogram coordinates, as localise_eval records it.
        if breasts[i] is not None:
            b = lb.inverse_box(breasts[i])
            rec.update(breast_y0=b[0], breast_y1=b[1], breast_x0=b[2], breast_x1=b[3])
        # Mask-level agreement at the bundle's own threshold, on the map the
        # box was derived from (breast filter applied when the rule uses it).
        pmap = _apply_breast(m, breasts[i]) if cfg["breast_filter"] else m
        # Top-k candidate components of the SAME map, ranked by summed probability,
        # in canvas and mammogram coordinates for BoxProvider(mode="matched"|"all").
        cands = topk_components(
            pmap, config.LOC_TOPK, threshold=cfg["threshold"], min_px=cfg["min_px"]
        )
        cands_canvas[i] = [c[0] for c in cands]
        for j in range(config.LOC_TOPK):
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
        if gts[i] is not None:
            gt_canvas[i] = gts[i]
        pm = pmap > cfg["threshold"]
        gm = np.asarray(msks[i]) > 0.5
        inter = float((pm & gm).sum())
        rec["mask_iou"] = inter / max(1.0, float((pm | gm).sum()))
        rec["dice"] = 2 * inter / max(1.0, float(pm.sum() + gm.sum()))
        rec["pred_fg_frac"] = float(pm.mean())
        image_key = str(r["lesion_id"]).rsplit("_", 1)[0]
        if image_key not in packed_pred:
            packed_pred[image_key] = (np.packbits(pm), pm.shape)
        gt = gts[i]
        if gt is None:
            rec["status"] = "no_gt"
            rows.append(rec)
            continue
        g = lb.inverse_box(gt)
        rec.update(gt_y0=g[0], gt_y1=g[1], gt_x0=g[2], gt_x1=g[3])
        if box is None:
            rec.update(status="miss", box_iou=0.0, box_iou_mammogram=0.0, centroid_hit=False)
        else:
            p = lb.inverse_box(box)
            rec.update(
                status="ok",
                pred_y0=p[0],
                pred_y1=p[1],
                pred_x0=p[2],
                pred_x1=p[3],
                box_iou=iou,
                box_iou_mammogram=_iou(g, p),
                centroid_hit=box_centroid_hit(box, gt),
            )
            # Error tuple for the jitter rung, as localise_eval records it.
            gh, gw = g[1] - g[0] + 1, g[3] - g[2] + 1
            ph, pw = p[1] - p[0] + 1, p[3] - p[2] + 1
            rec.update(
                dy_frac=(((p[0] + p[1]) - (g[0] + g[1])) / 2) / gh,
                dx_frac=(((p[2] + p[3]) - (g[2] + g[3])) / 2) / gw,
                scale_ratio_h=ph / gh,
                scale_ratio_w=pw / gw,
            )
        rows.append(rec)
    df = _add_any_lesion_iou(pd.DataFrame(rows))
    df = _add_matched_topk(df, gt_canvas, cands_canvas, meta)
    df = _add_image_dice(df, msks, packed_pred)
    for k, v in cfg.items():
        df[f"bundle_{k}"] = v
    return df


def candidates(args: argparse.Namespace) -> None:
    """The top --k surviving components of the selected ensemble map, one row per candidate
    per IMAGE, with features and a label: the re-ranker's training data."""
    from src.localise import Letterbox

    _check_splits([args.split], False)
    sel = json.loads((BUNDLE / "bundle_selection.json").read_text())
    cfg = sel["selected"]
    ens = tuple(cfg["ensemble"].split("+"))
    _assert_member_frames(ens)
    for r in ens:
        for f in ("plain", "flip"):
            if not (MAPS / f"{r}_{args.split}_{f}.npy").exists():
                raise SystemExit(f"missing cache {MAPS / f'{r}_{args.split}_{f}.npy'}")

    meta, gts, breasts = _load_split_side_tables(args.split, args.limit)
    maps = {
        r: (
            np.load(MAPS / f"{r}_{args.split}_plain.npy", mmap_mode="r"),
            np.load(MAPS / f"{r}_{args.split}_flip.npy", mmap_mode="r"),
        )
        for r in ens
    }

    # One map per image; keep the first row of each image and the GT of all of them.
    first_row: dict[str, int] = {}
    gt_by_image: dict[str, list] = {}
    for i, r in enumerate(meta.itertuples()):
        img = str(r.image_id)
        first_row.setdefault(img, i)
        if gts[i] is not None:
            gt_by_image.setdefault(img, []).append(gts[i])

    rows = []
    t0 = time.time()
    for n_done, (img, i) in enumerate(first_row.items(), 1):
        r = meta.iloc[i]
        lb = Letterbox(
            scale=float(r["scale"]),
            pad_y=int(r["pad_y"]),
            pad_x=int(r["pad_x"]),
            orig_h=int(r["orig_h"]),
            orig_w=int(r["orig_w"]),
            out_h=int(r["out_h"]),
            out_w=int(r["out_w"]),
        )
        plain, flip = _load_row_maps(maps, int(i))
        m = _ensemble_map(plain, flip, ens, cfg["tta"])
        pmap = _apply_breast(m, breasts[i]) if cfg["breast_filter"] else m
        boxes, sizes, sums, peaks = _components(pmap, cfg["threshold"])
        keep = [j for j in range(len(boxes)) if sizes[j] >= cfg["min_px"]]
        # Rank by the selected configuration's own rule, so candidate 1 is the box an
        # equal-weight selection deploys (this map is unweighted); the gain over it is
        # read off the rank.
        score = {"largest": sizes, "sum": sums, "peak": peaks}[cfg["rule"]]
        keep.sort(key=lambda j: float(score[j]), reverse=True)
        keep = keep[: args.k]

        gt_list = gt_by_image.get(img, [])
        breast = breasts[i]
        # Per-candidate member support: how well each member's OWN selected box agrees
        # with this component, since a component can survive on one member alone.
        member_boxes = []
        for r_ in ens:
            mm = _ensemble_map(plain, flip, (r_,), cfg["tta"])
            mb, _ = _select_box(
                mm,
                gt=None,
                breast=breast,
                tta=cfg["tta"],
                breast_filter=cfg["breast_filter"],
                rule=cfg["rule"],
                threshold=cfg["threshold"],
                min_px=cfg["min_px"],
            )
            member_boxes.append(mb)
        for rank, j in enumerate(keep, 1):
            cb = boxes[j]
            mb = lb.inverse_box(cb)
            area = float(sizes[j])
            cy, cx = (cb[0] + cb[1]) / 2.0, (cb[2] + cb[3]) / 2.0
            if breast is not None:
                # Distance from the candidate's centre to the nearest breast-box edge,
                # negative outside: scanner border and tape artefacts sit at the edge.
                d = min(cy - breast[0], breast[1] - cy, cx - breast[2], breast[3] - cx)
                bh, bw = breast[1] - breast[0] + 1, breast[3] - breast[2] + 1
                d_norm = float(d) / max(1.0, min(bh, bw) / 2.0)
            else:
                d, d_norm = np.nan, np.nan
            # A candidate is positive against ANY lesion on the image: the re-ranker's
            # job is to find a lesion, not a particular one.
            ious = [_iou(g, cb) for g in gt_list]
            best_iou = max(ious) if ious else 0.0
            # A member that produced no box supports nothing and scores 0 rather than
            # being dropped, so support and disagreement stay on one scale.
            supports = [(_iou(mb, cb) if mb is not None else 0.0) for mb in member_boxes]
            rows.append(
                {
                    "split": args.split,
                    "image_id": img,
                    "patient_id": r["patient_id"],
                    "cand_rank": rank,
                    "y0": mb[0],
                    "y1": mb[1],
                    "x0": mb[2],
                    "x1": mb[3],
                    "canvas_y0": cb[0],
                    "canvas_y1": cb[1],
                    "canvas_x0": cb[2],
                    "canvas_x1": cb[3],
                    "peak_prob": float(peaks[j]),
                    "mean_prob": float(sums[j] / max(area, 1.0)),
                    "sum_prob": float(sums[j]),
                    "area_px": area,
                    "dist_to_breast_edge": float(d),
                    "dist_to_breast_edge_norm": d_norm,
                    "member_support_mean": float(np.mean(supports)),
                    "member_disagreement": float(1.0 - np.mean(supports)),
                    "n_members_supporting": int(sum(x >= 0.3 for x in supports)),
                    "n_members": len(member_boxes),
                    "orig_h": int(r["orig_h"]),
                    "orig_w": int(r["orig_w"]),
                    "n_gt_on_image": len(gt_list),
                    "best_iou_any_gt": float(best_iou),
                    "is_rule_pick": int(rank == 1),
                    "label": int(best_iou >= args.iou_positive),
                }
            )
        if n_done % 200 == 0 or n_done == len(first_row):
            print(
                f"[candidates] {args.split}: {n_done}/{len(first_row)} images "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else BUNDLE / f"candidates_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"-> {out}")

    n_img = len(first_row)
    pos = int(df["label"].sum())
    print(
        f"[candidates] {args.split}: {len(df)} candidates over {n_img} images "
        f"({len(df) / max(n_img, 1):.2f} per image, k={args.k}, "
        f"rule={cfg['rule']} threshold={cfg['threshold']} min_px={cfg['min_px']})"
    )
    print(
        f"[candidates] class balance: {pos} positive / {len(df) - pos} negative "
        f"({pos / max(len(df), 1):.1%} positive at IoU >= {args.iou_positive})"
    )
    with_pos = int(df.groupby("image_id")["label"].max().sum())
    print(
        f"[candidates] images with at least one positive candidate: "
        f"{with_pos}/{n_img} ({with_pos / max(n_img, 1):.1%}) "
        f"— this is the re-ranker's ceiling on this split"
    )
    # Both rates are over ALL images, including those with no surviving candidate:
    # scoring only the images that produced one would flatter the rule.
    top1 = df[df["cand_rank"] == 1]
    n_top1_pos = int(top1["label"].sum())
    n_no_cand = n_img - len(top1)
    print(
        f"[candidates] the current rule's own pick is positive on "
        f"{n_top1_pos}/{n_img} images ({n_top1_pos / max(n_img, 1):.1%}) "
        f"— the baseline to beat "
        f"({n_no_cand} images produced no candidate)"
    )
    print(
        f"[candidates] rescuable: {with_pos - n_top1_pos}/{n_img} images "
        f"({(with_pos - n_top1_pos) / max(n_img, 1):.1%}) hold a positive candidate the "
        f"rule did not pick — the most a re-ranker could gain"
    )


def select(args: argparse.Namespace) -> None:
    """Re-derive the selection from the EXISTING sweep under a recorded rule, without
    rescoring, then rebuild the tables with the full metric set and gate on val."""
    from src.localise_eval import report

    grid = pd.read_csv(BUNDLE / "sweep_configs.csv")
    full = max(grid["ensemble"].unique(), key=lambda e: e.count("+"))
    runs = tuple(full.split("+"))
    sub = grid[grid["ensemble"] == full] if args.ensemble == "all" else grid
    best = int(sub["train_mean_iou"].idxmax())
    cfg = {
        k: grid.loc[best, k]
        for k in ("ensemble", "tta", "breast_filter", "rule", "threshold", "min_px")
    }
    cfg = {
        k: (
            bool(v)
            if isinstance(v, (bool, np.bool_))
            else int(v) if k == "min_px" else float(v) if k == "threshold" else str(v)
        )
        for k, v in cfg.items()
    }
    # The weights column travels with the selection, so the rebuilt table is the
    # weighting that was selected rather than an equal-weight one.
    wlabel = grid.loc[best, "weights"] if "weights" in grid.columns else "equal"
    wvec = _weights_for_label(runs, wlabel)
    cfg["weights"] = "equal" if wvec is None else str(wlabel)
    prev_path = BUNDLE / "bundle_selection.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else None
    grid["selected"] = grid["cfg_id"] == best
    grid.to_csv(BUNDLE / "sweep_configs.csv", index=False)
    rule = (
        "ensemble set fixed a priori to all swept runs; threshold, min-px, rule and TTA "
        "selected on train"
        if args.ensemble == "all"
        else "free selection over runs and post-processing on train"
    )
    print(
        f"[select] rule: {rule}\n[select] cfg {best} {cfg}: "
        f"train {grid.loc[best, 'train_mean_iou']:.4f} "
        f"val {grid.loc[best, 'val_mean_iou']:.4f} (val rank "
        f"{int((grid['val_mean_iou'] > grid.loc[best, 'val_mean_iou']).sum()) + 1}/{len(grid)})"
    )
    results = {
        "selected_cfg_id": best,
        "selected": cfg,
        "selection_rule": rule,
        "runs": list(runs),
        "train": {
            "mean_iou": float(grid.loc[best, "train_mean_iou"]),
            "detection": float(grid.loc[best, "train_detection"]),
        },
        "val": {
            "mean_iou": float(grid.loc[best, "val_mean_iou"]),
            "detection": float(grid.loc[best, "val_detection"]),
        },
    }
    if prev is not None and prev.get("selected") != cfg:
        results["superseded_selection"] = {
            k: prev.get(k)
            for k in ("selected_cfg_id", "selected", "selection_rule", "val", "gate_bundle_vs_run1")
            if k in prev
        }
    summaries, tables = [], {}
    for split in ("val", "train"):
        meta, gts, breasts = _load_split_side_tables(split, args.limit)
        df = _bundle_boxes(split, runs, cfg, meta, gts, breasts, weights=wvec)
        df.to_csv(BUNDLE / f"bundle_boxes_{split}.csv", index=False)
        print(f"-> {BUNDLE / f'bundle_boxes_{split}.csv'}")
        s = report(df, split)
        s["bundle"] = cfg
        summaries.append(s)
        tables[split] = df
    (BUNDLE / "bundle_metrics.json").write_text(json.dumps(summaries, indent=2))
    print(f"-> {BUNDLE / 'bundle_metrics.json'}")
    reference = pd.read_csv(args.reference)
    if args.limit is not None:
        reference = reference[reference["lesion_id"].isin(tables["val"]["lesion_id"])]
    print("\n[gate] selected bundle vs run 1, validation:")
    results["gate_bundle_vs_run1"] = _gate_tables(tables["val"], reference, "bundle", "run1")
    prev_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"-> {prev_path}")


def metrics(args: argparse.Namespace) -> None:
    """Rebuild the selected configuration's tables with the full metric set
    and write bundle_metrics.json in the localise_eval summary format."""
    from src.localise_eval import report

    sel = json.loads((BUNDLE / "bundle_selection.json").read_text())
    cfg = sel["selected"]
    runs = tuple(cfg["ensemble"].split("+"))
    wvec = _weights_for_label(runs, cfg.get("weights", "equal"))
    summaries = []
    for split in ("val", "train"):
        meta, gts, breasts = _load_split_side_tables(split, args.limit)
        df = _bundle_boxes(split, runs, cfg, meta, gts, breasts, weights=wvec)
        df.to_csv(BUNDLE / f"bundle_boxes_{split}.csv", index=False)
        print(f"-> {BUNDLE / f'bundle_boxes_{split}.csv'}")
        s = report(df, split)
        s["bundle"] = cfg
        summaries.append(s)
    out = BUNDLE / "bundle_metrics.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"-> {out}")


def _summ(iou: np.ndarray, det: np.ndarray) -> dict:
    ok = ~np.isnan(iou)
    v = np.where(ok, iou, 0.0)
    return {
        "mean_iou": float(v[ok].mean()),
        "detection": float(det[ok].mean()),
        "detect_at_iou50": float((v[ok] >= 0.5).mean()),
        "n": int(ok.sum()),
    }


def _assert_member_frames(runs: tuple[str, ...]) -> None:
    """Every member's cached maps must come from the store it was trained on: the maps
    share one canvas, but the stores differ in intensity normalisation."""
    print("[sweep] member normalisation frames:")
    bad = []
    for r in runs:
        # A manifest with no tensors_dir is v1.
        mpath = MAPS / f"{r}_manifest.json"
        actual = (
            json.loads(mpath.read_text()).get("tensors_dir", str(TENSORS))
            if mpath.exists()
            else str(TENSORS)
        )
        expected = MEMBER_TENSORS.get(r)
        # A member cached on a pod records that machine's mount point; translate
        # the known ones back to the store they were uploaded from, and say so.
        resolved, alias = _resolve_frame(actual)
        # Resolve both: a relative --tensors-dir names the same store as the
        # absolute default and must not read as a different frame.
        same = expected is not None and resolved.resolve() == Path(expected).resolve()
        mark = "unlisted" if expected is None else ("ok" if same else "WRONG")
        via = f" (pod mount {alias} -> {resolved})" if alias else ""
        print(f"    {r}: {actual}{via} [{mark}]")
        if mark == "WRONG":
            bad.append(f"{r}: cached from {actual}, expected {expected}")
        elif mark == "unlisted":
            bad.append(
                f"{r}: not in MEMBER_TENSORS, so its frame cannot be checked; "
                f"add an entry naming the store it was predicted from"
            )
    if bad:
        raise SystemExit(
            "[sweep] refusing to sweep: a member's normalisation frame is "
            "wrong or unverifiable.\n  "
            + "\n  ".join(bad)
            + "\n  Re-cache the member with the right --tensors-dir, or add its "
            "entry to MEMBER_TENSORS (which records which frame each member "
            "belongs to). An unlisted member is refused rather than swept "
            "unchecked."
        )


def sweep(args: argparse.Namespace) -> None:
    runs = tuple(args.tag)
    wvecs = _weight_vectors(runs, getattr(args, "weighting", "none"))
    if len(wvecs) > 1:
        print(
            f"[sweep] weighting: {len(wvecs)} vectors "
            f"({getattr(args, 'weighting')}) on the full {len(runs)}-member ensemble",
            flush=True,
        )
    for r in runs:
        for split in ALLOWED_SPLITS:
            for f in ("plain", "flip"):
                if not (MAPS / f"{r}_{split}_{f}.npy").exists():
                    raise SystemExit(
                        f"missing cache {MAPS / f'{r}_{split}_{f}.npy'}: run cache first"
                    )
    _assert_member_frames(runs)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(_grid(runs, wvecs))
    grid.insert(0, "cfg_id", np.arange(len(grid)))

    scored = {}
    for split in ("train", "val"):
        meta, gts, breasts, iou, det = _score_split(split, runs, args.workers, args.limit, wvecs)
        np.save(BUNDLE / f"sweep_iou_{split}.npy", iou)
        print(f"-> {BUNDLE / f'sweep_iou_{split}.npy'}")
        scored[split] = (meta, gts, breasts, iou, det)
        for k in ("mean_iou", "detection", "detect_at_iou50"):
            grid[f"{split}_{k}"] = [(_summ(iou[:, j], det[:, j])[k]) for j in range(len(grid))]

    # Reference rows: the single-box rule of each run alone must reproduce that run's
    # scored table (runs 1, 3, 4 and 4b) up to 8-bit quantisation.
    ref_rows = {}
    for r in runs:
        sel = grid[
            (grid["ensemble"] == r)
            & (grid["weights"] == "equal")
            & (~grid["tta"])
            & (~grid["breast_filter"])
            & (grid["rule"] == REFERENCE_RULE["rule"])
            & np.isclose(grid["threshold"], REFERENCE_RULE["threshold"])
            & (grid["min_px"] == REFERENCE_RULE["min_px"])
        ]
        ref_rows[r] = int(sel["cfg_id"].iloc[0])
        line = (
            f"[sweep] reference rule on {r}: train {sel['train_mean_iou'].iloc[0]:.4f} "
            f"val {sel['val_mean_iou'].iloc[0]:.4f} (cfg {ref_rows[r]})"
        )
        # Hard check against each run's scored table, within 0.002: 8-bit quantisation
        # may move the 4th decimal.
        table = {
            "run1": ROOT / "artifacts",
            "run3": ROOT / "outputs" / "results" / "localiser_run3_eval",
            "run4": ROOT / "outputs" / "results" / "localiser_run4_eval",
            "run4b": ROOT / "outputs" / "results" / "localiser_run4b_eval",
        }.get(r)
        if table is not None and args.limit is None:
            for split in ("train", "val"):
                p = table / f"localiser_boxes_{split}.csv"
                if not p.exists():
                    print(f"[sweep] reference check skipped for {r} {split}: no table at {p}")
                    continue
                t = pd.read_csv(p)
                t = t[t["status"] != "no_gt"]
                expect = float(t["box_iou"].fillna(0).mean())
                got = float(sel[f"{split}_mean_iou"].iloc[0])
                line += f"  [{split} table {expect:.4f}]"
                if abs(got - expect) > 0.002:
                    raise SystemExit(
                        f"{line}\nreference rule on {r} does not reproduce the "
                        f"{split} table ({got:.4f} vs {expect:.4f}); the cache or "
                        f"the grid labelling is wrong - stop"
                    )
        elif table is not None:
            print(f"[sweep] reference check skipped for {r}: --limit scores a subset")
        print(line)

    # SELECTION ON TRAIN. Ties (identical train means) go to the first row in
    # grid order, which is the plainer configuration.
    best = int(grid["train_mean_iou"].idxmax())
    cfg = {
        k: grid.loc[best, k]
        for k in ("ensemble", "tta", "breast_filter", "rule", "threshold", "min_px")
    }
    cfg = {
        k: (
            bool(v)
            if isinstance(v, (bool, np.bool_))
            else int(v) if k == "min_px" else float(v) if k == "threshold" else str(v)
        )
        for k, v in cfg.items()
    }
    cfg["weights"] = str(grid.loc[best, "weights"]) if "weights" in grid.columns else "equal"
    grid["selected"] = grid["cfg_id"] == best
    grid.to_csv(BUNDLE / "sweep_configs.csv", index=False)
    print(f"-> {BUNDLE / 'sweep_configs.csv'}")

    print("\n[sweep] top 10 configurations by TRAIN mean IoU (val shown for information only):")
    cols = [
        "cfg_id",
        "ensemble",
        "tta",
        "breast_filter",
        "rule",
        "threshold",
        "min_px",
        "train_mean_iou",
        "train_detection",
        "val_mean_iou",
        "val_detection",
    ]
    print(grid.sort_values("train_mean_iou", ascending=False).head(10)[cols].to_string(index=False))
    print(f"\n[sweep] SELECTED on train: cfg {best} {cfg}")
    print(
        f"        train {grid.loc[best, 'train_mean_iou']:.4f} "
        f"(det {grid.loc[best, 'train_detection']:.3f})"
        f"  val {grid.loc[best, 'val_mean_iou']:.4f} (det {grid.loc[best, 'val_detection']:.3f})"
    )

    # Selected boxes for both splits. The weight vector reaches the rebuild here too,
    # so the exported table is the weighting that was selected.
    sweep_wvec = _weights_for_label(runs, cfg.get("weights", "equal"))
    tables = {}
    for split in ("val", "train"):
        meta, gts, breasts, _, _ = scored[split]
        df = _bundle_boxes(split, runs, cfg, meta, gts, breasts, weights=sweep_wvec)
        _assert_export_matches_selection(
            df,
            split,
            (None if args.limit is not None else grid.loc[best, f"{split}_mean_iou"]),
            "sweep",
        )
        df.to_csv(BUNDLE / f"bundle_boxes_{split}.csv", index=False)
        print(f"-> {BUNDLE / f'bundle_boxes_{split}.csv'}")
        tables[split] = df

    # GATE ON VALIDATION against the frozen run-1 table, plus run 3 vs run 1.
    reference = pd.read_csv(args.reference)
    if args.limit is not None:  # smoke only: the gate needs identical lesion sets
        reference = reference[reference["lesion_id"].isin(tables["val"]["lesion_id"])]
    results = {
        "selected_cfg_id": best,
        "selected": cfg,
        "train": _summ(scored["train"][3][:, best], scored["train"][4][:, best]),
        "val": _summ(scored["val"][3][:, best], scored["val"][4][:, best]),
        "reference_cfg_ids": ref_rows,
        "reference_rule_from_cache": {
            r: {
                "train": float(grid.loc[ref_rows[r], "train_mean_iou"]),
                "val": float(grid.loc[ref_rows[r], "val_mean_iou"]),
            }
            for r in runs
        },
    }
    print("\n[gate] bundle (selected on train) vs run 1, validation:")
    results["gate_bundle_vs_run1"] = _gate_tables(tables["val"], reference, "bundle", "run1")
    run3 = ROOT / "outputs" / "results" / "localiser_run3_eval" / "localiser_boxes_val.csv"
    if run3.exists():
        print("\n[gate] run 3 (single box, localise_eval table) vs run 1, validation:")
        run3_table = pd.read_csv(run3)
        if args.limit is not None:
            run3_table = run3_table[run3_table["lesion_id"].isin(reference["lesion_id"])]
        results["gate_run3_vs_run1"] = _gate_tables(run3_table, reference, "run3", "run1")
    else:
        print(f"\n[gate] run 3 vs run 1 skipped: no table at {run3}")
    out = BUNDLE / "bundle_selection.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"-> {out}")


# apply — the selected bundle's boxes for one split, as a candidate boxes folder

# The keys `apply`'s auto-cache Namespace reads out of a member's cache manifest.
_CACHE_REQUIRED = ("weights", "encoder", "base_filters", "depth")

# Frame keys the auto-cache Namespace does not carry: a member declaring one would
# be re-predicted in a different frame from its other splits.
_CACHE_UNSUPPORTED_FRAME = ("tiled_inference", "tile_overlap", "map_canvas")


def _assert_auto_cacheable(runs: tuple[str, ...], split: str) -> list[str]:
    """Members of `runs` whose `split` maps are missing, refusing before the first
    cache if any of them cannot be auto-cached, so a refusal leaves nothing on disk."""
    # Every deficient member is named at once: one trip per member would write maps
    # for the members that came before it.
    need, bad = [], []
    for r in runs:
        if (MAPS / f"{r}_{split}_plain.npy").exists() and (MAPS / f"{r}_{split}_flip.npy").exists():
            continue
        need.append(r)
        mpath = MAPS / f"{r}_manifest.json"
        if not mpath.exists():
            bad.append(f"  {r}: no cache manifest at {mpath}")
            continue
        try:
            man = json.loads(mpath.read_text())
        except json.JSONDecodeError as e:
            bad.append(f"  {r}: cache manifest is not readable JSON ({e})")
            continue
        # Framework first and independently of the keys: the auto-cache builds Keras
        # through localise_eval._load_model and cannot load a torch member.
        fw = str(man.get("framework", "keras")).lower()
        if fw not in ("keras", "tensorflow", "tf"):
            bad.append(
                f"  {r}: framework is {man.get('framework')!r}; apply's auto-cache "
                f"builds Keras (localise_eval._load_model) and cannot load it"
            )
            continue
        missing = [k for k in _CACHE_REQUIRED if man.get(k) is None]
        # Truthy, not present: write_maps records tiled_inference False and
        # tile_overlap None for a whole-image member, which declares nothing.
        frame = [k for k in _CACHE_UNSUPPORTED_FRAME if man.get(k)]
        if missing:
            bad.append(
                f"  {r}: manifest lacks {', '.join(missing)}"
                + (f" (framework: {man['framework']})" if "framework" in man else "")
            )
        if frame:
            bad.append(
                f"  {r}: declares {', '.join(f'{k}={man[k]!r}' for k in frame)}, "
                f"which apply's auto-cache does not pass through"
            )
    if bad:
        raise SystemExit(
            f"[apply] REFUSED before caching anything: "
            f"{len(set(l.split(':')[0].strip() for l in bad))} "
            f"of the {len(need)} member(s) needing {split} maps cannot be auto-cached.\n"
            + "\n".join(bad)
            + f"\n\nNothing was written. Produce the missing {split} maps with the tool that can "
            f"(a PyTorch member: localiser_bundle.py write-torch-maps), put them in {MAPS}, "
            f"and re-run. Caching only the members that CAN be cached is not an option here: it "
            f"would leave a partial {split} set behind and the next run would adopt it."
        )
    return need


def apply(args: argparse.Namespace) -> None:
    """Write localiser_boxes_{split}.csv for the SELECTED configuration into the folder
    BoxProvider("unet", boxes_dir=...) reads, predicting any maps the cache lacks."""
    from src.localise_eval import report, _CAND_FIELDS

    _check_splits([args.split], args.allow_test)
    sel = json.loads((BUNDLE / "bundle_selection.json").read_text())
    cfg = sel["selected"]
    # A boxes folder is one selection: a folder that already holds boxes may only
    # receive further splits of the SAME configuration.
    mp = Path(args.boxes_dir) / "bundle_manifest.json"
    if mp.exists():
        prior = json.loads(mp.read_text()).get("selected")
        if prior != cfg:
            raise SystemExit(
                f"[apply] {mp} was written from selection {prior}, but the current "
                f"{BUNDLE / 'bundle_selection.json'} selects {cfg}. Restore that selection "
                f"into the bundle folder or pass a new --boxes-dir; nothing written."
            )
    runs = tuple(cfg["ensemble"].split("+"))
    # Before anything is predicted or written: every member that needs maps for
    # this split must be auto-cacheable. Refuses naming all of them at once.
    for r in _assert_auto_cacheable(runs, args.split):
        man = json.loads((MAPS / f"{r}_manifest.json").read_text())
        # A member's frame comes from its own manifest, never from the caller: a
        # missing split is filled in the frame its other splits were cached in.
        tdir = man.get("tensors_dir", str(TENSORS))
        print(f"[apply] caching {r} {args.split} from {man['weights']} (frame {tdir})")
        cache(
            argparse.Namespace(
                tag=r,
                weights=man["weights"],
                base_filters=man["base_filters"],
                depth=man["depth"],
                encoder=man["encoder"],
                split=[args.split],
                tensors_dir=tdir,
                batch_size=args.batch_size,
                limit=None,
                cpu=args.cpu,
                images_only=getattr(args, "images_only", False),
                allow_test=args.allow_test,
            )
        )
    meta, gts, breasts = _load_split_side_tables(args.split, args.limit)
    # The weight vector must reach the rebuild here: the boxes this writes are the
    # ones the endpoint is computed on, and the folder guard above compares `weights`.
    apply_wvec = _weights_for_label(runs, cfg.get("weights", "equal"))
    print(
        f"[apply] weights: {cfg.get('weights', 'equal')}"
        f"{'' if apply_wvec is None else ' -> ' + str(tuple(round(x, 6) for x in apply_wvec))}"
    )
    df = _bundle_boxes(args.split, runs, cfg, meta, gts, breasts, weights=apply_wvec)
    # The test split has no recorded reference, so its guarantee rests on these same
    # checks having passed for val and train into the SAME folder.
    _assert_export_matches_selection(
        df,
        args.split,
        (
            None
            if args.limit is not None or args.split == "test"
            else sel.get(args.split, {}).get("mean_iou")
        ),
        "apply",
    )
    out = Path(args.boxes_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"localiser_boxes_{args.split}.csv"
    df.to_csv(path, index=False)
    print(f"-> {path}")
    from src.localise_eval import K as _K

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
    ] + [f"cand{j + 1}_{f}" for j in range(_K) for f in _CAND_FIELDS]
    kpath = out / f"localiser_boxes_k{_K}_{args.split}.csv"
    df[[c for c in k3_cols if c in df.columns]].to_csv(kpath, index=False)
    print(f"-> {kpath}")
    s = report(df, args.split)
    man = {
        "selected": cfg,
        "selected_cfg_id": sel.get("selected_cfg_id"),
        "source_bundle": str(BUNDLE),
        "runs": {
            r: (
                json.loads((MAPS / f"{r}_manifest.json").read_text()).get("weights")
                if (MAPS / f"{r}_manifest.json").exists()
                else None
            )
            for r in runs
        },
        "summaries": {},
    }
    mp = out / "bundle_manifest.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        man["summaries"] = old.get("summaries", {})
    man["summaries"][args.split] = s
    mp.write_text(json.dumps(man, indent=2, default=float))
    print(f"-> {mp}")
    if args.split == "val":
        # The jitter rung resamples the observed validation error tuples and fallback
        # rate, in the form localise_eval exports them.
        scored = df[df["status"] != "no_gt"]
        ok = df[df["status"] == "ok"]
        err = {
            "fitted_on": "val",
            "n_fitted_on": int(len(ok)),
            "n_lesions": int(len(scored)),
            "fallback_rate": float((scored["status"] == "miss").mean()),
            "samples": ok[["dy_frac", "dx_frac", "scale_ratio_h", "scale_ratio_w"]]
            .to_numpy()
            .tolist(),
            "source": "localiser_bundle.py apply",
            "selected": cfg,
        }
        ep = out / "localiser_error_model.json"
        ep.write_text(json.dumps(err, indent=2))
        print(f"-> {ep}")


# gate


def _aligned_iou(cand: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per-lesion box IoU of both tables on the same scored lesions, misses 0."""
    for name, t in (("candidate", cand), ("reference", ref)):
        if "split" in t.columns and not set(t["split"].unique()) <= set(ALLOWED_SPLITS):
            raise SystemExit(
                f"{name} table carries rows outside {ALLOWED_SPLITS}; the test "
                f"partition is frozen"
            )
    c = cand[cand["status"] != "no_gt"][["lesion_id", "patient_id", "label", "box_iou"]]
    r = ref[ref["status"] != "no_gt"][["lesion_id", "box_iou"]]
    m = c.merge(r, on="lesion_id", how="inner", suffixes=("_cand", "_ref"), validate="1:1")
    if len(m) != len(c) or len(m) != len(r):
        raise SystemExit(
            f"lesion sets differ: candidate {len(c)}, reference {len(r)}, common {len(m)}"
        )
    m["box_iou_cand"] = m["box_iou_cand"].fillna(0.0)
    m["box_iou_ref"] = m["box_iou_ref"].fillna(0.0)
    m["diff"] = m["box_iou_cand"] - m["box_iou_ref"]
    return m


def _gate_tables(cand: pd.DataFrame, ref: pd.DataFrame, cand_name: str, ref_name: str) -> dict:
    """Paired patient-cluster bootstrap 95 % CI of mean(candidate - reference), paired
    by lesion so both localisers are resampled on the same patients in every draw."""
    from src.evaluate import bootstrap_ci

    m = _aligned_iou(cand, ref)
    diff = m["diff"].to_numpy(float)
    point, lo, hi = bootstrap_ci(
        m["label"].to_numpy(int),
        diff,
        lambda y, d: float(np.mean(d)),
        groups=m["patient_id"].to_numpy(),
    )
    res = {
        "candidate": cand_name,
        "reference": ref_name,
        "n_lesions": int(len(m)),
        "n_patients": int(m["patient_id"].nunique()),
        "n_iterations": int(config.BOOTSTRAP_ITERATIONS),
        "mean_iou_candidate": float(m["box_iou_cand"].mean()),
        "mean_iou_reference": float(m["box_iou_ref"].mean()),
        "mean_diff": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_lesions_improved": int((diff > 0).sum()),
        "n_lesions_worse": int((diff < 0).sum()),
        "passes": bool(lo > 0.0),
    }
    print(
        f"  {cand_name} {res['mean_iou_candidate']:.4f} vs {ref_name} "
        f"{res['mean_iou_reference']:.4f} diff {point:+.4f} "
        f"95% CI [{lo:+.4f}, {hi:+.4f}] n={res['n_lesions']} lesions / "
        f"{res['n_patients']} patients improved {res['n_lesions_improved']} "
        f"worse {res['n_lesions_worse']} -> {'PASS' if res['passes'] else 'FAIL'} "
        f"(the CI must lie above zero)"
    )
    return res


def gate(args: argparse.Namespace) -> None:
    cand = pd.read_csv(args.candidate)
    ref = pd.read_csv(args.reference)
    name = args.label or Path(args.candidate).resolve().parent.name
    ref_path = Path(args.reference).resolve()
    ref_name = "run1" if ref_path.parent == (ROOT / "artifacts").resolve() else ref_path.parent.name
    res = _gate_tables(cand, ref, name, ref_name)
    res["candidate_table"] = str(args.candidate)
    res["reference_table"] = str(args.reference)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else BUNDLE / f"gate_{name}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"-> {out}")


# write-torch-maps — maps for a PyTorch member, from a checkpoint that exists

# Trainer flags, refused because this verb only scores. --batch-size is absent
# deliberately: here it batches inference, not optimisation.
_TRAINING_ARGS = (
    "--epochs",
    "--lr",
    "--seed",
    "--patience",
    "--train-cache",
    "--steps-per-epoch",
    "--optimizer",
    "--loss",
    "--augment",
    "--resume",
    "--write-boxes",
    "--out-dir",
    "--encoder",
    "--output-prior",
)


def _require_torch(verb: str) -> None:
    """Refuse a torch verb under the Keras interpreter, before any work happens; the
    alternative is a bare ModuleNotFoundError from an import buried in the handler."""
    import importlib.util

    missing = [
        m for m in ("torch", "segmentation_models_pytorch") if importlib.util.find_spec(m) is None
    ]
    if missing:
        raise SystemExit(
            f"{verb} REFUSED: {', '.join(missing)} not importable under this interpreter "
            f"({sys.executable}).\n"
            f"This verb needs the torch environment, e.g.\n"
            f"    PYTHONPATH=. .torch-env/bin/python notebooks/scripts/localiser_bundle.py "
            f"{verb} ...\n"
            f"(requirements-torch.txt pins it.)"
        )


def _refuse_training_args(argv: list[str]) -> None:
    """Refuse a trainer flag by name. argparse runs first and rejects every flag the verb
    does not define, so this fires only for a flag the verb's parser also defines."""
    hit = [a for a in argv if a.split("=")[0] in _TRAINING_ARGS]
    if hit:
        raise SystemExit(
            f"write-torch-maps REFUSED: {', '.join(hit)} belongs to a trainer. "
            "This verb scores an existing checkpoint; to train, use "
            "src.localise_torch.train."
        )


def _encoder_from_manifest(ckpt: Path) -> tuple[str, dict]:
    """`(encoder, manifest)` for the run that produced `ckpt`, refusing rather than
    guessing: a wrong encoder surfaces as a shape error pages from the cause."""
    mp_ = ckpt.parent / "manifest.json"
    if not mp_.exists():
        raise SystemExit(
            f"write-torch-maps REFUSED: no manifest.json beside {ckpt}, and the encoder "
            f"is read from it. Put one at {mp_}."
        )
    man = json.loads(mp_.read_text())
    enc = man.get("encoder")
    if not enc:
        raise SystemExit(f"write-torch-maps REFUSED: {mp_} records no 'encoder'.")
    # The run manifest records the version too ("pytorch-2.8.0"); the cache
    # manifest write_maps writes records a bare "pytorch". Match the family.
    if not str(man.get("framework", "")).lower().startswith(("pytorch", "torch")):
        raise SystemExit(
            f"write-torch-maps REFUSED: {mp_} records framework "
            f"{man.get('framework')!r}, not pytorch."
        )
    return enc, man


def write_torch_maps(args: argparse.Namespace) -> None:
    """`src.localise_torch.evaluate.write_maps` on an existing checkpoint, so a
    further split's maps can be written without running a training job."""
    _refuse_training_args(sys.argv[1:])
    # The same rule _check_splits applies, for the same reason.
    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "write-torch-maps REFUSED: --split test needs --allow-test, which "
            "only the test-map job and run_test_pass.sh pass."
        )

    _require_torch("write-torch-maps")
    import torch
    import segmentation_models_pytorch as smp
    from src.localise_torch.evaluate import write_maps as _write_maps

    ckpt = Path(args.weights)
    if not ckpt.exists():
        raise SystemExit(f"write-torch-maps REFUSED: no checkpoint at {ckpt}")
    encoder, man = _encoder_from_manifest(ckpt)
    tiled = bool(man.get("tiled_inference", False))

    # in_channels=3 because the member was trained on ImageNet-normalised 3-channel
    # input; encoder_weights is None because the state dict supplies every weight.
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.to(args.device).eval()
    print(
        f"[write-torch-maps] {args.tag}: {encoder} from {ckpt} on {args.device}, "
        f"{'tiled' if tiled else 'whole-image'}, frame {args.canvas[0]}x{args.canvas[1]}"
    )

    maps_dir = Path(args.maps_dir)
    plain = maps_dir / f"{args.tag}_{args.split}_plain.npy"
    flip = maps_dir / f"{args.tag}_{args.split}_flip.npy"
    started = time.time()
    with torch.no_grad():
        r = _write_maps(
            model,
            args.tensors_dir,
            args.split,
            args.tag,
            maps_dir,
            device=args.device,
            batch=args.batch_size,
            out_hw=tuple(args.canvas),
            limit=args.limit,
            tiled=tiled,
            weights=str(ckpt),
            encoder=encoder,
        )

    # Exit status from the post-conditions: "write_maps returned" is not evidence
    # that the maps are on disk.
    bad = []
    for q in (plain, flip):
        if not q.exists():
            bad.append(f"  {q} was not written")
        elif q.stat().st_mtime < started:
            bad.append(
                f"  {q} is older than this run (mtime {q.stat().st_mtime:.0f} "
                f"< start {started:.0f}) — a previous file was left in place"
            )
    mpath = maps_dir / f"{args.tag}_manifest.json"
    if not mpath.exists():
        bad.append(f"  {mpath} was not written")
    else:
        m = json.loads(mpath.read_text())
        if args.split not in m.get("splits", {}):
            bad.append(f"  {mpath} records no '{args.split}' split")
        for k in ("weights", "encoder"):
            if not m.get(k):
                bad.append(f"  {mpath} records no {k!r} — apply reads it")
    if bad:
        print("write-torch-maps FAILED its post-conditions:", *bad, sep="\n", file=sys.stderr)
        raise SystemExit(1)

    # The device is recorded here rather than inside write_maps, which the trainer
    # shares: a control comparing maps across machines needs to know what made them.
    import json as _json

    if mpath.exists():
        _m = _json.loads(mpath.read_text())
        try:
            import torch as _t

            _dev = (
                _t.cuda.get_device_name(0)
                if _t.cuda.is_available()
                else (
                    "MPS"
                    if getattr(_t.backends, "mps", None) and _t.backends.mps.is_available()
                    else "CPU"
                )
            )
        except Exception:
            _dev = "unknown"
        _m["gpu_name"] = _dev
        _m["inference_path"] = "torch no_grad, eager"
        mpath.write_text(_json.dumps(_m, indent=2))

    print(f"-> {plain}")
    print(f"-> {flip}")
    print(f"-> {mpath}")
    print(f"[write-torch-maps] {r['n']} rows in {time.time() - started:.0f} s")


# torch-member-boxes — one selection rule for every member's box table
def _in_channels_from_state_dict(sd) -> tuple[int, str]:
    """Input channels of the encoder's stem convolution, read from the weights: the
    stem's name differs by encoder, so the shape is what is read, not the name."""
    for k, v in sd.items():
        if getattr(v, "ndim", 0) == 4 and int(v.shape[1]) in (1, 3):
            return int(v.shape[1]), k
    raise SystemExit(
        "torch-member-boxes REFUSED: no stem convolution found in the "
        "checkpoint, so its input channel count cannot be read."
    )


def torch_member_boxes(args: argparse.Namespace) -> None:
    """Re-derive a PyTorch member's box tables from its checkpoint, under the same
    largest-component rule every Keras member's table uses. No training."""
    import shutil

    _require_torch("torch-member-boxes")
    import torch
    import segmentation_models_pytorch as smp

    from src.localise_torch.evaluate import boxes_table

    ckpt = Path(args.weights)
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint: {ckpt}")
    out = ckpt.parent
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=dev)
    in_channels, stem = _in_channels_from_state_dict(state)
    if in_channels == 1:
        raise SystemExit(
            f"torch-member-boxes REFUSED: {ckpt} is a 1-channel member ({stem} has shape "
            f"{tuple(state[stem].shape)}), and the scoring path replicates every input to "
            f"three channels, so the forward pass cannot accept it. Score a 3-channel "
            f"checkpoint instead."
        )
    model = smp.Unet(
        encoder_name=args.encoder, encoder_weights=None, in_channels=in_channels, classes=1
    ).to(dev)
    model.load_state_dict(state)
    print(f"[boxes] {out.name}: loaded {ckpt.name} on {dev} ({in_channels}-channel, from {stem})")

    for split in args.split:
        if not (Path(args.tensors_dir) / f"{split}_images.npy").exists():
            print(f"[boxes] {split}: not in {args.tensors_dir}, skipping")
            continue
        q = out / f"localiser_boxes_{split}.csv"
        # Keep an existing table under a superseded name, once, rather than overwrite it.
        if q.exists():
            old = out / f"localiser_boxes_{split}_peakrule_superseded.csv"
            if not old.exists():
                shutil.copy2(q, old)
                print(f"[boxes] kept the previous table -> {old}")
        t = boxes_table(
            model, args.tensors_dir, split, device=dev, batch=args.batch_size, limit=args.limit
        )
        t.to_csv(q, index=False)
        scored = t[t["status"] != "no_gt"]
        mean_iou = float(scored["box_iou"].fillna(0).mean())
        print(
            f"[boxes] {split}: {len(t)} rows, mean box IoU (miss-inclusive) "
            f"{mean_iou:.4f}, detected {int(t['detected'].sum())}/{len(t)}"
        )
        print(f"-> {q}")


# reexport-selection — the table a weighted selection actually names
def _column_mean_iou(col: np.ndarray) -> float:
    """The sweep's own aggregation: NaN rows excluded, misses already 0."""
    ok = ~np.isnan(col)
    return float(np.where(ok, col, 0.0)[ok].mean())


def reexport_selection(args: argparse.Namespace) -> None:
    """Export a weighted selection's real box table, rebuilt from the score arrays and
    the deterministic grid. It writes to --out only; val and train only."""
    runs = tuple(args.tag)
    wvecs = _weight_vectors(runs, args.weighting)
    grid = _grid(runs, wvecs)
    print(
        f"grid rebuilt: {len(grid)} configurations from {len(runs)} members "
        f"x {len(wvecs)} weight vectors ({args.weighting})"
    )

    iou_t = np.load(BUNDLE / "sweep_iou_train.npy", mmap_mode="r")
    iou_v = np.load(BUNDLE / "sweep_iou_val.npy", mmap_mode="r")
    print(f"score arrays : train {iou_t.shape}, val {iou_v.shape}")
    if iou_t.shape[1] != len(grid):
        raise SystemExit(
            f"grid width {len(grid)} does not match the score array "
            f"{iou_t.shape[1]} - the arrays are from a different sweep; stop"
        )

    row = grid[args.cfg]
    print(f"\ncfg {args.cfg}: {row}")
    tr, va = _column_mean_iou(np.asarray(iou_t[:, args.cfg])), _column_mean_iou(
        np.asarray(iou_v[:, args.cfg])
    )
    print(f"  from the score arrays : train {tr:.10f} val {va:.10f}")

    # The selection this cfg claims to be: argmax of train over full-ensemble rows.
    full = "+".join(runs)
    idx = [i for i, g in enumerate(grid) if g["ensemble"] == full]
    tr_all = np.array([_column_mean_iou(np.asarray(iou_t[:, i])) for i in idx])
    best = idx[int(tr_all.argmax())]
    print(
        f"  train argmax over the {len(idx)} full-ensemble rows: cfg {best} "
        f"({'MATCHES' if best == args.cfg else 'DOES NOT MATCH'} --cfg)"
    )
    if best != args.cfg:
        raise SystemExit("the requested cfg is not the train winner; stop")

    if args.expect:
        exp = json.loads(Path(args.expect).read_text())["selected"]
        for k in ("ensemble", "tta", "breast_filter", "rule", "threshold", "min_px"):
            if str(row[k]) != str(exp[k]):
                raise SystemExit(
                    f"recorded selection disagrees on {k}: "
                    f"grid {row[k]!r} vs recorded {exp[k]!r}; stop"
                )
        print("  recorded selection agrees on all six selection fields")

    wvec = _weights_for_label(runs, row["weights"])
    print(f"  weights label         : {row['weights']}")
    print(f"  weight vector         : {None if wvec is None else tuple(round(x, 6) for x in wvec)}")
    if wvec is None:
        print(
            "\n  NOTE: this configuration's winning weighting is equal weights, so the "
            "existing export is already correct for it."
        )

    cfg = {k: row[k] for k in ("ensemble", "tta", "breast_filter", "rule", "threshold", "min_px")}
    cfg["weights"] = row["weights"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print()
    for split in ("val", "train"):
        meta, gts, breasts = _load_split_side_tables(split, None)
        df = _bundle_boxes(split, runs, cfg, meta, gts, breasts, weights=wvec)
        df.to_csv(out / f"bundle_boxes_{split}.csv", index=False)
        m = df[df["status"] != "no_gt"]["box_iou"].fillna(0).mean()
        ref = tr if split == "train" else va
        print(
            f"  {split}: exported mean box IoU {m:.10f} | score array {ref:.10f} "
            f"| |diff| {abs(m - ref):.3e}"
        )
    (out / "bundle_selection.json").write_text(
        json.dumps(
            {
                "selected_cfg_id": args.cfg,
                "selected": cfg,
                "selection_rule": (
                    "ensemble set fixed a priori to all swept runs; weights, threshold, "
                    "min-px, rule and TTA selected on train"
                ),
                "runs": list(runs),
                "weight_vector": None if wvec is None else list(wvec),
                "train": {"mean_iou": tr},
                "val": {"mean_iou": va},
                "reexport_note": (
                    "exported through the fixed _bundle_boxes path that receives the weight "
                    "vector; supersedes the earlier equal-weight export"
                ),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bundle-dir",
        default=str(config.LOC_BUNDLE_DIR),
        help="bundle folder, read and written (smoke runs: point it at a scratch folder)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="predict and store 8-bit maps (GPU)")
    c.add_argument("--tag", required=True, help="cache name, e.g. run1, run3, run4")
    c.add_argument("--weights", required=True)
    c.add_argument("--base-filters", type=int, default=config.LOC_BASE_FILTERS)
    c.add_argument("--depth", type=int, default=config.LOC_DEPTH)
    c.add_argument("--encoder", choices=["none", "vgg16"], default="none")
    c.add_argument(
        "--tiled",
        action="store_true",
        help="tiled inference, for a member cached from a 2048x1152 store",
    )
    c.add_argument("--tile-overlap", type=int, default=64)
    c.add_argument(
        "--canvas",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="area-downsample each map to this frame before storing. Members "
        "are averaged in one frame, so a member predicted at 2048x1152 is "
        "stored at 1024x576 to sit beside the others. Boxes are still "
        "derived at full resolution by localise_eval; only the ensemble "
        "cache is downsampled.",
    )
    c.add_argument(
        "--tensors-dir",
        default=None,
        help="tensor store to predict from, i.e. the member's normalisation "
        "frame. Default outputs/localiser_tensors; predict a member from the cache it "
        "was trained on. Recorded in the run's cache manifest, where sweep "
        "asserts it.",
    )
    c.add_argument("--split", nargs="+", default=list(ALLOWED_SPLITS), help="one or more splits")
    # cache refuses the test split like every other reader; only the test-map job and
    # run_test_pass.sh pass the flag that permits it.
    c.add_argument(
        "--allow-test",
        action="store_true",
        help="permit --split test (the test-map job and run_test_pass.sh only)",
    )
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--limit", type=int, default=None, help="smoke: first N rows only")
    c.add_argument("--cpu", action="store_true", help="run inference on the CPU")
    c.add_argument(
        "--images-only",
        action="store_true",
        help="the store carries no {split}_masks.npy. Maps are still written; "
        "the cache-time single-box check is skipped by name in the manifest "
        "rather than computed against a dummy. For a store that ships "
        "without the mask array.",
    )
    c.set_defaults(fn=cache)

    rc = sub.add_parser(
        "recheck",
        help="recompute the cache-time single-box check from maps already "
        "on disk, in the frame they were stored in (CPU, no inference)",
    )
    rc.add_argument("--tag", required=True)
    rc.add_argument(
        "--split",
        nargs="+",
        default=list(ALLOWED_SPLITS),
        help="one or more splits in a single flag, e.g. --split val train",
    )
    rc.add_argument("--limit", type=int, default=None, help="smoke: first N rows")
    rc.set_defaults(fn=recheck)

    s = sub.add_parser(
        "sweep", help="score the grid from the cache, select on train, gate on val (CPU)"
    )
    s.add_argument(
        "--tag", nargs="+", default=list(config.LOC_BUNDLE_RUNS), help="the runs to sweep over"
    )
    s.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    s.add_argument("--reference", default=str(ROOT / "artifacts" / "localiser_boxes_val.csv"))
    s.add_argument("--limit", type=int, default=None, help="smoke: first N rows per split")
    s.add_argument(
        "--weighting",
        choices=["none", "ones", "coarse", "group", "proportional"],
        default="none",
        help="weighted map averaging over the full ensemble, selected on train. "
        "none (default) is the equal-weight sweep. ones is the equivalence "
        "probe: equal weights through the weighted code path. coarse is the "
        "per-member {0.5,1,2} grid while k<=6 and the per-group grid beyond "
        "that; group forces the per-group form. proportional needs per-member train "
        "IoUs, which sweep does not pass, so it adds no vector.",
    )
    s.set_defaults(fn=sweep)

    se = sub.add_parser(
        "select", help="re-select from the existing sweep under a recorded rule (CPU)"
    )
    se.add_argument(
        "--ensemble",
        choices=["all", "free"],
        required=True,
        help="all = ensemble set fixed a priori to every swept run; free = the sweep's rule",
    )
    se.add_argument("--reference", default=str(ROOT / "artifacts" / "localiser_boxes_val.csv"))
    se.add_argument("--limit", type=int, default=None, help="smoke: first N rows per split")
    se.set_defaults(fn=select)

    cd = sub.add_parser(
        "candidates",
        help="the top --k surviving components of the selected map, per image, "
        "with features and labels — the re-ranker's data (CPU)",
    )
    cd.add_argument("--split", required=True, choices=list(ALLOWED_SPLITS))
    cd.add_argument(
        "--k",
        type=int,
        default=5,
        help="candidates kept per image, ranked by the selected rule (default 5)",
    )
    cd.add_argument(
        "--iou-positive",
        type=float,
        default=0.30,
        help="a candidate is positive at this box IoU with ANY GT lesion on "
        "the image (default 0.30)",
    )
    cd.add_argument("--out", default=None, help="default: <bundle>/candidates_{split}.csv")
    cd.add_argument("--limit", type=int, default=None, help="smoke: first N rows")
    cd.set_defaults(fn=candidates)

    m = sub.add_parser(
        "metrics", help="full localise_eval metric set for the selected bundle (CPU)"
    )
    m.add_argument("--limit", type=int, default=None, help="smoke: first N rows per split")
    m.set_defaults(fn=metrics)

    a = sub.add_parser(
        "apply",
        help="write the selected bundle's boxes for one split into a candidate boxes folder",
    )
    a.add_argument("--split", required=True, choices=["val", "train", "test"])
    a.add_argument(
        "--boxes-dir",
        required=True,
        help="folder for localiser_boxes_{split}.csv (new name, never artifacts/)",
    )
    a.add_argument(
        "--allow-test", action="store_true", help="permit the test split (run_test_pass.sh only)"
    )
    a.add_argument("--batch-size", type=int, default=8)
    a.add_argument("--cpu", action="store_true")
    a.add_argument(
        "--images-only",
        action="store_true",
        help="passed through to any auto-cache: the store has no mask array",
    )
    a.add_argument("--limit", type=int, default=None, help="smoke: first N rows")
    a.set_defaults(fn=apply)

    g = sub.add_parser(
        "gate", help="paired patient-bootstrap CI of a candidate's val box IoU vs run 1"
    )
    g.add_argument("--candidate", required=True, help="localiser_boxes_val.csv of the candidate")
    g.add_argument("--reference", default=str(ROOT / "artifacts" / "localiser_boxes_val.csv"))
    g.add_argument("--label", default=None, help="label in the output (default: parent folder)")
    g.add_argument("--out", default=None)
    g.set_defaults(fn=gate)

    w = sub.add_parser(
        "write-torch-maps",
        help="maps for a PyTorch member from an existing checkpoint (inference only)",
    )
    w.add_argument("--weights", required=True, help="the member's best.pt")
    w.add_argument("--tag", required=True, help="member name in the bundle, e.g. l7a")
    w.add_argument("--split", required=True, choices=["train", "val", "test"])
    w.add_argument(
        "--allow-test",
        action="store_true",
        help="permit --split test (the test-map job and run_test_pass.sh only)",
    )
    w.add_argument(
        "--tensors-dir", required=True, help="per-lesion tensor store holding {split}_images.npy"
    )
    w.add_argument("--maps-dir", default=None, help="default: <bundle>/maps")
    w.add_argument(
        "--canvas",
        nargs=2,
        type=int,
        default=[1024, 576],
        metavar=("H", "W"),
        help="frame the maps are stored in; must match the other members",
    )
    w.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    w.add_argument("--batch-size", type=int, default=4, help="tiled inference only")
    w.add_argument("--limit", type=int, default=None, help="smoke: first N rows")
    w.set_defaults(fn=write_torch_maps)

    tb = sub.add_parser(
        "torch-member-boxes", help="re-derive a PyTorch member's box tables from its checkpoint"
    )
    tb.add_argument(
        "--weights", required=True, help="the member's best.pt; the tables are written beside it"
    )
    tb.add_argument("--encoder", required=True)
    tb.add_argument("--tensors-dir", required=True)
    tb.add_argument("--split", nargs="+", default=["val", "train"], help="one or more splits")
    tb.add_argument("--batch-size", type=int, default=8)
    tb.add_argument("--limit", type=int, default=None)
    tb.set_defaults(fn=torch_member_boxes)

    rx = sub.add_parser(
        "reexport-selection",
        help="export a weighted selection's real box table from the score arrays",
    )
    rx.add_argument("--tag", nargs="+", required=True, help="the swept members, in order")
    rx.add_argument(
        "--weighting", default="group", choices=["none", "ones", "coarse", "group", "proportional"]
    )
    rx.add_argument("--cfg", type=int, required=True, help="the selected configuration id")
    rx.add_argument("--out", required=True)
    rx.add_argument("--expect", default=None, help="bundle_selection.json to check against")
    rx.set_defaults(fn=reexport_selection)

    args = ap.parse_args()
    _set_bundle_dir(args.bundle_dir)
    if getattr(args, "maps_dir", None) is None and args.cmd == "write-torch-maps":
        args.maps_dir = str(MAPS)
    args.fn(args)


if __name__ == "__main__":
    main()

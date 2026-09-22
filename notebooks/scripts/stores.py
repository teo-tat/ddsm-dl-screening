#!/usr/bin/env python
"""stores.py — build, copy and re-check every tensor store the project trains from.

A store is a decoded, un-standardised copy of what the pipeline feeds a model: one
channel, float32 in [0,1], frame order. The test partition is never exported.

  crop per-lesion crops, or the whole mammogram with --source full
  fusion the paired stores: view, context and the single-stream control
  candidates one row per localiser candidate component — the re-ranker's set
  notest-copy a test-free copy of the v1 per-lesion localiser store, for upload
  verify do the stored rows still equal what the current code produces?
  norms scratch-mode normalisation statistics over the training crops
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

data_loader = None  # bound by _backend() on first use


def _backend():
    """Import data_loader (and through it TensorFlow) on use, and fix the seeds, so
    a filesystem-only subcommand such as notest-copy loads no framework."""
    global data_loader
    if data_loader is None:
        from src import data_loader as _module

        data_loader = _module
    config.set_seeds()
    return data_loader


BUNDLE = config.LOC_BUNDLE_DIR
# The candidate-row features the store records, with the per-member box agreement and
# the rule's own pick flag; the re-ranker reads a subset of them.
FEATURES = (
    "peak_prob",
    "mean_prob",
    "sum_prob",
    "area_px",
    "dist_to_breast_edge",
    "dist_to_breast_edge_norm",
    "member_support_mean",
    "member_disagreement",
    "n_members_supporting",
    "cand_rank",
    "is_rule_pick",
)
SPLIT_ORDER = ("val", "train")


# Shared helpers
def _sha(path: Path) -> str:
    """First 16 hex digits of a file's SHA-256, which is what manifests record."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha_full(path: Path, cap: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(cap):
            h.update(b)
    return h.hexdigest()


def _ordered(splits) -> list[str]:
    """val before train, whatever order they were asked for: the fidelity check reads
    the validation rows, so writing them first means a failed export still has them."""
    return [s for s in SPLIT_ORDER if s in set(splits)]


def _build_datasets(normalisation: str, target_size=None, **kw):
    """data_loader.build_datasets with the config paths, plus the export-only knobs.
    target_size is a parameter so two stores can differ only in the final resize."""
    return data_loader.build_datasets(
        train_csv=config.TRAIN_CSV,
        test_csv=config.TEST_CSV,
        train_img_dir=config.TRAIN_IMG_DIR,
        test_img_dir=config.TEST_IMG_DIR,
        train_roi_dir=config.TRAIN_ROI_DIR,
        test_roi_dir=config.TEST_ROI_DIR,
        target_size=target_size or config.TARGET_SIZE,
        val_fraction=config.VAL_FRACTION,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        normalisation=normalisation,
        **kw,
    )


def _collect(ds, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    n = 0
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
        n += len(x)
        if limit is not None and n >= limit:
            break
    x = np.concatenate(xs)[:limit]
    y = np.concatenate(ys)[:limit]
    return x.astype(np.float32), y.astype(np.int64)


def _crop_kw(
    box_source: str | None, boxes_dir: str | None, margin: float, mask_overlay: bool, source: str
) -> tuple[dict, str | None]:
    """(kwargs, crop_sizing) for the source switch both the exporter and the re-check
    use, so a store is rebuilt through exactly the call that built it."""
    if source == "full":
        return dict(source="full"), None
    provider = None
    crop_sizing = "mask_margin"
    if box_source:
        # Train and val only: no store ever holds test rows.
        from src.boxes import provider_for

        provider = provider_for(box_source, boxes_dir=boxes_dir, splits=("train", "val"))
        crop_sizing = "box_provider"
    return (
        dict(
            source="crop",
            crop_sizing=crop_sizing,
            crop_margin=margin,
            mask_overlay=mask_overlay,
            box_provider=provider,
        ),
        crop_sizing,
    )


# crop — per-lesion crops, and the whole-image store
def cmd_crop(args) -> None:
    _backend()
    full = args.source == "full"
    if full and (args.box_source or args.mask_overlay):
        raise SystemExit(
            "--source full takes no box source and no mask overlay: it is the whole image"
        )
    if args.box_source and not args.tag:
        raise SystemExit("--box-source needs --tag so the stores do not collide")

    target_size = (
        (args.image_size, args.image_size) if args.image_size else tuple(config.TARGET_SIZE)
    )
    src_kw, crop_sizing = _crop_kw(
        args.box_source, args.boxes_dir, args.margin, args.mask_overlay, args.source
    )

    tag = (
        args.tag
        or ("full_image" if full else None)
        or (f"m{int(round(args.margin * 100)):03d}" + ("_overlay" if args.mask_overlay else ""))
    )
    out = Path(args.out) if args.out else config.OUTPUTS_DIR / f"crop_store_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Raw cached crops, frame order, [0,1]: shuffle and standardisation off. One call
    # for both sources; the frame key differs — a crop row is a lesion, a full row an image.
    train_ds, val_ds, _, _, train_split = _build_datasets(
        "scratch",
        target_size=target_size,
        augment=False,
        shuffle_train=False,
        standardize=False,
        **src_kw,
    )
    frames = data_loader.split_frames(args.source)
    key = "image file path" if full else "lesion_id"
    assert (
        frames["train"][key].tolist() == train_split[key].tolist()
    ), "split_frames and build_datasets disagree on the training frame"

    manifest = {
        "margin": args.margin,
        "mask_overlay": args.mask_overlay,
        "mask_dilation_px": config.ROI_MASK_DILATION_PX if args.mask_overlay else None,
        "intensity_norm": config.INTENSITY_NORM if full else config.CROP_INTENSITY_NORM,
        "target_size": list(target_size),
        "source": args.source,
        "unit": "image" if full else "lesion",
        "crop_sizing": None if full else crop_sizing,
        "box_source": args.box_source,
        "boxes_dir": args.boxes_dir,
        "store_tag": tag,
        "channels": 1,
        "value_range": [0.0, 1.0],
        "partition_seed": config.SEED,
        "splits": {},
        "files": {},
    }
    datasets = {"val": val_ds, "train": train_ds}
    for split in _ordered(args.split):
        x, y = _collect(datasets[split], args.limit)
        frame = frames[split].reset_index(drop=True)
        if args.limit is None and len(x) != len(frame):
            raise SystemExit(
                f"{split}: dataset yielded {len(x)} rows, frame has {len(frame)}; "
                f"a row failed to load and the store would be misaligned"
            )
        frame = frame.iloc[: len(x)]
        if not (frame["label"].to_numpy() == y).all():
            raise SystemExit(f"{split}: labels from the dataset differ from the frame")
        meta = (
            frame[["image file path", "patient_id", "label"]]
            if full
            else (
                frame[["lesion_id", "image_id", "patient_id", "label"]]
                if "image_id" in frame.columns
                else frame[["lesion_id", "patient_id", "label"]]
            )
        )
        np.save(out / f"{split}_images.npy", x)
        meta.to_csv(out / f"{split}_meta.csv", index=False)
        manifest["splits"][split] = {
            "n": int(len(x)),
            "n_malignant": int(y.sum()),
            "shape": list(x.shape[1:]),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }
        manifest["files"][f"{split}_images.npy"] = _sha(out / f"{split}_images.npy")
        print(
            f"[export] {split}: {x.shape} mean {x.mean():.4f} std {x.std():.4f} "
            f"({time.time() - t0:.0f} s)"
        )
        print(f"-> {out / f'{split}_images.npy'}")
        print(f"-> {out / f'{split}_meta.csv'}")

    # Fidelity check: the standard validation pipeline (standardised) must
    # equal the exported rows after the same standardisation.
    if "val" in set(args.split):
        chk_kw, _ = _crop_kw(
            args.box_source, args.boxes_dir, args.margin, args.mask_overlay, args.source
        )
        _, val_std, _, _, _ = _build_datasets(
            "scratch", target_size=target_size, augment=False, **chk_kw
        )
        xs = np.load(out / "val_images.npy", mmap_mode="r")
        mean, std = (
            data_loader._scratch_norm_stats(config.INTENSITY_NORM, "full")
            if full
            else data_loader._scratch_norm_stats(config.CROP_INTENSITY_NORM, "crop")
        )
        n_check = 0
        for xb, _ in val_std.take(2):
            xb = xb.numpy()
            ref = (np.asarray(xs[n_check : n_check + len(xb)]) - mean) / std
            if len(ref) < len(xb):
                break
            if not np.allclose(xb, ref, atol=1e-5):
                raise SystemExit(
                    "fidelity check FAILED: exported val rows differ from the " "standard pipeline"
                )
            n_check += len(xb)
        manifest["fidelity_check"] = {"split": "val", "rows_compared": int(n_check), "passed": True}
        print(
            f"[export] fidelity check passed on {n_check} validation rows "
            f"(standard pipeline == store after standardisation)"
        )
    else:
        manifest["fidelity_check"] = {"skipped": "val not exported"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {out / 'manifest.json'}")


# fusion — the paired stores
def raw_stream(
    frame: pd.DataFrame, margin: float, *, lookups: tuple, crop_sizing: str, provider
) -> tf.data.Dataset:
    """One crop per row, single channel, in [0,1], nothing after the cache: it is
    ``fusion._stream`` with standardisation off, which leaves the geometry unchanged."""
    lookup, mask_lookup, full_lookup = lookups
    ds = data_loader._make_dataset(
        frame,
        lookup,
        config.TARGET_SIZE,
        1,
        config.SEED,
        normalisation="scratch",
        augment=False,
        shuffle=False,
        intensity_norm=config.CROP_INTENSITY_NORM,
        source="crop",
        path_col="cropped image file path",
        pad_square=config.CROP_PAD_SQUARE,
        crop_sizing=crop_sizing,
        crop_margin=margin,
        mask_lookup=mask_lookup,
        full_lookup=full_lookup,
        box_provider=provider,
        standardize=False,
    )
    return ds.unbatch()


def collect_pairs(ds: tf.data.Dataset, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Materialise an unbatched stream into (images, labels)."""
    xs, ys = [], []
    n = 0
    for x, y in ds:
        xs.append(x.numpy()[None, ...])
        ys.append(np.asarray(y).reshape(1))
        n += 1
        if limit is not None and n >= limit:
            break
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    return x, y


def fusion_fidelity(run, out: Path, split: str = "val", n_batches: int = 2) -> int:
    """The standard fusion pipeline must equal the store once the channel is replicated
    and the ImageNet statistics applied, which is what the training side does."""
    # FusionRun.dataset turns augmentation, shuffling and both orderings off for any
    # split that is not train, which is what a like-for-like comparison needs.
    ds, _ = run.dataset(split)
    xa = np.load(out / f"{split}_images_a.npy", mmap_mode="r")
    xb = (
        np.load(out / f"{split}_images_b.npy", mmap_mode="r")
        if (out / f"{split}_images_b.npy").exists()
        else None
    )
    mean = np.asarray(config.NORM_STATS["imagenet_mean"], np.float32)
    std = np.asarray(config.NORM_STATS["imagenet_std"], np.float32)

    def prep(block: np.ndarray) -> np.ndarray:
        return (np.repeat(np.asarray(block), 3, axis=-1) - mean) / std

    n = 0
    for batch in ds.take(n_batches):
        x, _ = batch
        got_a = (x[0] if isinstance(x, (tuple, list)) else x).numpy()
        ref_a = prep(xa[n : n + len(got_a)])
        if len(ref_a) < len(got_a):
            break
        if not np.allclose(got_a, ref_a, atol=1e-5):
            raise SystemExit(f"fidelity check FAILED on crop A of {split}")
        if xb is not None and isinstance(x, (tuple, list)):
            got_b = x[1].numpy()
            ref_b = prep(xb[n : n + len(got_b)])
            if not np.allclose(got_b, ref_b, atol=1e-5):
                raise SystemExit(f"fidelity check FAILED on crop B of {split}")
        n += len(got_a)
    return n


def cmd_fusion(args) -> None:
    _backend()
    from src import fusion

    # FusionRun owns the pair construction, the tag rule, the margins and the box
    # provider; test_boxes=False keeps the provider on train and val.
    run = fusion.FusionRun(
        args.model,
        args.pair,
        args.margin,
        args.box_source,
        single_mass_only=args.single_mass_only,
        rows=args.rows,
        batch_size=None,
        tag_suffix="",
        boxes_dir=args.boxes_dir,
        test_boxes=False,
    )
    tag = args.tag or run.tag
    out = Path(args.out) if args.out else config.OUTPUTS_DIR / f"fusion_store_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    lookups = run.lookups
    paired = args.pair != "control"

    t0 = time.time()
    manifest = {
        "store_tag": tag,
        "fusion_tag": run.tag,
        "pair": args.pair,
        "rows": run.rows,
        "margin_a": run.margin,
        "margin_b": run.margin_b,
        "box_source": args.box_source,
        "boxes_dir": args.boxes_dir,
        "single_mass_only": args.single_mass_only,
        "crop_sizing": run.crop_sizing,
        "paired": paired,
        "intensity_norm": config.CROP_INTENSITY_NORM,
        "target_size": list(config.TARGET_SIZE),
        "channels": 1,
        "value_range": [0.0, 1.0],
        "partition_seed": config.SEED,
        "note": (
            "channel replication, ImageNet standardisation, shuffling, "
            "per-stream augmentation, both orderings and swap averaging "
            "happen at training time, not in this store"
        ),
        "splits": {},
        "files": {},
    }

    for split in _ordered(args.split):
        pairs = run.pairs[split].reset_index(drop=True)
        fa, fb = run.frames_for(split)
        xa, ya = collect_pairs(
            raw_stream(
                fa, run.margin, lookups=lookups, crop_sizing=run.crop_sizing, provider=run.provider
            ),
            args.limit,
        )
        if args.limit is None and len(xa) != len(pairs):
            raise SystemExit(
                f"{split}: stream A yielded {len(xa)} rows, pair table has "
                f"{len(pairs)}; a lesion failed to load and the store "
                f"would be misaligned"
            )
        keep = pairs.iloc[: len(xa)]
        if not (keep["label"].to_numpy() == ya).all():
            raise SystemExit(f"{split}: labels from stream A differ from the pair table")
        np.save(out / f"{split}_images_a.npy", xa)
        manifest["files"][f"{split}_images_a.npy"] = _sha(out / f"{split}_images_a.npy")
        print(
            f"[export] {split} A: {xa.shape} mean {xa.mean():.4f} std {xa.std():.4f} "
            f"({time.time() - t0:.0f} s)"
        )
        print(f"-> {out / f'{split}_images_a.npy'}")

        if paired:
            xb, _ = collect_pairs(
                raw_stream(
                    fb,
                    run.margin_b,
                    lookups=lookups,
                    crop_sizing=run.crop_sizing,
                    provider=run.provider,
                ),
                args.limit,
            )
            if len(xb) != len(xa):
                raise SystemExit(f"{split}: stream B yielded {len(xb)} rows against A's {len(xa)}")
            np.save(out / f"{split}_images_b.npy", xb)
            manifest["files"][f"{split}_images_b.npy"] = _sha(out / f"{split}_images_b.npy")
            print(
                f"[export] {split} B: {xb.shape} mean {xb.mean():.4f} std {xb.std():.4f} "
                f"({time.time() - t0:.0f} s)"
            )
            print(f"-> {out / f'{split}_images_b.npy'}")

        keep.to_csv(out / f"{split}_meta.csv", index=False)
        print(f"-> {out / f'{split}_meta.csv'}")
        manifest["splits"][split] = {
            "n": int(len(xa)),
            "n_malignant": int(ya.sum()),
            "shape": list(xa.shape[1:]),
        }

    if args.limit is None and "val" in set(args.split):
        n = fusion_fidelity(run, out)
        manifest["fidelity_check"] = {"split": "val", "rows_compared": int(n), "passed": True}
        print(
            f"[export] fidelity check passed on {n} validation rows "
            f"(standard fusion pipeline == store after standardisation)"
        )
    else:
        manifest["fidelity_check"] = {
            "skipped": "smoke run" if args.limit is not None else "val not exported"
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {out / 'manifest.json'}")


# candidates — one row per localiser candidate component
def _candidate_crop(full: np.ndarray, box, margin: float, target_size) -> np.ndarray:
    """The classifier's crop path, from data_loader's own functions."""
    arr = data_loader._crop_box(full, box, margin)
    arr = data_loader._normalise_intensity(arr, config.CROP_INTENSITY_NORM)
    return data_loader._pad_and_resize(arr, target_size, pad_square=config.CROP_PAD_SQUARE).astype(
        np.float32
    )


def export_candidate_split(
    split: str, margin: float, target_size, out: Path, limit: int | None
) -> dict:
    cand = pd.read_csv(BUNDLE / f"candidates_{split}.csv")
    if limit is not None:
        cand = cand.iloc[:limit]
    # The frame's path column is relative; resolve it through the same directory
    # lookup the real pipeline and precompute_localiser use, keyed on image_id.
    paths = {
        Path(k).parts[0]: v
        for k, v in data_loader._build_combined_lookup(
            config.TRAIN_IMG_DIR, config.TEST_IMG_DIR
        ).items()
    }
    missing = set(cand["image_id"]) - set(paths)
    if missing:
        raise SystemExit(
            f"{len(missing)} candidate image_ids have no file on disk, "
            f"e.g. {sorted(missing)[:3]}"
        )

    x = np.zeros((len(cand), target_size[0], target_size[1], 1), np.float32)
    t0 = time.time()
    # Group by image: one DICOM decode serves every candidate on that image.
    for n, (img, grp) in enumerate(cand.groupby("image_id", sort=False), 1):
        full = data_loader._read_full(str(paths[img]))
        for pos, r in zip(grp.index, grp.itertuples()):
            x[cand.index.get_loc(pos)] = _candidate_crop(
                full, (int(r.y0), int(r.y1), int(r.x0), int(r.x1)), margin, target_size
            )
        if n % 100 == 0 or n == cand["image_id"].nunique():
            print(
                f"[candstore] {split}: {n}/{cand['image_id'].nunique()} images "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{split}_images.npy", x)
    cand.to_csv(out / f"{split}_meta.csv", index=False)
    pos = int(cand["label"].sum())
    print(f"-> {out / f'{split}_images.npy'} {x.shape}")
    print(
        f"[candstore] {split}: {len(cand)} candidates, "
        f"{pos} positive / {len(cand) - pos} negative "
        f"({pos / max(len(cand), 1):.1%} positive), "
        f"{cand['image_id'].nunique()} images, {cand['patient_id'].nunique()} patients"
    )
    return {
        "n": int(len(cand)),
        "n_positive": pos,
        "n_negative": int(len(cand) - pos),
        "positive_rate": round(pos / max(len(cand), 1), 4),
        "n_images": int(cand["image_id"].nunique()),
        "n_patients": int(cand["patient_id"].nunique()),
        "shape": list(x.shape[1:]),
    }


def candidate_fidelity(out: Path, frames: dict) -> dict:
    """Rank-1 crops of single-lesion images must equal the rung-3 store's crops: on
    such an image the rule's own pick is candidate rank 1, so they agree bit for bit."""
    ref_dir = config.OUTPUTS_DIR / "crop_store_rung3_unet_bundle"
    if not (ref_dir / "val_images.npy").exists():
        return {"ran": False, "reason": f"{ref_dir} not present"}
    ref_x = np.load(ref_dir / "val_images.npy", mmap_mode="r")
    ref_m = pd.read_csv(ref_dir / "val_meta.csv")
    cand = pd.read_csv(out / "val_meta.csv")
    x = np.load(out / "val_images.npy", mmap_mode="r")

    frame = frames["val"]
    per_image = frame.groupby("image_id")["lesion_id"].count()
    single = set(per_image[per_image == 1].index)
    top1 = cand[(cand["cand_rank"] == 1) & (cand["image_id"].isin(single))]
    lookup = {r.image_id: i for i, r in enumerate(ref_m.itertuples()) if r.image_id in single}

    checked = matched = 0
    worst = 0.0
    for r in top1.itertuples():
        j = lookup.get(r.image_id)
        if j is None:
            continue
        a = np.asarray(x[cand.index.get_loc(r.Index)], np.float32)
        b = np.asarray(ref_x[j], np.float32)
        if a.shape != b.shape:
            continue
        checked += 1
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        matched += int(d == 0.0)
        if checked >= 40:
            break
    print(
        f"[candstore] fidelity: {matched}/{checked} rank-1 crops byte-identical to "
        f"crop_store_rung3_unet_bundle (max abs diff {worst:.2e})"
    )
    return {
        "ran": True,
        "checked": checked,
        "byte_identical": matched,
        "max_abs_diff": worst,
        "passed": bool(checked and matched == checked),
    }


def cmd_candidates(args) -> None:
    _backend()
    ts = (args.image_size, args.image_size) if args.image_size else config.TARGET_SIZE
    out = Path(args.out)
    frames = data_loader.split_frames(source="crop")
    manifest = {
        "store_tag": out.name,
        "margin": args.margin,
        "intensity_norm": config.CROP_INTENSITY_NORM,
        "pad_square": config.CROP_PAD_SQUARE,
        "target_size": list(ts),
        "channels": 1,
        "value_range": [0.0, 1.0],
        "candidates_from": str(BUNDLE / "candidates_{split}.csv"),
        "features": list(FEATURES),
        "label_rule": "box IoU >= 0.30 with any GT lesion on the image",
        "partition_seed": config.SEED,
        "splits": {},
        "files": {},
    }
    for split in _ordered(args.split):
        manifest["splits"][split] = export_candidate_split(split, args.margin, ts, out, args.limit)
        manifest["files"][f"{split}_images.npy"] = _sha(out / f"{split}_images.npy")
    if "val" in set(args.split) and args.limit is None:
        manifest["fidelity_check"] = candidate_fidelity(out, frames)
        if not manifest["fidelity_check"].get("passed", True):
            raise SystemExit(
                "[candstore] FIDELITY FAILED: the crop path does not reproduce "
                "crop_store_rung3_unet_bundle; store not trusted"
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {out / 'manifest.json'}")


# notest-copy — a test-free copy of the v1 per-lesion localiser store
def cmd_notest_copy(args) -> int:
    """Copy val and train of the v1 per-lesion store, then assert nothing matching
    `test*` came with them, so the test tensors never reach a training machine."""
    # Train as well as val: the ensemble rule is selected on the train split, so a
    # val-only export would leave every member ungateable.
    src, dst = Path(args.src), Path(args.dst)
    splits = tuple(_ordered(args.split))
    if not src.exists():
        print(f"source store missing: {src}", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)
    wanted = [f"{s}_{k}" for s in splits for k in ("images.npy", "masks.npy", "meta.csv")]
    for name in wanted:
        s, d = src / name, dst / name
        if not s.exists():
            print(f"MISSING in source: {name}", file=sys.stderr)
            return 2
        if d.exists() and d.stat().st_size == s.stat().st_size:
            print(f"  present, size matches, skipping copy: {name}")
            continue
        print(f"  copying {name} ({s.stat().st_size / 2**30:.2f} GiB)", flush=True)
        shutil.copy2(s, d)

    # the assertion this subcommand exists for
    stray = sorted(p.name for p in dst.iterdir() if p.name.lower().startswith("test"))
    extra = sorted(p.name for p in dst.iterdir() if p.name not in wanted)
    print("\n--- assertions ---")
    print(
        f"  [{'PASS' if not stray else 'FAIL'}] no test* file in the export"
        + (f" — found {stray}" if stray else "")
    )
    print(
        f"  [{'PASS' if not extra else 'FAIL'}] nothing but the expected files"
        + (f" — extra {extra}" if extra else "")
    )
    ok = not stray and not extra
    for name in wanted:
        a, b = (src / name).stat().st_size, (dst / name).stat().st_size
        same = a == b
        ok &= same
        print(f"  [{'PASS' if same else 'FAIL'}] {name}: {b:,} B")
    # Row counts, so a truncated copy cannot pass as a complete one.
    for s in splits:
        n_src = len(np.load(src / f"{s}_images.npy", mmap_mode="r"))
        n_dst = len(np.load(dst / f"{s}_images.npy", mmap_mode="r"))
        same = n_src == n_dst
        ok &= same
        print(f"  [{'PASS' if same else 'FAIL'}] {s}: {n_dst} rows (source {n_src})")
    # One checksum, on the smallest payload, so identity is provable not assumed.
    h_src, h_dst = _sha_full(src / "val_meta.csv"), _sha_full(dst / "val_meta.csv")
    ok &= h_src == h_dst
    print(
        f"  [{'PASS' if h_src == h_dst else 'FAIL'}] val_meta.csv sha256 identical — {h_dst[:16]}"
    )

    total = sum((dst / n).stat().st_size for n in wanted)
    print(f"\n  export size: {total / 2**30:.2f} GiB")
    print(f"  -> {dst}")
    print(
        "\n"
        + (
            "EXPORT SAFE TO UPLOAD: contains no test partition."
            if ok
            else "EXPORT FAILED ITS ASSERTIONS — do not upload."
        )
    )
    return 0 if ok else 1


# verify — do the stored rows still equal what the current code produces?
def recheck_crop(store: Path, n: int) -> dict:
    """Byte-level comparison of the first `n` val and train rows against the current pipeline."""
    m = json.loads((store / "manifest.json").read_text())
    # The same source switch the exporter uses: a whole-image store is rebuilt
    # through source="full", never through the crop path.
    src_kw, _ = _crop_kw(
        m.get("box_source"),
        m.get("boxes_dir"),
        m.get("margin"),
        m.get("mask_overlay", False),
        "full" if m.get("source") == "full" else "crop",
    )
    if m.get("source") != "full":
        src_kw["crop_sizing"] = m["crop_sizing"]
    train_ds, val_ds, _, _, _ = _build_datasets(
        "scratch",
        target_size=tuple(m["target_size"]),
        augment=False,
        shuffle_train=False,
        standardize=False,
        **src_kw,
    )
    out = {}
    for split, ds in (("val", val_ds), ("train", train_ds)):
        live, _ = _collect(ds, n)
        stored = np.asarray(np.load(store / f"{split}_images.npy", mmap_mode="r")[: len(live)])
        d = np.abs(live.astype(np.float64) - stored.astype(np.float64))
        out[split] = {
            "rows": int(len(live)),
            "byte_identical": bool(np.array_equal(live, stored)),
            "max_abs_diff": float(d.max()),
            "rows_differing": int((d.reshape(len(d), -1).max(1) > 0).sum()),
        }
    return out


def recheck_fusion(store: Path, n_batches: int) -> dict:
    """The fusion exporter's own check (standardised, atol 1e-5, val only) under the
    current code."""
    from src import fusion

    m = json.loads((store / "manifest.json").read_text())
    run = fusion.FusionRun(
        "vgg16",
        m["pair"],
        m["margin_a"],
        m["box_source"],
        single_mass_only=m["single_mass_only"],
        rows=m["rows"],
        batch_size=None,
        tag_suffix="",
        boxes_dir=m.get("boxes_dir"),
        test_boxes=False,
    )
    try:
        rows = fusion_fidelity(run, store, "val", n_batches=n_batches)
        return {"val": {"rows": int(rows), "passed_atol_1e-5_standardised": True}}
    except SystemExit as e:
        return {"val": {"passed_atol_1e-5_standardised": False, "error": str(e)}}


def cmd_verify(args) -> int:
    _backend()
    report, failed = {}, []
    for name in args.store:
        store = config.OUTPUTS_DIR / name
        kind = "fusion" if name.startswith("fusion_store_") else "crop"
        res = (
            recheck_fusion(store, max(1, args.limit // 32))
            if kind == "fusion"
            else recheck_crop(store, args.limit)
        )
        report[name] = {"kind": kind, **res}
        for split, r in res.items():
            ok = r.get("byte_identical", r.get("passed_atol_1e-5_standardised"))
            if not ok:
                failed.append(f"{name}:{split}")
            detail = (
                f"byte_identical={r['byte_identical']} max|diff|={r['max_abs_diff']:.3e} "
                f"rows_differing={r['rows_differing']}/{r['rows']}"
                if kind == "crop"
                else f"atol-1e-5 standardised: {ok} ({r.get('rows', 0)} rows)"
            )
            print(f"  {name:46} {split:5} {detail}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"-> {args.out}")
    if failed:
        print(f"STORES THAT DO NOT MATCH THE CURRENT PIPELINE: {', '.join(failed)}")
        return 1
    print("all stores match the current pipeline on the rows checked")
    return 0


# norms — scratch-mode normalisation statistics over the training crops
def _load_one(row, path_col, strategy, crop_sizing, crop_lk, mask_lk, full_lk):
    """Return the finalised [0,1] array for one lesion, or None if unmatched."""
    import pydicom

    if crop_sizing == "mask_margin":
        lesion = Path(row[path_col]).parts[0]
        img_name = re.sub(r"_\d+$", "", lesion)
        full_p, mask_p = full_lk.get(img_name), mask_lk.get(lesion)
        if full_p is None or mask_p is None:
            return None
        full = pydicom.dcmread(str(full_p), force=True)
        arr = full.pixel_array.astype(np.float32)
        if getattr(full, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
        mask = pydicom.dcmread(str(mask_p), force=True).pixel_array
        arr = data_loader._crop_to_mask_bbox(arr, mask, config.CROP_MARGIN_FRAC)
    else:
        p = crop_lk.get(data_loader._resolve_lookup_key(row[path_col], "crop"))
        if p is None:
            return None
        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
    arr = data_loader._normalise_intensity(arr, strategy)
    return data_loader._pad_and_resize(arr, config.TARGET_SIZE)


def cmd_norms(args) -> None:
    """Global pixel mean and std in [0,1] over the training crops alone, which feed the
    per-strategy scratch stats in config; held-out crops never inform normalisation. Each
    crop is zero-padded to a square, which is how the stored stats were measured; the
    pipeline stretches crops to the input instead."""
    dl = _backend()

    path_col = "cropped image file path"
    pool = dl._pool_frames(
        dl._prepare_frame(config.TRAIN_CSV, path_col),
        dl._prepare_frame(config.TEST_CSV, path_col),
        path_col,
    )
    train_split, _, _ = dl._three_way_split(
        pool, test_fraction=config.TEST_FRACTION, val_fraction=config.VAL_FRACTION, seed=config.SEED
    )

    crop_lk = dl._build_combined_roi_lookup(config.TRAIN_ROI_DIR, config.TEST_ROI_DIR, pick="crop")
    mask_lk = full_lk = {}
    if args.crop_sizing == "mask_margin":
        mask_lk = dl._build_combined_roi_lookup(
            config.TRAIN_ROI_DIR, config.TEST_ROI_DIR, pick="mask"
        )
        _full = dl._build_combined_lookup(config.TRAIN_IMG_DIR, config.TEST_IMG_DIR)
        full_lk = {Path(k).parts[0]: v for k, v in _full.items()}

    total_sum = total_sqsum = total_n = 0.0
    loaded = skipped = 0
    for _, row in train_split.iterrows():
        arr = _load_one(row, path_col, args.strategy, args.crop_sizing, crop_lk, mask_lk, full_lk)
        if arr is None:
            skipped += 1
            continue
        total_sum += float(arr.sum())
        total_sqsum += float((arr.astype(np.float64) ** 2).sum())
        total_n += arr.size
        loaded += 1

    if total_n == 0:
        raise RuntimeError(
            f"No crops loaded ({skipped} skipped) — the lookups resolved "
            f"nothing. Check TRAIN_ROI_DIR / TEST_ROI_DIR in config."
        )
    if skipped > 0.01 * (loaded + skipped):
        print(
            f"WARNING: {skipped} of {loaded + skipped} lesions unmatched "
            f"({skipped / (loaded + skipped):.2%}) — stats are computed on "
            f"a biased subset if these are not random."
        )

    mean = total_sum / total_n
    std = float(np.sqrt(max(total_sqsum / total_n - mean**2, 0.0)))
    print(f"\nstrategy={args.strategy} crop_sizing={args.crop_sizing}")
    print(f"Crops loaded: {loaded} skipped: {skipped}")
    print(f"mean = {mean:.4f}")
    print(f"std  = {std:.4f}")

    const = {"percentile": "PERCENTILE", "minmax": "MINMAX", "clahe": "CLAHE"}.get(
        args.strategy, "CROP"
    )
    matches_pipeline = (
        args.strategy == config.CROP_INTENSITY_NORM and args.crop_sizing == config.CROP_SIZING
    )
    if matches_pipeline:
        print("\nPaste into config.py (strategy and crop sizing match; padding is not checked):")
        print(f"  {const}_NORM_MEAN: float = {mean:.4f}")
        print(f"  {const}_NORM_STD: float  = {std:.4f}")
    else:
        print(
            f"\nDIAGNOSTIC RUN — do not paste. This measured "
            f"strategy={args.strategy}/{args.crop_sizing}, but the pipeline "
            f"runs {config.CROP_INTENSITY_NORM}/{config.CROP_SIZING}, so "
            f"these stats do not describe it."
        )


# CLI
def _add_split(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--split",
        nargs="+",
        default=["val", "train"],
        choices=["val", "train"],
        help="one or more of val and train; test is never exported",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("crop", help="per-lesion crops, or the whole image with --source full")
    p.add_argument("--margin", type=float, default=config.CROP_MARGIN_FRAC)
    p.add_argument("--mask-overlay", action="store_true")
    p.add_argument(
        "--out",
        default=None,
        help="default outputs/crop_store_{--tag}; without --tag, crop_store_full_image "
        "for --source full, else crop_store_m{margin x 100, three digits}[_overlay]",
    )
    p.add_argument(
        "--box-source",
        default=None,
        choices=["oracle", "unet", "cam", "jitter", "shuffled", "whole"],
        help="export a ladder condition instead of the oracle mask_margin "
        "crops; sets crop_sizing to box_provider",
    )
    p.add_argument(
        "--boxes-dir", default=None, help="candidate boxes folder for --box-source unet|jitter"
    )
    p.add_argument(
        "--tag",
        default=None,
        help="store name suffix, e.g. rung3_unet_bundle; required with "
        "--box-source so conditions do not collide",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="square crop resolution; default config.TARGET_SIZE (224). "
        "Same rows and boxes, only the final resize changes.",
    )
    p.add_argument(
        "--source",
        choices=["crop", "full"],
        default="crop",
        help="full: the whole mammogram, one row per image, built by the same "
        "data_loader.build_datasets call the full-image trainer and "
        "src.evaluate --source full use. Default crop: unchanged.",
    )
    p.add_argument("--limit", type=int, default=None, help="smoke: first N rows per split")
    _add_split(p)
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("fusion", help="the paired stores: view, context, control")
    p.add_argument("--model", default="vgg16")
    p.add_argument("--pair", required=True, choices=["view", "context", "control"])
    p.add_argument("--rows", default=None, choices=["paired", "all"])
    p.add_argument("--margin", type=float, default=config.CROP_MARGIN_FRAC)
    p.add_argument("--box-source", default="oracle", choices=["oracle", "unet"])
    p.add_argument("--boxes-dir", default=None)
    p.add_argument("--single-mass-only", action="store_true")
    p.add_argument("--tag", default=None, help="store name; default: the fusion tag")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke: first N pairs per split")
    _add_split(p)
    p.set_defaults(func=cmd_fusion)

    p = sub.add_parser("candidates", help="one row per localiser candidate component")
    p.add_argument("--margin", type=float, default=config.CROP_MARGIN_FRAC)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument(
        "--out",
        required=True,
        help="store directory, e.g. outputs/candidate_store_rung3_unet_bundle",
    )
    p.add_argument("--limit", type=int, default=None, help="smoke: first N candidates")
    _add_split(p)
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("notest-copy", help="test-free copy of the v1 per-lesion localiser store")
    p.add_argument("--src", default=str(config.OUTPUTS_DIR / "localiser_tensors"))
    p.add_argument("--dst", default=str(config.OUTPUTS_DIR / "localiser_tensors_v1_notest"))
    _add_split(p)
    p.set_defaults(func=cmd_notest_copy)

    p = sub.add_parser("verify", help="re-check stored rows against the current code")
    p.add_argument("--store", nargs="+", required=True, help="store directory names under outputs/")
    p.add_argument("--limit", type=int, default=64, help="crop stores: rows per split")
    p.add_argument("--out", default=str(config.RESULTS_DIR / "store_fidelity_recheck.json"))
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "norms",
        help="scratch-mode normalisation stats over the training crops, zero-padded to "
        "squares as the stored stats were; the pipeline stretches crops instead",
    )
    p.add_argument(
        "--strategy",
        default=config.CROP_INTENSITY_NORM,
        choices=["fixed_max", "percentile", "minmax", "clahe"],
    )
    p.add_argument(
        "--crop-sizing", default=config.CROP_SIZING, choices=["provided_tight", "mask_margin"]
    )
    p.set_defaults(func=cmd_norms)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())

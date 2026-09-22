"""Precompute letterboxed (image, mask) tensors for the localiser: one pass over
the DICOMs, after which training reads arrays instead of decoding, and the store
is portable to another machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_loader import (
    _prepare_frame,
    _pool_frames,
    _three_way_split,
    _read_full,
    _normalise_intensity,
    _build_combined_roi_lookup,
    _build_combined_lookup,
)
from .localise import letterbox
from pathlib import Path
import pydicom

OUT = config.OUTPUTS_DIR / "localiser_tensors"


def _read_mask(path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    m = ds.pixel_array
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        m = m.max() - m
    return (m > (m.max() * 0.5 if m.max() > 0 else 0)).astype(np.float32)


def main(
    merged: bool = False,
    splits=("train", "val", "test"),
    out_dir: str | None = None,
    canvas_h: int = config.LOC_INPUT_H,
    canvas_w: int = config.LOC_INPUT_W,
) -> None:
    """Letterboxed (image, mask) tensors, one row per lesion; merged=True puts every
    lesion of a mammogram in one mask, so training rows carry no contradictory masks.
    """
    # out_dir lets a rebuild write beside an existing cache instead of over it.
    out = Path(out_dir) if out_dir else (config.LOC_TENSORS_MERGED_DIR if merged else OUT)
    out.mkdir(parents=True, exist_ok=True)
    config.set_seeds(config.SEED)

    pool = _pool_frames(
        _prepare_frame(config.TRAIN_CSV, "cropped image file path"),
        _prepare_frame(config.TEST_CSV, "cropped image file path"),
        "cropped image file path",
    )
    frames = dict(
        zip(
            ("train", "val", "test"),
            _three_way_split(
                pool,
                test_fraction=config.TEST_FRACTION,
                val_fraction=config.VAL_FRACTION,
                seed=config.SEED,
            ),
        )
    )

    masks = _build_combined_roi_lookup(config.TRAIN_ROI_DIR, config.TEST_ROI_DIR, pick="mask")
    fulls = {
        Path(k).parts[0]: v
        for k, v in _build_combined_lookup(config.TRAIN_IMG_DIR, config.TEST_IMG_DIR).items()
    }

    for name in splits:
        frame = frames[name]
        if merged:
            groups = [(img, g) for img, g in frame.groupby("image_id", sort=True)]
        else:
            groups = [(r.image_id, frame.iloc[[i]]) for i, r in enumerate(frame.itertuples())]
        n = len(groups)
        imgs = np.zeros((n, canvas_h, canvas_w), np.float16)
        msks = np.zeros((n, canvas_h, canvas_w), np.uint8)
        meta = []
        for i, (image_id, g) in enumerate(groups):
            full = _normalise_intensity(
                _read_full(str(fulls[image_id])),
                "percentile",
                percentile_high=config.LOC_PERCENTILE_HIGH,
            )
            img_lb, lb = letterbox(full, canvas_h, canvas_w)
            union = np.zeros((canvas_h, canvas_w), np.float32)
            for lesion_id in g["lesion_id"]:
                msk_lb, _ = letterbox(
                    _read_mask(masks[lesion_id]), canvas_h, canvas_w, is_mask=True
                )
                union = np.maximum(union, msk_lb)
            imgs[i] = np.clip(img_lb, 0, 1).astype(np.float16)
            msks[i] = (union > 0.5).astype(np.uint8)
            labels = sorted(set(int(v) for v in g["label"]))
            meta.append(
                {
                    "lesion_id": (
                        g["lesion_id"].iloc[0] if not merged else "|".join(g["lesion_id"])
                    ),
                    "image_id": image_id,
                    "patient_id": g["patient_id"].iloc[0],
                    "label": max(labels),
                    "label_mixed": len(labels) > 1,
                    "n_lesions": int(len(g)),
                    **lb.as_dict(),
                }
            )
            if (i + 1) % 100 == 0:
                print(f"  {name}: {i + 1}/{n}")

        np.save(out / f"{name}_images.npy", imgs)
        np.save(out / f"{name}_masks.npy", msks)
        pd.DataFrame(meta).to_csv(out / f"{name}_meta.csv", index=False)
        print(
            f"{name}: {n} {'images' if merged else 'lesion rows'} "
            f"({len(frame)} lesions), "
            f"{(imgs.nbytes + msks.nbytes) / 1e9:.2f} GB, "
            f"mean fg {msks.mean():.5f}"
        )

    print(f"\nWritten to {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged",
        action="store_true",
        help="one row per mammogram, all its lesions in one mask",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument(
        "--canvas-h",
        type=int,
        default=config.LOC_INPUT_H,
        help="letterbox canvas height. The Letterbox record in the metadata carries "
        "the scale, so boxes map back to mammogram coordinates at any canvas size.",
    )
    ap.add_argument(
        "--canvas-w", type=int, default=config.LOC_INPUT_W, help="letterbox canvas width"
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write here instead of the default store (outputs/localiser_tensors, "
        "or outputs/localiser_tensors_merged with --merged)",
    )
    a = ap.parse_args()
    main(
        merged=a.merged,
        splits=tuple(a.splits),
        out_dir=a.out_dir,
        canvas_h=a.canvas_h,
        canvas_w=a.canvas_w,
    )

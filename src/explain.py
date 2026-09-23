"""Does the crop classifier attend to the annotated lesion? Grad-CAM shows
where the evidence for the model's call lies; it is compared here with the
ground-truth ROI mask inside the same crop, split by whether the call was correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from scipy.stats import mannwhitneyu

from . import config
from .cam_localise import _split_model, grad_cam
from .data_loader import (
    _build_combined_lookup,
    _build_combined_roi_lookup,
    _crop_box,
    _normalise_intensity,
    _pad_and_resize,
    _read_full,
    _scratch_norm_stats,
)
from .evaluate import optimal_threshold
from .models import MODEL_REGISTRY, build_model

ART = Path(__file__).resolve().parent.parent / "artifacts"
_DEVICE = "/GPU:0"


def _read_mask(path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    m = ds.pixel_array
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        m = m.max() - m
    return (m > (m.max() * 0.5 if m.max() > 0 else 0)).astype(np.float32)


def _match_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a mask to the mammogram grid; some ROI masks are
    stored at a different scale from their mammogram.
    """
    if mask.shape == tuple(shape):
        return mask
    with tf.device("/CPU:0"):
        r = tf.image.resize(mask[..., None], list(shape), method="nearest").numpy()[..., 0]
    return (r > 0.5).astype(np.float32)


def _to_canvas(arr: np.ndarray, pad_square: bool) -> np.ndarray:
    return _pad_and_resize(arr, config.TARGET_SIZE, pad_square=pad_square)[..., 0]


def build_input(
    full: np.ndarray, mask: np.ndarray, box, margin: float, normalisation: str, intensity_norm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(model input, display crop in [0,1], mask on the canvas, mask_in_crop), built by
    the training pipeline's own functions; mask_in_crop is how much lesion the crop holds.
    """
    mask = _match_shape(mask, full.shape)
    if box is None:  # fallback: whole mammogram, padded
        crop, mcrop, pad = full, mask, True
    else:
        crop, mcrop, pad = (_crop_box(full, box, margin), _crop_box(mask, box, margin), False)
    mask_in_crop = float(mcrop.sum() / max(mask.sum(), 1.0))
    disp = _normalise_intensity(crop, intensity_norm)
    canvas = _to_canvas(disp, pad)  # [224,224] in [0,1]
    mcanvas = (_to_canvas(mcrop, pad) > 0.5).astype(np.float32)
    x = canvas[..., None]
    if normalisation == "imagenet":
        x = np.repeat(x, 3, axis=-1)
        mean = np.asarray(config.NORM_STATS["imagenet_mean"], np.float32)
        std = np.asarray(config.NORM_STATS["imagenet_std"], np.float32)
    else:
        mean, std = _scratch_norm_stats(intensity_norm, "crop")
    return ((x - mean) / std).astype(np.float32), canvas, mcanvas, mask_in_crop


def overlap(cam: np.ndarray, mask: np.ndarray, tau: float) -> dict:
    """CAM against the lesion mask: attention_in_mask is the threshold-free fraction of
    CAM mass inside it; cam_iou uses the CAM binarised at `tau` x its maximum.
    """
    total = float(cam.sum())
    inside = float((cam * mask).sum()) / total if total > 0 else float("nan")
    b = cam > tau * cam.max() if cam.max() > 0 else np.zeros_like(cam, bool)
    m = mask > 0.5
    inter, union = float((b & m).sum()), float((b | m).sum())
    ys, xs = np.nonzero(cam == cam.max()) if cam.max() > 0 else (np.array([0]), np.array([0]))
    peak_in_mask = bool(m[int(ys.mean()), int(xs.mean())]) if m.any() else False
    return dict(
        attention_in_mask=inside,
        cam_iou=inter / union if union else 0.0,
        peak_in_mask=peak_in_mask,
        mask_frac_of_canvas=float(m.mean()),
        peak_y=int(ys.mean()),
        peak_x=int(xs.mean()),
    )


def run(
    model_name: str,
    tag: str,
    weights: Path,
    *,
    split: str,
    source: str,
    margin: float,
    tau: float,
    layer: str | None,
    n_show: int = 2,
    boxes_dir: str | None = None,
) -> dict:
    rows = pd.read_csv(ART / f"localiser_boxes_{split}.csv")
    fulls = {
        Path(k).parts[0]: v
        for k, v in _build_combined_lookup(config.TRAIN_IMG_DIR, config.TEST_IMG_DIR).items()
    }
    masks = _build_combined_roi_lookup(config.TRAIN_ROI_DIR, config.TEST_ROI_DIR, pick="mask")
    from .boxes import provider_for

    provider = provider_for(
        source,
        boxes_dir=boxes_dir,
        splits=("train", "val", "test") if split == "test" else ("train", "val"),
    )
    normalisation = MODEL_REGISTRY[model_name]["normalisation"]
    intensity_norm = config.CROP_INTENSITY_NORM

    with tf.device(_DEVICE):
        model = build_model(model_name)
        model.load_weights(str(weights))
        forward, layer_name = _split_model(model, layer)
    print(
        f"[explain] {tag} <- {weights.name}; CAM layer {layer_name}; boxes={source}; "
        f"margin={margin:.2f}; split={split}"
    )

    recs, keep, skipped = [], {}, 0
    for i, r in enumerate(rows.itertuples()):
        fp, mp = fulls.get(r.image), masks.get(r.lesion_id)
        if fp is None or mp is None:
            skipped += 1
            continue
        full = _read_full(str(fp))
        box = provider.get(r.lesion_id, full.shape)
        x, disp, mcanvas, mic = build_input(
            full, _read_mask(mp), box, margin, normalisation, intensity_norm
        )
        with tf.device(_DEVICE):
            cam, p = grad_cam(forward, tf.constant(x[None]))
            cam = tf.image.resize(
                cam[..., None], list(config.TARGET_SIZE), method="bilinear"
            ).numpy()[0, ..., 0]
        rec = dict(
            split=split,
            lesion_id=r.lesion_id,
            patient_id=r.patient_id,
            label=int(r.label),
            y_prob=float(p[0]),
            box_source=source,
            detected=box is not None,
            mask_in_crop=mic,
            **overlap(cam, mcanvas, tau),
        )
        bright = disp >= np.quantile(disp, 0.99)
        rec["peak_on_bright_1pct"] = bool(bright[rec["peak_y"], rec["peak_x"]])
        recs.append(rec)
        keep[r.lesion_id] = (disp, cam, mcanvas)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")
    if skipped:
        print(f"[explain] skipped {skipped} of {len(rows)} lesions with no mammogram or mask file")

    df = pd.DataFrame(recs)
    # The operating point always comes from validation: on test it is the threshold
    # src.evaluate fitted there, on validation it is fitted here on the same rows.
    if split == "test":
        thr = float(
            pd.read_csv(config.RESULTS_DIR / f"{tag}_predictions_val.csv")["threshold"].iloc[0]
        )
        thr_source = "validation (src.evaluate)"
    else:
        thr = optimal_threshold(
            df["label"].to_numpy(),
            df["y_prob"].to_numpy(),
            target_sensitivity=config.TARGET_SENSITIVITY,
        )
        thr_source = "validation"
    df["y_pred"] = (df["y_prob"] >= thr).astype(int)
    df["correct"] = df["y_pred"] == df["label"]
    df["outcome"] = np.select(
        [(df.label == 1) & df.correct, (df.label == 0) & df.correct, (df.label == 0) & ~df.correct],
        ["TP", "TN", "FP"],
        default="FN",
    )
    out_csv = config.RESULTS_DIR / f"explain_{tag}_{split}.csv"
    df.to_csv(out_csv, index=False)

    def _grp(mask):
        s = df[mask]
        return dict(
            n=int(len(s)),
            attention_in_mask=float(s["attention_in_mask"].mean()),
            cam_iou=float(s["cam_iou"].mean()),
            peak_in_mask=float(s["peak_in_mask"].mean()),
            mask_in_crop=float(s["mask_in_crop"].mean()),
            peak_on_bright=float(s["peak_on_bright_1pct"].mean()),
        )

    # Conditioned within the predicted class: pooling across classes confounds correctness
    # with class composition. grad_cam targets the class on its side of p = 0.5, so the
    # predicted_malignant cases with threshold <= p < 0.5 carry a CAM for benign.
    within = {
        "predicted_malignant": dict(TP=_grp(df.outcome == "TP"), FP=_grp(df.outcome == "FP")),
        "predicted_benign": dict(TN=_grp(df.outcome == "TN"), FN=_grp(df.outcome == "FN")),
    }
    a_ok, a_bad = (
        df.loc[df.correct, "attention_in_mask"].dropna(),
        df.loc[~df.correct, "attention_in_mask"].dropna(),
    )
    u = mannwhitneyu(a_ok, a_bad, alternative="two-sided") if len(a_ok) and len(a_bad) else None
    n_nan = int(df["attention_in_mask"].isna().sum())
    if n_nan:
        print(
            f"[explain] {n_nan} lesions with an all-zero CAM have no attention_in_mask; "
            "left out of its means and the Mann-Whitney test"
        )
    # Grounding conditioned on the lesion being inside the crop: where the box excludes
    # it, attention_in_mask is mechanically 0 and measures the localiser, not the CAM.
    present = df["mask_in_crop"] > 0
    conditioned = {
        "mask_in_crop_gt_0": _grp(present),
        "mask_in_crop_ge_0.5": _grp(df["mask_in_crop"] >= 0.5),
        "mask_in_crop_ge_0.9": _grp(df["mask_in_crop"] >= 0.9),
        "n_crops_excluding_lesion": int((~present).sum()),
        "frac_crops_excluding_lesion": round(float((~present).mean()), 4),
    }
    summary = dict(
        tag=tag,
        split=split,
        box_source=source,
        margin=margin,
        tau=tau,
        cam_layer=layer_name,
        threshold=float(thr),
        threshold_source=thr_source,
        n=int(len(df)),
        n_correct=int(df.correct.sum()),
        chance_attention=float(df["mask_frac_of_canvas"].mean()),
        all=_grp(df.index == df.index),
        correct=_grp(df.correct),
        incorrect=_grp(~df.correct),
        conditioned_on_mask_in_crop=conditioned,
        by_outcome={o: _grp(df.outcome == o) for o in ("TP", "TN", "FP", "FN")},
        mannwhitney_attention_correct_vs_incorrect=(
            None if u is None else dict(U=float(u.statistic), p=float(u.pvalue))
        ),
        within_predicted_class=within,
    )
    (config.RESULTS_DIR / f"explain_{tag}_{split}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n--- {tag} on {split} (n={len(df)}, thr {thr:.3f}) ---")
    print(f"chance level (mask fraction of canvas): {summary['chance_attention']:.3f}")
    g_all, g_in = summary["all"], conditioned["mask_in_crop_gt_0"]
    print(
        f"attention_in_mask {g_all['attention_in_mask']:.3f} unconditioned "
        f"-> {g_in['attention_in_mask']:.3f} on the {g_in['n']} crops that contain the "
        f"lesion ({conditioned['n_crops_excluding_lesion']} of {len(df)} exclude it "
        f"entirely, {conditioned['frac_crops_excluding_lesion']:.1%})"
    )
    for k in ("correct", "incorrect"):
        g = summary[k]
        print(
            f"{k:<9} n={g['n']:3d} attention_in_mask {g['attention_in_mask']:.3f} "
            f"cam_iou {g['cam_iou']:.3f} peak_in_mask {g['peak_in_mask']:.3f} "
            f"mask_in_crop {g['mask_in_crop']:.3f}"
        )
    for o in ("TP", "TN", "FP", "FN"):
        g = summary["by_outcome"][o]
        print(
            f"  {o} n={g['n']:3d} attention_in_mask {g['attention_in_mask']:.3f} "
            f"cam_iou {g['cam_iou']:.3f}"
        )
    print(
        "CAM peak on top-1% brightest pixels: "
        f"{df['peak_on_bright_1pct'].mean():.3f} overall, "
        f"{df.loc[df.label == 1, 'peak_on_bright_1pct'].mean():.3f} malignant, "
        f"{df.loc[df.label == 0, 'peak_on_bright_1pct'].mean():.3f} benign"
    )
    if u is not None:
        print(f"Mann-Whitney attention_in_mask correct vs incorrect: p = {u.pvalue:.4f}")
    else:
        print("Mann-Whitney attention_in_mask correct vs incorrect: skipped, a group is empty")

    # Overlay figure: per outcome, the n_show most confident cases, highest y_prob for TP
    # and FP and lowest for TN and FN.
    fig, axes = plt.subplots(4, n_show, figsize=(2.6 * n_show, 10.5), squeeze=False)
    for row_i, o in enumerate(("TP", "TN", "FP", "FN")):
        ids = (
            df[df.outcome == o]
            .sort_values("y_prob", ascending=(o in ("TN", "FN")))["lesion_id"]
            .tolist()[:n_show]
        )
        for col_i in range(n_show):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i >= len(ids):
                continue
            disp, cam, m = keep[ids[col_i]]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam, cmap="jet", alpha=0.35, vmin=0, vmax=1)
            ax.contour(m, levels=[0.5], colors="lime", linewidths=1.0)
            rr = df[df.lesion_id == ids[col_i]].iloc[0]
            ax.set_title(f"{o}  p={rr.y_prob:.2f}  att={rr.attention_in_mask:.2f}", fontsize=8)
    fig.suptitle(f"{tag}: Grad-CAM vs ground-truth mask (green), {split}", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"gradcam_{tag}_{split}.png", dpi=200)
    plt.close(fig)
    print(f"-> {out_csv}")
    return summary


def _refuse_test_outside_pass(split: str) -> None:
    """Refuses the test split unless the test pass has set its token
    (test_pass_guard.authorised)."""
    from .test_pass_guard import authorised

    if split == "test" and not authorised():
        raise SystemExit(
            "REFUSED: --split test decodes test mammograms and masks; only "
            "notebooks/scripts/run_test_pass.sh may run it"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--tag-suffix", default="", help="appended to {model}_crop to form the tag")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument(
        "--box-source",
        default="oracle",
        choices=["oracle", "unet", "cam", "jitter", "shuffled", "whole"],
    )
    ap.add_argument("--margin", type=float, default=config.CROP_MARGIN_FRAC)
    ap.add_argument("--tau", type=float, default=config.CAM_THRESHOLD)
    ap.add_argument("--layer", default=None)
    ap.add_argument(
        "--boxes-dir",
        default=None,
        help="candidate localiser folder: localiser_boxes_{split}.csv for --box-source unet, "
        "localiser_error_model.json for jitter; default: the frozen artefacts",
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag "
        "that no longer exists fails in a second.",
    )
    args = ap.parse_args()
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return

    global _DEVICE
    if args.cpu:
        _DEVICE = "/CPU:0"
    _refuse_test_outside_pass(args.split)
    if args.split == "test":
        print(
            "[explain] WARNING: test split requested - only under the pre-registered "
            "single test pass."
        )
    config.set_seeds()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_crop{args.tag_suffix}"
    # config.resolve_weights searches WEIGHTS_DIR, then the CUDA landing folders, and
    # names every folder it searched when it finds nothing.
    if args.weights:
        weights = Path(args.weights)
        if not weights.exists():
            raise SystemExit(f"weights not found: {weights}")
    else:
        try:
            weights = config.resolve_weights(tag)
        except FileNotFoundError as e:
            raise SystemExit(str(e))
    run(
        args.model,
        tag,
        weights,
        split=args.split,
        source=args.box_source,
        margin=args.margin,
        tau=args.tau,
        layer=args.layer,
        boxes_dir=args.boxes_dir,
    )


if __name__ == "__main__":
    main()

"""The localisation-degradation ladder: seven training runs of one architecture that
differ only in the crop box: oracle tight, oracle at the margin m*, localiser box,
CAM box, jittered box, shuffled box, whole image.
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

from . import config
from .boxes import LADDER, BoxProvider, provider_for
from .localise import box_iou

ART = Path(__file__).resolve().parent.parent / "artifacts"


# k > 1 trains the localiser-box rung on the greedily matched top-k candidate instead
# of the box table's single box, under its own tag suffix; 1 = single box.
RUNG3_K: int = 1
RUNG3_BOXES_DIR: str | None = None
# Tag suffix of the localiser-box rung, and of the jittered-box rung when
# RUNG3_BOXES_DIR is set, e.g. "_bundle"; "" leaves those tags bare.
RUNG3_SUFFIX: str = ""
# Suffix for every other rung (1, 2, 4, 6, 7, and 5 without RUNG3_BOXES_DIR), so they
# come from the same ledger as the localiser-box and jittered-box rungs.
LEDGER_SUFFIX: str = ""


def rung_tag(arch: str, rung: int) -> str:
    source = LADDER[rung][0]
    suffix = LEDGER_SUFFIX
    if rung == 3:
        suffix = (f"_k{RUNG3_K}" if RUNG3_K > 1 else "") + RUNG3_SUFFIX
    elif rung == 5 and RUNG3_BOXES_DIR:
        suffix = (
            RUNG3_SUFFIX  # the jittered box mirrors the candidate's error, so it shares the suffix
        )
    return f"{arch}_crop_rung{rung}_{source}{suffix}"


# The shuffled box is always built on all three splits, the provider it is trained on;
# train/val donors do not depend on the test table (boxes.BoxProvider).
SHUFFLED_SPLITS: tuple[str, ...] = ("train", "val", "test")


def rung_provider(source: str, splits=("train", "val", "test")) -> BoxProvider:
    """The box provider for a rung; `splits` limits which box tables are read, except
    for the shuffled box, which always reads SHUFFLED_SPLITS.
    """
    if source == "shuffled":
        return provider_for(source, boxes_dir=None, splits=SHUFFLED_SPLITS)
    if source == "unet" and RUNG3_K > 1:
        return BoxProvider(
            "unet", k=RUNG3_K, mode="matched", boxes_dir=RUNG3_BOXES_DIR, splits=splits
        )
    return provider_for(source, boxes_dir=RUNG3_BOXES_DIR, splits=splits)


def provider_splits(source: str, split: str) -> tuple[str, ...]:
    """The split set measured_iou builds a rung's provider on (recorded in the CSV)."""
    if source == "shuffled":
        return SHUFFLED_SPLITS
    if RUNG3_BOXES_DIR and split != "test":
        return ("train", "val")
    return ("train", "val", "test")


def rung_margin(rung: int, m_star: float) -> float:
    m = LADDER[rung][1]
    return m_star if m is None else float(m)


# Measured localisation quality of each rung's boxes (validation)


def measured_iou(source: str, split: str = "val") -> dict:
    """Miss-inclusive box IoU of a box source against the ground-truth boxes of `split`,
    from the same artefacts the crops are cut from. Oracle scores 1, the whole image 0.
    """
    gt_path = ART / f"localiser_boxes_{split}.csv"
    gt = pd.read_csv(gt_path)
    gt = gt[gt["status"] != "no_gt"]
    gt_source = str(gt_path.relative_to(ART.parent))
    if source == "oracle":
        return dict(
            iou=1.0, detection=1.0, n=int(len(gt)), gt_source=gt_source, provider_splits="-"
        )
    if source == "whole":
        return dict(
            iou=0.0, detection=0.0, n=int(len(gt)), gt_source=gt_source, provider_splits="-"
        )
    # With a candidate folder, validation builds the provider on train + val, since the
    # folder's test table is written for the test stage; that stage needs the table, or
    # every test lesion scores as a miss, IoU 0.
    splits = provider_splits(source, split)
    prov = rung_provider(source, splits=splits)
    ious, det = [], 0
    for r in gt.itertuples():
        g = (int(r.gt_y0), int(r.gt_y1), int(r.gt_x0), int(r.gt_x1))
        b = prov.get(r.lesion_id, (int(r.orig_h), int(r.orig_w)))
        if b is None:
            ious.append(0.0)
        else:
            det += 1
            ious.append(box_iou(b, g))
    boxes_from = (
        "candidate folder " + str(RUNG3_BOXES_DIR)
        if (RUNG3_BOXES_DIR and source in ("unet", "jitter"))
        else "artifacts/"
    )
    return dict(
        iou=float(np.mean(ious)),
        detection=det / len(gt),
        n=int(len(gt)),
        gt_source=gt_source,
        provider_splits="+".join(splits),
        boxes_from=boxes_from,
    )


# Stages


def stage_train(arch: str, m_star: float, rungs: list[int]) -> None:
    from .models import _SCRATCH_MODELS
    from .train import train_baseline, train_transfer, _ensure_dirs

    _ensure_dirs()
    for k in rungs:
        source, _, desc = LADDER[k]
        tag = rung_tag(arch, k)
        if (config.WEIGHTS_DIR / f"{tag}_best.weights.h5").exists():
            print(f"[ladder] rung {k} ({source}) already trained -> {tag}; skipping")
            continue
        if source == "cam":
            man = ART / "cam_manifest.json"
            if not man.exists():
                raise FileNotFoundError(
                    "rung 4 needs artifacts/cam_boxes.csv from "
                    "src.cam_localise on the whole-image transfer model"
                )
            m = json.loads(man.read_text())
            if m.get("model") in _SCRATCH_MODELS:
                raise RuntimeError(
                    f"cam_boxes.csv was produced from the scratch model "
                    f"'{m.get('model')}' - regenerate from the transfer model"
                )
        margin = rung_margin(k, m_star)
        print(f"\n[ladder] rung {k}: {desc} (source={source}, margin={margin:.2f}) -> {tag}")
        kwargs = dict(
            source="crop",
            crop_sizing="box_provider",
            box_provider=rung_provider(source, splits=("train", "val")),
            crop_margin=margin,
            tag_suffix=tag.replace(f"{arch}_crop", "", 1),
        )
        if arch in _SCRATCH_MODELS:
            train_baseline(arch, **kwargs)
        else:
            train_transfer(arch, **kwargs)


def stage_val(arch: str, m_star: float, rungs: list[int]) -> pd.DataFrame:
    from .models import _SCRATCH_MODELS

    rows = []
    for k in rungs:
        source, _, desc = LADDER[k]
        tag = rung_tag(arch, k)
        hist = config.RESULTS_DIR / (
            f"{tag}_history.json" if arch in _SCRATCH_MODELS else f"{tag}_finetune_history.json"
        )
        if not hist.exists():
            print(f"[ladder] rung {k}: no history at {hist.name}; skipping")
            continue
        h = json.loads(hist.read_text())
        va, vl = np.asarray(h["val_auc"]), np.asarray(h["val_loss"])
        loc = measured_iou(source)
        # Seed replicates, tags {tag}_s<N>: the spread of their best val AUC is the
        # only variance estimate the validation figure has.
        seed_best = [float(va.max())]
        for sp in sorted(config.RESULTS_DIR.glob(f"{tag}_s[0-9]*{hist.name[len(tag):]}")):
            hs = json.loads(sp.read_text())
            seed_best.append(float(np.asarray(hs["val_auc"]).max()))
        rows.append(
            dict(
                rung=k,
                source=source,
                description=desc,
                margin=rung_margin(k, m_star),
                tag=tag,
                val_auc_best=float(va.max()),
                epoch_best=int(va.argmax()),
                val_auc_at_min_loss=float(va[vl.argmin()]),
                box_iou_val=loc["iou"],
                detection_val=loc["detection"],
                iou_gt_source=loc["gt_source"],
                iou_provider_splits=loc["provider_splits"],
                iou_boxes_from=loc.get("boxes_from", "-"),
                n_seeds=len(seed_best),
                val_auc_seed_mean=float(np.mean(seed_best)),
                val_auc_seed_sd=(
                    float(np.std(seed_best, ddof=1)) if len(seed_best) > 1 else np.nan
                ),
            )
        )
    if not rows:
        raise SystemExit(f"REFUSED: no rung of {arch} has a history; nothing written")
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / f"ladder_{arch}_val.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    seeded = df["n_seeds"] > 1
    if seeded.any():
        # Error bar = ±1 seed SD where replicates exist, none elsewhere.
        df["_lo"] = df["val_auc_best"] - df["val_auc_seed_sd"].fillna(0.0)
        df["_hi"] = df["val_auc_best"] + df["val_auc_seed_sd"].fillna(0.0)
        title = (
            f"{arch}: validation AUC across the ladder (single seed; "
            f"±1 seed SD on rungs {', '.join(str(r) for r in df.loc[seeded, 'rung'])})"
        )
        _plot(
            df,
            arch,
            y="val_auc_best",
            lo="_lo",
            hi="_hi",
            title=title,
            path=config.FIGURES_DIR / f"ladder_{arch}_val.png",
        )
    else:
        _plot(
            df,
            arch,
            y="val_auc_best",
            lo=None,
            hi=None,
            title=f"{arch}: validation AUC across the ladder (single seed, no CI)",
            path=config.FIGURES_DIR / f"ladder_{arch}_val.png",
        )
    return df


def stage_test(arch: str, m_star: float, rungs: list[int]) -> pd.DataFrame:
    from .analysis import auc_ci

    rows = []
    for k in rungs:
        source, _, desc = LADDER[k]
        tag = rung_tag(arch, k)
        tbl = config.RESULTS_DIR / f"{tag}_predictions_test.csv"
        if not tbl.exists():
            print(f"[ladder] rung {k}: no test table for {tag}; skipping")
            continue
        r = auc_ci(pd.read_csv(tbl))
        loc = measured_iou(source, split="test")
        rows.append(
            dict(
                rung=k,
                source=source,
                description=desc,
                tag=tag,
                margin=rung_margin(k, m_star),
                auc=r["auc"],
                ci_lo=r["ci"][0],
                ci_hi=r["ci"][1],
                n=r["n"],
                box_iou_test=loc["iou"],
                detection_test=loc["detection"],
                iou_gt_source=loc["gt_source"],
                iou_provider_splits=loc["provider_splits"],
                iou_boxes_from=loc.get("boxes_from", "-"),
            )
        )
    if not rows:
        raise SystemExit(f"REFUSED: no rung of {arch} has a test table; nothing written")
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / f"ladder_{arch}_test.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _plot(
        df,
        arch,
        y="auc",
        lo="ci_lo",
        hi="ci_hi",
        title=f"{arch}: test AUC across the ladder (patient-bootstrap 95% CI)",
        path=config.FIGURES_DIR / f"ladder_{arch}_test.png",
        iou_col="box_iou_test",
    )
    return df


def _plot(
    df: pd.DataFrame,
    arch: str,
    *,
    y: str,
    lo,
    hi,
    title: str,
    path: Path,
    iou_col: str = "box_iou_val",
) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    err = None
    if lo and hi:
        err = [df[y] - df[lo], df[hi] - df[y]]
    labels = [f"{k}\n{s}" for k, s in zip(df["rung"], df["source"])]
    ax = axes[0]
    ax.errorbar(range(len(df)), df[y], yerr=err, fmt="o-", capsize=3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(
        config.BIRADS["auc_roc"],
        ls="--",
        c="grey",
        lw=0.8,
        label=f"BI-RADS {config.BIRADS['auc_roc']:.3f}",
    )
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("rung")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.errorbar(df[iou_col], df[y], yerr=err, fmt="o", capsize=3)
    for _, r in df.iterrows():
        ax.annotate(
            f"{int(r['rung'])}",
            (r[iou_col], r[y]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_xlabel("measured miss-inclusive box IoU of the crop source")
    ax.set_ylabel("AUC-ROC")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[ladder] figure -> {path}")


def _refuse_test_stage() -> None:
    """Refuses the test stage unless the freeze token is set."""
    from .test_pass_guard import authorised

    if not authorised():
        raise SystemExit(
            "REFUSED: --stage test reads the test tables; only "
            "notebooks/scripts/run_test_pass.sh may run it"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", required=True, help="classifier architecture")
    ap.add_argument(
        "--margin", type=float, required=True, help="crop margin m* of every rung but rung 1"
    )
    ap.add_argument("--stage", choices=["train", "val", "test"], default="train")
    ap.add_argument("--rungs", type=int, nargs="+", default=sorted(LADDER))
    ap.add_argument(
        "--rung3-k",
        type=int,
        default=1,
        help="rung 3 on the matched top-k candidate (tag suffix _k<k>); 1 = single box",
    )
    ap.add_argument(
        "--boxes-dir",
        default=None,
        help="candidate boxes folder: localiser_boxes_{split}.csv for single-box "
        "rung 3, or localiser_boxes_k3_{split}.csv for --rung3-k > 1, and "
        "localiser_error_model.json for rung 5. Default: the frozen artefacts.",
    )
    ap.add_argument(
        "--rung3-suffix",
        default="",
        help="tag suffix for rung 3 trained on a candidate, e.g. _bundle "
        "(tag vgg16_crop_rung3_unet_bundle), and for rung 5 with --boxes-dir. "
        "Default: none.",
    )
    ap.add_argument(
        "--ledger-suffix",
        default="",
        help="tag suffix for rungs 1, 2, 4, 6, 7 (and 5 without --boxes-dir), e.g. "
        "config.LEDGER_SUFFIX, so they come from the same ledger as rung 3. Default: none.",
    )
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
    global RUNG3_K, RUNG3_BOXES_DIR, RUNG3_SUFFIX, LEDGER_SUFFIX
    RUNG3_K, RUNG3_BOXES_DIR, RUNG3_SUFFIX = (args.rung3_k, args.boxes_dir, args.rung3_suffix)
    LEDGER_SUFFIX = args.ledger_suffix
    if RUNG3_BOXES_DIR and RUNG3_K == 1 and not RUNG3_SUFFIX:
        raise SystemExit(
            "--boxes-dir for single-box rung 3 needs --rung3-suffix, so the "
            "candidate's weights never overwrite the rung-3 tag of the frozen artefacts"
        )
    if args.stage == "test":
        _refuse_test_stage()
    config.set_seeds()
    {"train": stage_train, "val": stage_val, "test": stage_test}[args.stage](
        args.arch, args.margin, args.rungs
    )


if __name__ == "__main__":
    main()

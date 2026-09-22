#!/usr/bin/env python
"""figures.py — every figure and summary table the report is built from.

Every product is read from files on disk, and a missing input is reported rather than
plotted. Model outputs are read on validation only. data-flow counts every partition,
test labels included, so the loader refuses it unless the test pass has set its token
(test_pass_guard.authorised).

  report the six report figures, from the ablation bars to the silent sizes
  matrix one row per trained classifier tag, decoded from tag and histories
  stages the Chollet stage-progression table, one pipeline, one seed
  ablations the single-variable ablation table, against the replica baseline
  crops one validation lesion seen through every box source of the ladder
  data-flow the partition and consumer diagram, counted from the CSVs at run time
  system the static system diagram, from config alone
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.boxes import LADDER  # noqa: E402

ART = ROOT / "artifacts"


def _discovered(what: str, n: int, where, *, required: bool = False) -> int:
    """Print the count and the search location, then refuse an empty required
    discovery: a tool that found nothing has checked nothing."""
    places = where if isinstance(where, (list, tuple)) else [where]
    loc = ", ".join(str(p) for p in places)
    print(f"[discovered] {n} {what} under {loc}")
    if required and n == 0:
        raise SystemExit(
            f"REFUSED: no {what} found under {loc}. An empty discovery is not a "
            f"pass — this run checked nothing."
        )
    return n


# Bound by _configure() from the command line, so every input folder and the
# ledger suffix are arguments rather than constants baked into the figures.
RES = config.RESULTS_DIR
FIG = config.FIGURES_DIR
BUNDLE = config.LOC_BUNDLE_DIR
SUFFIX = ""


def _configure(args) -> None:
    """Bind the folders and the ledger suffix this invocation reads and writes."""
    global RES, FIG, BUNDLE, SUFFIX
    RES = Path(getattr(args, "results_dir", RES))
    FIG = Path(getattr(args, "out", FIG)) if getattr(args, "out", None) else FIG
    BUNDLE = Path(getattr(args, "bundle_dir", BUNDLE))
    SUFFIX = getattr(args, "tag_suffix", "") or ""


def _registry():
    """MODEL_REGISTRY, imported on use: src.models pulls in TensorFlow, and most
    of these figures are drawn from CSV and JSON alone."""
    from src.models import MODEL_REGISTRY

    return MODEL_REGISTRY


# The six report figures
matplotlib.use("Agg")


def ablation_figure() -> None:
    p = RES / "ablation_table.csv"
    if not p.exists():
        print("[figures] ablation_table.csv missing - skipped")
        return
    t = pd.read_csv(p)
    t = t[t["task"].astype(str).str.startswith("A4")].copy()
    ref = t[t["task"] == "A4.0"]
    if ref.empty:
        print("[figures] replica baseline row missing - skipped")
        return
    ref_auc = float(ref["val_auc"].iloc[0])
    rows = t[t["task"] != "A4.0"].sort_values("delta_vs_replica")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    nf = config.VAL_AUC_NOISE_FLOOR
    ax.axvspan(-nf, nf, color="0.9", zorder=0, label=f"single-seed noise floor ±{nf:.2f}")
    ax.axvline(0, color="0.4", lw=0.8)
    colours = [
        "#d62728" if d < -nf else "#1f77b4" if d > nf else "0.55" for d in rows["delta_vs_replica"]
    ]
    ax.barh(rows["change"], rows["delta_vs_replica"], color=colours, height=0.6)
    for y, (d, v, cap) in enumerate(
        zip(rows["delta_vs_replica"], rows["val_auc"], rows["hit_cap"])
    ):
        ax.text(
            d + (0.004 if d >= 0 else -0.004),
            y,
            f"{v:.3f}" + (" (cap)" if cap else ""),
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=8,
        )
    lo = float(rows["delta_vs_replica"].min())
    ax.set_xlim(min(lo - 0.035, -nf - 0.01), nf + 0.03)
    ax.set_xlabel(f"validation AUC minus replica baseline ({ref_auc:.3f}); single seed per cell")
    ax.set_title("Regularised CNN, single-variable ablations at m = 0.20 (A4)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    out = FIG / "ablation_val.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def localiser_distribution() -> None:
    """Per-lesion validation box IoU of run 1 (frozen), run 4 and the promoted
    ensemble: histogram plus the three-way outcome split (no box / IoU 0 / IoU > 0)."""
    sources = {
        "run 1 (frozen)": ART / "localiser_boxes_val.csv",
        "run 4": RES / "localiser_run4_eval" / "localiser_boxes_val.csv",
        "ensemble 1+3+4 (rung 3)": BUNDLE / "boxes" / "localiser_boxes_val.csv",
    }
    avail = {k: pd.read_csv(v) for k, v in sources.items() if v.exists()}
    for k, v in sources.items():
        if not v.exists():
            print(f"[figures] localiser_distribution: {k} table missing ({v}) - series skipped")
    if not avail:
        print("[figures] no localiser tables - skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    bins = np.linspace(0, 1, 21)
    summary = []
    for name, df in avail.items():
        sc = df[df["status"] != "no_gt"]
        iou = pd.to_numeric(sc["box_iou"], errors="coerce").fillna(0.0).to_numpy()
        miss = (sc["status"] == "miss").sum()
        zero = ((sc["status"] == "ok") & (iou == 0)).sum()
        hit = (iou > 0).sum()
        axes[0].hist(
            iou[iou > 0], bins=bins, histtype="step", lw=1.6, label=f"{name}: mean {iou.mean():.3f}"
        )
        summary.append((name, miss / len(sc), zero / len(sc), hit / len(sc), iou.mean()))
    axes[0].set_xlabel("box IoU with the annotated lesion (lesions with IoU > 0 only)")
    axes[0].set_ylabel("validation lesions (n = 243)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("Localiser error is bimodal: a near-hit or a gross miss", fontsize=10)
    names = [s[0] for s in summary]
    miss = np.array([s[1] for s in summary])
    zero = np.array([s[2] for s in summary])
    hit = np.array([s[3] for s in summary])
    axes[1].barh(names, miss, color="0.35", label="no box (fallback)")
    axes[1].barh(names, zero, left=miss, color="#d62728", label="box, IoU = 0")
    axes[1].barh(names, hit, left=miss + zero, color="#2ca02c", label="box, IoU > 0")
    for i, (m, z, h) in enumerate(zip(miss, zero, hit)):
        axes[1].text(1.01, i, f"{h:.0%} hit", va="center", fontsize=8)
    axes[1].set_xlim(0, 1.18)
    axes[1].set_xlabel("share of validation lesions")
    axes[1].legend(
        fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3
    )
    axes[1].invert_yaxis()
    fig.tight_layout()
    out = FIG / "localiser_iou_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def stage6_curves() -> None:
    """Train vs validation AUC and validation loss per epoch for the scaled CNN
    runs and the regularised CNN: Chollet stage 6 into stage 7."""
    runs = {
        "scaled, no aug, LR 1e-3": RES / f"scaled_crop_m020{SUFFIX}_history.json",
        "scaled, no aug, LR 5e-4": RES / f"scaled_crop_m020_lr5e-4{SUFFIX}_history.json",
        "scaled, augmented (superseded)": config.OUTPUTS_DIR
        / "_superseded"
        / "scaled_crop_m020_augmented"
        / f"scaled_crop_m020{SUFFIX}_history.json",
        "regularised (stage 7)": RES / f"regularised_crop_m020{SUFFIX}_history.json",
    }
    avail = {k: json.loads(v.read_text()) for k, v in runs.items() if v.exists()}
    for k, v in runs.items():
        if not v.exists():
            print(f"[figures] stage6_curves: {k} history missing ({v}) - curve skipped")
    if not avail:
        print("[figures] no scratch histories - skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for i, (name, h) in enumerate(avail.items()):
        c = f"C{i}"
        e = np.arange(1, len(h["val_auc"]) + 1)
        axes[0].plot(e, h["auc"], color=c, lw=1.2, label=f"{name}: train")
        axes[0].plot(e, h["val_auc"], color=c, lw=1.2, ls="--", label=f"{name}: val")
        axes[1].plot(e, h["val_loss"], color=c, lw=1.2, label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_ylim(0.45, 1.02)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    axes[0].set_title("Stage 6: capacity without regularisation", fontsize=10)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation loss")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("Validation loss diverges without augmentation", fontsize=10)
    fig.tight_layout()
    out = FIG / "stage6_curves.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


_MISSING: list[str] = []


def _load(rel: str):
    p = RES / rel
    if not p.exists():
        _MISSING.append(rel)
        return None
    return json.loads(p.read_text())


def localiser_gate_ladder() -> None:
    """Every localiser candidate's gate delta with its CI on one axis, so a null
    and a detectable harm can be told apart at a glance."""
    gates = [
        ("4c vs run1", "localiser_bundle/gate_run4c_vs_run1.json"),
        ("L5 vs run4", "localiser_bundle/gate_l5_vs_run4.json"),
        ("L6 vs run4", "localiser_bundle/gate_l6_vs_run4.json"),
        ("L6b vs run4", "localiser_bundle/gate_l6b_vs_run4.json"),
        ("ens4+L6", "localiser_bundle/gate_ens4l6_vs_ens3.json"),
        ("ens4+L6b", "localiser_bundle/gate_ens4l6b_vs_ens3.json"),
        ("re-ranker", "localiser_bundle/gate_rerank_features.json"),
        ("box fusion A", "localiser_bundle/gate_boxfusion_A_vs_ens3.json"),
        ("hybrid", "localiser_bundle/gate_hybrid_vs_ens3.json"),
        ("scale TTA", "localiser_bundle/gate_scale_tta_vs_ens3.json"),
    ]
    rows = [(lab, _load(rel)) for lab, rel in gates]
    rows = [(lab, g) for lab, g in rows if g]
    if not rows:
        print("[figures] localiser_gate_ladder: no gate files, skipped")
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.6))
    for i, (lab, g) in enumerate(rows):
        d, lo, hi = g["mean_diff"], g["ci_lower"], g["ci_upper"]
        ok = g.get("passes", False)
        ax.plot([lo, hi], [i, i], color="0.35", lw=1.6, zorder=2)
        ax.plot(
            [d],
            [i],
            "o",
            ms=6,
            zorder=3,
            color="#2e7d32" if ok else ("#c62828" if hi < 0 else "0.25"),
        )
    ax.axvline(0, color="0.6", lw=1, ls="--", zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("change in per-lesion box IoU vs the reference (95 % CI)")
    ax.set_title("Every localiser gate: nothing passes, and two are detectable harm", fontsize=10)
    fig.tight_layout()
    out = FIG / "localiser_gate_ladder.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def tta_gradient() -> None:
    """Change in box IoU under scale TTA against the widest scale applied."""
    files = ["scale_tta/scale_tta_val.json", "scale_tta_narrow/scale_tta_val.json"]
    have = [_load(f) for f in files]
    if not any(have):
        print("[figures] tta_gradient: no scale_tta results, skipped")
        return
    d = {}
    for h in have:
        if h:
            d[round(max(h["scales"]) - 1.0, 3)] = h["delta"]
    if not d:
        print("[figures] tta_gradient: no deltas, skipped")
        return
    xs = sorted(d)
    ys = [d[x] for x in xs]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([x * 100 for x in xs], ys, "o-", color="#c62828")
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("scale range applied at test time (%)")
    ax.set_ylabel("change in box IoU vs no TTA")
    ax.set_title("Scale TTA harms monotonically in the amount of scale", fontsize=10)
    fig.tight_layout()
    out = FIG / "tta_gradient.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def l6_silent_sizes() -> None:
    """Where L6 goes silent: the size-quartile mix of silent vs the rest."""
    d = _load("l6_silent_images.json")
    if not d or "size_quartile_mix" not in d:
        print("[figures] l6_silent_sizes: no l6_silent_images.json, skipped")
        return
    q = d["size_quartile_mix"]
    labs = ["Q1", "Q2", "Q3", "Q4"]
    silent = [q.get("True", {}).get(k, 0) for k in labs]
    rest = [q.get("False", {}).get(k, 0) for k in labs]
    x = np.arange(len(labs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - w / 2, silent, w, label=f"silent (n={d['silent']['n']})", color="#c62828")
    ax.bar(x + w / 2, rest, w, label=f"the rest (n={d['rest']['n']})", color="0.55")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("share of the group")
    ax.set_xlabel("lesion size quartile")
    ax.set_title("L6 goes silent on the smallest lesions", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "l6_silent_sizes.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


# The experiment matrix
def _scratch_archs() -> set:
    """Registry names trained from scratch; computed on use, so drawing a figure
    from CSV alone does not import TensorFlow."""
    return {n for n, m in _registry().items() if m["normalisation"] == "scratch"}


_STAGE_SUFFIXES = (
    "_frozen_history.json",
    "_finetune_history.json",
    "_full_history.json",
    "_history.json",
)


def _tag_of(path: Path) -> tuple[str, str]:
    """(tag, stage) from a history filename."""
    for s in _STAGE_SUFFIXES:
        if path.name.endswith(s):
            return (path.name[: -len(s)], s.replace("_history.json", "").lstrip("_") or "single")
    raise ValueError(path)


def _best(history_path: Path) -> dict:
    h = json.loads(history_path.read_text())
    va = np.asarray(h["val_auc"], float)
    if va.size == 0:
        return {}
    i = int(va.argmax())
    ta = np.asarray(h.get("auc", []), float)
    return dict(
        val_auc=round(float(va[i]), 4),
        epoch=i + 1,
        epochs=int(va.size),
        train_auc=round(float(ta[i]), 4) if ta.size > i else np.nan,
    )


def decode(tag: str) -> dict:
    """Everything the tag encodes, with config defaults where it is silent."""
    arch = tag.split("_")[0]
    rest = tag[len(arch) :]
    row: dict = dict(tag=tag, arch=arch, kind="scratch" if arch in _scratch_archs() else "transfer")
    if "_fusion_" in tag:
        row["pipeline"] = "fusion"
        m = re.search(r"_fusion_(\w+?)(_unet)?(_single)?(_smoke)?$", tag)
        mode = re.search(r"_fusion_([a-z]+)", tag).group(1)
        row["crop_source"] = "unet" if "_unet" in tag else "oracle"
        row["margin"] = (
            f"{config.CROP_MARGIN_FRAC:.2f} + {config.FUSION_CONTEXT_MARGIN:.2f}"
            if mode == "context"
            else f"{config.CROP_MARGIN_FRAC:.2f}"
        )
        row["variant"] = f"fusion mode = {mode}" + (
            " (single-mass pairs)" if "_single" in tag else ""
        )
        rest = ""
    elif rest.startswith("_crop"):
        row["pipeline"] = "crop"
        rest = rest[len("_crop") :]
        rung = re.match(r"_rung(\d)_([a-z]+)(_k\d)?", rest)
        if rung:
            k = int(rung.group(1))
            row["crop_source"] = LADDER[k][0]
            m = LADDER[k][1]
            row["margin"] = f"{(config.CROP_MARGIN_FRAC if m is None else m):.2f}"
            row["variant"] = f"ladder rung {k}" + (
                f", top-{rung.group(3)[2:]} matched" if rung.group(3) else ""
            )
            rest = rest[rung.end() :]
        else:
            row["crop_source"] = "oracle"
            mm = re.search(r"_m(\d{3})", rest)
            row["margin"] = f"{int(mm.group(1)) / 100:.2f}" if mm else "legacy"
            if mm:
                rest = rest.replace(mm.group(0), "", 1)
    else:
        row["pipeline"] = "full"
        row["crop_source"] = "none"
        row["margin"] = ""
        rest = rest.replace("_full", "", 1)

    ab = re.search(r"_no_(dropout|l2|se|aug)", rest)
    row["ablation"] = ab.group(1) if ab else ""
    rest = rest.replace(ab.group(0), "", 1) if ab else rest
    row["aug"] = "off" if row["ablation"] == "aug" else "default"

    lr = re.search(r"_lr([0-9]+e-?[0-9]+|[0-9.]+)", rest)
    if lr:
        row["lr"] = lr.group(1)
        rest = rest.replace(lr.group(0), "", 1)
    elif row["kind"] == "transfer":
        row["lr"] = f"{config.FINETUNE_LR:g} (fine-tune)"
    else:
        row["lr"] = f"{(config.REGULARISED_LR if arch == 'regularised' else config.INITIAL_LR):g}"

    row["overlay"] = "yes" if "_overlay" in rest else ""
    rest = rest.replace("_overlay", "", 1)
    sd = re.search(r"_s(\d+)$", rest)
    row["seed"] = int(sd.group(1)) if sd else config.SEED
    rest = rest[: sd.start()] if sd else rest
    rest = rest.strip("_")
    if rest:
        row["variant"] = row.get("variant", "") + ("; " if row.get("variant") else "") + rest
    row.setdefault("variant", "")
    return row


def collect() -> pd.DataFrame:
    rows = {}
    for folder, status in ((RES, "current"), (RES / "_superseded", "superseded")):
        for p in sorted(folder.glob("*_history.json")):
            tag, stage = _tag_of(p)
            r = rows.setdefault((tag, status), dict(decode(tag), status=status))
            b = _best(p)
            if stage == "frozen":
                r["frozen_val_auc"] = b.get("val_auc", np.nan)
            elif stage in ("finetune", "single"):
                r.update(b)
            elif stage == "full" and "val_auc" not in r:
                # Without stage histories the concatenated one is used, so the best
                # epoch is over frozen + fine-tune.
                r.update(b)
                r["variant"] += (
                    "; " if r["variant"] else ""
                ) + "full history only (best over both stages)"
    for (tag, status), r in rows.items():
        if status != "current":
            continue
        # allow_metal: a laptop run's own weights count, even where a CUDA twin exists.
        try:
            config.resolve_weights(tag, allow_metal=True)
        except config.WeightsRefused:
            raise  # two differing files for one tag: an error, not missing weights
        except FileNotFoundError:
            r["status"] = "current (no weights)"
    cols = [
        "tag",
        "status",
        "pipeline",
        "arch",
        "kind",
        "crop_source",
        "margin",
        "aug",
        "lr",
        "overlay",
        "ablation",
        "seed",
        "variant",
        "frozen_val_auc",
        "val_auc",
        "epoch",
        "epochs",
        "train_auc",
    ]
    df = pd.DataFrame(list(rows.values()))
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    order = {"crop": 0, "fusion": 1, "full": 2}
    df["_o"] = df["pipeline"].map(order)
    df = df.sort_values(["status", "_o", "arch", "tag"]).drop(columns="_o")
    return df[cols].reset_index(drop=True)


# The stage-progression table
STAGE_ROWS = [
    # (stage, label, tag, history file suffix, notes)
    (
        "5",
        "Baseline CNN (scratch)",
        "baseline_crop_m020",
        "_history.json",
        "Chollet 5: beat a baseline",
    ),
    (
        "6",
        "Scaled CNN (scratch, no regularisation, no augmentation)",
        "scaled_crop_m020",
        "_history.json",
        "Chollet 6: overfit probe at INITIAL_LR 1e-3",
    ),
    (
        "6",
        "Scaled CNN, LR matched to stage 7 (5e-4)",
        "scaled_crop_m020_lr5e-4",
        "_history.json",
        "Chollet 6: same model at REGULARISED_LR, removes the LR confound",
    ),
    (
        "7",
        "Regularised CNN (scratch, SE + L2 + dropout)",
        "regularised_crop_m020",
        "_history.json",
        "Chollet 7",
    ),
    (
        "8",
        "VGG16 (ImageNet transfer)",
        "vgg16_crop_m020",
        "_finetune_history.json",
        "C2 winner; ladder architecture",
    ),
    (
        "8",
        "EfficientNetB0 (ImageNet transfer)",
        "efficientnet_crop_m020",
        "_finetune_history.json",
        "corrected input scale",
    ),
    ("8", "DenseNet121 (ImageNet transfer)", "densenet121_crop_m020", "_finetune_history.json", ""),
    ("8", "ResNet50 (ImageNet transfer)", "resnet50_crop_m020", "_finetune_history.json", ""),
    (
        "A",
        "VGG16 whole image (Pipeline A)",
        "vgg16_full",
        "_finetune_history.json",
        "no localisation; image-level rows",
    ),
]


# The ablation table
REPLICA = "regularised_crop_m020_kg"

LOCAL_BASELINE = "regularised_crop_m020"

KNOBS = [
    (r"_no_dropout", "head dropout 0.3 -> 0", "A4.1"),
    (r"_no_l2", "L2 3e-5 -> 0", "A4.2"),
    (r"_no_se", "SE blocks removed", "A4.3"),
    (r"_no_aug", "augmentation off", "A4.4"),
    (r"_lr2e-3", "LR 5e-4 -> 2e-3", "A4.5"),
    (r"_overlay_lrmatched", "dilated-mask overlay on the crop", "A4.6"),
]


def _from_history(path: Path) -> dict:
    h = json.loads(path.read_text())
    va = np.asarray(h["val_auc"], float)
    b = int(va.argmax())
    return dict(
        val_auc_best=float(va[b]),
        best_epoch=b + 1,
        epochs_run=int(va.size),
        epochs_cap=np.nan,
        train_auc_at_best=float(h["auc"][b]) if "auc" in h else np.nan,
    )


def load_run(results_dir: Path, tag: str) -> dict | None:
    m = results_dir / f"{tag}_manifest.json"
    h = results_dir / f"{tag}_history.json"
    if m.exists():
        d = json.loads(m.read_text())
        return {
            k: d.get(k)
            for k in (
                "val_auc_best",
                "best_epoch",
                "epochs_run",
                "epochs_cap",
                "train_auc_at_best",
                "tf_version",
                "wall_time_s",
            )
        }
    if h.exists():
        return _from_history(h)
    return None


def knob_of(tag: str) -> tuple[str, str]:
    for pat, desc, tid in KNOBS:
        if re.search(pat, tag):
            return desc, tid
    return "", ""


def build_ablations(results_dir: Path) -> pd.DataFrame:
    rows = []
    replica = load_run(results_dir, REPLICA)
    ref = replica["val_auc_best"] if replica else np.nan
    tags = sorted(
        {
            p.name.replace("_manifest.json", "").replace("_history.json", "")
            for p in results_dir.glob("regularised_crop_m020*_kg_*.json")
        }
    )
    for tag in tags:
        r = load_run(results_dir, tag)
        if r is None:
            continue
        desc, tid = ("replica baseline (reference)", "A4.0") if tag == REPLICA else knob_of(tag)
        delta = r["val_auc_best"] - ref if tag != REPLICA and not np.isnan(ref) else np.nan
        hit = (
            r.get("epochs_cap") is not None
            and not pd.isna(r.get("epochs_cap"))
            and r["epochs_run"] >= r["epochs_cap"]
        )
        verdict = (
            ""
            if tag == REPLICA
            else (
                "hit cap: lower bound"
                if hit
                else (
                    "within noise"
                    if abs(delta) < config.VAL_AUC_NOISE_FLOOR
                    else "worse than replica" if delta < 0 else "better than replica"
                )
            )
        )
        rows.append(
            dict(
                task=tid,
                tag=tag,
                change=desc,
                val_auc=round(r["val_auc_best"], 4),
                delta_vs_replica=(round(delta, 4) if not np.isnan(delta) else np.nan),
                best_epoch=r["best_epoch"],
                epochs_run=r["epochs_run"],
                epochs_cap=r.get("epochs_cap"),
                hit_cap=bool(hit),
                train_auc_at_best=(
                    round(r["train_auc_at_best"], 4)
                    if r.get("train_auc_at_best") is not None
                    else np.nan
                ),
                verdict=verdict,
            )
        )
    local = load_run(results_dir, LOCAL_BASELINE)
    if local:
        rows.append(
            dict(
                task="stage 7",
                tag=LOCAL_BASELINE,
                change="local baseline (stage table row; not the ablation reference)",
                val_auc=round(local["val_auc_best"], 4),
                delta_vs_replica=np.nan,
                best_epoch=local["best_epoch"],
                epochs_run=local["epochs_run"],
                epochs_cap=config.MAX_EPOCHS,
                hit_cap=local["epochs_run"] >= config.MAX_EPOCHS,
                train_auc_at_best=round(local["train_auc_at_best"], 4),
                verdict="reference only",
            )
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {
            "A4.0": 0,
            "A4.1": 1,
            "A4.2": 2,
            "A4.3": 3,
            "A4.4": 4,
            "A4.5": 5,
            "A4.6": 6,
            "stage 7": 9,
        }
        df = df.sort_values(by="task", key=lambda s: s.map(order).fillna(8)).reset_index(drop=True)
    return df


# The ladder crop panels
matplotlib.use("Agg")

CROP_SPLITS = ("train", "val")

CROP_PANELS = [  # (key, title, source, margin)
    ("oracle_tight", "rung 1: oracle, m = 0", "oracle", 0.0),
    ("oracle_mstar", f"rung 2: oracle, m = {config.CROP_MARGIN_FRAC:.2f}", "oracle", None),
    (
        "oracle_context",
        f"oracle, m = {config.FUSION_CONTEXT_MARGIN:.2f} (context)",
        "oracle",
        config.FUSION_CONTEXT_MARGIN,
    ),
    ("unet", "rung 3: U-Net box", "unet", None),
    ("cam", "rung 4: Grad-CAM box", "cam", None),
    ("jitter", "rung 5: oracle + sampled U-Net error", "jitter", None),
    ("shuffled", "rung 6: shuffled donor box", "shuffled", None),
    ("whole", "rung 7: whole image", "whole", None),
]

CROP_COLOURS = {
    "oracle": "#2ca02c",
    "unet": "#1f77b4",
    "cam": "#ff7f0e",
    "jitter": "#9467bd",
    "shuffled": "#d62728",
}


def pick_lesion() -> str:
    """The validation lesion whose U-Net box IoU is closest to the median among
    those where both localisers produced a box: a typical case, not a best one."""
    unet = pd.read_csv(ART / "localiser_boxes_val.csv")
    cam = pd.read_csv(ART / "cam_boxes.csv")
    cam = cam[cam["split"] == "val"]
    both = unet[unet["status"] == "ok"].merge(
        cam[cam["status"] == "ok"][["lesion_id"]], on="lesion_id"
    )
    med = both["box_iou_mammogram"].median()
    both = both.assign(d=(both["box_iou_mammogram"] - med).abs()).sort_values(["d", "lesion_id"])
    return str(both["lesion_id"].iloc[0])


def gather_crop(lesion_id: str) -> tuple[np.ndarray, dict]:
    from src.boxes import BoxProvider
    from src.data_loader import _build_combined_lookup, _normalise_intensity, _read_full

    image_id = lesion_id.rsplit("_", 1)[0]
    fulls = {
        Path(k).parts[0]: v
        for k, v in _build_combined_lookup(config.TRAIN_IMG_DIR, config.TEST_IMG_DIR).items()
    }
    full = _normalise_intensity(_read_full(str(fulls[image_id])), config.CROP_INTENSITY_NORM)
    shape = full.shape[:2]
    providers = {
        s: BoxProvider(s, splits=CROP_SPLITS)
        for s in ("oracle", "unet", "cam", "jitter", "shuffled")
    }
    boxes = {s: p.get(lesion_id, shape) for s, p in providers.items()}
    return full, boxes


def draw_crop_examples(lesion_id: str, full: np.ndarray, boxes: dict, out: Path) -> dict:
    m_star = config.CROP_MARGIN_FRAC
    gt = boxes["oracle"]
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 9.0))
    axes = axes.ravel()
    # Whole mammogram, downsampled for display, with every box; legend below.
    ds = 8
    ax = axes[0]
    ax.imshow(full[::ds, ::ds], cmap="gray", vmin=0, vmax=1)
    handles = []
    for s, b in boxes.items():
        if b is None:
            continue
        y0, y1, x0, x1 = b
        h = Rectangle(
            (x0 / ds, y0 / ds),
            (x1 - x0) / ds,
            (y1 - y0) / ds,
            fill=False,
            lw=1.6 if s == "oracle" else 1.0,
            edgecolor=CROP_COLOURS[s],
            label=s,
        )
        ax.add_patch(h)
        handles.append(h)
    ax.legend(
        handles=handles,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        handlelength=1.2,
    )
    ax.set_title(
        "whole mammogram, every box\n" + lesion_id.replace("Mass-Training_", ""), fontsize=7.5
    )
    ax.set_xticks([])
    ax.set_yticks([])

    record = {
        "lesion_id": lesion_id,
        "image_shape": list(full.shape[:2]),
        "boxes": {s: (list(map(int, b)) if b is not None else None) for s, b in boxes.items()},
        "panels": {},
    }
    from src.data_loader import _crop_box
    from src.localise import box_iou

    for ax, (key, title, source, margin) in zip(axes[1:], CROP_PANELS):
        m = m_star if margin is None else margin
        b = None if source == "whole" else boxes[source]
        if source == "whole" or b is None:
            crop = full
            note = (
                "no localisation"
                if source == "whole"
                else "localiser miss: falls back to whole image"
            )
            iou = None
        else:
            crop = _crop_box(full, b, m)
            iou = box_iou(gt, b) if gt is not None else None
            note = (
                f"{crop.shape[0]}x{crop.shape[1]} px"
                if source == "oracle"
                else f"IoU vs oracle {iou:.2f}, {crop.shape[0]}x{crop.shape[1]} px"
            )
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\n{note}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        record["panels"][key] = {
            "source": source,
            "margin": m,
            "box": (list(map(int, b)) if b is not None else None),
            "crop_shape": list(crop.shape[:2]),
            "iou_vs_oracle": iou,
        }
    fig.tight_layout(h_pad=1.4)
    fig.savefig(out, dpi=200)
    print(f"-> {out}")
    return record


# The data-flow diagram
matplotlib.use("Agg")


def gather_data_flow() -> dict:
    raw_tr = pd.read_csv(config.TRAIN_CSV)
    raw_te = pd.read_csv(config.TEST_CSV)
    from src.data_loader import split_frames

    les = split_frames("crop")
    img = split_frames("full")
    cs = pd.read_csv(config._ROOT / config.BIRADS["comparison_set"])
    cs_n = cs["partition"].str.lower().value_counts().to_dict()
    d = dict(
        raw_rows_train=int(len(raw_tr)),
        raw_rows_test=int(len(raw_te)),
        raw_patients=int(pd.concat([raw_tr, raw_te])["patient_id"].nunique()),
        lesions=int(sum(len(f) for f in les.values())),
        images=int(sum(len(f) for f in img.values())),
        patients=int(sum(f["patient_id"].nunique() for f in img.values())),
        partitions={},
    )
    for sp in ("train", "val", "test"):
        d["partitions"][sp] = dict(
            patients=int(img[sp]["patient_id"].nunique()),
            images=int(len(img[sp])),
            images_malignant=int(img[sp]["label"].sum()),
            lesions=int(len(les[sp])),
            lesions_malignant=int(les[sp]["label"].sum()),
            birads_comparison=int(cs_n.get(sp, 0)),
        )
    return d


def draw_data_flow(d: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4", fs=8.5, bold_first=True):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#444", lw=0.9
            )
        )
        lines = text.split("\n")
        ax.text(
            x + w / 2,
            y + h - 0.16,
            lines[0],
            ha="center",
            va="top",
            fontsize=fs,
            fontweight="bold" if bold_first else "normal",
        )
        ax.text(
            x + w / 2,
            y + h - 0.44,
            "\n".join(lines[1:]),
            ha="center",
            va="top",
            fontsize=fs - 0.5,
            linespacing=1.35,
        )

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.9),
        )

    # Row 1: raw CSVs
    box(
        1.0,
        6.2,
        3.6,
        0.85,
        f"CBIS-DDSM mass CSVs (Lee et al., 2017)\n"
        f"train-set rows {d['raw_rows_train']:,}  |  test-set rows {d['raw_rows_test']:,}"
        f"  |  {d['raw_patients']} patients",
    )
    box(
        6.4,
        6.2,
        3.6,
        0.85,
        "Official split discarded\n"
        "BI-RADS-stratified at abnormality level;\nnot patient-disjoint (R3)",
        fc="#fbeaea",
    )
    arrow(2.8, 6.2, 2.8, 5.65)

    # Row 2: pooled units
    box(
        0.6,
        4.7,
        4.4,
        0.95,
        "Pooled and de-duplicated\n"
        f"{d['lesions']:,} abnormalities (one ROI mask each)\n"
        f"{d['images']:,} mammograms  ·  {d['patients']} patients",
    )
    box(
        6.0,
        4.7,
        4.4,
        0.95,
        "ONE patient-level partition\n"
        f"StratifiedGroupKFold on patient_id, seed {config.SEED}, ~70/15/15,\n"
        "derived on the abnormality pool; every row set inherits it",
    )
    arrow(5.0, 5.17, 6.0, 5.17)
    for x in (1.85, 5.5, 9.15):
        arrow(8.2 if x != 1.85 else 8.2, 4.7, x, 3.75)

    # Row 3: partitions
    cols = {"train": (0.35, "Train"), "val": (4.0, "Validation"), "test": (7.65, "Test")}
    for sp, (x, name) in cols.items():
        p = d["partitions"][sp]
        text = (
            f"{name}\n{p['patients']} patients\n"
            f"{p['images']} mammograms ({p['images_malignant']} malignant)\n"
            f"{p['lesions']} lesions ({p['lesions_malignant']} malignant)"
        )
        box(x, 2.45, 3.0, 1.3, text, fc="#eaf1fb" if sp != "test" else "#eaf7ea")

    # Row 4: what consumes each partition
    box(
        0.35,
        0.55,
        3.0,
        1.45,
        "Consumers\nlocaliser training (masks merged per image)\n"
        "classifier training · normalisation constants\nCAM-source classifier",
        fs=8,
    )
    box(
        4.0,
        0.55,
        3.0,
        1.45,
        "Consumers\nevery selection: margin m*, architecture,\n"
        "box thresholds, operating point,\nerror model, temperature",
        fs=8,
    )
    p = d["partitions"]["test"]
    box(
        7.65,
        0.55,
        3.0,
        1.45,
        "Consumers\nONE pre-registered pass\n"
        f"BI-RADS comparison set: {p['birads_comparison']} mammograms\n"
        "(BI-RADS 0 excluded); paired DeLong",
        fs=8,
        fc="#eaf7ea",
    )
    for x in (1.85, 5.5, 9.15):
        arrow(x, 2.45, x, 2.0)

    ax.text(
        5.5,
        0.15,
        "Unit of analysis: Pipeline A one row per mammogram; Pipeline B (crops, localiser, ladder) "
        "one row per abnormality. Lesion-level scores are aggregated to image level (max) "
        "for the BI-RADS comparison.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# The system diagram
matplotlib.use("Agg")


def draw_system_diagram(path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#f4f4f4", fs=8.5, ec="#444", lw=0.9, tdx=0.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw
            )
        )
        ax.text(
            x + w / 2 + tdx,
            y + h - 0.14,
            title,
            ha="center",
            va="top",
            fontsize=fs,
            fontweight="bold",
        )
        if body:
            ax.text(
                x + w / 2,
                y + h - 0.40,
                body,
                ha="center",
                va="top",
                fontsize=fs - 0.8,
                linespacing=1.3,
            )

    def arrow(x0, y0, x1, y1, ls="-", color="#444"):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, linestyle=ls),
        )

    def rung(x, y, n):
        ax.add_patch(plt.Circle((x, y), 0.15, fc="#333", ec="none"))
        ax.text(
            x, y, str(n), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold"
        )

    # Input
    box(
        0.3,
        3.2,
        1.7,
        1.2,
        "Mammogram",
        "full resolution\n~5,000 × 3,000 px\n(digitised film)",
        fc="#ffffff",
    )

    # Stage 1: box sources
    ax.text(
        4.55,
        7.35,
        "STAGE 1 — where the box comes from",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.add_patch(
        FancyBboxPatch(
            (2.5, 0.75),
            4.1,
            6.45,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            fc="none",
            ec="#999",
            lw=0.8,
            ls="--",
        )
    )

    W = 3.5  # source boxes stop at x=6.2; the GT-box route runs at x=6.42
    box(
        2.7,
        5.75,
        W,
        1.15,
        "Oracle: ground-truth ROI mask",
        "radiologist annotation → bbox\ndevelopment condition only (R4)",
        fc="#fbeaea",
        tdx=0.3,
    )
    rung(2.95, 6.72, 1)
    rung(3.3, 6.72, 2)

    box(
        2.7,
        4.25,
        W,
        1.25,
        "U-Net localiser (supervised)",
        f"{config.LOC_INPUT_H}×{config.LOC_INPUT_W} letterbox → mask → threshold "
        f"{config.LOC_MASK_THRESHOLD}\nlargest component → bbox → inverse letterbox",
        fc="#eaf1fb",
        tdx=0.3,
    )
    rung(2.95, 5.32, 3)

    box(
        2.7,
        2.75,
        W,
        1.25,
        "Grad-CAM localiser (weakly supervised)",
        "whole-image classifier (224 px) → CAM\n"
        f"τ × max ({config.CAM_THRESHOLD}) → largest component → bbox",
        fc="#eaf1fb",
        tdx=0.3,
    )
    rung(2.95, 3.82, 4)

    box(
        2.7,
        1.0,
        W,
        1.5,
        "Controls: oracle box perturbed",
        "jitter — resampled observed U-Net error tuples\n"
        "+ measured fallback rate\n"
        "shuffled — another lesion's box, same view,\nplaced within this breast",
        fc="#f6f0e6",
        tdx=0.3,
    )
    rung(2.95, 2.32, 5)
    rung(3.3, 2.32, 6)

    for y in (6.3, 4.85, 3.35, 1.75):
        arrow(2.0, 3.8, 2.7, y)

    # The oracle box feeds the controls: a dashed route down the inside of the panel
    xr = 6.42
    arrow(6.2, 6.1, xr, 6.1, ls="--")
    arrow(xr, 6.1, xr, 2.3, ls="--")
    arrow(xr, 2.3, 6.2, 2.3, ls="--")
    ax.text(
        xr + 0.02,
        3.9,
        "GT box → controls",
        fontsize=7,
        color="#666",
        rotation=90,
        ha="center",
        va="center",
        bbox=dict(fc="white", ec="none", pad=1.5),
    )

    # Shared geometry
    ax.text(8.55, 7.35, "SHARED GEOMETRY", ha="center", fontsize=9.5, fontweight="bold")
    box(
        7.3,
        4.25,
        2.5,
        1.7,
        "Box → crop",
        f"expand by m* = {config.CROP_MARGIN_FRAC:.0%} each side\n"
        "clamp to image · crop at full res\npercentile window p1–p99\n"
        "resize 224×224, no padding",
        fc="#eef7ee",
    )
    box(
        7.3,
        2.35,
        2.5,
        1.35,
        "No box → fallback",
        "whole mammogram, padded\nto square, counted\n(rung 7 = 100% fallback)",
        fc="#eef7ee",
    )
    rung(7.55, 3.47, 7)
    for y in (6.3, 4.85, 3.35, 1.75):
        arrow(6.2, y, 7.3, 5.1)
    arrow(6.6, 3.0, 7.3, 3.0, ls="--")
    ax.text(6.95, 3.12, "no box", fontsize=7, color="#666", ha="center", va="bottom")

    # Stage 2
    ax.text(11.0, 7.35, "STAGE 2 — classify", ha="center", fontsize=9.5, fontweight="bold")
    box(
        10.3,
        4.25,
        1.95,
        1.7,
        "VGG16",
        "ImageNet init\nfrozen → fine-tune\nblock5, BN frozen\nselected on val AUC",
        fc="#eaf1fb",
    )
    arrow(9.8, 5.1, 10.3, 5.1)
    arrow(9.8, 3.0, 10.3, 4.6)
    box(
        10.3,
        2.35,
        1.95,
        1.35,
        "p(malignant)",
        "threshold fitted on\nvalidation (sens ≥ 0.90)\ncarried to test",
        fc="#ffffff",
    )
    arrow(11.27, 4.25, 11.27, 3.7)
    box(
        10.3,
        0.75,
        1.95,
        1.2,
        "Analyses",
        "AUC vs BI-RADS · calibration\nprevalence · decision curve\nGrad-CAM · MC-Dropout",
        fc="#ffffff",
        fs=8,
    )
    arrow(11.27, 2.35, 11.27, 1.95)

    # Ladder key
    ax.text(
        0.3,
        2.6,
        "Ladder rungs (one classifier,\none margin, one geometry;\nonly the box source differs)",
        fontsize=7.8,
        va="top",
    )
    key = [
        (1, "oracle, m = 0"),
        (2, "oracle, m = m*"),
        (3, "U-Net"),
        (4, "Grad-CAM"),
        (5, "jitter control"),
        (6, "shuffled control"),
        (7, "whole image"),
    ]
    for i, (n, t) in enumerate(key):
        y = 1.75 - i * 0.22
        rung(0.45, y, n)
        ax.text(0.7, y, t, fontsize=7.5, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# The subcommands
def cmd_report(args) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ablation_figure()
    localiser_distribution()
    stage6_curves()
    localiser_gate_ladder()
    tta_gradient()
    l6_silent_sizes()
    # Acceptance: a missing input must be visible. A figure drawn from a default
    # is worse than a figure not drawn.
    if _MISSING:
        print(f"\n[figures] {len(_MISSING)} input(s) NOT FOUND:")
        for m in _MISSING:
            print(f"  [NO SOURCE] {m}")
        raise SystemExit(1)
    print("\nacceptance: every gate, TTA and silent-size input was found; no [NO SOURCE] markers")


def cmd_matrix(args) -> None:
    df = collect()
    # The matrix is a verdict about the set of runs on disk; an empty one would
    # read as "no runs exist" rather than "this run found none".
    _discovered("trained tags (histories)", len(df), [RES, RES / "_superseded"], required=True)
    out = Path(args.out_table)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(
        f"-> {out} ({len(df)} tags: "
        f"{(df['status'].str.startswith('current')).sum()} current, "
        f"{(df['status'] == 'superseded').sum()} superseded)"
    )
    show = df[df["status"].str.startswith("current")]
    print(
        "\n| tag | pipeline | source | margin | LR | ablation | overlay "
        "| frozen AUC | val AUC | epoch/epochs | train AUC | variant |"
    )
    print("|---|---|---|---|---|---|---|---:|---:|---:|---:|---|")
    for r in show.itertuples():
        fz = "" if pd.isna(r.frozen_val_auc) else f"{r.frozen_val_auc:.3f}"
        va = "" if pd.isna(r.val_auc) else f"{r.val_auc:.3f}"
        ta = "" if pd.isna(r.train_auc) else f"{r.train_auc:.3f}"
        ep = "" if pd.isna(r.epoch) else f"{int(r.epoch)}/{int(r.epochs)}"
        print(
            f"| {r.tag} | {r.pipeline} | {r.crop_source} | {r.margin} | {r.lr} | {r.ablation} | "
            f"{r.overlay} | {fz} | {va} | {ep} | {ta} | {r.variant} |"
        )


def cmd_stages(args) -> None:
    out_rows = []
    for stage, label, tag, hist_suffix, note in STAGE_ROWS:
        tagged = f"{tag}{SUFFIX}"
        p = RES / f"{tagged}{hist_suffix}"
        if not p.exists():
            out_rows.append(dict(stage=stage, model=label, tag=tagged, note=f"MISSING {p.name}"))
            continue
        h = json.loads(p.read_text())
        va, vl = np.asarray(h["val_auc"]), np.asarray(h["val_loss"])
        ta = np.asarray(h.get("auc", [np.nan] * len(va)))
        i = int(va.argmax())
        out_rows.append(
            dict(
                stage=stage,
                model=label,
                tag=tagged,
                val_auc_best=round(float(va[i]), 4),
                epoch_best=i + 1,
                train_auc_at_best=round(float(ta[i]), 4) if len(ta) > i else np.nan,
                val_auc_at_min_loss=round(float(va[vl.argmin()]), 4),
                epochs=len(va),
                note=note,
            )
        )
    df = pd.DataFrame(out_rows)
    out = Path(args.out_table)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(
        "\n| Stage | Model | val AUC (best) | train AUC at that epoch | val AUC at min loss "
        "| epochs |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for r in out_rows:
        if "val_auc_best" in r:
            print(
                f"| {r['stage']} | {r['model']} | {r['val_auc_best']:.3f} | "
                f"{r['train_auc_at_best']:.3f} | "
                f"{r['val_auc_at_min_loss']:.3f} | {r['epochs']} |"
            )
        else:
            print(f"| {r['stage']} | {r['model']} | — | — | — | {r['note']} |")
    print(f"-> {out}")


def cmd_ablations(args) -> None:
    rd = RES
    df = build_ablations(rd)
    _discovered("ablation runs (regularised_crop_m020*_kg_*)", len(df), rd, required=True)
    if df.empty:
        raise SystemExit(f"no _kg runs under {rd}: unpack ablations_kg.zip into outputs/ first")
    out = Path(args.out_table)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"-> {out}")
    print(
        f"\nnoise floor {config.VAL_AUC_NOISE_FLOOR:.2f} AUC (single seed); replica = {REPLICA}\n"
    )
    print(
        "| task | change | val AUC | delta vs replica | epoch / run / cap | train AUC | verdict |"
    )
    print("|---|---|---:|---:|---:|---:|---|")
    for r in df.itertuples():
        d = "" if pd.isna(r.delta_vs_replica) else f"{r.delta_vs_replica:+.4f}"
        cap = "" if pd.isna(r.epochs_cap) else f"{int(r.epochs_cap)}"
        ta = "" if pd.isna(r.train_auc_at_best) else f"{r.train_auc_at_best:.3f}"
        print(
            f"| {r.task} | {r.change} | {r.val_auc:.4f} | {d} | "
            f"{int(r.best_epoch)} / {int(r.epochs_run)} / {cap} | {ta} | {r.verdict} |"
        )


def cmd_crops(args) -> None:
    lesion_id = args.lesion or pick_lesion()
    val_ids = set(pd.read_csv(ART / "localiser_boxes_val.csv")["lesion_id"])
    if lesion_id not in val_ids:
        raise SystemExit(f"{lesion_id} is not a validation lesion")
    print(f"[crop_examples] lesion {lesion_id}")
    full, boxes = gather_crop(lesion_id)
    out = FIG / "crop_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = draw_crop_examples(lesion_id, full, boxes, out)
    side = out.with_suffix(".json")
    side.write_text(json.dumps(record, indent=2))
    print(f"-> {side}")


def cmd_data_flow(args) -> None:
    d = gather_data_flow()
    FIG.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_table)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(d, indent=2))
    draw_data_flow(d, FIG / "data_flow.png")
    print(json.dumps(d, indent=2))
    print(f"-> {out_json}")
    print(f"-> {FIG / 'data_flow.png'}")


def cmd_system(args) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    draw_system_diagram(FIG / "system_diagram.png")
    print(f"-> {FIG / 'system_diagram.png'}")


# CLI
def _add_common(p, *, table_default=None, needs_results=True):
    p.add_argument("--out", default=str(config.FIGURES_DIR), help="folder for the figures")
    if needs_results:
        p.add_argument(
            "--results-dir",
            default=str(config.RESULTS_DIR),
            help="the tables and histories to read",
        )
        p.add_argument(
            "--tag-suffix",
            default="",
            help="ledger suffix appended to the tags that stages and the report's stage-6 "
            "curves read; the other products ignore it",
        )
    if table_default:
        p.add_argument(
            "--out-table",
            default=str(config.RESULTS_DIR / table_default),
            help="path of the table this writes",
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("report", help="the six report figures")
    _add_common(p)
    p.add_argument("--bundle-dir", default=str(config.LOC_BUNDLE_DIR))
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("matrix", help="one row per trained classifier tag")
    _add_common(p, table_default="experiment_matrix.csv")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("stages", help="the Chollet stage-progression table")
    _add_common(p, table_default="stage_table.csv")
    p.set_defaults(func=cmd_stages)

    p = sub.add_parser("ablations", help="the single-variable ablation table (A4)")
    _add_common(p, table_default="ablation_table.csv")
    p.set_defaults(func=cmd_ablations)

    p = sub.add_parser("crops", help="one lesion through every box source of the ladder")
    _add_common(p, needs_results=False)
    p.add_argument("--lesion", default=None, help="validation lesion id (default: the median rule)")
    p.set_defaults(func=cmd_crops)

    p = sub.add_parser("data-flow", help="the partition and consumer diagram")
    _add_common(p, table_default="data_flow.json", needs_results=False)
    p.set_defaults(func=cmd_data_flow)

    p = sub.add_parser("system", help="the static system diagram")
    _add_common(p, needs_results=False)
    p.set_defaults(func=cmd_system)

    args = ap.parse_args(argv)
    _configure(args)
    args.func(args)


if __name__ == "__main__":
    main()

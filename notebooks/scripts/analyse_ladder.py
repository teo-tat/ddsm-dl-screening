#!/usr/bin/env python
"""analyse_ladder.py — where the localisation ladder's error falls, and on which lesions.

The rungs differ only in the box the classifier is given, so the analysis asks which
lesions get a bad box and whether an oracle crop would have rescued them.

  errors per-lesion join of rungs 2/3/5, boxes, ground truth and a jitter replay;
    AUC in IoU bins
  subtlety one row per CBIS subtlety grade and per density: miss rate, bad-box rate,
    rung 3 vs rung 2
  silent-images lesions a member never fires on (peak below the threshold), by lesion
    size and density
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

ART = ROOT / "artifacts"
BINS = ["none", "0", "(0, 0.3)", "[0.3, 0.5)", ">= 0.5"]
MIN_N_FOR_CI = 20
RUNGS = ("2", "3", "5")


def _tags(pairs, model: str, tag_suffix: str) -> dict[str, str]:
    """`--tag rung2=... rung3=...` over the default tag rule; the overrides are keyed
    because a rung is a role, and one landing on the wrong rung moves a contrast silently.
    """
    tags = {
        "2": f"{model}_crop_rung2_oracle",
        "3": f"{model}_crop_rung3_unet{tag_suffix}",
        "5": f"{model}_crop_rung5_jitter{tag_suffix}",
    }
    for p in pairs or []:
        key, _, val = p.partition("=")
        k = key.replace("rung", "")
        if k not in RUNGS or not val:
            raise SystemExit(f"--tag takes rung2=<tag>, rung3=<tag> or rung5=<tag>; got {p!r}")
        tags[k] = val
    return tags


# Errors: the per-lesion error analysis
def iou_bin(detected: bool, iou: float) -> str:
    if not detected:
        return "none"
    if iou <= 0.0:
        return "0"
    if iou < 0.3:
        return "(0, 0.3)"
    if iou < 0.5:
        return "[0.3, 0.5)"
    return ">= 0.5"


def _auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    """AUC with a patient-cluster CI where the subset supports one."""
    from src.evaluate import _auc_roc_fn, bootstrap_ci

    n = int(len(y))
    if n == 0 or len(np.unique(y)) < 2:
        return {"n": n, "auc": None, "ci": None, "note": "single class or empty"}
    if n < MIN_N_FOR_CI or len(np.unique(groups)) < 5:
        return {
            "n": n,
            "auc": float(roc_auc_score(y, p)),
            "ci": None,
            "note": f"n < {MIN_N_FOR_CI} or too few patients for a CI",
        }
    auc, lo, hi = bootstrap_ci(y.astype(int), p.astype(float), _auc_roc_fn, groups=groups)
    return {"n": n, "auc": float(auc), "ci": [float(lo), float(hi)]}


def _load_table(tag: str, split: str) -> pd.DataFrame:
    p = config.RESULTS_DIR / f"{tag}_predictions_{split}.csv"
    if not p.exists():
        raise SystemExit(
            f"no prediction table {p}. Write it with: python -m src.evaluate ... "
            f"--tag-suffix {tag.split('_crop', 1)[-1]} --split {split}"
        )
    df = pd.read_csv(p)
    need = {"lesion_id", "patient_id", "y_true", "y_prob"}
    if not need <= set(df.columns):
        raise SystemExit(f"{p} lacks {need - set(df.columns)}")
    return df


def build_frame(boxes_dir: Path, split: str, tags: dict[str, str]) -> pd.DataFrame:
    """One row per lesion: scores of rungs 2/3/5, localiser outcome, jitter replay, metadata."""
    from src import data_loader
    from src.boxes import provider_for
    from src.localise import box_iou

    t2, t3, t5 = (_load_table(tags[k], split) for k in RUNGS)
    base = t2[["lesion_id", "patient_id", "y_true", "y_prob"]].rename(columns={"y_prob": "p2"})
    for name, t in (("p3", t3), ("p5", t5)):
        s = t.set_index("lesion_id")
        if set(s.index) != set(base["lesion_id"]):
            raise SystemExit(f"{name}: lesion set differs from rung 2's table")
        base[name] = s.loc[base["lesion_id"], "y_prob"].to_numpy()
        if not (s.loc[base["lesion_id"], "y_true"].to_numpy() == base["y_true"].to_numpy()).all():
            raise SystemExit(f"{name}: labels disagree with rung 2's table")

    # Localiser outcome from the promoted box table (mammogram coordinates).
    bx = pd.read_csv(boxes_dir / f"localiser_boxes_{split}.csv").set_index("lesion_id")
    bx = bx[bx["status"] != "no_gt"]
    missing = set(base["lesion_id"]) - set(bx.index)
    if missing:
        raise SystemExit(
            f"{len(missing)} lesions absent from the box table, e.g. {sorted(missing)[:3]}"
        )
    bx = bx.loc[base["lesion_id"]]
    base["loc_status"] = bx["status"].to_numpy()
    base["loc_detected"] = bx["detected"].astype(bool).to_numpy()
    iou_col = "box_iou_mammogram" if "box_iou_mammogram" in bx.columns else "box_iou"
    base["loc_iou"] = pd.to_numeric(bx[iou_col], errors="coerce").fillna(0.0).to_numpy()
    base.loc[~base["loc_detected"], "loc_iou"] = 0.0
    base["loc_bin"] = [iou_bin(d, i) for d, i in zip(base["loc_detected"], base["loc_iou"])]
    for c in ("dy_frac", "dx_frac", "scale_ratio_h", "scale_ratio_w", "n_lesions_on_image"):
        if c in bx.columns:
            base[c] = pd.to_numeric(bx[c], errors="coerce").to_numpy()

    # Ground truth in mammogram coordinates, and lesion size.
    gt = (
        pd.read_csv(ART / f"localiser_boxes_{split}.csv")
        .set_index("lesion_id")
        .loc[base["lesion_id"]]
    )
    gt_boxes = {
        lid: (int(r.gt_y0), int(r.gt_y1), int(r.gt_x0), int(r.gt_x1)) for lid, r in gt.iterrows()
    }
    base["gt_area_frac"] = (
        ((gt["gt_y1"] - gt["gt_y0"] + 1) * (gt["gt_x1"] - gt["gt_x0"] + 1))
        / (gt["orig_h"] * gt["orig_w"])
    ).to_numpy()

    # Replay rung 5's draw: the jitter provider is deterministic per lesion id.
    splits = ("train", "val", "test") if split == "test" else ("train", "val")
    jit = provider_for("jitter", boxes_dir=boxes_dir, splits=splits)
    jit_iou, jit_det = [], []
    for lid, r in gt.iterrows():
        b = jit.get(lid, (int(r.orig_h), int(r.orig_w)))
        if b is None:
            jit_det.append(False)
            jit_iou.append(0.0)
        else:
            jit_det.append(True)
            jit_iou.append(float(box_iou(b, gt_boxes[lid])))
    base["jit_detected"], base["jit_iou"] = jit_det, jit_iou
    base["jit_bin"] = [iou_bin(d, i) for d, i in zip(jit_det, jit_iou)]

    # CBIS metadata from the partition frames; shape and margins are not in the prediction tables.
    meta = data_loader.split_frames(source="crop")[split].set_index("lesion_id")
    for c in ("subtlety", "breast_density", "mass shape", "mass margins"):
        if c in meta.columns:
            base[c] = meta.loc[base["lesion_id"], c].to_numpy()
    return base


def bin_table(df: pd.DataFrame, bin_col: str) -> dict:
    out = {}
    g = df["patient_id"].to_numpy()
    for b in BINS:
        m = (df[bin_col] == b).to_numpy()
        sub = df[m]
        out[b] = {
            "n": int(m.sum()),
            "malignant_frac": (float(sub["y_true"].mean()) if m.any() else None),
            **{
                f"rung{k}": _auc(sub["y_true"].to_numpy(), sub[f"p{k}"].to_numpy(), g[m])
                for k in RUNGS
            },
        }
    return out


def victims_contrast(df: pd.DataFrame, iou_col: str, det_col: str, thr: float) -> dict:
    """AUC of rungs 2, 3 and 5 on the lesions a box source handles badly, against the rest."""
    bad = (~df[det_col].to_numpy()) | (df[iou_col].to_numpy() < thr)
    g = df["patient_id"].to_numpy()
    res = {"rule": f"no box or IoU < {thr}", "n_bad": int(bad.sum()), "n_good": int((~bad).sum())}
    for k in RUNGS:
        res[f"rung{k}_on_bad"] = _auc(
            df["y_true"][bad].to_numpy(), df[f"p{k}"][bad].to_numpy(), g[bad]
        )
        res[f"rung{k}_on_good"] = _auc(
            df["y_true"][~bad].to_numpy(), df[f"p{k}"][~bad].to_numpy(), g[~bad]
        )
    return res


def rates_by(df: pd.DataFrame, col: str, bad: np.ndarray) -> list[dict]:
    rows = []
    for v, sub in df.assign(_bad=bad).groupby(col, dropna=False):
        rows.append(
            {
                col: (None if pd.isna(v) else (v if isinstance(v, str) else float(v))),
                "n": int(len(sub)),
                "bad_box_rate": float(sub["_bad"].mean()),
                "malignant_frac": float(sub["y_true"].mean()),
            }
        )
    return rows


def figure(df: pd.DataFrame, res: dict, path: Path, split: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(BINS))
    for ax, key, title in (
        (axes[0, 0], "by_localiser_iou", "binned by the localiser's box IoU (rung 3's boxes)"),
        (axes[0, 1], "by_jitter_iou", "binned by rung 5's replayed jitter IoU"),
    ):
        for k, mk in (("2", "o"), ("3", "s"), ("5", "^")):
            vals = [res[key][b][f"rung{k}"]["auc"] for b in BINS]
            ns = [res[key][b]["n"] for b in BINS]
            y = [v if v is not None else np.nan for v in vals]
            lo = [
                (v - res[key][b][f"rung{k}"]["ci"][0] if res[key][b][f"rung{k}"]["ci"] else 0)
                for v, b in zip(y, BINS)
            ]
            hi = [
                (res[key][b][f"rung{k}"]["ci"][1] - v if res[key][b][f"rung{k}"]["ci"] else 0)
                for v, b in zip(y, BINS)
            ]
            ax.errorbar(
                x + (int(k) - 3) * 0.08,
                y,
                yerr=[lo, hi],
                fmt=mk + "-",
                capsize=3,
                label=f"rung {k}",
            )
            if k == "2":
                for xi, n in zip(x, ns):
                    ax.annotate(f"n={n}", (xi, 0.42), ha="center", fontsize=7, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(BINS, fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    bad_loc = (~df["loc_detected"].to_numpy()) | (df["loc_iou"].to_numpy() < 0.3)
    for col, mk in (("subtlety", "o"), ("breast_density", "s")):
        if col in df.columns:
            r = pd.DataFrame(rates_by(df, col, bad_loc)).dropna(subset=[col])
            ax.plot(r[col], r["bad_box_rate"], mk + "-", label=f"by {col}")
    ax.set_ylabel("rate of no box or IoU < 0.3 (localiser)")
    ax.set_xlabel("CBIS grade")
    ax.set_title("where the localiser gives up", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    d = df["p3"] - df["p2"]
    for lab, c in ((1, "tab:red"), (0, "tab:blue")):
        m = df["y_true"] == lab
        ax.scatter(
            df.loc[m, "loc_iou"], d[m], s=10, alpha=0.6, c=c, label="malignant" if lab else "benign"
        )
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_xlabel("localiser box IoU (mammogram coordinates)")
    ax.set_ylabel("score(rung 3) − score(rung 2)")
    ax.set_title("per-lesion cost of the realised box", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Rung 3 vs rung 5 on {split}: "
        "is the localiser's error selective in the classifier's favour?",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"-> {path}")


def _refuse_test_outside_pass(split: str) -> None:
    """Refuses the test split unless the test pass has set its token
    (test_pass_guard.authorised)."""
    from src.test_pass_guard import authorised

    if split == "test" and not authorised():
        raise SystemExit(
            "REFUSED: --split test reads the stored test prediction and box tables; only "
            "notebooks/scripts/run_test_pass.sh may run it"
        )


def cmd_errors(args) -> None:
    from src.compare import auc_diff_ci

    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "REFUSED: the test partition is frozen: --allow-test is for run_test_pass.sh only"
        )
    _refuse_test_outside_pass(args.split)
    config.set_seeds()
    tags = _tags(args.tag, args.model, args.tag_suffix)
    df = build_frame(Path(args.boxes_dir), args.split, tags)
    y, g = df["y_true"].to_numpy(int), df["patient_id"].to_numpy()

    res = {
        "split": args.split,
        "n": int(len(df)),
        "tags": tags,
        "boxes_dir": args.boxes_dir,
        "overall": {f"rung{k}": _auc(y, df[f"p{k}"].to_numpy(), g) for k in RUNGS},
        "paired_auc_differences": {},
    }
    for a, b in (("3", "5"), ("2", "3"), ("2", "5")):
        pa, pb = df[f"p{a}"].to_numpy(), df[f"p{b}"].to_numpy()
        res["paired_auc_differences"][f"rung{a}_minus_rung{b}"] = {
            "diff": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "ci": auc_diff_ci(y, pa, pb, g),
        }
    res["localiser_outcome"] = {
        "detection": float(df["loc_detected"].mean()),
        "mean_iou_incl": float(df["loc_iou"].mean()),
        "bins": df["loc_bin"].value_counts().reindex(BINS, fill_value=0).to_dict(),
    }
    res["jitter_replay"] = {
        "detection": float(df["jit_detected"].mean()),
        "mean_iou_incl": float(df["jit_iou"].mean()),
        "bins": df["jit_bin"].value_counts().reindex(BINS, fill_value=0).to_dict(),
    }
    res["by_localiser_iou"] = bin_table(df, "loc_bin")
    res["by_jitter_iou"] = bin_table(df, "jit_bin")
    res["victims"] = {
        "localiser": victims_contrast(df, "loc_iou", "loc_detected", 0.3),
        "jitter": victims_contrast(df, "jit_iou", "jit_detected", 0.3),
    }
    bad_loc = (~df["loc_detected"].to_numpy()) | (df["loc_iou"].to_numpy() < 0.3)
    df["gt_size_quartile"] = pd.qcut(
        df["gt_area_frac"], 4, labels=["q1 smallest", "q2", "q3", "q4 largest"]
    ).astype(str)
    df["multi_lesion_image"] = (
        (df["n_lesions_on_image"] > 1) if "n_lesions_on_image" in df.columns else False
    )
    res["bad_box_rates"] = {
        c: rates_by(df, c, bad_loc)
        for c in (
            "subtlety",
            "breast_density",
            "mass shape",
            "mass margins",
            "gt_size_quartile",
            "multi_lesion_image",
            "y_true",
        )
        if c in df.columns
    }
    # Score-level view: mean (rung 3 - rung 2) and (rung 5 - rung 2) by class.
    df["d3"], df["d5"] = df["p3"] - df["p2"], df["p5"] - df["p2"]
    res["score_change_by_class"] = {
        str(lab): {
            "mean_rung3_minus_rung2": float(df.loc[df.y_true == lab, "d3"].mean()),
            "mean_rung5_minus_rung2": float(df.loc[df.y_true == lab, "d5"].mean()),
        }
        for lab in (0, 1)
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.out_figure)
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ladder_error_analysis_{args.split}{args.out_suffix}"
    out_json = out_dir / f"{stem}.json"
    out_json.write_text(json.dumps(res, indent=2, default=float))
    print(f"-> {out_json}")
    per_lesion = out_dir / f"{stem}_per_lesion.csv"
    df.to_csv(per_lesion, index=False)
    print(f"-> {per_lesion}")
    figure(df, res, fig_dir / f"{stem}.png", args.split)

    o, d = res["overall"], res["paired_auc_differences"]
    print(
        f"\n[{args.split}] n={len(df)} rung2 {o['rung2']['auc']:.4f} rung3 {o['rung3']['auc']:.4f} "
        f"rung5 {o['rung5']['auc']:.4f}"
    )
    for k, v in d.items():
        print(f"  {k}: {v['diff']:+.4f} CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    v = res["victims"]
    for src in ("localiser", "jitter"):
        vb, vg = v[src]["rung2_on_bad"], v[src]["rung2_on_good"]
        print(
            f"  oracle (rung 2) on {src}'s victims (n={v[src]['n_bad']}): "
            f"{vb['auc'] if vb['auc'] is None else round(vb['auc'], 4)} vs the rest "
            f"(n={v[src]['n_good']}): {vg['auc'] if vg['auc'] is None else round(vg['auc'], 4)}"
        )


# Subtlety: the stratified table
def _rates(df: pd.DataFrame, prob: str, thr_col: str) -> dict:
    y = df["y_true"].to_numpy(int)
    pred = (df[prob].to_numpy(float) >= df[thr_col].to_numpy(float)).astype(int)
    out = {"accuracy": float((pred == y).mean())}
    out["sensitivity"] = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    out["specificity"] = float((1 - pred[y == 0]).mean()) if (y == 0).any() else np.nan
    out["auc"] = float(roc_auc_score(y, df[prob])) if len(np.unique(y)) == 2 else np.nan
    return out


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    groups = [(v, g) for v, g in df.groupby(by, dropna=False)] + [("all", df)]
    for v, g in groups:
        r3, r2 = _rates(g, "p3", "thr3"), _rates(g, "p2", "thr2")
        rows.append(
            {
                by: v,
                "n": int(len(g)),
                "malignant_frac": round(float(g["y_true"].mean()), 3),
                "unet_miss_rate": round(float((~g["detected"]).mean()), 3),
                "unet_bad_box_rate": round(float(g["bad_box"].mean()), 3),
                "unet_mean_box_iou": round(float(g["box_iou"].mean()), 3),
                "rung3_accuracy": round(r3["accuracy"], 3),
                "rung3_sensitivity": round(r3["sensitivity"], 3),
                "rung3_specificity": round(r3["specificity"], 3),
                "rung3_auc": round(r3["auc"], 3),
                "rung2_accuracy": round(r2["accuracy"], 3),
                "rung2_auc": round(r2["auc"], 3),
            }
        )
    return pd.DataFrame(rows)


def cmd_subtlety(args) -> None:
    tags = _tags(args.tag, args.model, args.tag_suffix)
    R = config.RESULTS_DIR
    t3 = pd.read_csv(R / f"{tags['3']}_predictions_val.csv")
    t2 = pd.read_csv(R / f"{tags['2']}_predictions_val.csv").set_index("lesion_id")
    bx = pd.read_csv(Path(args.boxes_dir) / "localiser_boxes_val.csv").set_index("lesion_id")
    df = t3.rename(columns={"y_prob": "p3", "threshold": "thr3"})
    df["p2"] = t2.loc[df["lesion_id"], "y_prob"].to_numpy()
    df["thr2"] = t2.loc[df["lesion_id"], "threshold"].to_numpy()
    if not (t2.loc[df["lesion_id"], "y_true"].to_numpy() == df["y_true"].to_numpy()).all():
        raise SystemExit("rung 2 and rung 3 tables disagree on labels")
    b = bx.loc[df["lesion_id"]]
    df["detected"] = b["detected"].astype(bool).to_numpy()
    iou_col = "box_iou_mammogram" if "box_iou_mammogram" in b.columns else "box_iou"
    df["box_iou"] = pd.to_numeric(b[iou_col], errors="coerce").fillna(0.0).to_numpy()
    df.loc[~df["detected"], "box_iou"] = 0.0
    df["bad_box"] = (~df["detected"]) | (df["box_iou"] < 0.3)
    if "subtlety" not in df.columns:
        raise SystemExit("prediction table lacks a subtlety column")

    out_frames = []
    for by in ("subtlety", "breast_density"):
        if by in df.columns:
            t = table(df, by)
            t.insert(0, "stratified_by", by)
            t = t.rename(columns={by: "stratum"})
            out_frames.append(t)
            print(
                f"\n**By {by}** (validation, n = {len(df)}; rung 3 at its val threshold "
                f"{df['thr3'].iloc[0]:.3f}, rung 2 at {df['thr2'].iloc[0]:.3f})\n"
            )
            cols = [
                "stratum",
                "n",
                "malignant_frac",
                "unet_miss_rate",
                "unet_bad_box_rate",
                "unet_mean_box_iou",
                "rung3_accuracy",
                "rung3_sensitivity",
                "rung3_specificity",
                "rung3_auc",
                "rung2_accuracy",
                "rung2_auc",
            ]
            print("| " + " | ".join(cols) + " |")
            print("|" + "---:|" * len(cols))
            for r in t[cols].itertuples(index=False):
                print("| " + " | ".join(str(v) for v in r) + " |")
        else:
            print(f"\n[subtlety] no {by} column; skipped")
    out = pd.concat(out_frames, ignore_index=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n-> {path}")


# Silent images: lesions a member does not fire on at all
def cmd_silent_images(args) -> None:
    """Peaks come from tiled inference at the training scale: area-downsampling the
    ensemble map averages peaks away and would manufacture silence."""
    from src import data_loader

    d = Path(args.tensors_dir)
    meta = pd.read_csv(d / "val_meta.csv")
    msks = np.load(d / "val_masks.npy", mmap_mode="r")
    n = len(meta) if args.limit is None else min(args.limit, len(meta))

    if args.peaks and not Path(args.peaks).exists():
        raise SystemExit(f"REFUSED: --peaks {args.peaks} does not exist")
    if args.peaks:
        pk = pd.read_csv(args.peaks)
        ids = meta["lesion_id"].to_numpy()[:n]
        if (
            "lesion_id" not in pk.columns
            or len(pk) < n
            or (pk["lesion_id"].to_numpy()[:n] != ids).any()
        ):
            raise SystemExit(
                f"REFUSED: the lesion_id rows of {args.peaks} do not align with the first "
                f"{n} lesions of {d / 'val_meta.csv'}"
            )
        peaks = pk["peak"].to_numpy(float)[:n]
    else:
        from src.localise import predict_tiled
        from src.localise_eval import _load_model

        model = _load_model(
            args.weights,
            config.LOC_BASE_FILTERS,
            config.LOC_DEPTH,
            "vgg16",
            input_hw=(config.LOC_PATCH_SIZE, config.LOC_PATCH_SIZE),
        )
        imgs = np.load(d / "val_images.npy", mmap_mode="r")
        peaks = np.empty(n, float)
        for i in range(n):
            p = predict_tiled(
                model, np.asarray(imgs[i], np.float32), overlap=args.overlap, batch=args.batch_size
            )
            peaks[i] = float(p.max())
            if (i + 1) % 40 == 0:
                print(f"[silent] {i + 1}/{n}", flush=True)
        pd.DataFrame({"lesion_id": meta["lesion_id"][:n], "peak": peaks}).to_csv(
            Path(args.out).with_name("l6_val_peaks.csv"), index=False
        )

    # Lesion size on the member's canvas, and density from the partition frames.
    side = np.array([np.sqrt(float((np.asarray(msks[i]) > 0).sum())) for i in range(n)])
    frame = data_loader.split_frames(source="crop")["val"].set_index("lesion_id")
    dens = frame.reindex(meta["lesion_id"][:n])["breast_density"].to_numpy()

    df = pd.DataFrame(
        {
            "lesion_id": meta["lesion_id"][:n],
            "peak": peaks,
            "lesion_side_px": side,
            "breast_density": dens,
            "label": meta["label"][:n].to_numpy(),
        }
    )
    df["size_quartile"] = pd.qcut(df["lesion_side_px"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df["silent"] = df["peak"] < args.threshold

    n_sil = int(df["silent"].sum())
    print(
        f"\n--- silent images: peak < {args.threshold} on {n_sil}/{n} "
        f"({n_sil / max(n, 1):.1%}) validation lesions ---\n"
    )
    out = {
        "weights": args.weights,
        "threshold": args.threshold,
        "n": int(n),
        "n_silent": n_sil,
        "silent_rate": round(n_sil / max(n, 1), 4),
    }

    print(f"{'group':10s} {'n':>4s} {'median side px':>15s} {'malignant':>10s}")
    for lbl, sub in (("silent", df[df.silent]), ("rest", df[~df.silent])):
        print(
            f"{lbl:10s} {len(sub):4d} {sub.lesion_side_px.median():15.1f} "
            f"{sub.label.mean():10.3f}"
        )
        out[lbl] = {
            "n": int(len(sub)),
            "median_side_px": round(float(sub.lesion_side_px.median()), 1),
            "malignant_rate": round(float(sub.label.mean()), 4),
        }

    print("\nsize quartile mix (share of each group):")
    q = pd.crosstab(df["silent"], df["size_quartile"], normalize="index")
    print(q.round(3).to_string())
    out["size_quartile_mix"] = {
        str(k): {str(c): round(float(v), 4) for c, v in row.items()} for k, row in q.iterrows()
    }

    if df["breast_density"].notna().any():
        print("\nbreast density mix (share of each group):")
        dq = pd.crosstab(df["silent"], df["breast_density"], normalize="index")
        print(dq.round(3).to_string())
        out["density_mix"] = {
            str(k): {str(c): round(float(v), 4) for c, v in row.items()} for k, row in dq.iterrows()
        }
    else:
        print("\nbreast density unavailable in the val frame")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


# CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("errors", help="per-lesion error analysis of rungs 2/3/5")
    p.add_argument("--model", default="vgg16")
    p.add_argument(
        "--tag-suffix", default="_bundle", help="rung-3/5 tag suffix of the promoted localiser"
    )
    p.add_argument(
        "--tag",
        nargs="+",
        default=None,
        help="override a rung's tag: rung2=<tag> rung3=<tag> rung5=<tag>",
    )
    p.add_argument(
        "--boxes-dir",
        required=True,
        help="promoted boxes folder (localiser_boxes_{split}.csv, error model)",
    )
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--allow-test", action="store_true", help="run_test_pass.sh only")
    p.add_argument(
        "--out", default=str(config.RESULTS_DIR), help="folder for the JSON and the per-lesion CSV"
    )
    p.add_argument("--out-figure", default=str(config.FIGURES_DIR), help="folder for the PNG")
    p.add_argument("--out-suffix", default="", help="extra suffix on the output file names")
    p.set_defaults(func=cmd_errors)

    p = sub.add_parser("subtlety", help="miss rate and rung-3 accuracy by subtlety and density")
    p.add_argument("--model", default="vgg16")
    p.add_argument("--tag-suffix", default="_bundle")
    p.add_argument(
        "--tag", nargs="+", default=None, help="override a rung's tag: rung2=<tag> rung3=<tag>"
    )
    p.add_argument("--boxes-dir", required=True)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "subtlety_table_val.csv"))
    p.set_defaults(func=cmd_subtlety)

    p = sub.add_parser("silent-images", help="lesions a member does not fire on at all")
    p.add_argument("--weights", required=True)
    p.add_argument("--tensors-dir", default=str(config.OUTPUTS_DIR / "localiser_tensors_v3"))
    p.add_argument("--threshold", type=float, default=config.LOC_MASK_THRESHOLD)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--peaks",
        default=None,
        help="reuse a peaks CSV already computed, instead of running inference again",
    )
    p.add_argument("--out", default=str(config.RESULTS_DIR / "l6_silent_images.json"))
    p.set_defaults(func=cmd_silent_images)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

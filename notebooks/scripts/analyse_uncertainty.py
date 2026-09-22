#!/usr/bin/env python
"""Can the localiser flag its own failures, and is acting on it worth it?
Descriptive, enters no family; test data are refused outside the test pass.

  scores two disagreement scores per lesion against localisation failure, as AUROC
  deferral the retained set when the most-disagreeing cases are referred, against random
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def bundle(bundle_dir: str | None = None):
    """The bundle module beside this one, pointed at bundle_dir for its maps and
    selection, so the scores reuse its own map loading and box selection.
    """
    spec = importlib.util.spec_from_file_location(
        "localiser_bundle", Path(__file__).resolve().parent / "localiser_bundle.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("localiser_bundle", m)
    spec.loader.exec_module(m)
    if bundle_dir:
        m._set_bundle_dir(bundle_dir)
    return m


# Scores: does the ensemble's disagreement predict its own failures?
def _member_boxes_and_maps(lb, split: str, ens: tuple[str, ...], cfg: dict, limit: int | None):
    """Per lesion: each member's own selected box, the unweighted ensemble map's box, and
    the member maps' SD inside that box."""
    meta, gts, breasts = lb._load_split_side_tables(split, limit)
    maps = {
        r: (
            np.load(lb.MAPS / f"{r}_{split}_plain.npy", mmap_mode="r"),
            np.load(lb.MAPS / f"{r}_{split}_flip.npy", mmap_mode="r"),
        )
        for r in ens
    }
    n = len(meta)
    per_member: list[list] = []
    ens_boxes: list = []
    map_sd: list[float] = []
    for i in range(n):
        plain, flip = lb._load_row_maps(maps, i)
        member_maps = [lb._ensemble_map(plain, flip, (r,), cfg["tta"]) for r in ens]
        boxes = []
        for m in member_maps:
            b, _ = lb._select_box(
                m,
                gt=None,
                breast=breasts[i],
                tta=cfg["tta"],
                breast_filter=cfg["breast_filter"],
                rule=cfg["rule"],
                threshold=cfg["threshold"],
                min_px=cfg["min_px"],
            )
            boxes.append(b)
        per_member.append(boxes)

        em = lb._ensemble_map(plain, flip, ens, cfg["tta"])
        eb, _ = lb._select_box(
            em,
            gt=None,
            breast=breasts[i],
            tta=cfg["tta"],
            breast_filter=cfg["breast_filter"],
            rule=cfg["rule"],
            threshold=cfg["threshold"],
            min_px=cfg["min_px"],
        )
        ens_boxes.append(eb)
        # SD inside the unweighted ensemble map's box; the breast box when it produced
        # none, so the fallback cases still get a score instead of a missing value.
        region = eb if eb is not None else breasts[i]
        stack = np.stack(member_maps).astype(np.float32)
        if region is not None:
            stack = stack[:, region[0] : region[1] + 1, region[2] : region[3] + 1]
        map_sd.append(float(stack.std(axis=0).mean()) if stack.size else np.nan)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"[uncertainty] {split}: {i + 1}/{n} lesions", flush=True)
    return meta.reset_index(drop=True), per_member, ens_boxes, map_sd


def _box_disagreement(lb, boxes: list) -> float:
    """1 - mean pairwise IoU; a member with no box contributes 0 to its pairs."""
    ious = []
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a] is None or boxes[b] is None:
                ious.append(0.0)
            else:
                ious.append(lb._iou(boxes[a], boxes[b]))
    return 1.0 - float(np.mean(ious)) if ious else np.nan


def _cluster_bootstrap_auc(
    y: np.ndarray, s: np.ndarray, groups: np.ndarray, draws: int = 1000, seed: int = config.SEED
) -> tuple[float, float]:
    """Patient-cluster bootstrap CI for an AUROC: resample patients, not lesions."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, ss))
    if len(vals) < draws:
        print(f"[uncertainty] bootstrap: skipped {draws - len(vals)} of {draws} one-class draws")
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cmd_scores(args) -> None:
    """Box and map disagreement per lesion, from the cached member maps, against the
    gate's own outcome: miss-inclusive localisation failure (box IoU 0).
    """
    lb = bundle(args.bundle_dir)
    sel = json.loads((lb.BUNDLE / "bundle_selection.json").read_text())
    cfg = sel["selected"]
    ens = tuple(cfg["ensemble"].split("+"))
    lb._assert_member_frames(ens)
    print(f"[uncertainty] promoted configuration {sel.get('selected_cfg_id')}: {cfg}")

    meta, per_member, ens_boxes, map_sd = _member_boxes_and_maps(
        lb, args.split, ens, cfg, args.limit
    )

    boxes_path = Path(args.boxes_dir) / f"localiser_boxes_{args.split}.csv"
    promoted = pd.read_csv(boxes_path)
    promoted = promoted.set_index("lesion_id").loc[meta["lesion_id"].to_numpy()].reset_index()
    # Outcome: miss-inclusive localisation failure — the ensemble's per-lesion box IoU
    # with the ground truth equal to 0, fallbacks included, as the gate scores it.
    iou = promoted["box_iou"].fillna(0.0).to_numpy(float)
    fail = (iou <= 0.0).astype(int)

    df = pd.DataFrame(
        {
            "lesion_id": meta["lesion_id"],
            "patient_id": meta["patient_id"],
            "box_disagreement": [_box_disagreement(lb, b) for b in per_member],
            "map_disagreement": map_sd,
            "ensemble_box_iou": iou,
            "localisation_failure": fail,
            "n_members_with_box": [sum(b is not None for b in bl) for bl in per_member],
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"-> {out}")

    rate = fail.mean()
    print(
        f"\n[uncertainty] {args.split}: {len(df)} lesions, "
        f"{int(fail.sum())} miss-inclusive failures ({rate:.1%}), "
        f"{meta['patient_id'].nunique()} patients"
    )
    if len(np.unique(fail)) < 2:
        raise SystemExit("[uncertainty] outcome has one class; AUROC undefined")

    summary = {
        "split": args.split,
        "n": int(len(df)),
        "n_failures": int(fail.sum()),
        "failure_rate": round(float(rate), 4),
        "n_patients": int(meta["patient_id"].nunique()),
        "configuration": cfg,
        "draws": args.draws,
        "scores": {},
    }
    for score in ("box_disagreement", "map_disagreement"):
        s = df[score].to_numpy(float)
        ok = ~np.isnan(s)
        if not ok.all():
            print(f"[uncertainty] {score}: {int((~ok).sum())} lesions with no score left out")
        a = float(roc_auc_score(fail[ok], s[ok]))
        lo, hi = _cluster_bootstrap_auc(
            fail[ok], s[ok], df["patient_id"].to_numpy()[ok], args.draws
        )
        summary["scores"][score] = {
            "auroc": round(a, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "n_scored": int(ok.sum()),
        }
        verdict = "informative" if lo > 0.5 else "not distinguishable from chance"
        print(f"[uncertainty] {score}: AUROC {a:.4f} [{lo:.4f}, {hi:.4f}] -> {verdict}")

    jpath = out.with_suffix(".json")
    jpath.write_text(json.dumps(summary, indent=2))
    print(f"-> {jpath}")


# Deferral: would acting on the disagreement help?
def _refuse_test_outside_pass(tables: dict[str, pd.DataFrame]) -> None:
    """Refuses tables with test-partition rows (a partition or split column equal to
    test) unless the test pass has set its token (test_pass_guard.authorised)."""
    from src.test_pass_guard import authorised

    if authorised():
        return
    test = [
        name
        for name, df in tables.items()
        for col in ("partition", "split")
        if col in df.columns and (df[col].astype(str) == "test").any()
    ]
    if test:
        raise SystemExit(
            f"REFUSED: test-partition tables ({', '.join(sorted(set(test)))}) are read only "
            "by notebooks/scripts/run_test_pass.sh"
        )


def _retained_metrics(sub: pd.DataFrame) -> dict:
    """The three retained-set quantities; AUC is nan if one class is left."""
    y = sub["y_true"].to_numpy(int)
    out = {
        "n": int(len(sub)),
        "failure_rate": float((sub["ensemble_box_iou"] <= 0).mean()),
        "mean_box_iou": float(sub["ensemble_box_iou"].mean()),
    }
    out["classifier_auc"] = (
        float(roc_auc_score(y, sub["y_prob"].to_numpy(float))) if len(np.unique(y)) > 1 else np.nan
    )
    return out


def cmd_deferral(args) -> None:
    """The retained set at each deferral rate, scored three ways against a random
    baseline of the same size; only distance from that band is evidence.
    """
    unc = pd.read_csv(args.uncertainty)
    pred = pd.read_csv(args.predictions)
    _refuse_test_outside_pass({"uncertainty": unc, "predictions": pred})
    df = unc.merge(pred[["lesion_id", "y_true", "y_prob"]], on="lesion_id", how="inner")
    if len(df) != len(unc):
        raise SystemExit(
            f"join lost rows: {len(unc)} uncertainty vs {len(df)} joined; "
            "the tables are not the same lesions"
        )
    print(
        f"[deferral] {len(df)} lesions, "
        f"{int((df['ensemble_box_iou'] <= 0).sum())} miss-inclusive failures, "
        f"ranking by {args.score}"
    )

    rng = np.random.default_rng(config.SEED)
    order = np.argsort(-df[args.score].to_numpy(float))  # most uncertain first
    rows = []
    for d in np.arange(0.0, args.max_deferral + 1e-9, args.step):
        n_def = int(round(d * len(df)))
        keep = order[n_def:]
        m = _retained_metrics(df.iloc[keep])
        rec = {"deferral_rate": round(float(d), 3), "n_deferred": n_def, **m}
        if n_def == 0:
            for k in ("failure_rate", "mean_box_iou", "classifier_auc"):
                rec[f"random_{k}"] = m[k]
                rec[f"random_{k}_lo"] = rec[f"random_{k}_hi"] = m[k]
        else:
            draws = [
                _retained_metrics(df.iloc[rng.permutation(len(df))[n_def:]])
                for _ in range(args.draws)
            ]
            for k in ("failure_rate", "mean_box_iou", "classifier_auc"):
                v = np.array([x[k] for x in draws], float)
                if np.isnan(v).any():
                    print(
                        f"[deferral] {k} at {d:.2f}: skipped {int(np.isnan(v).sum())} of "
                        f"{args.draws} random draws with one class retained"
                    )
                rec[f"random_{k}"] = float(np.nanmean(v))
                rec[f"random_{k}_lo"] = float(np.nanpercentile(v, 2.5))
                rec[f"random_{k}_hi"] = float(np.nanpercentile(v, 97.5))
        rows.append(rec)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"-> {args.out}")

    print(
        f"\n{'defer':>6s} {'n':>4s} | {'fail':>6s} {'random':>15s} | "
        f"{'boxIoU':>6s} {'random':>15s} | {'clfAUC':>6s} {'random':>15s}"
    )
    for r in rows:
        print(
            f"{r['deferral_rate']:6.2f} {r['n']:4d} | "
            f"{r['failure_rate']:6.3f} "
            f"[{r['random_failure_rate_lo']:.3f},{r['random_failure_rate_hi']:.3f}] | "
            f"{r['mean_box_iou']:6.3f} "
            f"[{r['random_mean_box_iou_lo']:.3f},{r['random_mean_box_iou_hi']:.3f}] | "
            f"{r['classifier_auc']:6.3f} "
            f"[{r['random_classifier_auc_lo']:.3f},{r['random_classifier_auc_hi']:.3f}]"
        )

    # A rule is useful only where it leaves the random band; say so explicitly
    # rather than leaving it to the reader's eye.
    verdicts = {}
    for k, better in (
        ("failure_rate", "lower"),
        ("mean_box_iou", "higher"),
        ("classifier_auc", "higher"),
    ):
        outside = [
            r["deferral_rate"]
            for r in rows
            if r["n_deferred"] > 0
            and (r[k] < r[f"random_{k}_lo"] if better == "lower" else r[k] > r[f"random_{k}_hi"])
        ]
        verdicts[k] = outside
        print(
            f"\n{k}: outside the random band at deferral rates "
            f"{outside if outside else 'NONE — null result on this measure'}"
        )

    # Composition of the deferred and retained sets: a bad crop scores low, which is
    # correct on a benign lesion and wrong on a malignant one, so a deferred set rich in
    # failures should separate the classes poorly.
    n_def = int(round(args.composition_at * len(df)))
    parts = {"deferred": df.iloc[order[:n_def]], "retained": df.iloc[order[n_def:]]}
    comp = {"deferral_rate": args.composition_at, "n_deferred": n_def}
    print(f"\n=== composition at {args.composition_at:.0%} deferral ===")
    print(
        f"{'set':10s} {'n':>4s} {'malig':>6s} {'prev':>6s} {'AUC':>7s} "
        f"{'mean p|benign':>14s} {'mean p|malig':>13s} {'gap':>7s} {'fail rate':>10s}"
    )
    for name, sub in parts.items():
        y = sub["y_true"].to_numpy(int)
        p_ = sub["y_prob"].to_numpy(float)
        a = float(roc_auc_score(y, p_)) if len(np.unique(y)) > 1 else float("nan")
        mb = float(p_[y == 0].mean()) if (y == 0).any() else float("nan")
        mm = float(p_[y == 1].mean()) if (y == 1).any() else float("nan")
        fr = float((sub["ensemble_box_iou"] <= 0).mean())
        comp[name] = {
            "n": int(len(sub)),
            "n_malignant": int(y.sum()),
            "prevalence": round(float(y.mean()), 4),
            "auc": None if np.isnan(a) else round(a, 4),
            "mean_prob_benign": round(mb, 4),
            "mean_prob_malignant": round(mm, 4),
            "score_gap": round(mm - mb, 4),
            "failure_rate": round(fr, 4),
        }
        print(
            f"{name:10s} {len(sub):4d} {int(y.sum()):6d} {y.mean():6.3f} {a:7.4f} "
            f"{mb:14.4f} {mm:13.4f} {mm - mb:7.4f} {fr:10.3f}"
        )
    d_gap, r_gap = comp["deferred"]["score_gap"], comp["retained"]["score_gap"]
    d_auc, r_auc = comp["deferred"]["auc"], comp["retained"]["auc"]
    supported = d_auc is not None and r_auc is not None and d_auc < r_auc and d_gap < r_gap
    comp["hypothesis_supported"] = bool(supported)
    print(
        "\nhypothesis (deferred separates worse; benign and malignant both score low): "
        f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}"
    )
    print(f"   deferred AUC {d_auc} vs retained {r_auc}; score gap {d_gap} vs {r_gap}")

    (Path(args.out).with_suffix(".json")).write_text(
        json.dumps(
            {
                "n": int(len(df)),
                "score": args.score,
                "random_draws": args.draws,
                "outside_random_band": verdicts,
                "composition_12b": comp,
                "curve": rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"-> {Path(args.out).with_suffix('.json')}")


# CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scores", help="does member disagreement predict localisation failure?")
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "train"],
        help="val or train; test is refused",
    )
    p.add_argument(
        "--boxes-dir",
        required=True,
        help="boxes folder of the configuration in --bundle-dir's bundle_selection.json; "
        "its localiser_boxes_{split}.csv gives the outcome (box_iou). The promoted folder "
        "is boxes_dir in refs/promoted_pointer.json",
    )
    p.add_argument("--bundle-dir", default=str(config.LOC_BUNDLE_DIR))
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None, help="smoke: first N lesions")
    p.add_argument(
        "--out",
        required=True,
        help="scores table to write (.csv); the summary is written beside it as .json",
    )
    p.set_defaults(func=cmd_scores)

    p = sub.add_parser("deferral", help="the retained set under deferral, against random")
    p.add_argument("--uncertainty", required=True, help="the scores table this tool wrote")
    p.add_argument(
        "--predictions",
        required=True,
        help="the classifier's prediction table (lesion_id, y_true, y_prob)",
    )
    p.add_argument(
        "--score",
        default="box_disagreement",
        help="scores-table column to rank by, highest first",
    )
    p.add_argument("--max-deferral", type=float, default=0.5)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--draws", type=int, default=200, help="random-deferral draws per rate")
    p.add_argument(
        "--composition-at",
        type=float,
        default=0.25,
        help="deferral rate at which the label composition and score separation of the "
        "deferred and retained sets are reported",
    )
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_deferral)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

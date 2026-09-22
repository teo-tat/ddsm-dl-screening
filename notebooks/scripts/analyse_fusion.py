#!/usr/bin/env python
"""analyse_fusion.py — does a second view help, and where does a learned combiner earn it?

Every comparison is on the same pairs, paired at the pair level, so none of it is
confounded by scoring fusion and rung 3 on different evaluation sets.

  controls rung 3 restricted to the fusion pairs, scored per pair: one view only, and the
    mean of both
  compare the fusion system against those controls, Holm-corrected, plus the oracle-box gap
  stratify the pairs split by whether either view's box failed, and where fusion's advantage
    sits
"""

from __future__ import annotations

import argparse
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

RESULTS = config.RESULTS_DIR


def auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def _cluster_bootstrap(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, draws: int = 1000, seed: int = config.SEED
) -> tuple[float, float]:
    """Patient-cluster bootstrap CI for an AUC: resample patients, not pairs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(draws):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return (
        (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        if vals
        else (float("nan"), float("nan"))
    )


def paired_bootstrap(y, pa, pb, groups, draws=1000, seed=config.SEED):
    """Patient-cluster bootstrap of AUC(a) - AUC(b) on the same resample."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    d = []
    for _ in range(draws):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
        int(sum(x > 0 for x in d)),
        len(d),
    )


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down, returned in the input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def _refuse_test_outside_pass(split: str | None, tables: dict[str, pd.DataFrame]) -> None:
    """Refuses test-partition tables unless the test pass has set its token
    (test_pass_guard.authorised): `split` is the --split flag, and a table records its rows
    in a partition or split column."""
    from src.test_pass_guard import authorised

    if authorised():
        return
    test = [
        name
        for name, df in tables.items()
        for col in ("partition", "split")
        if col in df.columns and (df[col].astype(str) == "test").any()
    ]
    if split == "test" or test:
        raise SystemExit(
            "REFUSED: test-partition tables"
            + (f" ({', '.join(sorted(set(test)))})" if test else "")
            + " are read only by notebooks/scripts/run_test_pass.sh"
        )


# Controls: rung 3 restricted to the fusion pairs
def cmd_controls(args) -> None:
    """Rung 3 on the fusion store's own pair rows, scored per pair two ways: view_a_only
    (one view against two) and mean_a_b (naive averaging, the floor learned fusion must beat).
    """
    pairs = pd.read_csv(args.pairs)
    r3 = pd.read_csv(args.rung3).set_index("lesion_id")
    _refuse_test_outside_pass(args.split, {"pairs": pairs, "rung 3": r3})
    print(f"[pairs] {len(pairs)} pairs from {args.pairs}")
    print(f"[pairs] rung 3 table: {len(r3)} lesions from {Path(args.rung3).name}")

    missing = [c for c in ("lesion_id_a", "lesion_id_b", "label") if c not in pairs.columns]
    if missing:
        raise SystemExit(f"pair table lacks {missing}")
    for col in ("lesion_id_a", "lesion_id_b"):
        absent = set(pairs[col]) - set(r3.index)
        if absent:
            raise SystemExit(
                f"{len(absent)} {col} not in the rung-3 table, e.g. {sorted(absent)[:3]}"
            )

    pa = r3.loc[pairs["lesion_id_a"], "y_prob"].to_numpy(float)
    pb = r3.loc[pairs["lesion_id_b"], "y_prob"].to_numpy(float)
    ya = r3.loc[pairs["lesion_id_a"], "y_true"].to_numpy(int)
    yb = r3.loc[pairs["lesion_id_b"], "y_true"].to_numpy(int)
    y = pairs["label"].to_numpy(int)
    if not np.array_equal(ya, y):
        raise SystemExit("crop A's rung-3 label disagrees with the pair label; rows are misaligned")
    # The two views of one breast should carry one diagnosis. Report it if they
    # do not, because then the pair label is not a property of the breast.
    n_disagree = int((ya != yb).sum())
    print(
        f"[pairs] pairs whose two views carry different labels: {n_disagree}"
        + ("  <- the pair label is crop A's, as fusion defines it" if n_disagree else "")
    )

    groups = pairs["patient_id"].to_numpy()
    out = {
        "pairs": str(args.pairs),
        "rung3_table": str(args.rung3),
        "n_pairs": int(len(pairs)),
        "n_malignant": int(y.sum()),
        "n_patients": int(pd.Series(groups).nunique()),
        "n_pairs_with_discordant_view_labels": n_disagree,
        "controls": {},
    }

    print(f"\n{'control':14s} {'AUC':>7s}  {'95% CI':>18s}   what it answers")
    for name, p_, what in (
        ("view_a_only", pa, "one view vs two"),
        ("mean_a_b", (pa + pb) / 2.0, "learned fusion vs naive averaging"),
    ):
        a = float(roc_auc_score(y, p_))
        lo, hi = _cluster_bootstrap(y, p_, groups, args.draws)
        out["controls"][name] = {"auc": round(a, 6), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"{name:14s} {a:7.4f}  [{lo:.4f}, {hi:.4f}]   {what}")

    # Context only: the --rung3 table's AUC on all its lesions. The launchers pass the
    # three-seed ensemble of the localiser-box classifier (rung 3, ens3). It is not
    # comparable with the rows above, which cover the pairs only.
    full = float(roc_auc_score(r3["y_true"], r3["y_prob"]))
    out["rung3_all_lesions_for_context"] = round(full, 6)
    print(
        f"\nrung 3 on all {len(r3)} lesions, context only (different rows, not a control): "
        f"{full:.4f}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")

    tbl = pd.DataFrame(
        {
            "lesion_id_a": pairs["lesion_id_a"],
            "lesion_id_b": pairs["lesion_id_b"],
            "patient_id": groups,
            "y_true": y,
            "y_prob_view_a_only": pa,
            "y_prob_mean_a_b": (pa + pb) / 2.0,
        }
    )
    tpath = out_dir / f"{args.tag}_predictions_{args.split}.csv"
    tbl.to_csv(tpath, index=False)
    print(f"-> {tpath}")


# Compare: the deployable system against its controls
def cmd_compare(args) -> None:
    """Two pre-specified tests, Holm-corrected across the two: fusion ens3 against naive
    two-view averaging, and against one-view rung 3. The oracle-box gap is descriptive.
    """
    from src.evaluate import delong_test

    fus = pd.read_csv(args.fusion)
    ctl = pd.read_csv(args.controls)
    orc = pd.read_csv(args.oracle_fusion)
    _refuse_test_outside_pass(args.split, {"fusion": fus, "controls": ctl, "oracle fusion": orc})
    # Every table that records its partition must record this split and only it; the
    # controls table has no partition column and is tied to the split by its pair ids.
    bad = []
    for name, path, df in (
        ("fusion", args.fusion, fus),
        ("oracle fusion", args.oracle_fusion, orc),
    ):
        parts = sorted(map(str, df["partition"].unique())) if "partition" in df.columns else None
        if parts != [args.split]:
            bad.append(f"  {name}: partition {parts} in {path}")
    if bad:
        raise SystemExit(
            f"REFUSED: --split {args.split}, but not every table belongs to it:\n" + "\n".join(bad)
        )
    key = "lesion_id" if "lesion_id" in fus.columns else "lesion_id_a"
    fus = fus.set_index(key)
    orc = orc.set_index("lesion_id" if "lesion_id" in orc.columns else "lesion_id_a")
    ctl = ctl.set_index("lesion_id_a")
    ids = ctl.index.to_numpy()
    for name, df in (("fusion", fus), ("oracle fusion", orc)):
        missing = set(ids) - set(df.index)
        if missing:
            raise SystemExit(f"{len(missing)} pairs missing from the {name} table")

    y = ctl["y_true"].to_numpy(int)
    groups = ctl["patient_id"].to_numpy()
    for name, df in (("fusion", fus), ("oracle fusion", orc)):
        if not np.array_equal(df.loc[ids, "y_true"].to_numpy(int), y):
            raise SystemExit(f"labels disagree between the {name} and control tables")

    series = {
        "fusion_ens3_unet": fus.loc[ids, "y_prob"].to_numpy(float),
        "naive_averaging": ctl["y_prob_mean_a_b"].to_numpy(float),
        "one_view_rung3": ctl["y_prob_view_a_only"].to_numpy(float),
        "fusion_ens3_oracle": orc.loc[ids, "y_prob"].to_numpy(float),
    }
    aucs = {k: float(roc_auc_score(y, v)) for k, v in series.items()}

    print(
        f"=== {len(y)} {args.split} pairs, {pd.Series(groups).nunique()} patients, "
        f"{int(y.sum())} malignant ===\n"
    )
    for k, v in aucs.items():
        print(f"  {k:22s} {v:.4f}")

    # The two pre-specified tests, Holm-corrected across the two.
    tests = [
        ("fusion_ens3_unet", "naive_averaging", "(i)  learned fusion vs naive two-view averaging"),
        ("fusion_ens3_unet", "one_view_rung3", "(ii) learned fusion vs one view"),
    ]
    rows, pvals = [], []
    for a, b, label in tests:
        d = delong_test(y, series[a], series[b])
        p = float(d["p_value"] if "p_value" in d else d.get("p"))
        lo, hi, n_pos, n_draw = paired_bootstrap(y, series[a], series[b], groups, args.draws)
        rows.append(
            {
                "test": label,
                "a": a,
                "b": b,
                "auc_a": aucs[a],
                "auc_b": aucs[b],
                "diff": aucs[a] - aucs[b],
                "ci95": [lo, hi],
                "p_delong": p,
                "bootstrap_frac_positive": n_pos / max(n_draw, 1),
            }
        )
        pvals.append(p)
    for r, ph in zip(rows, holm(pvals)):
        r["p_holm"] = ph
        r["significant_holm_005"] = bool(ph < 0.05)

    print("\n=== Holm family of two (pre-specified) ===")
    for r in rows:
        print(f"  {r['test']}")
        print(
            f"      {r['auc_a']:.4f} vs {r['auc_b']:.4f}  diff {r['diff']:+.4f} "
            f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        )
        print(
            f"      DeLong p {r['p_delong']:.4f}  Holm {r['p_holm']:.4f}  "
            f"-> {'SIGNIFICANT' if r['significant_holm_005'] else 'not significant'}"
        )

    # Descriptive, outside the family: what the localiser still costs the pair system.
    lo, hi, n_pos, n_draw = paired_bootstrap(
        y, series["fusion_ens3_oracle"], series["fusion_ens3_unet"], groups, args.draws
    )
    gap = aucs["fusion_ens3_oracle"] - aucs["fusion_ens3_unet"]
    desc = {
        "a": "fusion_ens3_oracle",
        "b": "fusion_ens3_unet",
        "auc_a": aucs["fusion_ens3_oracle"],
        "auc_b": aucs["fusion_ens3_unet"],
        "diff": gap,
        "ci95": [lo, hi],
        "in_holm_family": False,
    }
    print("\n=== Descriptive, outside the family ===")
    print(
        f"  oracle-box fusion vs localiser-box fusion: {aucs['fusion_ens3_oracle']:.4f} vs "
        f"{aucs['fusion_ens3_unet']:.4f}  gap {gap:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    )
    print("  (the localiser's residual cost to the paired system)")

    out = {
        "n_pairs": int(len(y)),
        "n_patients": int(pd.Series(groups).nunique()),
        "n_malignant": int(y.sum()),
        "draws": args.draws,
        "aucs": aucs,
        "holm_family": rows,
        "descriptive_oracle_gap": desc,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


# Stratify: where the combiner's advantage sits
def cmd_stratify(args) -> None:
    """Splits the pairs on both views' boxes — `both_ok` (both IoU > 0) against `any_failed`
    — and asks where the learned combiner beats averaging. Descriptive; selects nothing.
    """
    from src.evaluate import bootstrap_ci

    boxes_path = Path(args.boxes_dir) / "localiser_boxes_val.csv"
    fus = pd.read_csv(args.fusion)
    ctl = pd.read_csv(args.controls)
    box = pd.read_csv(boxes_path)
    _refuse_test_outside_pass(None, {"fusion": fus, "controls": ctl})
    box = box[box["status"] != "no_gt"][["lesion_id", "box_iou"]].copy()
    box["box_iou"] = box["box_iou"].fillna(0.0)
    iou = dict(zip(box["lesion_id"], box["box_iou"]))

    # The fusion table is one row per pair, keyed by crop A's lesion_id; the control
    # table carries both ids.
    df = ctl.merge(
        fus[["lesion_id", "y_prob"]].rename(
            columns={"lesion_id": "lesion_id_a", "y_prob": "y_prob_fusion"}
        ),
        on="lesion_id_a",
        validate="1:1",
    )
    if len(df) < len(ctl):
        print(f"[stratify] {len(ctl) - len(df)} control pairs have no fusion row; skipped")
    missing = [l for l in list(df["lesion_id_a"]) + list(df["lesion_id_b"]) if l not in iou]
    if missing:
        raise SystemExit(
            f"{len(missing)} lesion ids are absent from the box table; "
            f"the pairs and the boxes are not the same condition - stop"
        )
    df["iou_a"] = df["lesion_id_a"].map(iou)
    df["iou_b"] = df["lesion_id_b"].map(iou)
    df["stratum"] = np.where((df["iou_a"] > 0) & (df["iou_b"] > 0), "both_ok", "any_failed")

    out = {
        "n_pairs": int(len(df)),
        "boxes": str(boxes_path),
        "fusion": args.fusion,
        "controls": args.controls,
        "strata": {},
    }
    print(f"pairs {len(df)}  |  boxes {boxes_path.parent.name}")
    for s in ("both_ok", "any_failed"):
        d = df[df["stratum"] == s]
        if len(d) == 0:
            print(f"  {s}: empty")
            continue
        y = d["y_true"].to_numpy(int)
        af, ag = auc(y, d["y_prob_fusion"]), auc(y, d["y_prob_mean_a_b"])
        diff = af - ag
        point, lo, hi = bootstrap_ci(
            y,
            np.c_[d["y_prob_fusion"].to_numpy(), d["y_prob_mean_a_b"].to_numpy()],
            lambda yy, pp: auc(yy, pp[:, 0]) - auc(yy, pp[:, 1]),
            groups=d["patient_id"].to_numpy(),
        )
        out["strata"][s] = {
            "n": int(len(d)),
            "n_malig": int(y.sum()),
            "n_patients": int(d["patient_id"].nunique()),
            "auc_fusion": af,
            "auc_avg": ag,
            "diff": diff,
            "ci95": [float(lo), float(hi)],
        }
        print(
            f"  {s:11} n={len(d):3} malig={int(y.sum()):3}  fusion {af:.4f}  "
            f"averaging {ag:.4f}  diff {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"-> {args.out}")


# CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("controls", help="rung 3 restricted to the fusion pairs, scored per pair")
    p.add_argument(
        "--pairs",
        required=True,
        help="the fusion pair table (a store's {split}_meta.csv or "
        "artifacts/fusion_pairs_view_{split}.csv); defines the rows, order and labels",
    )
    p.add_argument("--rung3", required=True, help="rung 3's per-lesion prediction table")
    p.add_argument(
        "--tag",
        required=True,
        help="tag of the control table this writes, e.g. rung3_on_pairs",
    )
    p.add_argument(
        "--out",
        default=str(RESULTS),
        help="folder for {tag}.json and {tag}_predictions_{split}.csv",
    )
    p.add_argument("--draws", type=int, default=1000)
    # Names the per-pair table by the split it was computed on.
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="the split the pairs and rung-3 tables come from; names the table",
    )
    p.set_defaults(func=cmd_controls)

    p = sub.add_parser("compare", help="the fusion system against its controls, Holm-corrected")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--fusion", required=True, help="learned fusion ens3 table, localiser boxes")
    p.add_argument("--controls", required=True, help="the controls table for the same split")
    p.add_argument("--oracle-fusion", required=True, help="learned fusion ens3 table, oracle boxes")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("stratify", help="the combiner's advantage where a view's box failed")
    p.add_argument("--fusion", required=True)
    p.add_argument("--controls", required=True)
    p.add_argument(
        "--boxes-dir",
        required=True,
        help="boxes folder; localiser_boxes_val.csv inside it defines the strata",
    )
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_stratify)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

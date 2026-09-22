#!/usr/bin/env python
"""Applies the classifier-ceiling decision rule to validation prediction tables.

ceiling the bar from the m = 0.20 seeds, and every configuration against it
read-session one session's ledger: ladder contrasts, recovery, ceiling, Family A
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

RESULTS = ROOT / "outputs" / "results"
DEFAULT_OUT = RESULTS / "ceiling_26"

# The bar comes from five seeds at m = 0.20 on the box_provider(oracle) route, and
# is recomputed from them before the rule touches any configuration.
BASELINE_TAGS = [
    "vgg16_crop_rung2_oracle_cuda",
    "vgg16_crop_rung2_oracle_s1_cuda",
    "vgg16_crop_rung2_oracle_s2_cuda",
    "vgg16_crop_rung2_oracle_s3",
    "vgg16_crop_rung2_oracle_s4",
]
SEED28 = "vgg16_crop_rung2_oracle_cuda"
FLOOR = 0.01

# The margin sweep on the operational route: three seeds per margin, so no margin
# rests on one run. m = 0.20's three are the first three baseline tags.
MARGINS = {
    "0.10": [
        "vgg16_crop_rung2_oracle_m010",
        "vgg16_crop_rung2_oracle_m010_s1",
        "vgg16_crop_rung2_oracle_m010_s2",
    ],
    "0.20 (incumbent)": BASELINE_TAGS[:3],
    "0.30": [
        "vgg16_crop_rung2_oracle_m030",
        "vgg16_crop_rung2_oracle_m030_s1",
        "vgg16_crop_rung2_oracle_m030_s2",
    ],
    "0.40": [
        "vgg16_crop_rung2_oracle_m040",
        "vgg16_crop_rung2_oracle_m040_s1",
        "vgg16_crop_rung2_oracle_m040_s2",
    ],
}
# Single-seed configurations, reported as descriptive.
SINGLE = {
    "c. patience 50": "vgg16_crop_rung2_oracle_p50",
    "d. resnet50": "resnet50_crop_rung2_oracle",
    "d. densenet121": "densenet121_crop_rung2_oracle",
    "d. efficientnet": "efficientnet_crop_rung2_oracle",
}
ENS5 = BASELINE_TAGS
ENS4B = [
    "vgg16_crop_rung2_oracle_cuda",
    "resnet50_crop_rung2_oracle",
    "densenet121_crop_rung2_oracle",
    "efficientnet_crop_rung2_oracle",
]
# The route comparison needs no runs; both sides are read from --results-dir.
ROUTE_MASK = [
    "vgg16_crop_m020_p25_cuda",
    "vgg16_crop_m020_p25_s1_cuda",
    "vgg16_crop_m020_p25_s2_cuda",
]


def table(tag: str, results: Path) -> Path | None:
    p = results / f"{tag}_predictions_val.csv"
    return p if p.exists() else None


def auc(tag: str, results: Path) -> float | None:
    p = table(tag, results)
    if p is None:
        return None
    d = pd.read_csv(p)
    return float(roc_auc_score(d["y_true"], d["y_prob"]))


def build_ensemble(tags: list[str], out_tag: str, results: Path) -> float | None:
    """Mean of y_prob over the rows the tables share, computed here because
    src.ensemble resolves its inputs from outputs/results/ only.
    """
    missing = [t for t in tags if table(t, results) is None]
    if missing:
        print(f"  {out_tag}: not built — missing {', '.join(missing)}")
        return None
    frames = []
    for t in tags:
        d = pd.read_csv(table(t, results)).drop_duplicates("lesion_id").set_index("lesion_id")
        frames.append(d)
    idx = frames[0].index
    for d in frames[1:]:
        idx = idx.intersection(d.index)
    y = frames[0].loc[idx, "y_true"].to_numpy()
    prob = np.mean([d.loc[idx, "y_prob"].to_numpy() for d in frames], axis=0)
    out = frames[0].loc[idx].copy()
    out["y_prob"] = prob
    out.reset_index().to_csv(results / f"{out_tag}_predictions_val.csv", index=False)
    return float(roc_auc_score(y, prob))


def verify_against_src_ensemble(results: Path) -> None:
    """One equivalence check against src.ensemble, on the rung-3 ens3 tables in results;
    skipped when any is absent."""
    ref = results / "vgg16_crop_rung3_unet_bundle_cuda_ens3_predictions_val.csv"
    parts = [
        "vgg16_crop_rung3_unet_bundle_cuda",
        "vgg16_crop_rung3_unet_bundle_s1_cuda",
        "vgg16_crop_rung3_unet_bundle_s2_cuda",
    ]
    if not ref.exists() or any(not (results / f"{t}_predictions_val.csv").exists() for t in parts):
        print("  (equivalence check skipped: the reference ens3 tables are not all present)")
        return
    frames = [
        pd.read_csv(results / f"{t}_predictions_val.csv")
        .drop_duplicates("lesion_id")
        .set_index("lesion_id")
        for t in parts
    ]
    idx = frames[0].index
    for d in frames[1:]:
        idx = idx.intersection(d.index)
    mine = np.mean([d.loc[idx, "y_prob"].to_numpy() for d in frames], axis=0)
    theirs = (
        pd.read_csv(ref)
        .drop_duplicates("lesion_id")
        .set_index("lesion_id")
        .loc[idx, "y_prob"]
        .to_numpy()
    )
    err = float(np.max(np.abs(mine - theirs)))
    if not err < 1e-9:  # the complement of OK below, so NaN is a mismatch too
        raise SystemExit(
            f"REFUSED: equivalence vs src.ensemble on the existing rung-3 ens3: "
            f"max |diff| {err:.2e} (MISMATCH)"
        )
    print(
        f"  equivalence vs src.ensemble on the existing rung-3 ens3: max |diff| {err:.2e} "
        f"({'OK' if err < 1e-9 else 'MISMATCH'})"
    )


# read-session — the declared reading of a one-session ledger
_RS_RESULTS = RESULTS  # replaced by cmd_read_session from --results-dir

DELTA = 0.03  # per row
ROBUST = 0.10  # a contrast's |dAUC| at or above this is robust
FLOOR_GATE = 0.794022  # the floor gate

# Figures from the previous ledger, printed beside the ones recomputed here.
PRIOR = {
    "ceiling_mean": 0.8725,
    "ceiling_sd": 0.0084,
    "ceiling_bar": 0.8977,
    "best_challenger": 0.8694,
    "recovery_fraction": 0.804,
    "residual_over_sd": 1.36,
}

LADDER = {  # rung -> tag stem, in the session's own naming
    1: "vgg16_crop_rung1_oracle_cuda0917",
    2: "vgg16_crop_rung2_oracle_cuda0917",
    3: "vgg16_crop_rung3_unet_promoted_cuda0917",
    4: "vgg16_crop_rung4_cam_cuda0917",
    5: "vgg16_crop_rung5_jitter_promoted_cuda0917",
    6: "vgg16_crop_rung6_shuffled_cuda0917",
    7: "vgg16_crop_rung7_whole_cuda0917",
}
OLD_BOX_RUNG3 = "vgg16_crop_rung3_unet_bundle_cuda0917"
FAMILY_A = {
    "vgg16": "vgg16_crop_m020_cuda0917",
    "resnet50": "resnet50_crop_m020_cuda0917",
    "densenet121": "densenet121_crop_m020_cuda0917",
    "efficientnet": "efficientnet_crop_m020_cuda0917",
}
SCRATCH = {
    "baseline": "baseline_crop_m020_cuda0917",
    "scaled": "scaled_crop_m020_cuda0917",
    "regularised": "regularised_crop_m020_cuda0917",
}
FUSION = {
    "view (oracle boxes)": "vgg16_fusion_view_cuda0917",
    "view (A24 boxes)": "vgg16_fusion_view_unet_promoted_cuda0917",
    "control (one stream)": "vgg16_fusion_control_cuda0917",
}
WHOLE_IMAGE = "vgg16_full_cuda0917"

# The robust ladder pairs, with the sign each is expected to keep.
ROBUST_PAIRS = [
    ((2, 4), "+"),
    ((2, 6), "+"),
    ((2, 7), "+"),
    ((3, 4), "+"),
    ((3, 6), "+"),
    ((3, 7), "+"),
    ((4, 6), "+"),
    ((6, 7), "-"),
]
RESTATED_PAIRS = [(2, 3), (4, 7)]


def exact_auc(tag: str) -> float | None:
    """Exact validation AUC from the prediction table, or None if absent."""
    p = _RS_RESULTS / f"{tag}_predictions_val.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return float(roc_auc_score(d["y_true"].to_numpy(int), d["y_prob"].to_numpy(float)))


def seeds_of(stem: str, extra: tuple[str, ...] = ("_s1", "_s2")) -> dict[str, float]:
    """{tag: exact AUC} for a stem's base run and its replicates."""
    out = {}
    for suffix in ("",) + extra:
        tag = f"{stem}{suffix}"
        v = exact_auc(tag)
        if v is not None:
            out[tag] = v
    return out


def summarise(stem: str, extra: tuple[str, ...] = ("_s1", "_s2")) -> dict:
    vals = seeds_of(stem, extra)
    a = np.array(list(vals.values()), float)
    return {
        "stem": stem,
        "tags": list(vals),
        "aucs": [round(v, 6) for v in vals.values()],
        "n_seeds": int(a.size),
        "mean": float(a.mean()) if a.size else None,
        "sd": float(a.std(ddof=1)) if a.size > 1 else None,
    }


def cmd_read_session(args) -> None:
    global _RS_RESULTS
    _RS_RESULTS = Path(args.results_dir)
    rows_json = Path(args.rows)
    out_path = Path(args.out)
    declared = json.loads(rows_json.read_text())
    have = sum(1 for r in declared if (_RS_RESULTS / f"{r['tag']}_predictions_val.csv").exists())
    print(
        f"[discovered] {have} of {len(declared)} declared rows have a prediction table "
        f"under {_RS_RESULTS}"
    )
    if have == 0:
        raise SystemExit("REFUSED: no scored rows — this reading would be of nothing.")
    if have < len(declared):
        print(
            f"  WARNING: {len(declared) - have} row(s) not scored; every figure below is "
            f"partial and must not be quoted."
        )

    res: dict = {
        "session": "_cuda0917",
        "delta_per_row": DELTA,
        "robust_threshold": ROBUST,
        "rows_declared": len(declared),
        "rows_scored": have,
        "prior": PRIOR,
    }

    # the ladder, three seeds per rung (rung 2 has five)
    print("\n--- the ladder, session means (validation, exact AUC) ---")
    print(f"{'rung':>4} {'n':>2} {'mean':>8} {'SD':>8} seeds")
    rungs = {}
    for k, stem in LADDER.items():
        extra = ("_s1", "_s2", "_s3", "_s4") if k == 2 else ("_s1", "_s2")
        s = summarise(stem, extra)
        rungs[k] = s
        if s["mean"] is not None:
            print(
                f"{k:>4} {s['n_seeds']:>2} {s['mean']:>8.4f} "
                f"{(s['sd'] if s['sd'] is not None else float('nan')):>8.4f} "
                + " ".join(f"{v:.4f}" for v in s["aucs"])
            )
    res["ladder"] = {str(k): v for k, v in rungs.items()}

    print("\n--- robust ladder contrasts (must keep sign) ---")
    contrasts = {}
    for (a, b), want in ROBUST_PAIRS:
        ma, mb = rungs[a]["mean"], rungs[b]["mean"]
        if ma is None or mb is None:
            print(
                f"  rung {a} - rung {b} skipped: no prediction table for any seed of "
                + ", ".join(rungs[k]["stem"] for k in (a, b) if rungs[k]["mean"] is None)
                + f" under {_RS_RESULTS}"
            )
            continue
        d = ma - mb
        kept = (d > 0) if want == "+" else (d < 0)
        contrasts[f"rung{a}_minus_rung{b}"] = {
            "diff": round(d, 6),
            "expected_sign": want,
            "sign_kept": bool(kept),
            "robust_now": abs(d) >= ROBUST,
        }
        flag = "sign kept" if kept else "SIGN CHANGED — report as a finding that invalidates δ"
        print(
            f"  rung {a} - rung {b}: {d:+.4f} (expected {want}) -> {flag}"
            + (
                ""
                if abs(d) >= ROBUST
                else f" [|dAUC| {abs(d):.4f} < {ROBUST}: not robust in this session]"
            )
        )
    print("\n--- contrasts restated from the session (nothing carried over) ---")
    for a, b in RESTATED_PAIRS:
        ma, mb = rungs[a]["mean"], rungs[b]["mean"]
        if ma is None or mb is None:
            print(
                f"  rung {a} - rung {b} skipped: no prediction table for any seed of "
                + ", ".join(rungs[k]["stem"] for k in (a, b) if rungs[k]["mean"] is None)
                + f" under {_RS_RESULTS}"
            )
            continue
        d = ma - mb
        contrasts[f"rung{a}_minus_rung{b}"] = {
            "diff": round(d, 6),
            "restated": True,
            "robust_now": abs(d) >= ROBUST,
        }
        print(
            f"  rung {a} - rung {b}: {d:+.4f} (restated; |dAUC| {abs(d):.4f} "
            f"{'>=' if abs(d) >= ROBUST else '<'} {ROBUST})"
        )
    res["ladder_contrasts"] = contrasts

    # the recovery figure
    print("\n--- the recovery figure, recomputed from session numbers only ---")
    ceiling, old_box, a24 = rungs[2], summarise(OLD_BOX_RUNG3), rungs[3]
    rec: dict = {"ceiling": ceiling, "old_box_rung3": old_box, "a24_rung3": a24}
    if None not in (ceiling["mean"], old_box["mean"], a24["mean"]):
        cost = ceiling["mean"] - old_box["mean"]
        recovered = a24["mean"] - old_box["mean"]
        residual = ceiling["mean"] - a24["mean"]
        sds = [s for s in (ceiling["sd"], old_box["sd"]) if s is not None]
        larger_sd = max(sds) if sds else None
        fraction = recovered / cost if cost else float("nan")
        ratio = residual / ceiling["sd"] if ceiling["sd"] else float("nan")
        interpretable = larger_sd is not None and cost >= 2 * larger_sd
        rec.update(
            cost=round(cost, 6),
            recovered=round(recovered, 6),
            residual=round(residual, 6),
            fraction=round(float(fraction), 4),
            residual_over_ceiling_sd=round(float(ratio), 3),
            larger_seed_sd=(round(larger_sd, 6) if larger_sd else None),
            cost_at_least_2sd=bool(interpretable),
        )
        print(
            f"  ceiling (rung 2, {ceiling['n_seeds']} seeds) {ceiling['mean']:.6f} "
            f"SD {ceiling['sd']:.6f}"
        )
        print(f"  rung 3 on the old boxes ({old_box['n_seeds']} seeds) {old_box['mean']:.6f}")
        print(f"  rung 3 on A24 ({a24['n_seeds']} seeds) {a24['mean']:.6f}")
        print(f"  cost {cost:+.6f} recovered {recovered:+.6f} residual {residual:+.6f}")
        if not interpretable:
            rec["verdict"] = (
                "uninterpretable: the cost is smaller than 2x the larger seed SD, "
                "so the fraction is not quotable"
            )
            print(
                f"  cost < 2 x larger seed SD ({2 * larger_sd:.6f}) -> the fraction is "
                f"UNINTERPRETABLE and is not quoted"
            )
        else:
            keeps = fraction >= 0.50
            rec["verdict"] = (
                "most of the localisation cost is recovered (fraction >= 50%)"
                if keeps
                else "the 'most of the cost is recovered' claim does NOT hold at this "
                "session's numbers (fraction < 50%)"
            )
            print(
                f"  recovery fraction {fraction:.1%} (prior {PRIOR['recovery_fraction']:.1%}) -> "
                f"{'claim holds' if keeps else 'claim does not hold; restate it'}"
            )
            print(
                f"  residual / ceiling seed SD = {ratio:.2f} "
                f"(prior {PRIOR['residual_over_sd']:.2f}) -> "
                + (
                    "'within seed noise' wording is permitted"
                    if ratio <= 1.0
                    else "NO 'within seed noise' wording"
                )
            )
            rec["within_seed_noise_permitted"] = bool(ratio <= 1.0)
    res["recovery_14a"] = rec

    # the ceiling and the bar
    print("\n--- the ceiling and the bar (the decision rule is not re-applied here) ---")
    if ceiling["mean"] is not None and ceiling["sd"] is not None:
        bar = ceiling["mean"] + max(0.01, 3 * ceiling["sd"])
        res["ceiling_26"] = {
            "mean": round(ceiling["mean"], 6),
            "sd": round(ceiling["sd"], 6),
            "bar": round(bar, 6),
            "n_seeds": ceiling["n_seeds"],
            "prior_mean": PRIOR["ceiling_mean"],
            "prior_sd": PRIOR["ceiling_sd"],
            "prior_bar": PRIOR["ceiling_bar"],
            "best_challenger_10sep": PRIOR["best_challenger"],
            "challenger_would_clear": bool(PRIOR["best_challenger"] > bar),
            "rule_reapplied": False,
            "note": "the challengers were not re-measured, so the decision "
            "rule cannot be applied like-for-like; m = 0.20 stands as decided",
        }
        print(
            f"  mean {ceiling['mean']:.6f} (prior {PRIOR['ceiling_mean']:.4f}) "
            f"SD {ceiling['sd']:.6f} (prior {PRIOR['ceiling_sd']:.4f})"
        )
        print(f"  bar {bar:.6f} (prior {PRIOR['ceiling_bar']:.4f})")
        print(
            f"  best prior challenger {PRIOR['best_challenger']:.4f} -> "
            + (
                "would clear the re-measured bar: reported as a cross-session OBSERVATION, "
                "not a decision"
                if PRIOR["best_challenger"] > bar
                else "does not clear the re-measured bar either"
            )
        )

    # Family A
    print("\n--- Family A (Holm family): three-seed means ---")
    fam = {k: summarise(v) for k, v in FAMILY_A.items()}
    for k, s in fam.items():
        if s["mean"] is not None:
            print(
                f"  {k:<13} {s['mean']:.4f} SD {s['sd']:.4f} "
                + " ".join(f"{v:.4f}" for v in s["aucs"])
            )
    pairs = {}
    names = list(FAMILY_A)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if fam[a]["mean"] is None or fam[b]["mean"] is None:
                print(
                    f"  Family A {a} - {b} skipped: no prediction table for any seed of "
                    + ", ".join(fam[k]["stem"] for k in (a, b) if fam[k]["mean"] is None)
                    + f" under {_RS_RESULTS}"
                )
                continue
            d = fam[a]["mean"] - fam[b]["mean"]
            pairs[f"{a}_minus_{b}"] = {"diff": round(d, 6), "robust_now": abs(d) >= ROBUST}
    res["family_a"] = {"runs": fam, "contrasts": pairs}
    print("  contrasts (|dAUC| >= 0.10 is robust; the rest are restated from this session):")
    for k, v in sorted(pairs.items(), key=lambda kv: -abs(kv[1]["diff"])):
        print(f"    {k:<28} {v['diff']:+.4f} {'robust' if v['robust_now'] else 'restated'}")

    # the floor gate
    print("\n--- floor gate at 0.794022 ---")
    if a24["mean"] is not None:
        passes = a24["mean"] >= FLOOR_GATE
        res["floor_gate"] = {
            "floor": FLOOR_GATE,
            "rung3_a24_mean": round(a24["mean"], 6),
            "passes": bool(passes),
        }
        print(
            f"  rung 3 on A24, three-seed mean {a24['mean']:.6f} vs floor {FLOOR_GATE:.6f} -> "
            + (
                "PASS: the pointer may be rewritten"
                if passes
                else "FAIL: the pointer is NOT rewritten and the result comes back"
            )
        )

    # the rest of the session, for the record
    res["scratch"] = {k: summarise(v) for k, v in SCRATCH.items()}
    res["fusion"] = {k: summarise(v) for k, v in FUSION.items()}
    res["whole_image"] = summarise(WHOLE_IMAGE)
    print("\n--- the rest of the session (for the record) ---")
    for label, group in (("scratch", res["scratch"]), ("fusion", res["fusion"])):
        for k, s in group.items():
            if s["mean"] is not None:
                print(
                    f"  {label:<8} {k:<22} {s['mean']:.4f} SD "
                    f"{(s['sd'] if s['sd'] is not None else float('nan')):.4f}"
                )
    if res["whole_image"]["mean"] is not None:
        print(
            f"  whole {'vgg16_full':<22} {res['whole_image']['mean']:.4f} SD "
            f"{res['whole_image']['sd']:.4f}"
        )

    out_path.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n-> {out_path}")


def cmd_ceiling(a) -> None:
    results = Path(a.results_dir)
    out_dir = Path(a.out)

    # the route transfer, from data already on disk
    print("--- the route transfer (no runs needed) ---")
    mask = [auc(t, results) for t in ROUTE_MASK]
    box3 = [auc(t, results) for t in BASELINE_TAGS[:3]]
    for label, vals in (
        ("mask_margin m020 p25", mask),
        ("box_provider oracle rung 2 (operational)", box3),
    ):
        if all(v is not None for v in vals):
            arr = np.array(vals)
            print(
                f"  {label:<42} mean {arr.mean():.4f} SD {arr.std(ddof=1):.4f} "
                f"[{' '.join(f'{v:.4f}' for v in arr)}]"
            )
    if all(v is not None for v in mask + box3):
        d = abs(np.mean(mask) - np.mean(box3))
        print(f"  the two routes differ by {d:.4f} at the shared margin.")
        print("  A difference within noise supports m = 0.20 only, not either route's margin")
        print("  response elsewhere.\n")
    else:
        print(
            "  route comparison skipped: "
            + ", ".join(t for t, v in zip(ROUTE_MASK + BASELINE_TAGS[:3], mask + box3) if v is None)
            + f" not in {results}\n"
        )

    # the bar, from the m = 0.20 baseline seeds
    base = [auc(t, results) for t in BASELINE_TAGS]
    have = [(t, v) for t, v in zip(BASELINE_TAGS, base) if v is not None]
    if len(have) < 3:
        raise SystemExit("REFUSED: need at least three of the five baseline tables")
    if len(have) < 5:
        print(
            f"WARNING: only {len(have)} of 5 baseline seeds present — the five-seed bar is "
            f"not complete."
        )
        print("         The bar below is provisional and the rule must NOT be applied yet.\n")
    arr = np.array([v for _, v in have])
    mean, sd = float(arr.mean()), float(arr.std(ddof=1))
    delta = max(FLOOR, 3 * sd)
    bar = mean + delta
    print(f"--- the bar, from {len(have)} seeds at m = 0.20 ---")
    print(f"  {' '.join(f'{v:.4f}' for v in arr)}")
    print(f"  mean {mean:.4f} SD {sd:.4f} -> bar = mean + max({FLOOR}, 3*SD) = {bar:.4f}")
    s28 = auc(SEED28, results)
    if s28 is not None and sd > 0:
        z = (s28 - mean) / sd
        verdict = "an outlier" if abs(z) > 2 else "not an outlier"
        print(f"  seed 28 = {s28:.4f}, {z:+.2f} SD from the {len(have)}-seed mean -> {verdict}\n")

    rows, winners = [], []

    def report(label, tag_or_tags, descriptive=False):
        tags = tag_or_tags if isinstance(tag_or_tags, list) else [tag_or_tags]
        vals = [auc(t, results) for t in tags]
        got = [v for v in vals if v is not None]
        if not got:
            print(f"{label:<26}{'not yet run':>15}")
            rows.append({"config": label, "tags": tags})
            return
        v = float(np.mean(got))
        clears = (not descriptive) and v > bar
        note = f" ({len(got)} seed{'s' if len(got) > 1 else ''})" if len(tags) > 1 else ""
        print(
            f"{label:<26}{v:>15.4f}{v - mean:>+10.4f}"
            f"{('descriptive' if descriptive else ('YES' if clears else 'no')):>13}{note}"
        )
        rows.append(
            {
                "config": label,
                "tags": tags,
                "exact_val_auc": v,
                "n_seeds": len(got),
                "delta_vs_mean": v - mean,
                "clears_bar": bool(clears),
                "descriptive": descriptive,
            }
        )
        if clears:
            winners.append((label, tags[0], v))

    print("--- (a) margin on the operational route, three seeds each ---")
    print(f"{'configuration':<26}{'mean exact AUC':>15}{'vs bar mean':>10}{'clears':>13}")
    for label, tags in MARGINS.items():
        report(f"a. margin {label}", tags)

    print("\n--- (c) and (d), single-seed and descriptive as declared ---")
    for label, tag in SINGLE.items():
        report(label, tag, descriptive=True)

    print("\n--- ensembles, built from the prediction tables ---")
    verify_against_src_ensemble(results)
    for out_tag, tags, label, desc in (
        ("vgg16_crop_rung2_oracle_ens5", ENS5, "b. ens5 (five seeds)", False),
        ("crop_rung2_oracle_ens4backbone", ENS4B, "d. four-backbone", True),
    ):
        v = build_ensemble(tags, out_tag, results)
        if v is not None:
            clears = (not desc) and v > bar
            print(
                f"{label:<26}{v:>15.4f}{v - mean:>+10.4f}"
                f"{('descriptive' if desc else ('YES' if clears else 'no')):>13}"
            )
            rows.append(
                {
                    "config": label,
                    "tags": [out_tag],
                    "exact_val_auc": v,
                    "delta_vs_mean": v - mean,
                    "clears_bar": bool(clears),
                    "descriptive": desc,
                }
            )
            if clears:
                winners.append((label, out_tag, v))

    print()
    if len(have) < 5:
        print(
            "DECISION: WITHHELD — the five-seed bar must finish first. The rule is not "
            "applied on a provisional bar."
        )
    elif winners:
        best = max(winners, key=lambda w: w[2])
        print(
            f"DECISION: {best[0]} clears the bar at {best[2]:.4f} and becomes the classifier "
            f"for the cascade's rung 3 and the fusion rerun."
        )
        print("  Every configuration compared here is on the operational route.")
        if best[0].startswith("a. margin"):
            print("  The margin winner must also be run at RUNG 3.")
    else:
        print(
            f"DECISION: nothing clears {bar:.4f}, so m = 0.20 stands: no configuration "
            "under the rule beats the m = 0.20 mean by more than max(0.01, 3 SD)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision.json").write_text(
        json.dumps(
            {
                "baseline_tags": [t for t, _ in have],
                "baseline_aucs": list(arr),
                "n_seeds": len(have),
                "mean": mean,
                "sd": sd,
                "floor": FLOOR,
                "delta": delta,
                "bar": bar,
                "seed28": s28,
                "seed28_z": (None if s28 is None or sd == 0 else (s28 - mean) / sd),
                "route_mask_margin": mask,
                "route_box_provider": box3,
                "configs": rows,
                "provisional": len(have) < 5,
                "winner": (max(winners, key=lambda w: w[2])[0] if winners else None),
                "rule": "replace the incumbent only if a configuration beats the m=0.20 "
                "five-seed mean by more than max(0.01, 3 x pooled five-seed SD), fixed before "
                "any run",
            },
            indent=2,
            default=float,
        )
    )
    pd.DataFrame(rows).to_csv(out_dir / "configurations.csv", index=False)
    print(f"\n-> {out_dir / 'decision.json'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("ceiling", help="apply the decision rule to the ceiling runs")
    # Required, with no default: the five-seed bar's seeds 3 and 4 live in
    # outputs/_cuda/s8_results/, not in outputs/results/.
    c.add_argument(
        "--results-dir",
        required=True,
        help="the prediction tables to decide on, e.g. outputs/_cuda/s8_results; "
        "the ensembles built here are written back into it, as src.ensemble would",
    )
    c.add_argument(
        "--out", default=str(DEFAULT_OUT), help="folder for decision.json and configurations.csv"
    )
    c.set_defaults(func=cmd_ceiling)

    r = sub.add_parser("read-session", help="apply the declared reading to a one-session ledger")
    r.add_argument(
        "--results-dir",
        default=str(RESULTS),
        help="the session's prediction tables (exact AUCs come from these)",
    )
    r.add_argument(
        "--rows",
        default=str(ROOT / "notebooks" / "scripts" / "session_cuda0917_rows.json"),
        help="the session's declared rows, for the completeness check",
    )
    r.add_argument("--out", default=str(RESULTS / "s0917_reading.json"))
    r.set_defaults(func=cmd_read_session)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

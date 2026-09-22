#!/usr/bin/env python
"""What the localiser ensemble's members contribute. Validation only, misses score 0.
Everything is descriptive except seed-null, the declared primary test.

  gate-c-strata where a member's gate (c) gain lands, across three fixed strata
  error-correlation per-lesion error correlation between members
  victim-rescue whether two passing members rescue the same lesions
  seed-null the paired difference of differences against a seed replicate
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

LB = config.LOC_BUNDLE_DIR
DEFAULT_REFERENCE = LB / "boxes" / "localiser_boxes_val.csv"  # E3, the promoted set
DEFAULT_GEOMETRY = ROOT / "artifacts" / "lesion_geometry.csv"
DEFAULT_OUT = config.RESULTS_DIR / "gate_c_strata"
VICTIM_THR = 0.3  # analyse_ladder.victims_contrast
RESCUE_THR = 0.5  # the detect@0.5 convention
MISS_THR = 0.5


def _discovered(what: str, n: int, where, *, required: bool = False) -> int:
    """Print what this run found and where; refuse an empty required discovery."""
    places = where if isinstance(where, (list, tuple)) else [where]
    loc = ", ".join(str(p) for p in places)
    print(f"[discovered] {n} {what} under {loc}")
    if required and n == 0:
        raise SystemExit(
            f"REFUSED: no {what} found under {loc}. An empty discovery is not a "
            f"pass — this run checked nothing."
        )
    return n


def bundle():
    """The bundle module beside this one: the gate's own paired IoU arithmetic,
    imported rather than restated so there is one implementation of it.
    """
    spec = importlib.util.spec_from_file_location(
        "localiser_bundle", Path(__file__).resolve().parent / "localiser_bundle.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("localiser_bundle", m)
    spec.loader.exec_module(m)
    return m


def iou_frame(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "split" in d.columns:
        d = d[d["split"] == "val"]
    d = d.drop_duplicates("lesion_id").set_index("lesion_id")
    out = pd.DataFrame({"box_iou": d["box_iou"].astype(float).fillna(0.0)})
    out["detected"] = d["detected"].astype(bool) if "detected" in d.columns else True
    return out


def _rel(path) -> str:
    """Repo-relative where possible, the path as given otherwise: a `--tag` path is
    usually already relative, and relative_to would raise on it.
    """
    q = Path(path)
    try:
        return str((q if q.is_absolute() else (ROOT / q)).resolve().relative_to(ROOT))
    except ValueError:
        return str(q)


def _sha16(path: Path) -> str:
    """First 16 hex digits of a file's SHA-256, as the store manifests record."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def iou_series(path: Path) -> pd.Series:
    """Per-lesion val box IoU, misses scored 0, indexed by lesion_id."""
    d = pd.read_csv(path)
    if "split" in d.columns:
        d = d[d["split"] == "val"]
    s = d.set_index("lesion_id")["box_iou"].astype(float).fillna(0.0)
    return s.groupby(level=0).first()


def _pairs(tags: list[str] | None, default: dict) -> dict:
    """`--tag label=path` entries, or the recorded default set."""
    if not tags:
        return {k: Path(v) for k, v in default.items()}
    out = {}
    for t in tags:
        label, _, path = t.partition("=")
        if not path:
            raise SystemExit(f"--tag takes label=path pairs; got {t!r}")
        out[label] = Path(path)
    return out


# gate-c-strata — where the gain lands
def block(df: pd.DataFrame, col: str, title: str, total: float) -> list[dict]:
    print(f"\n--- {title} ---")
    print(
        f"{'stratum':<28}{'n':>5}{'impr':>6}{'worse':>7}{'same':>6}"
        f"{'mean dIoU':>11}{'share of +total':>17}"
    )
    rows = []
    for v, sub in df.groupby(col, dropna=False, observed=True):
        d = sub["delta"]
        contrib = float(d.sum() / len(df))  # its part of the overall mean delta
        r = {
            "stratum": str(v),
            "n": int(len(sub)),
            "improved": int((d > 0).sum()),
            "worse": int((d < 0).sum()),
            "unchanged": int((d == 0).sum()),
            "mean_delta_in_stratum": float(d.mean()),
            "contribution_to_overall_mean": contrib,
            "share_of_overall": float(contrib / total) if total else float("nan"),
            "mean_iou_ref": float(sub["ref"].mean()),
            "mean_iou_cand": float(sub["cand"].mean()),
        }
        rows.append(r)
        print(
            f"{str(v):<28}{r['n']:>5}{r['improved']:>6}{r['worse']:>7}{r['unchanged']:>6}"
            f"{r['mean_delta_in_stratum']:>11.4f}{r['share_of_overall']*100:>16.1f}%"
        )
    return rows


def cmd_gate_c_strata(args) -> None:
    """Candidate is the bundle for runs 1+3+4+<member>, reference the promoted E3.
    Strata: E3's victim class, size quartile, prior-miss status at detect@0.5.
    """
    cand_p = Path(args.candidate)
    if not cand_p.is_absolute():  # so a relative --candidate prints
        cand_p = (ROOT / cand_p).resolve()
    if not cand_p.exists():
        raise SystemExit(f"candidate table not found: {cand_p}")
    ref_p = Path(args.reference)
    if not ref_p.is_absolute():  # so a relative --reference prints
        ref_p = (ROOT / ref_p).resolve()
    cand, ref = iou_frame(cand_p), iou_frame(ref_p)
    common = cand.index.intersection(ref.index)
    df = pd.DataFrame(
        {
            "cand": cand.loc[common, "box_iou"],
            "ref": ref.loc[common, "box_iou"],
            "ref_detected": ref.loc[common, "detected"],
        }
    )
    df["delta"] = df["cand"] - df["ref"]
    total = float(df["delta"].mean())

    print(f"candidate : {cand_p.relative_to(ROOT)} mean {df['cand'].mean():.4f}")
    print(f"reference : {ref_p.relative_to(ROOT)} (E3) mean {df['ref'].mean():.4f}")
    print(
        f"lesions   : {len(df)} improved {int((df['delta']>0).sum())} "
        f"worse {int((df['delta']<0).sum())} unchanged {int((df['delta']==0).sum())}"
    )
    print(f"overall mean delta: {total:+.4f}")

    # 1. E3's victim class
    df["victim"] = np.where(
        (~df["ref_detected"]) | (df["ref"] < VICTIM_THR),
        f"E3 victim (no box or IoU<{VICTIM_THR})",
        "E3 handles it",
    )
    # 2. size quartile, cut on the validation lesions only
    g = pd.read_csv(args.geometry).drop_duplicates("lesion_id").set_index("lesion_id")
    df["box_area_frac"] = g.reindex(df.index)["box_area_frac"]
    n_missing = int(df["box_area_frac"].isna().sum())
    df["size_q"] = pd.qcut(df["box_area_frac"], 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
    # 3. prior-miss at the detect@0.5 convention
    df["prior_miss"] = np.where(df["ref"] < MISS_THR, f"E3 miss (IoU<{MISS_THR})", "E3 hit")

    res = {
        "candidate": str(cand_p.relative_to(ROOT)),
        "name": args.label,
        "reference": str(ref_p.relative_to(ROOT)),
        "n_lesions": int(len(df)),
        "overall_mean_delta": total,
        "n_improved": int((df["delta"] > 0).sum()),
        "n_worse": int((df["delta"] < 0).sum()),
        "n_unchanged": int((df["delta"] == 0).sum()),
        "n_missing_geometry": n_missing,
        "victim_threshold": VICTIM_THR,
        "miss_threshold": MISS_THR,
    }
    res["by_victim_class"] = block(df, "victim", "1. E3 victim class", total)
    res["by_size_quartile"] = block(df, "size_q", "2. lesion size quartile (box_area_frac)", total)
    res["by_prior_miss"] = block(df, "prior_miss", "3. prior-miss status (detect@0.5)", total)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}_strata.json").write_text(json.dumps(res, indent=2))
    df.to_csv(out / f"{args.label}_per_lesion.csv")
    if n_missing:
        print(f"\nNOTE: {n_missing} lesions had no geometry row; they fall outside the quartiles.")
    for f in (f"{args.label}_strata.json", f"{args.label}_per_lesion.csv"):
        print(f"-> {out / f}")


# error-correlation — is decorrelation visible at all?

# label -> box table. Fixed members first; the others are found by glob on their
# name under outputs/_cuda/, so a member's folder suffix needs no edit here.
FIXED = {
    "run1": "artifacts/localiser_boxes_val.csv",
    "run3": "outputs/results/localiser_run3_eval/localiser_boxes_val.csv",
    "run4": "outputs/results/localiser_run4_eval/localiser_boxes_val.csv",
    "E3": "outputs/results/localiser_bundle/boxes/localiser_boxes_val.csv",
}
GLOBBED = ["l7v", "l7a", "l8a", "l7v_s1", "l7v_s2", "run4_s1", "run4_s2"]


def discover() -> dict[str, Path]:
    tables: dict[str, Path] = {}
    for label, rel in FIXED.items():
        p = ROOT / rel
        if p.exists():
            tables[label] = p
    for m in GLOBBED:
        hits = sorted(
            glob.glob(
                str(ROOT / "outputs" / "_cuda" / f"localiser_{m}_*" / "localiser_boxes_val.csv")
            )
        )
        # a member's own eval folder, for the Keras replicates
        hits += sorted(
            glob.glob(
                str(
                    ROOT
                    / "outputs"
                    / "_cuda"
                    / f"localiser_{m}_*"
                    / "eval"
                    / "localiser_boxes_val.csv"
                )
            )
        )
        if hits:
            tables[m] = Path(hits[0])
    missing = [m for m in (*FIXED, *GLOBBED) if m not in tables]
    if missing:
        print(f"[skipped] no box table found, so not correlated: {', '.join(missing)}")
    return tables


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi coefficient of two binary vectors; nan when either is constant."""
    n11 = float(((a == 1) & (b == 1)).sum())
    n10 = float(((a == 1) & (b == 0)).sum())
    n01 = float(((a == 0) & (b == 1)).sum())
    n00 = float(((a == 0) & (b == 0)).sum())
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float("nan") if den == 0 else (n11 * n00 - n10 * n01) / den


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(((a == 1) & (b == 1)).sum())
    union = float(((a == 1) | (b == 1)).sum())
    return float("nan") if union == 0 else inter / union


def cmd_error_correlation(args) -> None:
    """Descriptive, so no intervals: this says only whether decorrelation is visible.
    Members are discovered by default; `--tag label=path` names them instead.
    """
    tables = _pairs(args.tag, {}) if args.tag else discover()
    # The roots searched, not every file found: a reader checks the places, and
    # the member list itself is printed below and recorded in member_set.
    where = (
        [_rel(v) for v in tables.values()]
        if args.tag
        else [ROOT / "outputs" / "_cuda", ROOT / "artifacts", config.RESULTS_DIR]
    )
    _discovered(
        ("member box tables (declared)" if args.tag else "member box tables (discovered)"),
        len(tables),
        where,
        required=True,
    )
    if len(tables) < 2:
        raise SystemExit("fewer than two member tables found — nothing to correlate")
    iou = pd.DataFrame({k: iou_series(v) for k, v in tables.items()})
    before = len(iou)
    iou = iou.dropna()
    print(f"members: {', '.join(tables)}")
    for k, v in tables.items():
        print(f"  {k:8s} <- {_rel(v)}")
    print(
        f"lesions: {len(iou)} common to every member"
        + (f" (of {before} seen in at least one)" if before != len(iou) else "")
    )
    print("mean box IoU: " + "  ".join(f"{k} {iou[k].mean():.4f}" for k in iou.columns))

    miss = (iou < MISS_THR).astype(int)
    cols = list(iou.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "pearson_iou": float(iou[a].corr(iou[b], method="pearson")),
                    "spearman_iou": float(iou[a].corr(iou[b], method="spearman")),
                    "miss_jaccard": jaccard(miss[a].to_numpy(), miss[b].to_numpy()),
                    "miss_phi": phi(miss[a].to_numpy(), miss[b].to_numpy()),
                    "n_miss_a": int(miss[a].sum()),
                    "n_miss_b": int(miss[b].sum()),
                    "n_miss_both": int(((miss[a] == 1) & (miss[b] == 1)).sum()),
                }
            )
    pw = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pw.to_csv(out / "pairwise_val.csv", index=False)
    iou.to_csv(out / "per_lesion_iou_val.csv")

    def show(sub: pd.DataFrame, title: str) -> None:
        print(f"\n--- {title} ---")
        print(
            f"{'pair':<18}{'pearson':>9}{'spearman':>10}{'miss J':>9}{'miss phi':>10}"
            f"{'both miss':>11}"
        )
        for _, r in sub.sort_values("pearson_iou").iterrows():
            print(
                f"{r['a']+' / '+r['b']:<18}{r['pearson_iou']:>9.3f}{r['spearman_iou']:>10.3f}"
                f"{r['miss_jaccard']:>9.3f}{r['miss_phi']:>10.3f}"
                f"{r['n_miss_both']:>7d}/{min(r['n_miss_a'], r['n_miss_b']):d}"
            )

    keras = [c for c in cols if c in ("run1", "run3", "run4") or c.startswith("run4_s")]
    kk = pw[pw["a"].isin(keras) & pw["b"].isin(keras)]
    show(kk, "Keras members, pairwise")
    torch_m = [c for c in cols if c.startswith(("l7", "l8"))]
    kt = pw[
        (pw["a"].isin(keras) & pw["b"].isin(torch_m))
        | (pw["a"].isin(torch_m) & pw["b"].isin(keras))
    ]
    if len(kt):
        show(kt, "torch member(s) against the Keras members")
    else:
        print("\n[skipped] torch against Keras: the set lacks a torch or a Keras member")
    tt = pw[pw["a"].isin(torch_m) & pw["b"].isin(torch_m)]
    if len(tt):
        show(tt, "torch members, pairwise")
    else:
        print("\n[skipped] torch pairwise: fewer than two torch members")
    e3 = pw[(pw["a"] == "E3") | (pw["b"] == "E3")]
    if len(e3):
        show(e3, "each member against the promoted three-run ensemble E3")
    else:
        print("\n[skipped] against E3: no E3 table in the set")

    # The member set is part of the result: discovery can cover a different set, so
    # each table is recorded with the digest and row count it was read at.
    summary = {
        "members": {k: _rel(v) for k, v in tables.items()},
        "member_set": {
            "source": "declared via --tag" if args.tag else "discovered by glob",
            "n_members": len(tables),
            "labels": sorted(tables),
            "tables": {
                k: {
                    "path": _rel(v),
                    "sha256_16": _sha16(Path(v)),
                    "n_rows": int(len(pd.read_csv(v))),
                }
                for k, v in tables.items()
            },
        },
        "n_lesions": int(len(iou)),
        "miss_threshold": MISS_THR,
        "mean_box_iou": {k: float(iou[k].mean()) for k in cols},
        "mean_pearson_keras_pairwise": (float(kk["pearson_iou"].mean()) if len(kk) else None),
        "mean_pearson_torch_vs_keras": (float(kt["pearson_iou"].mean()) if len(kt) else None),
        "reading": (
            "lower torch-vs-Keras correlation than Keras-pairwise is consistent with "
            "the framework-diversity explanation and is evidence about mechanism, not a test "
            "of it. Comparable correlations mean the gate (c) pass is not obviously explained "
            "by decorrelation and the report says the mechanism is unexplained."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary["mean_pearson_torch_vs_keras"] is not None:
        print(
            f"\nmean Pearson — Keras pairwise {summary['mean_pearson_keras_pairwise']:.3f}, "
            f"torch vs Keras {summary['mean_pearson_torch_vs_keras']:.3f}"
        )
    for f in ("pairwise_val.csv", "per_lesion_iou_val.csv", "summary.json"):
        print(f"-> {out / f}")


# victim-rescue — do two passing members rescue the same lesions?
RESCUE_SOURCES = {
    "E3": str(LB / "boxes" / "localiser_boxes_val.csv"),
    "ens4(l7v)": str(LB / "selected_runs134l7v" / "bundle_boxes_val.csv"),
    "ens4(l7a)": str(LB / "selected_runs134l7a" / "bundle_boxes_val.csv"),
    "l7v alone": str(
        ROOT / "outputs" / "_cuda" / "localiser_l7v_vgg16_cachev1" / "localiser_boxes_val.csv"
    ),
    "l7a alone": str(
        ROOT / "outputs" / "_cuda" / "localiser_l7a_resnet34_cachev1" / "localiser_boxes_val.csv"
    ),
}


def cmd_victim_rescue(args) -> None:
    """An E3 victim has no box or IoU < 0.3, and is rescued when the ensemble box
    reaches IoU >= 0.5; `--tag label=path` replaces the five sources, under their labels.
    """
    sources = _pairs(args.tag, RESCUE_SOURCES)
    missing = [k for k, v in sources.items() if not Path(v).exists()]
    if missing:
        raise SystemExit("missing box tables: " + ", ".join(missing))
    df = pd.DataFrame({k: iou_series(Path(v)) for k, v in sources.items()}).dropna()
    victims = df.index[df["E3"] < VICTIM_THR]
    print(f"{len(df)} val lesions; {len(victims)} are E3 victims (IoU < {VICTIM_THR})\n")

    res = {
        "n_lesions": int(len(df)),
        "n_victims": int(len(victims)),
        "victim_threshold": VICTIM_THR,
        "rescue_threshold": RESCUE_THR,
        "pairs": {},
    }

    def rescued(col: str) -> set:
        return set(df.loc[victims].index[df.loc[victims, col] >= RESCUE_THR])

    for label, (a, b) in {
        "four-member ensembles": ("ens4(l7v)", "ens4(l7a)"),
        "the members alone": ("l7v alone", "l7a alone"),
    }.items():
        ra, rb = rescued(a), rescued(b)
        both, union = ra & rb, ra | rb
        only_a, only_b = ra - rb, rb - ra
        jac = len(both) / len(union) if union else float("nan")
        print(f"--- {label} ---")
        print(f"  {a:<12} rescues {len(ra):>3} of {len(victims)} victims")
        print(f"  {b:<12} rescues {len(rb):>3} of {len(victims)} victims")
        print(f"  both (overlap) {len(both):>3}")
        print(f"  only {a:<11} {len(only_a):>3}")
        print(f"  only {b:<11} {len(only_b):>3}")
        print(f"  union {len(union):>3} overlap/union = {jac:.2f}")
        print(
            f"  headroom: {len(union) - max(len(ra), len(rb)):>3} victims the better of the two "
            f"misses but the other rescues\n"
        )
        res["pairs"][label] = {
            "a": a,
            "b": b,
            "n_rescued_a": len(ra),
            "n_rescued_b": len(rb),
            "n_both": len(both),
            "n_only_a": len(only_a),
            "n_only_b": len(only_b),
            "n_union": len(union),
            "overlap_over_union": jac,
            "headroom_over_better": len(union) - max(len(ra), len(rb)),
            "rescued_both": sorted(both),
            "rescued_only_a": sorted(only_a),
            "rescued_only_b": sorted(only_b),
        }

    # victims neither touches: the floor the ensemble route cannot reach
    ra, rb = rescued("ens4(l7v)"), rescued("ens4(l7a)")
    never = sorted(set(victims) - (ra | rb))
    print(f"--- victims neither four-member ensemble rescues: {len(never)} of {len(victims)} ---")
    print("  these are the ceiling of the ensemble route as it stands.")
    res["n_victims_neither_rescues"] = len(never)
    res["victims_neither_rescues"] = never

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "victim_rescue_overlap.json").write_text(json.dumps(res, indent=2))
    print(f"\n-> {out / 'victim_rescue_overlap.json'}")


# seed-null — the paired difference of differences
def ens4_table(bundle_dir: Path, member: str) -> pd.DataFrame:
    """The four-member ensemble's validation box table for `member`."""
    p = bundle_dir / f"selected_runs134{member}" / "bundle_boxes_val.csv"
    if not p.exists():
        raise SystemExit(
            f"no four-member table for '{member}': {p} does not exist. "
            f"That member has not been gated, or its sweep did not finish."
        )
    return pd.read_csv(p)


def dod(
    lb,
    seed_tbl: pd.DataFrame,
    pass_tbl: pd.DataFrame,
    seed: str,
    m: str,
    seed_c_passed: bool | None,
) -> dict:
    """Paired patient-clustered bootstrap of mean(ens4[seed] - ens4[m]). Reading 1: the
    seed's own gate (c) failed and the CI is below 0; 3: below 0 otherwise; 2: it is not.
    """
    res = lb._gate_tables(seed_tbl, pass_tbl, f"ens4+{seed}", f"ens4+{m}")
    lo, hi = res["ci_lower"], res["ci_upper"]
    res["seed_c_passed"] = seed_c_passed
    if seed_c_passed is False and hi < 0:
        res["reading"] = (
            "1: the seed's own (c) FAILED and it is clearly below the pass — "
            "the gain requires a differently-implemented member. The "
            "difference is the implementation-diversity effect size."
        )
    elif hi < 0:
        res["reading"] = (
            "3: clearly smaller — this difference IS the implementation-diversity effect size"
        )
    elif lo > 0:
        res["reading"] = "2 a fortiori: the same-recipe replicate helps at least as much"
    else:
        res["reading"] = (
            "2: not distinguishable — a FAILURE TO DISTINGUISH, not a "
            "demonstration of equivalence"
        )
    return res


def cmd_seed_null(args) -> None:
    """Both (c) effects are measured on the same lesions against the same reference
    ensemble, so the reference cancels and what is left is paired per lesion.
    """
    lb = bundle()
    bundle_dir = Path(args.bundle_dir)
    seed_tbl = ens4_table(bundle_dir, args.seed_member)
    seed_gate = bundle_dir / f"gate_ens4{args.seed_member}_vs_ens3.json"
    seed_c_passed = (
        bool(json.loads(seed_gate.read_text())["passes"]) if seed_gate.exists() else None
    )
    out = {
        "seed_member": args.seed_member,
        "passes": args.passes,
        "seed_own_c_passed": seed_c_passed,
        "per_pass": {},
    }
    print(
        f"\n{args.seed_member}'s OWN gate (c): "
        f"{'PASS' if seed_c_passed else 'FAIL' if seed_c_passed is False else 'not yet run'}"
    )

    print(
        f"\n--- primary test: paired difference of differences, {args.seed_member} "
        f"vs each declared (c) pass ---"
    )
    for m in args.passes:
        out["per_pass"][m] = dod(
            lb, seed_tbl, ens4_table(bundle_dir, m), args.seed_member, m, seed_c_passed
        )
        print(f"    -> reading {out['per_pass'][m]['reading']}")

    # Against the passes' MEAN: the interval is computed on the per-lesion quantity
    # IoU_i[ens3+seed] - mean_m IoU_i[ens3+m], not assembled out of the per-pass ones.
    print(f"\n--- primary test: {args.seed_member} vs the MEAN of the passes ---")
    mean_tbl = None
    for m in args.passes:
        t = ens4_table(bundle_dir, m)[
            ["lesion_id", "patient_id", "label", "status", "box_iou"]
        ].copy()
        t["box_iou"] = t["box_iou"].fillna(0.0)
        mean_tbl = (
            t
            if mean_tbl is None
            else mean_tbl.merge(
                t[["lesion_id", "box_iou"]],
                on="lesion_id",
                how="inner",
                suffixes=("", f"_{m}"),
                validate="1:1",
            )
        )
    iou_cols = [c for c in mean_tbl.columns if c.startswith("box_iou")]
    if len(iou_cols) != len(args.passes):
        raise SystemExit(f"expected {len(args.passes)} IoU columns, got {iou_cols}")
    mean_tbl["box_iou"] = mean_tbl[iou_cols].mean(axis=1)
    out["vs_mean"] = dod(
        lb,
        seed_tbl,
        mean_tbl[["lesion_id", "patient_id", "label", "status", "box_iou"]],
        args.seed_member,
        "mean of " + "+".join(args.passes),
        seed_c_passed,
    )
    print(f"    -> reading {out['vs_mean']['reading']}")

    # Secondary, descriptive only: where the seed's own (c) effect sits relative to
    # the passes' effects. Its answer is driven by interval width.
    eff = {}
    for m in args.passes + [args.seed_member]:
        f = bundle_dir / f"gate_ens4{m}_vs_ens3.json"
        if f.exists():
            eff[m] = json.loads(f.read_text())
    if args.seed_member in eff and len(eff) > 1:
        passes_mean = float(np.mean([eff[m]["mean_diff"] for m in args.passes if m in eff]))
        g = eff[args.seed_member]
        contains = g["ci_lower"] <= passes_mean <= g["ci_upper"]
        out["secondary_descriptive"] = {
            "seed_c_effect": g["mean_diff"],
            "seed_ci": [g["ci_lower"], g["ci_upper"]],
            "passes_mean_effect": passes_mean,
            "ci_contains_passes_mean": contains,
        }
        print("\n--- secondary (DESCRIPTIVE ONLY) ---")
        print(
            f"  {args.seed_member} (c) effect {g['mean_diff']:+.4f} "
            f"[{g['ci_lower']:+.4f}, {g['ci_upper']:+.4f}]; the passes' mean effect "
            f"{passes_mean:+.4f}; CI contains it: {contains}"
        )
        print(
            "  Descriptive. Its answer is driven by interval width, which is why it "
            "is not the test."
        )
    else:
        print("\n[skipped] secondary: no gate (c) record for the seed member or the passes")

    dest = Path(args.out) if args.out else bundle_dir / f"dod_{args.seed_member}_vs_passes.json"
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


# CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gate-c-strata", help="where a member's gate (c) gain lands")
    p.add_argument("--candidate", required=True, help="the four-member bundle_boxes_val.csv")
    p.add_argument("--label", required=True, help="name in the output files, e.g. ens4l7v")
    p.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="E3's box table")
    p.add_argument(
        "--geometry",
        default=str(DEFAULT_GEOMETRY),
        help="artifacts/lesion_geometry.csv, for the size quartiles",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(func=cmd_gate_c_strata)

    p = sub.add_parser("error-correlation", help="per-lesion error correlation between members")
    p.add_argument(
        "--tag",
        nargs="+",
        default=None,
        help="label=path pairs; default: discover the known members",
    )
    p.add_argument("--out", default=str(config.RESULTS_DIR / "member_correlation"))
    p.set_defaults(func=cmd_error_correlation)

    p = sub.add_parser("victim-rescue", help="do two passing members rescue the same lesions?")
    p.add_argument(
        "--tag",
        nargs="+",
        default=None,
        help="label=path pairs; default: the five declared sources",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(func=cmd_victim_rescue)

    p = sub.add_parser("seed-null", help="the paired difference of differences")
    p.add_argument("--seed-member", required=True, help="run4_s1 or run4_s2")
    p.add_argument("--passes", nargs="+", default=["l7v", "l7a", "l8a", "l9"])
    p.add_argument("--bundle-dir", default=str(LB))
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_seed_null)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

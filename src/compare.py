"""The declared statistical comparisons: one primary endpoint against the BI-RADS
anchor, and a Holm-corrected family of pairwise DeLong tests among named models.
Every figure is computed from the prediction tables, so no model is re-scored here.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from . import config
from .analysis import to_image_level
from .evaluate import bootstrap_ci, delong_test, mcnemar_test

ROOT = Path(__file__).resolve().parent.parent


def _auc_diff_fn(y_true: np.ndarray, probs: np.ndarray, **_kw) -> float:
    """AUC(a) - AUC(b) for a (n, 2) column stack of the two models' scores."""
    return float(roc_auc_score(y_true, probs[:, 0]) - roc_auc_score(y_true, probs[:, 1]))


def auc_diff_ci(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, groups: np.ndarray
) -> list[float]:
    """Patient-cluster bootstrap 95 % CI of the paired AUC difference: DeLong treats
    rows as independent, and rows sharing a patient are not.
    """
    stacked = np.column_stack([np.asarray(prob_a, float), np.asarray(prob_b, float)])
    _, lo, hi = bootstrap_ci(np.asarray(y_true, int), stacked, _auc_diff_fn, groups=groups)
    return [float(lo), float(hi)]


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# "val" runs this same code over the validation tables as a dry run, and writes to
# outputs/results/validation/ so it cannot land beside a test result.
SPLIT: str = "test"


def load(tag: str) -> pd.DataFrame:
    p = config.RESULTS_DIR / f"{tag}_predictions_{SPLIT}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} - run `python -m src.evaluate ... --tag-suffix` first")
    return pd.read_csv(p)


def primary_endpoint(tag: str) -> dict:
    df = to_image_level(load(tag))
    cs = pd.read_csv(ROOT / config.BIRADS["comparison_set"])
    cs = cs[cs["partition"].str.lower() == SPLIT]["image file path"]
    on = df[df["image file path"].isin(cs)].copy()
    a = pd.to_numeric(on["assessment"].astype("object"), errors="coerce")
    on = on[a.notna() & (a != 0)]
    a = a[on.index]
    n_expected = int(config.BIRADS["n_test_comparison"]) if SPLIT == "test" else int(len(cs))
    y = on["y_true"].to_numpy(int)
    res = delong_test(y, on["y_prob"].to_numpy(float), a.to_numpy(float))
    out = dict(
        model=tag,
        unit="image",
        aggregation=config.IMAGE_SCORE_AGGREGATION,
        n=int(len(on)),
        n_expected=n_expected,
        n_malignant=int(y.sum()),
        auc_model=res["auc_a"],
        auc_birads=res["auc_b"],
        auc_diff=res["auc_diff"],
        auc_diff_ci_patient_bootstrap=auc_diff_ci(
            y, on["y_prob"].to_numpy(float), a.to_numpy(float), on[config.GROUP_COL].to_numpy()
        ),
        z=res["z"],
        p_value=res["p_value"],
        birads_anchor_from_config=config.BIRADS["auc_roc"],
        correction="none (single pre-registered comparison)",
    )
    # The anchor is recomputed on the rows actually scored, so a model covering a
    # subset is compared against BI-RADS over that same subset.
    out["birads_anchor_recomputed_on_scored_rows"] = True
    out["row_set"] = (
        f"images of the {SPLIT} BI-RADS comparison set that this model emits a "
        f"prediction for, after dropping BI-RADS 0"
    )
    if out["n"] != n_expected:
        out["warning"] = f"{out['n']} rows scored, expected {n_expected}"
        # A pair-level system is undefined on an image with no counterpart view, so
        # a shortfall is the system's domain rather than a sampling choice.
        out["row_set_note"] = (
            f"{out['n']} of {n_expected} comparison-set images. A view-fusion model predicts per "
            f"PAIR and its table carries crop A's meta columns, so it covers only images that (a) "
            f"have a counterpart view in the pair store and (b) are the A-side of that pair. The "
            f"BI-RADS anchor quoted here ({out['auc_birads']:.4f}) is recomputed on these same "
            f"rows and is NOT the whole-comparison-set anchor. Quote the two together or neither."
        )
    if SPLIT == "test" and abs(out["auc_birads"] - config.BIRADS["auc_roc"]) > 0.005:
        out["warning_birads"] = (
            "BI-RADS AUC recomputed here differs from config.BIRADS "
            "- comparison set and config are out of step"
        )
    return out


def secondary_family(tags: list[str]) -> dict:
    tables = {t: load(t) for t in tags}
    ref = tables[tags[0]]
    key = "lesion_id" if "lesion_id" in ref.columns else "image_id"
    for t in tags[1:]:
        if tables[t][key].tolist() != ref[key].tolist():
            raise RuntimeError(f"{t} rows differ from {tags[0]} - DeLong needs paired rows")
    y = ref["y_true"].to_numpy(int)
    groups = ref[config.GROUP_COL].to_numpy()
    rows = []
    for a, b in combinations(tags, 2):
        pa, pb = tables[a]["y_prob"].to_numpy(float), tables[b]["y_prob"].to_numpy(float)
        d = delong_test(y, pa, pb)
        m = mcnemar_test(
            y, pa, pb, float(tables[a]["threshold"].iloc[0]), float(tables[b]["threshold"].iloc[0])
        )
        rows.append(
            dict(
                model_a=a,
                model_b=b,
                auc_a=d["auc_a"],
                auc_b=d["auc_b"],
                auc_diff=d["auc_diff"],
                auc_diff_ci_patient_bootstrap=auc_diff_ci(y, pa, pb, groups),
                z=d["z"],
                p_delong=d["p_value"],
                mcnemar_chi2=m["chi2"],
                p_mcnemar=m["p_value"],
                b_a_wrong_b_right=m["b"],
                c_a_right_b_wrong=m["c"],
            )
        )
    adj = holm([r["p_delong"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_delong_holm"] = q
        r["significant_holm_005"] = bool(q < 0.05)
    return dict(
        unit=key.replace("_id", ""),
        n=int(len(y)),
        family_size=len(rows),
        comparisons=rows,
        note="DeLong p-values Holm-corrected within the family; McNemar descriptive",
    )


def _refuse_test_outside_pass(split: str) -> None:
    """Refuses the test split unless the test pass has set its token
    (test_pass_guard.authorised)."""
    from .test_pass_guard import authorised

    if split == "test" and not authorised():
        raise SystemExit(
            "REFUSED: --split test reads the stored test prediction tables; only "
            "notebooks/scripts/run_test_pass.sh may run it"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--primary", default=None, help="Tag of the pre-declared model for the primary endpoint"
    )
    ap.add_argument(
        "--models", nargs="*", default=[], help="Tags for the Holm-corrected pairwise family"
    )
    ap.add_argument("--out", default="comparison_report.json")
    ap.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="val = dry run on the validation tables into outputs/results/validation/",
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
    global SPLIT
    SPLIT = args.split
    _refuse_test_outside_pass(SPLIT)
    if SPLIT == "val":
        print("--- DRY RUN on validation tables: numbers are not results ---")

    report = {}
    if args.primary:
        p = primary_endpoint(args.primary)
        report["primary"] = p
        print(
            f"--- PRIMARY: {p['model']} vs BI-RADS on the comparison set "
            f"(n={p['n']} of {p['n_expected']}, image-level, {p['aggregation']}; "
            f"anchor recomputed on these rows) ---"
        )
        ci = p["auc_diff_ci_patient_bootstrap"]
        print(
            f"  AUC model {p['auc_model']:.4f} AUC BI-RADS {p['auc_birads']:.4f} "
            f"diff {p['auc_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] patient-bootstrap "
            f"z={p['z']:.3f} p={p['p_value']:.4f}"
        )
        for k in ("warning", "warning_birads"):
            if k in p:
                print(f"  WARNING: {p[k]}")
    if len(args.models) >= 2:
        s = secondary_family(args.models)
        report["secondary"] = s
        print(
            f"\n--- SECONDARY: {s['family_size']} pairwise DeLong, Holm-corrected "
            f"({s['unit']}-level, n={s['n']}) ---"
        )
        for r in s["comparisons"]:
            flag = "*" if r["significant_holm_005"] else " "
            ci = r["auc_diff_ci_patient_bootstrap"]
            print(
                f"  {r['model_a']:<26} vs {r['model_b']:<26} dAUC {r['auc_diff']:+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] p {r['p_delong']:.4f} "
                f"Holm {r['p_delong_holm']:.4f} {flag} "
                f"McNemar p {r['p_mcnemar']:.4f} "
                f"(b={r['b_a_wrong_b_right']}, c={r['c_a_right_b_wrong']})"
            )
    elif args.models:
        print("[compare] one --models tag: the pairwise family needs at least two; skipped")
    if not report:
        raise SystemExit("nothing to do: pass --primary and/or >= 2 --models")
    if SPLIT == "val":
        out_dir = config.RESULTS_DIR / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / args.out
    else:
        out = config.RESULTS_DIR / args.out
    report["split"] = SPLIT
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

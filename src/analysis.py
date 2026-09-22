"""Post-hoc analyses computed from the saved prediction tables, with no model re-scored:
BI-RADS comparison, calibration, PPV/NPV by prevalence, decision curve and subgroups.
Descriptive only; the primary hypothesis test is in compare.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from . import config
from .evaluate import bootstrap_ci, _auc_roc_fn

ROOT = Path(__file__).resolve().parent.parent  # config.BIRADS paths are repo-relative
EPS = 1e-7


# Loading and unit of analysis

SPLIT: str = "test"  # "val" = dry run: the validation table stands in for test
_RESULTS_SRC = (
    config.RESULTS_DIR
)  # tables are always read from here, even when outputs are redirected


def load_tables(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = pd.read_csv(_RESULTS_SRC / f"{tag}_predictions_val.csv")
    test = (
        pd.read_csv(_RESULTS_SRC / f"{tag}_predictions_test.csv") if SPLIT == "test" else val.copy()
    )
    for df, name in ((val, "val"), (test, "test")):
        for c in ("y_true", "y_prob", "patient_id", "threshold"):
            if c not in df.columns:
                raise KeyError(f"{tag} {name} table lacks '{c}'")
    return val, test


def is_lesion_level(df: pd.DataFrame) -> bool:
    return "cropped image file path" in df.columns


def to_image_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lesion rows to one row per mammogram. The label is the max over lesions,
    so malignant wins; subtlety takes the min, the least conspicuous lesion."""
    if not is_lesion_level(df):
        return df.copy()
    if config.IMAGE_SCORE_AGGREGATION != "max":
        raise NotImplementedError(config.IMAGE_SCORE_AGGREGATION)
    agg = {"y_prob": "max", "y_true": "max", "patient_id": "first", "threshold": "first"}
    for c, how in (
        ("subtlety", "min"),
        ("breast_density", "first"),
        ("assessment", "max"),
        ("image_id", "first"),
    ):
        if c in df.columns:
            agg[c] = how
    out = df.groupby("image file path", sort=False).agg(agg).reset_index()
    out["n_lesions"] = df.groupby("image file path", sort=False).size().to_numpy()
    return out


# Metrics


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    pred = p >= thr
    tp = np.sum(pred & (y == 1))
    fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    fp = np.sum(pred & (y == 0))
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with equal-width bins; returns the bin table too."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, total = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        total += n / len(p) * abs(acc - conf)
        rows.append(dict(bin=b, lo=edges[b], hi=edges[b + 1], n=n, confidence=conf, observed=acc))
    return float(total), pd.DataFrame(rows)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T minimising validation NLL."""
    z = _logit(p_val)

    def nll(t):
        q = np.clip(_sigmoid(z / t), EPS, 1 - EPS)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return _sigmoid(_logit(p) / t)


def shift_prevalence(p: np.ndarray, p_from: float, p_to: float) -> np.ndarray:
    """Prior shift on the logit scale: recalibrate probabilities to a new prevalence."""
    return _sigmoid(_logit(p) + np.log(p_to / (1 - p_to)) - np.log(p_from / (1 - p_from)))


def ppv_npv(sens: float, spec: float, prev: float) -> tuple[float, float]:
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev) + EPS)
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev + EPS)
    return float(ppv), float(npv)


def net_benefit(
    y: np.ndarray, p: np.ndarray, pt: np.ndarray, prev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Net benefit of 'biopsy if p >= pt' at prevalence `prev`, and of biopsy-all. Sample
    sensitivity and specificity are combined with `prev`, not the dataset's own."""
    w = pt / (1 - pt)
    nb = np.empty_like(pt)
    for i, t in enumerate(pt):
        s, sp = sens_spec(y, p, t)
        nb[i] = s * prev - (1 - sp) * (1 - prev) * w[i]
    nb_all = prev - (1 - prev) * w
    return nb, nb_all


def auc_ci(df: pd.DataFrame) -> dict:
    y, p, g = (
        df["y_true"].to_numpy(int),
        df["y_prob"].to_numpy(float),
        df["patient_id"].to_numpy(),
    )
    if len(np.unique(y)) < 2:
        return dict(n=int(len(df)), n_pos=int(y.sum()), auc=None, ci=[None, None])
    point, lo, hi = bootstrap_ci(y, p, _auc_roc_fn, groups=g)
    return dict(n=int(len(df)), n_pos=int(y.sum()), auc=float(point), ci=[float(lo), float(hi)])


# Analyses


def birads_comparison(test: pd.DataFrame) -> dict:
    """AUC on the frozen image-level comparison set (BI-RADS 0 rows excluded)."""
    cs = pd.read_csv(ROOT / config.BIRADS["comparison_set"])
    cs = cs[cs["partition"].str.lower() == SPLIT]
    img = to_image_level(test)
    if "image file path" not in img.columns:
        raise KeyError(
            "prediction table lacks 'image file path' - cannot join the BI-RADS comparison set"
        )
    on = img[img["image file path"].isin(cs["image file path"])]
    out = dict(
        unit="image",
        n_expected=(int(config.BIRADS["n_test_comparison"]) if SPLIT == "test" else int(len(cs))),
        n_matched=int(len(on)),
        aggregation=config.IMAGE_SCORE_AGGREGATION,
        model=auc_ci(on),
        full_partition=auc_ci(img),
        birads=dict(
            auc=config.BIRADS["auc_roc"],
            ci=list(config.BIRADS["auc_roc_ci"]),
            sens=config.BIRADS["sens"],
            spec=config.BIRADS["spec"],
        ),
    )
    if out["n_matched"] != out["n_expected"]:
        print(
            f"[analysis] WARNING: {out['n_matched']} comparison rows matched, "
            f"expected {out['n_expected']}"
        )
    return out


def calibration(val: pd.DataFrame, test: pd.DataFrame, tag: str) -> dict:
    """ECE, Brier and a reliability diagram before and after temperature scaling, the
    temperature being fitted on validation and applied to test."""
    yv, pv = val["y_true"].to_numpy(int), val["y_prob"].to_numpy(float)
    yt, pt = test["y_true"].to_numpy(int), test["y_prob"].to_numpy(float)
    t = fit_temperature(yv, pv)
    pt_cal = apply_temperature(pt, t)
    e0, tab0 = ece(yt, pt)
    e1, tab1 = ece(yt, pt_cal)
    out = dict(
        temperature=t,
        fitted_on="validation",
        before=dict(ece=e0, brier=brier(yt, pt)),
        after=dict(ece=e1, brier=brier(yt, pt_cal)),
        auc_unchanged=float(roc_auc_score(yt, pt)) if len(np.unique(yt)) > 1 else None,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, tab, title, e in ((axes[0], tab0, "before", e0), (axes[1], tab1, "after", e1)):
        ax.plot([0, 1], [0, 1], "--", c="grey", lw=0.8)
        ax.scatter(tab["confidence"], tab["observed"], s=8 + tab["n"] / 2, zorder=3)
        ax.plot(tab["confidence"], tab["observed"], lw=1)
        ax.set_title(f"{title} temperature scaling (ECE {e:.3f})", fontsize=9)
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed malignant fraction")
    fig.suptitle(f"{tag}: reliability on test (T = {t:.2f}, fitted on validation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"calibration_{tag}.png", dpi=200)
    plt.close(fig)
    return out


def fixed_point_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Confusion-matrix metrics at a fixed threshold, for comparability with literature
    tables. Never used for selection: at a fixed prevalence accuracy rewards the majority.
    """
    y = np.asarray(y, int)
    pred = np.asarray(p, float) >= thr
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(
        threshold=float(thr),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=(tp + tn) / max(len(y), 1),
        precision=float(precision),
        recall=float(recall),
        specificity=tn / max(tn + fp, 1),
        f1=float(f1),
        n_predicted_positive=tp + fp,
    )


def prevalence_table(test: pd.DataFrame) -> dict:
    y, p = test["y_true"].to_numpy(int), test["y_prob"].to_numpy(float)
    thr = float(test["threshold"].iloc[0])
    s, sp = sens_spec(y, p, thr)
    rows = {}
    for name, prev in config.PREVALENCE.items():
        ppv, npv = ppv_npv(s, sp, prev)
        rows[name] = dict(prevalence=prev, ppv=ppv, npv=npv, ppv_minus_prior=ppv - prev)
    return dict(
        threshold=thr,
        threshold_source="validation",
        sensitivity=float(s),
        specificity=float(sp),
        at=rows,
        fixed_point=fixed_point_metrics(y, p, thr),
    )


def birads_matched_operating_point(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Specificity at the BI-RADS baseline's own sensitivity, from the highest validation
    threshold reaching it, so the two are compared at one sensitivity."""
    from sklearn.metrics import roc_curve

    target = float(config.BIRADS["sens"])
    yv, pv = val["y_true"].to_numpy(int), val["y_prob"].to_numpy(float)
    _, tpr, thr = roc_curve(yv, pv)
    ok = np.where(tpr >= target)[0]
    t = float(thr[ok[0]]) if len(ok) else float(thr[np.argmax(tpr)])
    s, sp = sens_spec(test["y_true"].to_numpy(int), test["y_prob"].to_numpy(float), t)
    return dict(
        target_sensitivity=target,
        threshold=t,
        threshold_source="validation",
        sensitivity=float(s),
        specificity=float(sp),
        birads_specificity=float(config.BIRADS["spec"]),
    )


def decision_curve(val: pd.DataFrame, test: pd.DataFrame, tag: str, t: float) -> dict:
    """Net benefit of the model against biopsy-all and biopsy-none at assessment-clinic
    prevalence, probabilities being temperature-scaled and prior-shifted to it first."""
    y, p = test["y_true"].to_numpy(int), test["y_prob"].to_numpy(float)
    p_src = float(config.PREVALENCE["cbis_test"])
    p_dep = float(config.PREVALENCE["nhs_assessment"])
    q = shift_prevalence(apply_temperature(p, t), p_src, p_dep)
    pt = np.linspace(0.02, 0.60, 59)
    nb, nb_all = net_benefit(y, q, pt, p_dep)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pt, nb, label="model")
    ax.plot(pt, nb_all, "--", label="biopsy all")
    ax.axhline(0, c="grey", lw=0.8, label="biopsy none")
    ax.set_ylim(bottom=min(-0.02, float(np.nanmin(nb)) - 0.01))
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(f"{tag}: decision curve at {p_dep:.1%} prevalence", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"decision_curve_{tag}.png", dpi=200)
    plt.close(fig)

    better = pt[(nb > nb_all) & (nb > 0)]
    return dict(
        prevalence=p_dep,
        threshold_probabilities=pt.round(3).tolist(),
        net_benefit=nb.round(4).tolist(),
        net_benefit_all=nb_all.round(4).tolist(),
        model_beats_both_from=(round(float(better.min()), 3) if len(better) else None),
        model_beats_both_to=(round(float(better.max()), 3) if len(better) else None),
    )


def subgroups(test: pd.DataFrame, tag: str) -> dict:
    thr = float(test["threshold"].iloc[0])
    out = {}
    for col in ("subtlety", "breast_density"):
        if col not in test.columns:
            out[col] = None
            continue
        levels = sorted(test[col].dropna().unique())
        rows = []
        for lv in levels:
            sub = test[test[col] == lv]
            r = auc_ci(sub)
            s, sp = sens_spec(sub["y_true"].to_numpy(int), sub["y_prob"].to_numpy(float), thr)
            rows.append(dict(level=int(lv), **r, sensitivity=float(s), specificity=float(sp)))
        out[col] = rows

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col in zip(axes, ("subtlety", "breast_density")):
        rows = out.get(col) or []
        xs = [r["level"] for r in rows if r["auc"] is not None]
        ys = [r["auc"] for r in rows if r["auc"] is not None]
        lo = [r["auc"] - r["ci"][0] for r in rows if r["auc"] is not None]
        hi = [r["ci"][1] - r["auc"] for r in rows if r["auc"] is not None]
        # A point outside its percentile CI would give a negative bar, which matplotlib refuses.
        ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="o", capsize=3)
        for r in rows:
            if r["auc"] is not None:
                ax.annotate(
                    f"n={r['n']}",
                    (r["level"], r["auc"]),
                    textcoords="offset points",
                    xytext=(6, -10),
                    fontsize=7,
                )
        ax.axhline(0.5, c="grey", lw=0.8, ls=":")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.3, 1.0)
    fig.suptitle(f"{tag}: test AUC by subgroup (patient-bootstrap 95% CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"subgroups_{tag}.png", dpi=200)
    plt.close(fig)
    return out


# CLI


def run(tag: str) -> dict:
    val, test = load_tables(tag)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    unit = "lesion" if is_lesion_level(test) else "image"
    cal = calibration(val, test, tag)
    res = dict(
        tag=tag,
        unit=unit,
        n_val=int(len(val)),
        n_test=int(len(test)),
        birads_comparison=birads_comparison(test),
        calibration=cal,
        prevalence=prevalence_table(test),
        birads_matched=birads_matched_operating_point(val, test),
        decision_curve=decision_curve(val, test, tag, cal["temperature"]),
        subgroups=subgroups(test, tag),
        note="descriptive, uncorrected; the primary endpoint is in evaluate.py",
    )
    path = config.RESULTS_DIR / f"analysis_{tag}.json"
    path.write_text(json.dumps(res, indent=2))

    b = res["birads_comparison"]
    m = b["model"]
    print(f"\n--- {tag} ({unit}-level pipeline) ---")
    print(
        f"BI-RADS comparison set (n={b['n_matched']}, image-level, {b['aggregation']}): "
        f"AUC {m['auc']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] vs BI-RADS "
        f"{b['birads']['auc']:.3f} {b['birads']['ci']}"
    )
    print(
        f"calibration: T={cal['temperature']:.2f} ECE {cal['before']['ece']:.3f} -> "
        f"{cal['after']['ece']:.3f} Brier {cal['before']['brier']:.3f} -> "
        f"{cal['after']['brier']:.3f}"
    )
    pv = res["prevalence"]
    print(
        f"operating point thr={pv['threshold']:.3f}: sens {pv['sensitivity']:.3f} "
        f"spec {pv['specificity']:.3f}"
    )
    for k, v in pv["at"].items():
        print(
            f"  {k:<15} prev {v['prevalence']:.3f} PPV {v['ppv']:.3f} "
            f"({v['ppv_minus_prior']:+.3f} over prior) NPV {v['npv']:.3f}"
        )
    fpm = pv["fixed_point"]
    print(
        f"fixed point (Table 5 comparability): accuracy {fpm['accuracy']:.3f} "
        f"precision {fpm['precision']:.3f} recall {fpm['recall']:.3f} F1 {fpm['f1']:.3f} "
        f"(tp {fpm['tp']} fp {fpm['fp']} tn {fpm['tn']} fn {fpm['fn']})"
    )
    bm = res["birads_matched"]
    print(
        f"at BI-RADS sensitivity {bm['target_sensitivity']:.2f} (thr {bm['threshold']:.3f}): "
        f"sens {bm['sensitivity']:.3f} spec {bm['specificity']:.3f} "
        f"vs BI-RADS spec {bm['birads_specificity']:.3f}"
    )
    dc = res["decision_curve"]
    print(
        f"decision curve @ {dc['prevalence']:.1%}: model beats both defaults for "
        f"pt in [{dc['model_beats_both_from']}, {dc['model_beats_both_to']}]"
    )
    for col in ("subtlety", "breast_density"):
        rows = res["subgroups"].get(col)
        if rows:
            print(
                f"{col}: "
                + "  ".join(
                    (
                        f"{r['level']}: {r['auc']:.2f} (n={r['n']})"
                        if r["auc"] is not None
                        else f"{r['level']}: n/a (n={r['n']})"
                    )
                    for r in rows
                )
            )
    print(f"-> {path}")
    return res


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
        "--tag",
        required=True,
        nargs="+",
        help="prediction-table tags to analyse",
    )
    ap.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="val = dry run: every analysis on the validation table, with figures "
        "and JSON into outputs/results/validation/.",
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
        smoke = config.RESULTS_DIR / "validation"
        smoke.mkdir(parents=True, exist_ok=True)
        config.FIGURES_DIR = config.RESULTS_DIR = (
            smoke  # outputs only; tables come from _RESULTS_SRC
        )
        print("--- DRY RUN on validation tables: outputs under", smoke, "---")
    for tag in args.tag:
        run(tag)


if __name__ == "__main__":
    main()

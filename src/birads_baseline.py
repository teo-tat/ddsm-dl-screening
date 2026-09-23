"""The BI-RADS clinical baseline: the radiologist's recorded assessment scored as
an ordinal predictor on each partition's mammograms and thresholded at >= 4 (biopsy
indicated), with assessment 0 (incomplete: additional imaging and/or prior mammograms
needed) dropped as off-scale.
It writes artifacts/birads_*, and refuses to run while either file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import split_frames
from .evaluate import bootstrap_ci, _auc_roc_fn, _auc_pr_fn

ART = Path(__file__).resolve().parent.parent / "artifacts"
THRESHOLD = 4.0
Z95 = 1.959963985


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z95**2 / n
    c = (p + Z95**2 / (2 * n)) / d
    h = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score_partition(df: pd.DataFrame, name: str) -> dict:
    a = pd.to_numeric(df["assessment"].astype("object"), errors="coerce")
    keep = a.notna() & (a != 0)  # drop_zero
    sub, s = df[keep], a[keep].to_numpy(float)
    y, g = sub["label"].to_numpy(int), sub["patient_id"].to_numpy()
    pred = s >= THRESHOLD
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    auc, lo, hi = bootstrap_ci(y, s, _auc_roc_fn, groups=g)
    ap, plo, phi = bootstrap_ci(y, s, _auc_pr_fn, groups=g)
    return dict(
        partition=name,
        n_full=int(len(df)),
        n=int(len(sub)),
        n_excluded_birads0=int((~keep).sum()),
        n_malignant=int(y.sum()),
        n_benign=int((y == 0).sum()),
        n_patients=int(sub["patient_id"].nunique()),
        auc_roc=float(auc),
        auc_roc_ci=[float(lo), float(hi)],
        auc_pr=float(ap),
        auc_pr_ci=[float(plo), float(phi)],
        sens=tp / max(tp + fn, 1),
        sens_ci=list(wilson(tp, tp + fn)),
        spec=tn / max(tn + fp, 1),
        spec_ci=list(wilson(tn, tn + fp)),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        comparison_paths=sub["image file path"].tolist(),
    )


def main() -> None:
    present = [
        f"artifacts/{p.name}"
        for p in (ART / "birads_comparison_set.csv", ART / "birads_baseline.json")
        if p.exists()
    ]
    if present:
        raise SystemExit(
            f"REFUSED: {', '.join(present)} present; they are frozen records and this script "
            "would overwrite them"
        )
    config.set_seeds()
    frames = split_frames("full")
    out, rows = {}, []
    for name in ("train", "val", "test"):
        r = score_partition(frames[name], name)
        paths = r.pop("comparison_paths")
        out[name] = r
        rows.append(
            pd.DataFrame({"image file path": sorted(paths), "partition": name.capitalize()})
        )
        print(
            f"--- {name}: n={r['n']} of {r['n_full']} images "
            f"({r['n_excluded_birads0']} BI-RADS 0 excluded), "
            f"{r['n_malignant']} malignant / {r['n_benign']} benign, "
            f"{r['n_patients']} patients ---"
        )
        print(
            f"  AUC-ROC {r['auc_roc']:.3f} [{r['auc_roc_ci'][0]:.3f}, {r['auc_roc_ci'][1]:.3f}]"
            f"   AUC-PR {r['auc_pr']:.3f}"
        )
        print(
            f"  sens {r['sens']:.3f} [{r['sens_ci'][0]:.3f}, {r['sens_ci'][1]:.3f}]"
            f"   spec {r['spec']:.3f} [{r['spec_ci'][0]:.3f}, {r['spec_ci'][1]:.3f}]"
        )

    ART.mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(ART / "birads_comparison_set.csv", index=False)
    t = out["test"]
    cfg = dict(
        variant="drop_zero",
        threshold=THRESHOLD,
        auc_roc=round(t["auc_roc"], 3),
        auc_roc_ci=[round(v, 3) for v in t["auc_roc_ci"]],
        auc_pr=round(t["auc_pr"], 3),
        sens=round(t["sens"], 3),
        spec=round(t["spec"], 3),
        n_test_comparison=t["n"],
        n_test_full=t["n_full"],
        comparison_set="artifacts/birads_comparison_set.csv",
        partition_source="lesion-level patient partition (data_loader.patient_partition)",
    )
    (ART / "birads_baseline.json").write_text(
        json.dumps(dict(per_partition=out, config=cfg), indent=2)
    )
    print("\nPaste into config.BIRADS:")
    print(json.dumps(cfg, indent=4))


if __name__ == "__main__":
    main()

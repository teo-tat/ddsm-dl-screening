"""Average the members' probabilities (or logits) into one prediction table under a new tag.
Rows are aligned on lesion_id or image_id and the table keeps the schema of a
single run, so the ensemble is read downstream like any other run.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .evaluate import _auc_roc_fn, bootstrap_ci, optimal_threshold

METHODS: tuple[str, ...] = ("mean", "logit_mean")


def _key(df: pd.DataFrame) -> str:
    return "lesion_id" if "cropped image file path" in df.columns else "image_id"


def _combine(probs: np.ndarray, method: str) -> np.ndarray:
    """probs: (n_members, n_rows)."""
    if method == "mean":
        return probs.mean(axis=0)
    if method == "logit_mean":
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)).mean(axis=0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"method must be one of {METHODS}, not {method!r}")


def ensemble_split(
    tags: list[str], split: str, method: str, threshold: float | None
) -> tuple[pd.DataFrame, dict]:
    """Aligned mean of the members' tables for one split; threshold=None fits the
    operating point here (validation only), a float carries one in from validation.
    """
    tables = [pd.read_csv(config.RESULTS_DIR / f"{t}_predictions_{split}.csv") for t in tags]
    key = _key(tables[0])
    base = tables[0].set_index(key)
    probs = np.empty((len(tags), len(base)))
    for i, (t, tbl) in enumerate(zip(tags, tables)):
        tbl = tbl.set_index(key)
        if set(tbl.index) != set(base.index):
            raise SystemExit(
                f"{t} {split}: row set differs from {tags[0]} ({len(tbl)} vs {len(base)} rows)"
            )
        tbl = tbl.loc[base.index]
        if not (tbl["y_true"].to_numpy() == base["y_true"].to_numpy()).all():
            raise SystemExit(f"{t} {split}: labels differ from {tags[0]} on aligned rows")
        probs[i] = tbl["y_prob"].to_numpy(float)
    y = base["y_true"].to_numpy(int)
    p = _combine(probs, method)
    if threshold is None:
        threshold = optimal_threshold(y, p, target_sensitivity=config.TARGET_SENSITIVITY)
    out = base.reset_index()
    out["y_prob"] = p
    out["partition"] = split
    out["threshold"] = float(threshold)

    member_auc = {t: float(roc_auc_score(y, probs[i])) for i, t in enumerate(tags)}
    ens_auc, lo, hi = bootstrap_ci(y, p, _auc_roc_fn, groups=base["patient_id"].to_numpy())
    summary = {
        "split": split,
        "n": int(len(y)),
        "method": method,
        "member_auc": member_auc,
        "member_auc_mean": float(np.mean(list(member_auc.values()))),
        "ensemble_auc": float(ens_auc),
        "ensemble_auc_ci": [float(lo), float(hi)],
        "threshold": float(threshold),
    }
    return out, summary


def _refuse_test_outside_pass(splits: list[str]) -> None:
    """Refuses the test split unless the test pass has set its token
    (test_pass_guard.authorised)."""
    from .test_pass_guard import authorised

    if "test" in splits and not authorised():
        raise SystemExit(
            "REFUSED: --splits includes test, which reads the stored test prediction tables; "
            "only notebooks/scripts/run_test_pass.sh may run it"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tags", nargs="+", required=True, help="member tags (>= 2)")
    ap.add_argument("--out-tag", required=True, help="tag of the ensemble's tables")
    ap.add_argument("--method", choices=list(METHODS), default="mean")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument(
        "--allow-test",
        action="store_true",
        help="permit the test split; refused unless the test pass has set its token "
        "(test_pass_guard.authorised)",
    )
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag "
        "that no longer exists fails in a second.",
    )
    args = ap.parse_args(argv)
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    if len(args.tags) < 2:
        raise SystemExit("an ensemble needs at least two members")
    if "test" in args.splits and not args.allow_test:
        raise SystemExit("the test partition is frozen: --allow-test is for run_test_pass.sh only")
    if "test" in args.splits and "val" not in args.splits:
        raise SystemExit("the test threshold is carried from validation: include val in --splits")
    _refuse_test_outside_pass(args.splits)

    summaries, threshold = [], None
    for split in ("val", "test"):
        if split not in args.splits:
            continue
        tbl, s = ensemble_split(args.tags, split, args.method, threshold)
        threshold = s["threshold"]
        path = config.RESULTS_DIR / f"{args.out_tag}_predictions_{split}.csv"
        tbl.to_csv(path, index=False)
        summaries.append(s)
        print(
            f"[ensemble] {split}: members "
            + ", ".join(
                f"{t.split('_crop_')[-1] if '_crop_' in t else t} {a:.4f}"
                for t, a in s["member_auc"].items()
            )
        )
        print(
            f"[ensemble] {split}: mean of members {s['member_auc_mean']:.4f} ensemble "
            f"{s['ensemble_auc']:.4f} "
            f"[{s['ensemble_auc_ci'][0]:.4f}, {s['ensemble_auc_ci'][1]:.4f}] "
            f"patient-bootstrap threshold {threshold:.4f}"
        )
        print(f"-> {path}")
    meta = config.RESULTS_DIR / f"{args.out_tag}_ensemble.json"
    meta.write_text(
        json.dumps(
            {
                "out_tag": args.out_tag,
                "members": args.tags,
                "method": args.method,
                "splits": summaries,
            },
            indent=2,
        )
    )
    print(f"-> {meta}")


if __name__ == "__main__":
    main()

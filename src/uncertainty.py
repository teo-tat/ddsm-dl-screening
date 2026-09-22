"""MC-Dropout referral analysis (Gal & Ghahramani, 2016; Leibig et al., 2017):
dropout stays active at inference, N stochastic passes are run, and the spread
across passes is the uncertainty used to defer cases and score what is retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

from . import config
from .analysis import ppv_npv, sens_spec
from .data_loader import assert_load_health, reset_load_stats, split_frames
from .evaluate import optimal_threshold
from .models import MODEL_REGISTRY, _SCRATCH_MODELS, build_model
from .train import _build_datasets, get_normalisation_mode

EPS = 1e-7
FRACTIONS = np.round(np.arange(0.0, 0.51, 0.05), 2)
_DEVICE = "/GPU:0"


# Model surgery


def split_head(model: keras.Model):
    """(backbone, head_layers) for a transfer model; raises for scratch CNNs."""
    nested = [l for l in model.layers if isinstance(l, keras.Model)]
    if not nested:
        raise NotImplementedError(
            "MC-Dropout is implemented for the transfer architectures (nested backbone + head)."
        )
    backbone = nested[0]
    head = model.layers[model.layers.index(backbone) + 1 :]
    if not any(isinstance(l, keras.layers.Dropout) for l in head):
        raise RuntimeError("no Dropout layer in the head - nothing to sample")
    return backbone, head


def head_forward(head, feats: tf.Tensor, *, mc: bool) -> tf.Tensor:
    """Run the head with Dropout in training mode (mc=True) and BN in inference."""
    h = feats
    for layer in head:
        h = layer(h, training=(mc and isinstance(layer, keras.layers.Dropout)))
    return tf.reshape(h, [-1])


def collect(model_name: str, weights: Path, ds, n_passes: int, seed: int) -> dict:
    """Deterministic and MC-sampled predictions with the derived spread measures. The
    backbone runs once and only the head is resampled, so N passes cost about one pass.
    """
    with tf.device(_DEVICE):
        model = build_model(model_name)
        model.load_weights(str(weights))
        backbone, head = split_head(model)
        tf.random.set_seed(seed)
        y_all, det, mc = [], [], []
        for x, y in ds:
            feats = backbone(x, training=False)
            det.append(head_forward(head, feats, mc=False).numpy())
            samples = np.stack(
                [head_forward(head, feats, mc=True).numpy() for _ in range(n_passes)], axis=0
            )  # [N, B]
            mc.append(samples)
            y_all.append(y.numpy().ravel())
    y = np.concatenate(y_all).astype(int)
    p_det = np.concatenate(det).astype(float)
    S = np.concatenate(mc, axis=1).astype(float)  # [N, n]
    S = np.clip(S, EPS, 1 - EPS)
    mean = S.mean(0)
    h_mean = -(mean * np.log(mean) + (1 - mean) * np.log(1 - mean))
    h_each = -(S * np.log(S) + (1 - S) * np.log(1 - S)).mean(0)
    return dict(
        y=y,
        p_det=p_det,
        mc_mean=mean,
        mc_std=S.std(0),
        entropy=h_mean,
        mutual_info=h_mean - h_each,
        softmax_uncertainty=1.0 - np.abs(2 * p_det - 1),
    )


# Deferral analysis


def deferral_curve(
    y: np.ndarray, p: np.ndarray, u: np.ndarray, thr: float, cutoffs: np.ndarray, prev: float
) -> list[dict]:
    """Retained-case metrics when the cases with u > cutoff are deferred; `cutoffs` are
    validation quantiles of u, so the fraction actually deferred is reported.
    """
    rows = []
    for f, c in zip(FRACTIONS, cutoffs):
        keep = u <= c if f > 0 else np.ones_like(u, bool)
        yk, pk = y[keep], p[keep]
        auc = roc_auc_score(yk, pk) if len(np.unique(yk)) > 1 else float("nan")
        s, sp = sens_spec(yk, pk, thr)
        ppv, npv = ppv_npv(s, sp, prev)
        rows.append(
            dict(
                fraction_target=float(f),
                fraction_deferred=float(1 - keep.mean()),
                n_retained=int(keep.sum()),
                auc=float(auc),
                sensitivity=float(s),
                specificity=float(sp),
                ppv_at_prev=ppv,
                npv_at_prev=npv,
            )
        )
    return rows


def error_detection_auc(y: np.ndarray, p: np.ndarray, u: np.ndarray, thr: float) -> float:
    """AUROC of the uncertainty score for predicting a misclassification."""
    err = ((p >= thr).astype(int) != y).astype(int)
    return float(roc_auc_score(err, u)) if len(np.unique(err)) > 1 else float("nan")


def _decode_and_check(ds) -> None:
    """Decodes every input once and stops the run if more reads failed than
    config.MAX_LOAD_FAILURE_RATE allows; the pipeline caches the decoded inputs, so the
    MC passes read these same tensors."""
    reset_load_stats()
    for _ in ds:
        pass
    assert_load_health()


def run(
    model_name: str,
    tag: str,
    weights: Path,
    *,
    splits: list[str],
    source: str | None,
    margin: float,
    n_passes: int,
    boxes_dir: str | None = None,
) -> dict:
    norm = get_normalisation_mode(model_name)
    kwargs = dict(augment=False, source="crop", crop_margin=margin)
    if source and source != "oracle":
        from .boxes import provider_for

        kwargs.update(
            crop_sizing="box_provider",
            box_provider=provider_for(
                source,
                boxes_dir=boxes_dir,
                splits=(("train", "val", "test") if "test" in splits else ("train", "val")),
            ),
        )
    else:
        kwargs.update(crop_sizing="mask_margin")
    _, val_ds, test_ds, _, _ = _build_datasets(norm, **kwargs)
    frames = split_frames("crop")

    _decode_and_check(val_ds)
    data = {"val": collect(model_name, weights, val_ds, n_passes, config.SEED)}
    if "test" in splits:
        _decode_and_check(test_ds)
        data["test"] = collect(model_name, weights, test_ds, n_passes, config.SEED)

    # Fitted on validation, applied elsewhere unchanged.
    v = data["val"]
    thr = optimal_threshold(v["y"], v["mc_mean"], target_sensitivity=config.TARGET_SENSITIVITY)
    cut = {
        k: np.quantile(v[k], 1 - FRACTIONS)
        for k in ("mc_std", "mutual_info", "softmax_uncertainty")
    }
    prev = float(config.PREVALENCE["nhs_assessment"])

    out = dict(
        tag=tag,
        model=model_name,
        box_source=source or "oracle",
        margin=margin,
        n_passes=n_passes,
        threshold=float(thr),
        threshold_source="validation (mc_mean)",
        deployment_prevalence=prev,
        splits={},
    )
    for sp, d in data.items():
        fr = frames[sp]
        if len(fr) != len(d["y"]):
            raise RuntimeError(f"{sp}: {len(d['y'])} predictions vs {len(fr)} frame rows")
        tbl = pd.DataFrame(
            dict(
                lesion_id=fr["lesion_id"].to_numpy(),
                patient_id=fr["patient_id"].to_numpy(),
                y_true=d["y"],
                p_det=d["p_det"],
                mc_mean=d["mc_mean"],
                mc_std=d["mc_std"],
                entropy=d["entropy"],
                mutual_info=d["mutual_info"],
                softmax_uncertainty=d["softmax_uncertainty"],
                threshold=thr,
            )
        )
        tbl.to_csv(config.RESULTS_DIR / f"{tag}_mc_{sp}.csv", index=False)
        res = dict(
            n=int(len(d["y"])),
            auc_deterministic=float(roc_auc_score(d["y"], d["p_det"])),
            auc_mc_mean=float(roc_auc_score(d["y"], d["mc_mean"])),
            mc_std_median=float(np.median(d["mc_std"])),
            error_detection_auc={
                k: error_detection_auc(
                    d["y"], d["mc_mean"] if k != "softmax_uncertainty" else d["p_det"], d[k], thr
                )
                for k in ("mc_std", "mutual_info", "softmax_uncertainty")
            },
            deferral={
                k: deferral_curve(
                    d["y"],
                    d["mc_mean"] if k != "softmax_uncertainty" else d["p_det"],
                    d[k],
                    thr,
                    cut[k],
                    prev,
                )
                for k in ("mc_std", "mutual_info", "softmax_uncertainty")
            },
        )
        out["splits"][sp] = res
        _plot(res, tag, sp)
        print(f"\n=== {tag} on {sp} (n={res['n']}, N={n_passes} passes, thr {thr:.3f}) ===")
        print(
            f"AUC deterministic {res['auc_deterministic']:.4f} | MC mean {res['auc_mc_mean']:.4f}"
            f" | median MC std {res['mc_std_median']:.4f}"
        )
        print(
            "misclassification-detection AUROC: "
            + "  ".join(f"{k} {v:.3f}" for k, v in res["error_detection_auc"].items())
        )
        print(
            f"{'deferred':>9} | {'AUC mc_std':>10} {'AUC MI':>8} {'AUC softmax':>11} | "
            f"{'sens':>6} {'spec':>6} {'PPV@26%':>8}  (mc_std)"
        )
        for i, f in enumerate(FRACTIONS):
            a, b, c = (
                res["deferral"][k][i] for k in ("mc_std", "mutual_info", "softmax_uncertainty")
            )
            print(
                f"{a['fraction_deferred']:>9.2f} | {a['auc']:>10.4f} {b['auc']:>8.4f} "
                f"{c['auc']:>11.4f} | "
                f"{a['sensitivity']:>6.3f} {a['specificity']:>6.3f} {a['ppv_at_prev']:>8.3f}"
            )
    path = config.RESULTS_DIR / f"uncertainty_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")
    return out


def _plot(res: dict, tag: str, split: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = {
        "mc_std": "MC-Dropout std",
        "mutual_info": "MC mutual information",
        "softmax_uncertainty": "softmax confidence (baseline)",
    }
    for k, lab in labels.items():
        rows = res["deferral"][k]
        x = [r["fraction_deferred"] for r in rows]
        axes[0].plot(x, [r["auc"] for r in rows], "o-", ms=3, label=lab)
        axes[1].plot(x, [r["sensitivity"] for r in rows], "o-", ms=3, label=f"sens, {lab}")
        axes[1].plot(x, [r["specificity"] for r in rows], "s--", ms=3, label=f"spec, {lab}")
        axes[2].plot(x, [r["ppv_at_prev"] for r in rows], "o-", ms=3, label=lab)
    axes[0].axhline(res["auc_deterministic"], c="grey", lw=0.8, ls=":", label="no deferral")
    axes[2].axhline(config.PREVALENCE["nhs_assessment"], c="grey", lw=0.8, ls=":", label="prior")
    for ax, t in zip(
        axes,
        (
            "AUC on retained cases",
            "sens / spec on retained cases",
            f"PPV at {config.PREVALENCE['nhs_assessment']:.1%} prevalence",
        ),
    ):
        ax.set_xlabel("fraction deferred")
        ax.set_title(t, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle(f"{tag}: deferral analysis on {split}", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"deferral_{tag}_{split}.png", dpi=200)
    plt.close(fig)


def _refuse_test_outside_pass(splits: list[str]) -> None:
    """Refuses the test split unless the test pass has set its token
    (test_pass_guard.authorised)."""
    from .test_pass_guard import authorised

    if "test" in splits and not authorised():
        raise SystemExit(
            "REFUSED: --splits includes test; only notebooks/scripts/run_test_pass.sh may run it"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model", required=True, choices=[m for m in MODEL_REGISTRY if m not in _SCRATCH_MODELS]
    )
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    ap.add_argument(
        "--box-source",
        default=None,
        choices=["oracle", "unet", "cam", "jitter", "shuffled", "whole"],
    )
    ap.add_argument("--margin", type=float, default=config.CROP_MARGIN_FRAC)
    ap.add_argument("--n-passes", type=int, default=50)
    ap.add_argument(
        "--boxes-dir",
        default=None,
        help="candidate localiser folder: localiser_boxes_{split}.csv for --box-source unet, "
        "localiser_error_model.json for jitter; default: the frozen artefacts",
    )
    ap.add_argument("--cpu", action="store_true")
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

    global _DEVICE
    if args.cpu:
        _DEVICE = "/CPU:0"
    _refuse_test_outside_pass(args.splits)
    if "test" in args.splits:
        print(
            "[uncertainty] WARNING: test split requested - only under the "
            "pre-registered single test pass."
        )
    config.set_seeds()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_crop{args.tag_suffix}"
    # config.resolve_weights is the shared lookup: WEIGHTS_DIR, then each landing area.
    if args.weights:
        weights = Path(args.weights)
        if not weights.exists():
            raise SystemExit(f"weights not found: {weights}")
    else:
        try:
            weights = config.resolve_weights(tag)
        except FileNotFoundError as e:
            raise SystemExit(str(e))
    run(
        args.model,
        tag,
        weights,
        splits=args.splits,
        source=args.box_source,
        margin=args.margin,
        n_passes=args.n_passes,
        boxes_dir=args.boxes_dir,
    )


if __name__ == "__main__":
    main()

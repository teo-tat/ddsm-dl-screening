"""The single test pass for one model: the operating point is fitted on validation
and test is scored once at that point, writing a metric suite with patient-cluster
bootstrap CIs, per-row prediction tables and a model card."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from . import config
from . import data_loader
from .data_loader import reset_load_stats, assert_load_health
from .models import MODEL_REGISTRY, build_model, get_normalisation_mode

# Prediction collection

TTA_MODES: tuple[str, ...] = ("none", "flip", "flip_rot")


def _tta_views(images: tf.Tensor, tta: str) -> list[tf.Tensor]:
    """The deterministic views averaged at inference: "flip" fits every path, since
    left and right breasts are mirror images; "flip_rot" fits the crop path only."""
    if tta == "none":
        return [images]
    flipped = tf.image.flip_left_right(images)
    if tta == "flip":
        return [images, flipped]
    if tta == "flip_rot":
        return [tf.image.rot90(v, k) for v in (images, flipped) for k in range(4)]
    raise ValueError(f"tta must be one of {TTA_MODES}, not {tta!r}")


def collect_predictions(
    model: keras.Model, ds: tf.data.Dataset, tta: str = "none"
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on *ds* and return ``(y_true, y_prob)``, scoring through the
    eager path because the Metal backend drops ReLU from the compiled graph."""
    y_true_parts, y_prob_parts = [], []
    for images, labels in ds:
        views = _tta_views(images, tta)
        probs = tf.reduce_mean(tf.stack([model(v, training=False) for v in views]), axis=0)
        y_true_parts.append(labels.numpy())
        y_prob_parts.append(probs.numpy().ravel())

    return (np.concatenate(y_true_parts).astype(int), np.concatenate(y_prob_parts).astype(float))


# Optimal threshold (clinical target)


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = config.TARGET_SENSITIVITY
) -> float:
    """Return the threshold at the first roc_curve point whose sensitivity reaches the target.
    roc_curve drops some collinear points by default, so this can be lower than the highest
    threshold reaching the target; on the rows it is fitted on, specificity is the same unless a
    positive and a negative share a score. Fit it on validation only: choosing it on the scored
    rows would inflate specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    valid_indices = np.where(tpr >= target_sensitivity)[0]

    if len(valid_indices) > 0:
        # roc_curve returns thresholds in descending order, so the first valid index
        # is the highest kept threshold meeting the target.
        best_idx = int(valid_indices[0])
    else:
        # Reached only when y_true has no positive: otherwise the last ROC point has tpr = 1.
        best_idx = int(np.argmax(tpr))

    return float(thresholds[best_idx])


# Metric functions (for bootstrap)


def _sensitivity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    tp = np.sum((y_pred >= threshold) & (y_true == 1))
    fn = np.sum((y_pred < threshold) & (y_true == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _specificity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    tn = np.sum((y_pred < threshold) & (y_true == 0))
    fp = np.sum((y_pred >= threshold) & (y_true == 0))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    correct = np.sum((y_pred >= threshold) == y_true.astype(bool))
    return float(correct / len(y_true)) if len(y_true) > 0 else 0.0


# Bootstrap confidence intervals


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    *,
    groups: np.ndarray,
    n_iterations: int = config.BOOTSTRAP_ITERATIONS,
    ci_level: float = config.BOOTSTRAP_CI_LEVEL,
    seed: int = config.SEED,
    **metric_kwargs,
) -> tuple[float, float, float]:
    """``(point_estimate, ci_lower, ci_upper)`` by the percentile method, resampling
    patients rather than rows: two lesions from one woman are not independent."""
    rng = np.random.RandomState(seed)
    point = metric_fn(y_true, y_prob, **metric_kwargs)

    groups = np.asarray(groups)
    if len(groups) != len(y_true):
        raise ValueError(f"groups has {len(groups)} entries for {len(y_true)} predictions.")

    unique = np.unique(groups)
    rows_of = {g: np.where(groups == g)[0] for g in unique}
    # Draws are stratified by whether a patient carries any malignant lesion, so no
    # resample can be single-class and collapse the AUC.
    malignant = np.array([bool(np.any(y_true[rows_of[g]] == 1)) for g in unique])
    pos_patients, neg_patients = unique[malignant], unique[~malignant]

    scores = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        drawn = []
        if len(pos_patients):
            drawn.append(rng.choice(pos_patients, size=len(pos_patients), replace=True))
        if len(neg_patients):
            drawn.append(rng.choice(neg_patients, size=len(neg_patients), replace=True))
        boot_idx = np.concatenate([rows_of[g] for g in np.concatenate(drawn)])

        try:
            scores[i] = metric_fn(y_true[boot_idx], y_prob[boot_idx], **metric_kwargs)
        except (ValueError, ZeroDivisionError):
            scores[i] = np.nan

    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return point, float("nan"), float("nan")
    alpha = 1.0 - ci_level
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _aligned_groups(split_frame: pd.DataFrame, n_predictions: int, partition: str) -> np.ndarray:
    """Return the patient id for each prediction row, halting on a length mismatch:
    a dropped row would shift every later pairing onto the wrong patient."""
    col = config.GROUP_COL
    if col not in split_frame.columns:
        raise KeyError(
            f"{partition} split frame has no '{col}' column, so predictions "
            f"cannot be grouped by patient for the bootstrap."
        )
    if len(split_frame) != n_predictions:
        raise RuntimeError(
            f"{partition}: {n_predictions} predictions but {len(split_frame)} "
            f"rows in the split frame. Rows were dropped when the dataset was "
            f"built, so predictions can no longer be matched to patients."
        )
    return split_frame[col].to_numpy()


_META_COLS = (
    "lesion_id",
    "image_id",
    "patient_id",
    "image file path",
    "cropped image file path",
    "label",
    "assessment",
    "subtlety",
    "breast_density",
    "breast density",
    "abnormality id",
    "left or right breast",
    "image view",
    "n_abnormalities",
    "labels_mixed",
)


def _prediction_table(
    frame: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, *, partition: str, threshold: float
) -> pd.DataFrame:
    """One row per prediction, with the join key, subgroup columns and patient id
    the later analyses need. Halts if frame and predictions cannot be aligned."""
    if len(frame) != len(y_true):
        raise RuntimeError(
            f"{partition}: {len(y_true)} predictions but "
            f"{len(frame)} frame rows - cannot align."
        )
    cols = [c for c in _META_COLS if c in frame.columns]
    out = frame[cols].reset_index(drop=True).copy()
    if "breast density" in out.columns and "breast_density" not in out.columns:
        out = out.rename(columns={"breast density": "breast_density"})
    if not np.array_equal(out["label"].to_numpy().astype(int), np.asarray(y_true)):
        raise RuntimeError(
            f"{partition}: frame labels disagree with dataset labels - rows are misaligned."
        )
    out["y_true"] = np.asarray(y_true).astype(int)
    out["y_prob"] = np.asarray(y_prob).astype(float)
    out["partition"] = partition
    out["threshold"] = float(threshold)
    return out


def _auc_roc_fn(y_true, y_prob, **_kw):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def _auc_pr_fn(y_true, y_prob, **_kw):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


# DeLong test (DeLong et al. 1988)


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Nonparametric test for the difference of two correlated AUCs, in the
    structural-component (placement-value) form of DeLong et al. (1988)."""
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos < 2 or n_neg < 2:
        return {
            "z": np.nan,
            "p_value": np.nan,
            "auc_a": np.nan,
            "auc_b": np.nan,
            "auc_diff": np.nan,
        }

    # Placement values V10[i] = P(positive i scored > a random negative)
    pos_a, neg_a = pa[pos_idx], pa[neg_idx]
    pos_b, neg_b = pb[pos_idx], pb[neg_idx]

    V10_a = np.array([np.mean(p > neg_a) + 0.5 * np.mean(p == neg_a) for p in pos_a])
    V10_b = np.array([np.mean(p > neg_b) + 0.5 * np.mean(p == neg_b) for p in pos_b])
    V01_a = np.array([np.mean(pos_a > n) + 0.5 * np.mean(pos_a == n) for n in neg_a])
    V01_b = np.array([np.mean(pos_b > n) + 0.5 * np.mean(pos_b == n) for n in neg_b])

    auc_a = float(np.mean(V10_a))
    auc_b = float(np.mean(V10_b))

    # 2×2 covariance matrix S = S10/n_pos + S01/n_neg
    S10 = np.cov(V10_a, V10_b)  # shape (2,2), ddof=1
    S01 = np.cov(V01_a, V01_b)
    S = S10 / n_pos + S01 / n_neg

    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return {"z": 0.0, "p_value": 1.0, "auc_a": auc_a, "auc_b": auc_b, "auc_diff": auc_a - auc_b}

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z)))

    return {
        "z": float(z),
        "p_value": p,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": float(auc_a - auc_b),
    }


# McNemar's test


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
) -> dict:
    """Continuity-corrected McNemar chi-squared, not the exact binomial test: do the
    two models make different errors at their respective operating points?"""
    dec_a = (pred_a >= threshold_a).astype(int)
    dec_b = (pred_b >= threshold_b).astype(int)

    correct_a = dec_a == y_true
    correct_b = dec_b == y_true

    # Discordant pairs
    b = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    c = int(np.sum(correct_a & ~correct_b))  # A right, B wrong

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}

    chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
    p = float(1 - stats.chi2.cdf(chi2, df=1))

    return {"chi2": chi2, "p_value": p, "b": b, "c": c}


# Full metric suite for one model


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    *,
    threshold: float,
    threshold_source: str,
    groups: np.ndarray,
) -> dict:
    """The metric suite with patient-level 95 % CIs. *threshold* is fitted
    elsewhere; threshold_source="test" is refused, only "validation" is reportable."""
    if threshold_source == "test":
        raise ValueError(
            "threshold_source='test' — the operating point may not be chosen "
            "on the partition it is scored against. Fit it on validation and "
            "pass it in."
        )
    if threshold_source != "validation":
        print(
            f"[evaluate] WARNING: threshold_source={threshold_source!r}. "
            f"Only 'validation' is reportable; this result is exploratory."
        )

    thresh = float(threshold)

    auc_val, auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, _auc_roc_fn, groups=groups)

    aucpr_val, aucpr_lo, aucpr_hi = bootstrap_ci(y_true, y_prob, _auc_pr_fn, groups=groups)

    sens_val, sens_lo, sens_hi = bootstrap_ci(
        y_true, y_prob, _sensitivity, groups=groups, threshold=thresh
    )

    spec_val, spec_lo, spec_hi = bootstrap_ci(
        y_true, y_prob, _specificity, groups=groups, threshold=thresh
    )

    acc_val, acc_lo, acc_hi = bootstrap_ci(
        y_true, y_prob, _accuracy, groups=groups, threshold=thresh
    )

    y_dec = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_dec, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results = {
        "model": model_name,
        "n_test_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "n_patients": int(len(np.unique(np.asarray(groups)))),
        "operating_threshold": round(thresh, 6),
        "threshold_source": threshold_source,
        "target_sensitivity": config.TARGET_SENSITIVITY,
        "bootstrap": {
            "unit": "patient",
            "n_iterations": config.BOOTSTRAP_ITERATIONS,
            "ci_level": config.BOOTSTRAP_CI_LEVEL,
        },
        "metrics": {
            "auc_roc": {
                "value": round(auc_val, 4),
                "ci_95_lower": round(auc_lo, 4),
                "ci_95_upper": round(auc_hi, 4),
            },
            "auc_pr": {
                "value": round(aucpr_val, 4),
                "ci_95_lower": round(aucpr_lo, 4),
                "ci_95_upper": round(aucpr_hi, 4),
            },
            "sensitivity": {
                "value": round(sens_val, 4),
                "ci_95_lower": round(sens_lo, 4),
                "ci_95_upper": round(sens_hi, 4),
            },
            "specificity": {
                "value": round(spec_val, 4),
                "ci_95_lower": round(spec_lo, 4),
                "ci_95_upper": round(spec_hi, 4),
            },
            "accuracy": {
                "value": round(acc_val, 4),
                "ci_95_lower": round(acc_lo, 4),
                "ci_95_upper": round(acc_hi, 4),
            },
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "baselines": {
            # BI-RADS reference values: context, not verdict.
            "birads_auc_roc": config.BIRADS["auc_roc"],
            "birads_auc_roc_ci": config.BIRADS["auc_roc_ci"],
            "birads_sensitivity": config.BIRADS["sens"],
            "birads_specificity": config.BIRADS["spec"],
            "birads_auc_pr": config.BIRADS["auc_pr"],
            "birads_variant": config.BIRADS["variant"],
            "birads_n": config.BIRADS["n_test_comparison"],
            # Signed differences, not pass/fail: the baseline covers a different
            # sample, and the verdict comes from a paired DeLong test.
            "delta_auc_roc": auc_val - config.BIRADS["auc_roc"],
            "delta_sensitivity": sens_val - config.BIRADS["sens"],
            "delta_auc_pr": aucpr_val - config.BIRADS["auc_pr"],
            "birads_comparison_is_paired": False,  # the paired test is in src.compare
        },
        "literature_context": {
            "wang_2024_auc_reference": config.LITERATURE_AUC_REFERENCE,
            "wang_2024_sensitivity_reference": config.LITERATURE_SENSITIVITY_REFERENCE,
            "wang_2024_specificity_reference": config.LITERATURE_SPECIFICITY_REFERENCE,
        },
    }
    return results


# Console reporting


def print_results(results: dict) -> None:
    name = results["model"]
    m = results["metrics"]
    bl = results["baselines"]

    print(f"\n{'='*60}")
    print(f"  Evaluation results: {name}")
    print(f"{'='*60}")
    print(
        f"  Test samples : {results['n_test_samples']} "
        f"(+{results['n_positive']} / −{results['n_negative']}) "
        f"across {results['n_patients']} patients"
    )
    print(
        f"  Threshold    : {results['operating_threshold']:.4f} "
        f"(fitted on {results['threshold_source']} at sensitivity ≥ "
        f"{results['target_sensitivity']:.2f})"
    )
    print(
        f"  CIs          : {results['bootstrap']['n_iterations']} resamples, "
        f"unit = {results['bootstrap']['unit']}\n"
    )

    header = f"  {'Metric':<16} {'Value':>8} {'95 % CI':>20} {'BI-RADS':>8} {'Δ':>8}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    # Δ is a signed difference, not a verdict: the model is not paired with the
    # baseline's comparison set. The paired DeLong test is reported separately.
    for key, label, bl_key in [
        ("sensitivity", "Sensitivity", "birads_sensitivity"),
        ("specificity", "Specificity", "birads_specificity"),
        ("auc_roc", "AUC-ROC", "birads_auc_roc"),
        ("auc_pr", "AUC-PR", "birads_auc_pr"),
        ("accuracy", "Accuracy", None),
    ]:
        v = m[key]
        ci_str = f"[{v['ci_95_lower']:.4f}, {v['ci_95_upper']:.4f}]"

        if bl_key is not None and bl.get(bl_key) is not None:
            baseline_val = float(bl[bl_key])
            bl_str = f"{baseline_val:>8.3f} {v['value'] - baseline_val:>+8.3f}"
        else:
            bl_str = f"{'—':>8} {'—':>8}"

        print(f"  {label:<16} {v['value']:>8.4f} {ci_str:>20} {bl_str}")

    print(
        f"\n  Δ is descriptive only: the BI-RADS verdict needs the paired "
        f"DeLong test\n  on the shared comparison set "
        f"(n = {bl['birads_n']}, variant = {bl['birads_variant']})."
    )

    cm = results["confusion_matrix"]
    print(f"\n  Confusion matrix (threshold = {results['operating_threshold']:.4f}):")
    print("                Pred −   Pred +")
    print(f"    Actual −    {cm['tn']:>5}    {cm['fp']:>5}")
    print(f"    Actual +    {cm['fn']:>5}    {cm['tp']:>5}")
    print()


# Single-model evaluation


def _save_model_card(
    model_name: str, weights_path: Path, model: keras.Model, run_config: dict
) -> None:
    """Save a JSON model card recording config, parameter counts, weights file and
    TF version: the settings a re-run of this evaluation must match."""
    card = {
        "model_name": model_name,
        "weights_file": str(weights_path),
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(tf.size(v).numpy() for v in model.trainable_variables)),
        "framework": {"tensorflow": tf.__version__, "python": sys.version},
        "training_config": {
            "seed": config.SEED,
            "target_size": list(config.TARGET_SIZE),
            "batch_size": config.BATCH_SIZE,
            "initial_lr": config.INITIAL_LR,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "val_fraction": config.VAL_FRACTION,
            "test_fraction": config.TEST_FRACTION,
            "class_weight_strategy": config.CLASS_WEIGHT_STRATEGY,
        },
        # The pipeline actually used at evaluation; it must match the training-time
        # values, or the loaded weights see a different input distribution.
        "eval_pipeline": run_config,
        "analysis_plan": {
            "target_sensitivity": config.TARGET_SENSITIVITY,
            "threshold_fitted_on": "validation",
            "bootstrap_unit": config.GROUP_COL,
            "birads_baseline": config.BIRADS,
        },
        "normalisation": config.NORM_STATS,
        "split": {
            "train_csv": str(config.TRAIN_CSV),
            "test_csv": str(config.TEST_CSV),
            "strategy": "patient-level StratifiedGroupKFold, pooled CSVs",
        },
    }

    card_path = config.RESULTS_DIR / f"{run_config.get('tag', model_name)}_model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"[evaluate] Model card saved → {card_path}")


def evaluate_single(
    model_name: str,
    weights_path: str | Path | None = None,
    save_predictions: bool = False,
    source: str = "full",
    clahe: bool = False,
    target_size: tuple[int, int] | None = None,
    pad_square: bool | None = None,
    batch_size: int | None = None,
    intensity_norm: str | None = None,
    crop_sizing: str = config.CROP_SIZING,
    crop_margin: float = config.CROP_MARGIN_FRAC,
    tag_suffix: str = "",
    mask_overlay: bool = False,
    box_provider=None,
    tta: str = "none",
    split: str = "test",
) -> dict:
    """Load a trained model, rebuild the evaluation split and compute all metrics.
    split="val" writes only the validation table and stops before any test row."""
    config.set_seeds()
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', not {split!r}")

    norm = get_normalisation_mode(model_name)
    # val_ds is needed as well as test_ds: the operating point is fitted on
    # validation and carried to test unchanged.
    _, val_ds, test_ds, test_split, _ = data_loader.build_datasets(
        train_csv=config.TRAIN_CSV,
        test_csv=config.TEST_CSV,
        train_img_dir=config.TRAIN_IMG_DIR,
        test_img_dir=config.TEST_IMG_DIR,
        train_roi_dir=config.TRAIN_ROI_DIR,
        test_roi_dir=config.TEST_ROI_DIR,
        target_size=target_size or config.TARGET_SIZE,
        batch_size=batch_size or config.BATCH_SIZE,
        normalisation=norm,
        augment=False,  # never augment test data
        source=source,
        clahe=clahe,
        pad_square=pad_square,
        intensity_norm=intensity_norm,
        crop_sizing=crop_sizing,
        crop_margin=crop_margin,
        mask_overlay=mask_overlay,
        box_provider=box_provider,
    )

    channels = MODEL_REGISTRY[model_name]["channels"]
    ts = target_size or config.TARGET_SIZE
    model = build_model(model_name, input_shape=(ts[0], ts[1], channels))
    # Not needed to load the weights or to score, which runs eagerly; it changes no prediction.
    model.compile(optimizer="adam", loss="binary_crossentropy")

    tag = f"{model_name}_crop" if source == "crop" else model_name
    tag = f"{tag}{tag_suffix}"
    if tta not in TTA_MODES:
        raise ValueError(f"tta must be one of {TTA_MODES}, not {tta!r}")

    if weights_path is None:
        # config.resolve_weights searches WEIGHTS_DIR, then the CUDA landing folders, and
        # names every folder it searched when it finds nothing.
        weights_path = config.resolve_weights(tag)

    weights_path = Path(weights_path)
    print(f"[evaluate] Loading weights from {weights_path}")
    model.load_weights(str(weights_path))
    # Weights resolve on the plain tag; outputs of a TTA run get their own tag.
    if tta != "none":
        tag = f"{tag}_tta_{tta}"
        print(f"[evaluate] test-time augmentation: {tta} -> outputs tagged {tag}")

    reset_load_stats()

    # 1. Fit the operating point on VALIDATION
    y_val, p_val = collect_predictions(model, val_ds, tta=tta)
    threshold = optimal_threshold(y_val, p_val, target_sensitivity=config.TARGET_SENSITIVITY)
    val_sens = _sensitivity(y_val, p_val, threshold)
    val_spec = _specificity(y_val, p_val, threshold)
    print(
        f"[evaluate] Operating point fitted on validation: "
        f"threshold={threshold:.4f} → sens={val_sens:.4f}, spec={val_spec:.4f} "
        f"(n={len(y_val)})"
    )

    frames = data_loader.split_frames(source=source)
    key = "lesion_id" if source == "crop" else "image_id"

    # 1b. Validation-only mode: the val table, then stop
    # test_ds is lazy and never iterated here, so no test pixel is decoded.
    if split == "val":
        assert_load_health()
        tbl = _prediction_table(frames["val"], y_val, p_val, partition="val", threshold=threshold)
        tbl_path = config.RESULTS_DIR / f"{tag}_predictions_val.csv"
        tbl.to_csv(tbl_path, index=False)
        val_auc = float(roc_auc_score(y_val, p_val))
        print(f"[evaluate] val prediction table ({len(tbl)} rows) → {tbl_path}")
        print(
            f"[evaluate] validation only: AUC {val_auc:.4f} (exact, n={len(y_val)}); "
            f"test not scored"
        )
        print(f"-> {tbl_path}")
        return {
            "tag": tag,
            "split": "val",
            "n": int(len(y_val)),
            "auc_roc": val_auc,
            "threshold": float(threshold),
            "sensitivity": round(val_sens, 4),
            "specificity": round(val_spec, 4),
        }

    # 2. Score TEST at that fixed point
    y_true, y_prob = collect_predictions(model, test_ds, tta=tta)
    assert_load_health()

    groups = _aligned_groups(test_split, len(y_true), partition="test")

    # 3. Per-row prediction tables (always written)
    # Every downstream analysis runs from these files alone.
    if frames["test"][key].tolist() != test_split[key].tolist():
        raise RuntimeError(
            "split_frames() and build_datasets() disagree on the "
            "test partition - refusing to write prediction tables."
        )
    for part, fr, yt, yp in (
        ("val", frames["val"], y_val, p_val),
        ("test", frames["test"], y_true, y_prob),
    ):
        tbl = _prediction_table(fr, yt, yp, partition=part, threshold=threshold)
        tbl_path = config.RESULTS_DIR / f"{tag}_predictions_{part}.csv"
        tbl.to_csv(tbl_path, index=False)
        print(f"[evaluate] {part} prediction table ({len(tbl)} rows) → {tbl_path}")

    if save_predictions:
        pred_path = config.RESULTS_DIR / f"{tag}_predictions.npz"
        np.savez(
            pred_path,
            y_true=y_true,
            y_prob=y_prob,
            groups=groups,
            y_val=y_val,
            p_val=p_val,
            threshold=np.array(threshold),
        )
        print(f"[evaluate] Predictions saved → {pred_path}")

    results = compute_all_metrics(
        y_true,
        y_prob,
        model_name,
        threshold=threshold,
        threshold_source="validation",
        groups=groups,
    )
    results["tag"] = tag
    results["validation_operating_point"] = {
        "n": int(len(y_val)),
        "sensitivity": round(val_sens, 4),
        "specificity": round(val_spec, 4),
    }
    print_results(results)

    # Filenames carry the full tag, not just the architecture, so a crop run and a
    # whole-image run of the same model do not overwrite each other.
    out_path = config.RESULTS_DIR / f"{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Results saved → {out_path}")

    _save_model_card(
        model_name,
        weights_path,
        model,
        run_config={
            "tag": tag,
            "source": source,
            "clahe": clahe,
            "target_size": list(target_size or config.TARGET_SIZE),
            "pad_square": pad_square,
            "batch_size": batch_size or config.BATCH_SIZE,
            "intensity_norm": intensity_norm,
            "crop_sizing": crop_sizing,
            "crop_margin": crop_margin,
            "mask_overlay": mask_overlay,
            "tag_suffix": tag_suffix,
            "normalisation": norm,
            "box_source": getattr(box_provider, "source", None),
            "tta": tta,
        },
    )

    return results


# Multi-model comparison


def compare_models(model_names: list[str]) -> None:
    """Evaluate every model, then run pairwise DeLong and McNemar tests; all models
    share one test partition, so the correlated-curve assumption holds."""
    config.set_seeds()

    # Each model needs its own test dataset: scratch and ImageNet backbones
    # take different channel counts.
    all_results: dict[str, dict] = {}
    all_y_true: dict[str, np.ndarray] = {}
    all_y_prob: dict[str, np.ndarray] = {}
    all_thresh: dict[str, float] = {}

    for name in model_names:
        try:
            results = evaluate_single(name, save_predictions=True)
            all_results[name] = results
            # Reload saved predictions for DeLong (avoids re-inference)
            preds = np.load(config.RESULTS_DIR / f"{name}_predictions.npz", allow_pickle=True)
            all_y_true[name] = preds["y_true"]
            all_y_prob[name] = preds["y_prob"]
            all_thresh[name] = results["operating_threshold"]
        except FileNotFoundError as e:
            print(f"[evaluate] SKIPPING {name}: {e}")
            continue

    evaluated = list(all_results)
    if len(evaluated) < 2:
        print("[evaluate] Fewer than 2 models available — skipping pairwise comparisons.")
        return

    # Sanity check: all models should have the same y_true
    ref_true = all_y_true[evaluated[0]]
    for name in evaluated[1:]:
        if not np.array_equal(ref_true, all_y_true[name]):
            print(
                f"[evaluate] WARNING: y_true mismatch between "
                f"{evaluated[0]} and {name} — DeLong test may be invalid."
            )

    # Pairwise DeLong tests
    print(f"\n{'='*60}")
    print("  Pairwise DeLong AUC-ROC comparisons")
    print(f"{'='*60}\n")

    delong_rows = []
    for a, b in combinations(evaluated, 2):
        result = delong_test(ref_true, all_y_prob[a], all_y_prob[b])
        sig = (
            "***"
            if result["p_value"] < 0.001
            else (
                "**" if result["p_value"] < 0.01 else ("*" if result["p_value"] < 0.05 else "n.s.")
            )
        )

        print(
            f"  {a:>15} vs {b:<15} "
            f"ΔAUC = {result['auc_diff']:+.4f} "
            f"z = {result['z']:.3f} "
            f"p = {result['p_value']:.4f} {sig}"
        )

        delong_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "auc_a": round(result["auc_a"], 4),
                "auc_b": round(result["auc_b"], 4),
                "auc_diff": round(result["auc_diff"], 4),
                "z_statistic": round(result["z"], 4),
                "p_value": round(result["p_value"], 4),
                "significant_005": result["p_value"] < 0.05,
            }
        )

    # Pairwise McNemar tests
    print(f"\n{'='*60}")
    print("  Pairwise McNemar tests (each model at its own validation-fitted operating point)")
    print(f"{'='*60}\n")

    mcnemar_rows = []
    for a, b in combinations(evaluated, 2):
        result = mcnemar_test(ref_true, all_y_prob[a], all_y_prob[b], all_thresh[a], all_thresh[b])
        sig = "*" if result["p_value"] < 0.05 else "n.s."

        print(
            f"  {a:>15} vs {b:<15} "
            f"χ² = {result['chi2']:.3f} "
            f"p = {result['p_value']:.4f} "
            f"b = {result['b']} c = {result['c']} {sig}"
        )

        mcnemar_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "chi2": round(result["chi2"], 4),
                "p_value": round(result["p_value"], 4),
                "b_a_wrong_b_right": result["b"],
                "c_a_right_b_wrong": result["c"],
                "significant_005": result["p_value"] < 0.05,
            }
        )

    # Ranking table
    print(f"\n{'='*60}")
    print("  Model ranking (by AUC-ROC)")
    print(f"{'='*60}\n")

    ranked = sorted(
        evaluated, key=lambda n: all_results[n]["metrics"]["auc_roc"]["value"], reverse=True
    )

    print(
        f"  {'Rank':>4} {'Model':<15} {'AUC-ROC':>8} {'Sens':>8} "
        f"{'Spec':>8} {'AUC-PR':>8} {'Acc':>8} {'> BI-RADS?':>10}"
    )
    print(f"  {'-'*85}")
    for rank, name in enumerate(ranked, 1):
        m = all_results[name]["metrics"]
        bl = all_results[name]["baselines"]
        exceeds = (
            f"ΔAUC {bl['delta_auc_roc']:+.3f}, "
            f"Δsens {bl['delta_sensitivity']:+.3f} "
            f"vs BI-RADS (n={bl['birads_n']}, not tested — "
            f"paired DeLong required)"
        )
        print(
            f"  {rank:>4} {name:<15} "
            f"{m['auc_roc']['value']:>8.4f} "
            f"{m['sensitivity']['value']:>8.4f} "
            f"{m['specificity']['value']:>8.4f} "
            f"{m['auc_pr']['value']:>8.4f} "
            f"{m['accuracy']['value']:>8.4f} "
            f"{exceeds:>10}"
        )
    print()

    # Save comparison report
    comparison = {
        "models_evaluated": evaluated,
        "individual_results": {n: all_results[n] for n in evaluated},
        "ranking_by_auc_roc": ranked,
        "delong_tests": delong_rows,
        "mcnemar_tests": mcnemar_rows,
        "holm_note": (
            "p-values above are UNCORRECTED. The pre-registered secondary "
            "family is the pairwise DeLong set; apply Holm before reporting."
        ),
    }
    out_path = config.RESULTS_DIR / "comparison_report.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[evaluate] Comparison report saved → {out_path}")


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CBIS-DDSM mass classifiers.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", type=str, choices=list(MODEL_REGISTRY), help="Evaluate a single model."
    )
    group.add_argument(
        "--compare",
        nargs="+",
        metavar="MODEL",
        help="Not supported: exits with a message naming src.compare, which runs comparisons.",
    )
    p.add_argument(
        "--weights", type=str, default=None, help="Path to weights file (auto-resolved if omitted)."
    )
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also write the test and validation predictions, patient ids and threshold to an "
        ".npz file (test split only).",
    )
    p.add_argument(
        "--source",
        choices=["full", "crop"],
        default="full",
        help="Image source: whole mammogram or ROI crop.",
    )
    # Must mirror the training-time input pipeline for the loaded weights.
    p.add_argument(
        "--image-size", type=int, default=None, help="Input resolution; must match training."
    )
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE; must match training.")
    p.add_argument(
        "--pad-square",
        dest="pad_square",
        action="store_true",
        help="Force square padding; must match training.",
    )
    p.add_argument(
        "--no-pad-square", dest="pad_square", action="store_false", help="Force no padding."
    )
    p.set_defaults(pad_square=None)  # None = decide by source
    p.add_argument(
        "--intensity-norm",
        choices=["fixed_max", "percentile", "minmax", "clahe"],
        default=None,
        help="Intensity strategy; must match training.",
    )
    p.add_argument(
        "--crop-sizing",
        choices=["provided_tight", "mask_margin", "box_provider"],
        default=config.CROP_SIZING,
        help="Crop derivation; must match training.",
    )
    p.add_argument(
        "--box-source",
        choices=["oracle", "unet", "cam", "jitter", "shuffled", "whole"],
        default=None,
        help="Ladder rung: crop from this box source (implies --crop-sizing "
        "box_provider). Boxes come from artifacts/ or --boxes-dir, not from masks.",
    )
    p.add_argument(
        "--crop-margin",
        type=float,
        default=config.CROP_MARGIN_FRAC,
        help="Mask-margin bbox fraction; must match training.",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Eval batch size.")
    p.add_argument(
        "--tag-suffix",
        type=str,
        default="",
        help="appended to {model}_crop or {model} to form the tag of the weights and outputs",
    )
    p.add_argument(
        "--mask-overlay",
        action="store_true",
        help="Apply dilated ROI mask overlay; must match training.",
    )
    p.add_argument(
        "--boxes-dir",
        type=str,
        default=None,
        help="candidate localiser folder: localiser_boxes_{split}.csv for --box-source unet, "
        "localiser_error_model.json for jitter; default: the frozen artefacts",
    )
    p.add_argument(
        "--tta",
        choices=list(TTA_MODES),
        default="none",
        help="test-time augmentation: average over flip (2 views) or "
        "flip_rot (8 dihedral views, crop path only). Outputs are tagged "
        "{tag}_tta_{mode}; default none = unchanged.",
    )
    p.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="test (default) = the single pre-registered pass, run only from "
        "notebooks/scripts/run_test_pass.sh. val = write the validation "
        "prediction table only, with the threshold fitted on those rows, "
        "and score no test row.",
    )
    p.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag "
        "that no longer exists fails in a second.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    config.set_seeds()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        raise SystemExit(
            "--compare is retired: run `python -m src.evaluate` once per "
            "model, then `python -m src.compare --primary ... --models ...`."
        )
    else:
        target_size = (args.image_size, args.image_size) if args.image_size else None
        provider, crop_sizing = None, args.crop_sizing
        if args.box_source:
            from .boxes import provider_for

            # A candidate boxes folder holds no test table, so a validation-only run
            # builds its provider on train + val; a test run keeps all three splits.
            splits = ("train", "val") if args.split == "val" else ("train", "val", "test")
            provider = provider_for(args.box_source, boxes_dir=args.boxes_dir, splits=splits)
            crop_sizing = "box_provider"
        evaluate_single(
            args.model,
            weights_path=args.weights,
            save_predictions=args.save_predictions,
            source=args.source,
            clahe=args.clahe,
            target_size=target_size,
            pad_square=args.pad_square,
            batch_size=args.batch_size,
            intensity_norm=args.intensity_norm,
            crop_sizing=crop_sizing,
            crop_margin=args.crop_margin,
            tag_suffix=args.tag_suffix,
            mask_overlay=args.mask_overlay,
            box_provider=provider,
            tta=args.tta,
            split=args.split,
        )


if __name__ == "__main__":
    main()

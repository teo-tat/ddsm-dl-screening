"""Two-crop fusion model. A view pair is one mass on the CC view and one on the MLO view
of the same breast; a context pair is one lesion cropped at two margins; the control is
the single-crop model on the same rows, so a comparison isolates the second stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

from . import config, data_loader
from .boxes import BoxProvider
from .models import build_fusion, build_model, unfreeze_for_finetuning
from .train import _compile, _ensure_dirs, _get_callbacks, _merge_histories, _save_history
from .evaluate import (
    _prediction_table,
    _sensitivity,
    _specificity,
    collect_predictions,
    compute_all_metrics,
    optimal_threshold,
)

ART = Path(__file__).resolve().parent.parent / "artifacts"
PairMode = Literal["view", "context", "control"]
BoxSource = Literal["oracle", "unet"]

_PAIR_COLS = (
    "split",
    "mode",
    "pair_kind",
    "lesion_id_a",
    "lesion_id_b",
    "patient_id",
    "side",
    "label",
)


# Pairs


def _view_pairs(frame: pd.DataFrame, single_mass_only: bool) -> tuple[pd.DataFrame, dict]:
    """CC<->MLO pairs within one split frame. A breast with one mass in each view pairs
    outright; an `abnormality id` match is the same breast, not verifiably the lesion.
    """
    rows, counts = [], dict(
        single_mass=0, id_match=0, disagree=0, no_counterpart=0, unmatched_multi=0
    )
    for (pid, side), grp in frame.groupby(["patient_id", "left or right breast"], sort=True):
        cc = grp[grp["image view"] == "CC"]
        mlo = grp[grp["image view"] == "MLO"]
        if cc.empty or mlo.empty:
            counts["no_counterpart"] += len(grp)
            continue
        if len(cc) == 1 and len(mlo) == 1:
            cand = [(cc.iloc[0], mlo.iloc[0], "single_mass")]
        elif single_mass_only:
            counts["unmatched_multi"] += len(grp)
            continue
        else:
            by_id = {int(r["abnormality id"]): r for _, r in mlo.iterrows()}
            cand, used = [], set()
            for _, a in cc.iterrows():
                b = by_id.get(int(a["abnormality id"]))
                if b is not None:
                    cand.append((a, b, "id_match"))
                    used.add(int(a["abnormality id"]))
            counts["unmatched_multi"] += len(grp) - 2 * len(cand)
        for a, b, kind in cand:
            if int(a["label"]) != int(b["label"]):
                counts["disagree"] += 1
                continue
            counts[kind] += 1
            rows.append(
                dict(
                    pair_kind=kind,
                    lesion_id_a=a["lesion_id"],
                    lesion_id_b=b["lesion_id"],
                    patient_id=pid,
                    side=side,
                    label=int(a["label"]),
                )
            )
    return pd.DataFrame(rows, columns=list(_PAIR_COLS[2:])), counts


def build_pairs(
    frames: dict[str, pd.DataFrame], mode: PairMode, rows: str, *, single_mass_only: bool = False
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Paired lesion ids per split, inheriting the patient partition: rows="paired" keeps
    the view-paired crop-A lesions, rows="all" every lesion. Each pair carries A's label.
    """
    from .test_pass_guard import authorised

    pairs, counts = {}, {}
    # Test pairs are built only for the authorised pass, the only caller that scores them.
    for split in [s for s in frames if s != "test" or authorised()]:
        f = frames[split]
        vp, c = _view_pairs(f, single_mass_only)
        if mode == "view":
            p = vp.copy()
        else:
            keep = (
                f
                if rows == "all"
                else f.set_index("lesion_id").loc[vp["lesion_id_a"]].reset_index()
            )
            p = pd.DataFrame(
                dict(
                    pair_kind=f"{mode}_{rows}",
                    lesion_id_a=keep["lesion_id"],
                    lesion_id_b=keep["lesion_id"],
                    patient_id=keep["patient_id"],
                    side=keep["left or right breast"],
                    label=keep["label"].astype(int),
                )
            )
        p.insert(0, "mode", mode)
        p.insert(0, "split", split)
        c["n_lesions"] = int(len(f))
        c["n_pairs"] = int(len(p))
        c["pair_rate"] = float(2 * len(vp) / max(1, len(f)))  # lesions covered by a view pair
        pairs[split], counts[split] = p.reset_index(drop=True), c
    names = list(pairs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(pairs[a]["patient_id"]) & set(pairs[b]["patient_id"])
            if shared:
                raise RuntimeError(f"{a}/{b} pairs share {len(shared)} patients - partition broken")
    return pairs, counts


def write_pairs(pairs: dict[str, pd.DataFrame], mode: PairMode, single_mass_only: bool) -> None:
    """Write artifacts/fusion_pairs_{mode}[_single]_{split}.csv once. An existing file is
    never regenerated, and a rebuild that differs from it halts the run."""
    tag = f"{mode}_single" if single_mass_only else mode
    for split, p in pairs.items():
        path = ART / f"fusion_pairs_{tag}_{split}.csv"
        if path.exists():
            old = pd.read_csv(path)
            same = (
                old["lesion_id_a"].tolist() == p["lesion_id_a"].tolist()
                and old["lesion_id_b"].tolist() == p["lesion_id_b"].tolist()
            )
            if not same:
                raise RuntimeError(
                    f"{path} exists and differs from the rebuilt pairs - "
                    f"frozen artefact, not overwriting"
                )
            print(f"[fusion] {path.name} exists and matches ({len(p)} pairs) - kept")
            continue
        p.to_csv(path, index=False)
        print(f"-> {path}")


def print_counts(counts: dict[str, dict]) -> None:
    for split, c in counts.items():
        print(
            f"[fusion] {split:5s} lesions {c['n_lesions']:4d} | pairs {c['n_pairs']:4d} | "
            f"view pair-rate {c['pair_rate']:.3f} | single-mass {c['single_mass']} | "
            f"id-match {c['id_match']} | label-disagree {c['disagree']} | "
            f"no counterpart view {c['no_counterpart']} | "
            f"multi-mass unmatched {c['unmatched_multi']}"
        )


# Datasets


def _lookups(crop_sizing: str) -> tuple[dict, dict | None, dict]:
    """ROI, mask and full-image path lookups for the crop pipeline."""
    lookup = data_loader._build_combined_roi_lookup(config.TRAIN_ROI_DIR, config.TEST_ROI_DIR)
    mask_lookup = (
        data_loader._build_combined_roi_lookup(
            config.TRAIN_ROI_DIR, config.TEST_ROI_DIR, pick="mask"
        )
        if crop_sizing == "mask_margin"
        else None
    )
    full = data_loader._build_combined_lookup(config.TRAIN_IMG_DIR, config.TEST_IMG_DIR)
    full_lookup = {Path(k).parts[0]: v for k, v in full.items()}
    return lookup, mask_lookup, full_lookup


def _stream(
    frame: pd.DataFrame,
    margin: float,
    *,
    lookups: tuple,
    crop_sizing: str,
    provider: BoxProvider | None,
) -> tf.data.Dataset:
    """One crop per row, decoded by the same crop function every rung uses. Unshuffled,
    un-augmented and unbatched, so two streams can be zipped row for row."""
    lookup, mask_lookup, full_lookup = lookups
    ds = data_loader._make_dataset(
        frame,
        lookup,
        config.TARGET_SIZE,
        1,
        config.SEED,
        normalisation="imagenet",
        augment=False,
        shuffle=False,
        intensity_norm=config.CROP_INTENSITY_NORM,
        source="crop",
        path_col="cropped image file path",
        pad_square=config.CROP_PAD_SQUARE,
        crop_sizing=crop_sizing,
        crop_margin=margin,
        mask_lookup=mask_lookup,
        full_lookup=full_lookup,
        box_provider=provider,
    )
    return ds.unbatch()


def swap_pairs(ds: tf.data.Dataset) -> tf.data.Dataset:
    """((a, b), y) -> ((b, a), y); shares the upstream cache, decodes nothing."""
    return ds.map(lambda x, y: ((x[1], x[0]), y))


def build_pair_dataset(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    margin_a: float,
    margin_b: float,
    *,
    augment: bool,
    shuffle: bool,
    batch_size: int,
    crop_sizing: str,
    provider: BoxProvider | None,
    lookups: tuple,
    both_orderings: bool = False,
) -> tf.data.Dataset:
    """tf.data pipeline yielding ((crop_a, crop_b), y), y = label of crop A. both_orderings
    adds every pair swapped, as a map over the cached streams, decoding no image twice.
    """
    a = _stream(frame_a, margin_a, lookups=lookups, crop_sizing=crop_sizing, provider=provider)
    b = _stream(frame_b, margin_b, lookups=lookups, crop_sizing=crop_sizing, provider=provider)
    ds = tf.data.Dataset.zip((a, b)).map(lambda xa, xb: ((xa[0], xb[0]), xa[1]))
    n = len(frame_a)
    if both_orderings:
        ds = ds.concatenate(swap_pairs(ds))
        n *= 2
    if shuffle:
        ds = ds.shuffle(buffer_size=n, seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        # Applied after standardisation: the crop path's imagenet-mode augmentation is
        # geometric only, so it commutes with the per-channel affine standardisation.
        aug = data_loader._make_batch_augment_fn("imagenet", source="crop")

        def _aug(x, y):
            xa, _ = aug(x[0], y)
            xb, _ = aug(x[1], y)  # second call -> independent draw
            return (xa, xb), y

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_single_dataset(
    frame: pd.DataFrame,
    margin: float,
    *,
    augment: bool,
    shuffle: bool,
    batch_size: int,
    crop_sizing: str,
    provider: BoxProvider | None,
    lookups: tuple,
) -> tf.data.Dataset:
    """The control's dataset: the standard single-crop pipeline on crop-A rows."""
    lookup, mask_lookup, full_lookup = lookups
    return data_loader._make_dataset(
        frame,
        lookup,
        config.TARGET_SIZE,
        batch_size,
        config.SEED,
        normalisation="imagenet",
        augment=augment,
        shuffle=shuffle,
        intensity_norm=config.CROP_INTENSITY_NORM,
        source="crop",
        path_col="cropped image file path",
        pad_square=config.CROP_PAD_SQUARE,
        crop_sizing=crop_sizing,
        crop_margin=margin,
        mask_lookup=mask_lookup,
        full_lookup=full_lookup,
        box_provider=provider,
    )


def _frames_for(frame: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-frame rows for crop A and crop B, in pair order, all meta columns kept."""
    idx = frame.set_index("lesion_id")
    fa = idx.loc[pairs["lesion_id_a"]].reset_index()
    fb = idx.loc[pairs["lesion_id_b"]].reset_index()
    if not np.array_equal(fa["label"].to_numpy(int), pairs["label"].to_numpy(int)):
        raise RuntimeError("crop-A labels disagree with the pair table")
    return fa, fb


# Run


class FusionRun:
    """Everything one tag needs: pairs, frames, datasets, model, paths."""

    # Default row set per mode; a non-default choice is spelled out in the tag.
    _DEFAULT_ROWS = {"view": "paired", "context": "all", "control": "paired"}

    def __init__(
        self,
        arch: str,
        pair: PairMode,
        margin: float,
        box_source: BoxSource,
        *,
        single_mass_only: bool,
        rows: str | None,
        batch_size: int | None,
        tag_suffix: str,
        smoke_rows: int = 0,
        boxes_dir: str | None = None,
        test_boxes: bool = True,
        unet_k: int = 1,
        unet_mode: str = "single",
    ):
        self.arch, self.pair, self.margin, self.box_source = (arch, pair, margin, box_source)
        self.boxes_dir = boxes_dir
        self.batch_size = batch_size or config.BATCH_SIZE
        self.smoke_rows = smoke_rows
        self.rows = rows or self._DEFAULT_ROWS[pair]
        if pair == "view" and self.rows != "paired":
            raise ValueError("view mode is defined on its own pairs; --rows all is meaningless")
        self.symmetric = pair == "view"  # both orderings in training, averaged at eval
        self.tag = (
            f"{arch}_fusion_{pair}"
            + ("_unet" if box_source == "unet" else "")
            + ("_single" if single_mass_only else "")
            + (f"_{self.rows}" if self.rows != self._DEFAULT_ROWS[pair] else "")
            + tag_suffix
        )
        self.frames = data_loader.split_frames(source="crop")
        self.pairs, self.counts = build_pairs(
            self.frames, pair, self.rows, single_mass_only=single_mass_only
        )
        self.single_mass_only = single_mass_only
        self.margin_b = config.FUSION_CONTEXT_MARGIN if pair == "context" else margin
        if box_source == "unet":
            from .boxes import provider_for

            self.crop_sizing, self.provider = "box_provider", provider_for(
                "unet",
                boxes_dir=boxes_dir,
                splits=("train", "val", "test") if test_boxes else ("train", "val"),
                k=unet_k,
                mode=unet_mode,
            )
        else:
            self.crop_sizing, self.provider = "mask_margin", None
        self._lookups = None

    # Lazy, because --dry-run must not touch any image.
    @property
    def lookups(self) -> tuple:
        if self._lookups is None:
            self._lookups = _lookups(self.crop_sizing)
        return self._lookups

    def frames_for(self, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        p = self.pairs[split]
        if self.smoke_rows:
            p = p.head(self.smoke_rows)
        return _frames_for(self.frames[split], p)

    def dataset(self, split: str) -> tuple[tf.data.Dataset, pd.DataFrame]:
        fa, fb = self.frames_for(split)
        train = split == "train"
        kw = dict(
            augment=train,
            shuffle=train,
            batch_size=self.batch_size,
            crop_sizing=self.crop_sizing,
            provider=self.provider,
            lookups=self.lookups,
        )
        if self.pair == "control":
            return build_single_dataset(fa, self.margin, **kw), fa
        return (
            build_pair_dataset(
                fa, fb, self.margin, self.margin_b, both_orderings=self.symmetric and train, **kw
            ),
            fa,
        )

    def build_model(self) -> keras.Model:
        shape = (config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)
        if self.pair == "control":
            return build_model(self.arch, input_shape=shape)
        return build_fusion(self.arch, input_shape=shape)

    @property
    def canonical_weights(self) -> Path:
        return config.WEIGHTS_DIR / f"{self.tag}_best.weights.h5"


def train(run: FusionRun, epochs: int | None = None) -> keras.Model:
    """Two-stage fine-tune on train_transfer's schedule: frozen trunk, then fine-tuning at
    FINETUNE_LR. Unlike train_transfer, it adds no eager validation-AUC callback."""
    _ensure_dirs()
    train_ds, fa_train = run.dataset("train")
    val_ds, _ = run.dataset("val")
    class_weight = (
        data_loader.compute_class_weights(fa_train)
        if config.CLASS_WEIGHT_STRATEGY == "balanced"
        else None
    )
    max_epochs = epochs or config.MAX_EPOCHS
    model = run.build_model()
    tag = run.tag
    print(
        f"\n{'=' * 60}\n  Training: {tag} (fusion, 2-stage, pair={run.pair}, "
        f"box={run.box_source}, margins {run.margin}/{run.margin_b})\n{'=' * 60}\n"
    )

    print(f"\n--- Stage 1: frozen trunk ({tag}) ---")
    _compile(model, lr=config.INITIAL_LR)
    model.summary()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=_get_callbacks(tag, stage="frozen"),
        class_weight=class_weight,
        verbose=2,
    )
    _save_history(h1, config.RESULTS_DIR / f"{tag}_frozen_history.json")
    best_frozen = config.WEIGHTS_DIR / f"{tag}_frozen_best.weights.h5"
    if best_frozen.exists():
        model.load_weights(str(best_frozen))
        print(f"[fusion] Loaded best frozen weights from {best_frozen}")

    print(f"\n--- Stage 2: fine-tuning ({tag}) ---")
    unfreeze_for_finetuning(model, backbone_name=run.arch)
    _compile(model, lr=config.FINETUNE_LR)
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=_get_callbacks(tag, stage="finetune"),
        class_weight=class_weight,
        verbose=2,
    )
    _save_history(h2, config.RESULTS_DIR / f"{tag}_finetune_history.json")
    merged = _merge_histories(
        {k: [float(v) for v in vs] for k, vs in h1.history.items()},
        {k: [float(v) for v in vs] for k, vs in h2.history.items()},
    )
    full = config.RESULTS_DIR / f"{tag}_full_history.json"
    full.write_text(json.dumps(merged, indent=2))
    print(f"-> {full}")

    best_ft = config.WEIGHTS_DIR / f"{tag}_finetune_best.weights.h5"
    if best_ft.exists():
        model.load_weights(str(best_ft))
    model.save_weights(str(run.canonical_weights))
    print(f"-> {run.canonical_weights}")
    va = h2.history.get("val_auc", [])
    if va:
        i = int(np.argmax(va))
        print(f"[fusion] {tag}: best fine-tune val_auc {va[i]:.4f} at epoch {i + 1}/{len(va)}")
    return model


def _load(run: FusionRun) -> keras.Model:
    model = run.build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    # config.resolve_weights searches WEIGHTS_DIR, then each CUDA landing area;
    # weights trained off-laptop stay where they landed and are not copied over.
    try:
        wp = config.resolve_weights(run.tag)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\n  - train first") from None
    model.load_weights(str(wp))
    print(f"[fusion] Loaded {wp}")
    return model


def _score(
    model: keras.Model, run: FusionRun, split: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data_loader.reset_load_stats()
    ds, fa = run.dataset(split)
    y, p = collect_predictions(model, ds)
    if run.symmetric:
        # Order-symmetric model: score (CC, MLO) and (MLO, CC) and average.
        y2, p2 = collect_predictions(model, swap_pairs(ds))
        if not np.array_equal(y, y2):
            raise RuntimeError(f"{split}: swapped ordering changed the label order")
        p = (p + p2) / 2.0
    data_loader.assert_load_health()
    if len(y) != len(fa):
        raise RuntimeError(f"{split}: {len(y)} predictions for {len(fa)} rows - misaligned")
    return y, p, fa


def stage_val(run: FusionRun, model: keras.Model | None = None) -> float:
    """Validation prediction table, and the threshold fitted on those rows."""
    model = model or _load(run)
    y, p, fa = _score(model, run, "val")
    thr = optimal_threshold(y, p)
    auc = roc_auc_score(y, p)
    tbl = _prediction_table(fa, y, p, partition="val", threshold=thr)
    out = config.RESULTS_DIR / f"{run.tag}_predictions_val.csv"
    tbl.to_csv(out, index=False)
    print(
        f"[fusion] {run.tag} val: n={len(y)} AUC {auc:.4f} threshold {thr:.4f} -> "
        f"sens {_sensitivity(y, p, thr):.4f} spec {_specificity(y, p, thr):.4f}"
    )
    print(f"-> {out}")
    return float(thr)


def stage_eval(run: FusionRun) -> dict:
    """TEST PASS ONLY. Val table, test table, results JSON in the evaluate.py schema,
    with the decision threshold fitted on validation rather than test."""
    print(
        "=" * 60 + "\n  TEST PASS: scoring the frozen test partition for " f"{run.tag}\n" + "=" * 60
    )
    model = _load(run)
    thr = stage_val(run, model)
    y, p, fa = _score(model, run, "test")
    tbl = _prediction_table(fa, y, p, partition="test", threshold=thr)
    out = config.RESULTS_DIR / f"{run.tag}_predictions_test.csv"
    tbl.to_csv(out, index=False)
    print(f"-> {out}")
    results = compute_all_metrics(
        y,
        p,
        run.arch,
        threshold=thr,
        threshold_source="validation",
        groups=fa[config.GROUP_COL].to_numpy(),
    )
    results.update(
        tag=run.tag,
        pair=run.pair,
        box_source=run.box_source,
        margins=[run.margin, run.margin_b],
        n_pairs=int(len(y)),
    )
    rp = config.RESULTS_DIR / f"{run.tag}_results.json"
    rp.write_text(json.dumps(results, indent=2, default=float))
    print(f"-> {rp}")
    return results


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", default="vgg16")
    ap.add_argument("--pair", choices=["view", "context", "control"], required=True)
    ap.add_argument(
        "--margin",
        type=float,
        default=config.CROP_MARGIN_FRAC,
        help="Crop-A margin (m*); context crop B uses config.FUSION_CONTEXT_MARGIN",
    )
    ap.add_argument(
        "--box-source",
        choices=["oracle", "unet"],
        default="oracle",
        help="oracle = mask_margin crops from the full-resolution masks; unet = the "
        "localiser's boxes through the box provider",
    )
    ap.add_argument(
        "--boxes-dir",
        default=None,
        help="candidate boxes folder for --box-source unet (localiser_boxes_{split}.csv); "
        "default: the frozen artefacts.",
    )
    ap.add_argument(
        "--unet-k",
        type=int,
        default=1,
        help="with --box-source unet: must be 1..LOC_TOPK; the box served does not depend on it",
    )
    ap.add_argument(
        "--unet-mode",
        choices=["single", "matched"],
        default="single",
        help="single = the box table's box; matched = the top-k candidate greedily matched "
        "to the ground truth (localiser_boxes_k3_{split}.csv)",
    )
    ap.add_argument(
        "--stage",
        choices=["train", "val", "eval"],
        default="train",
        help="train (then val table) | val (table from weights) | eval (TEST PASS only)",
    )
    ap.add_argument(
        "--single-mass-only",
        action="store_true",
        help="View pairs from single-mass breasts only (drops id-matched pairs)",
    )
    ap.add_argument(
        "--rows",
        choices=["paired", "all"],
        default=None,
        help="row set: view-paired crop-A lesions or every lesion. Defaults are "
        "view=paired, context=all, control=paired; a non-default choice is "
        "appended to the tag (e.g. vgg16_fusion_control_all).",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Cap per stage (default MAX_EPOCHS)")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="replicate seed for initialisation, augmentation and shuffle order; "
        "the tag gets _s<N>, as in src.train. The patient partition stays on "
        "config.SEED.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the pairs, write the view pair tables if absent, print counts; "
        "decode nothing",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch per stage on --smoke-rows pairs, tag suffix _smoke; writes no pair CSV",
    )
    ap.add_argument("--smoke-rows", type=int, default=16)
    ap.add_argument("--cpu", action="store_true", help="Run under /CPU:0 (Metal off)")
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag "
        "that no longer exists fails in a second.",
    )
    return ap.parse_args(argv)


def _refuse_test_stage() -> None:
    """Refuses the eval stage unless the test pass has set its token
    (test_pass_guard.authorised)."""
    from .test_pass_guard import authorised

    if not authorised():
        raise SystemExit(
            "REFUSED: --stage eval scores the test partition; only "
            "notebooks/scripts/run_test_pass.sh may run it"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return
    if args.stage == "eval":
        _refuse_test_stage()
    config.set_seeds()
    # The replicate seed is applied after the pairs are built, so every replicate sees
    # the same partition and pairs; only initialisation, augmentation and shuffling differ.
    suffix = args.tag_suffix + ("_smoke" if args.smoke else "")
    if args.seed is not None and args.seed != config.SEED:
        suffix += f"_s{args.seed}"
    run = FusionRun(
        args.arch,
        args.pair,
        args.margin,
        args.box_source,
        single_mass_only=args.single_mass_only,
        rows=args.rows,
        batch_size=args.batch_size,
        tag_suffix=suffix,
        smoke_rows=args.smoke_rows if args.smoke else 0,
        boxes_dir=args.boxes_dir,
        test_boxes=(args.stage == "eval"),
        unet_k=args.unet_k,
        unet_mode=args.unet_mode,
    )
    print(
        f"[fusion] tag {run.tag} pair={args.pair} rows={run.rows} box={args.box_source} "
        f"margins {run.margin}/{run.margin_b} crop_sizing={run.crop_sizing}"
        + (" both orderings in training, averaged at eval" if run.symmetric else "")
    )
    if args.seed is not None and args.seed != config.SEED:
        config.set_seeds(args.seed)
        print(
            f"[fusion] Replicate seed {args.seed} applied to initialisation, augmentation "
            f"and shuffle (partition and pairs unchanged, seed {config.SEED})"
        )
    print_counts(run.counts)
    if not args.smoke and args.pair == "view":
        write_pairs(run.pairs, "view", args.single_mass_only)
    if args.dry_run:
        return

    def _go() -> None:
        if args.stage == "train":
            model = train(run, epochs=1 if args.smoke else args.epochs)
            stage_val(run, model)
        elif args.stage == "val":
            stage_val(run)
        else:
            stage_eval(run)

    if args.cpu:
        with tf.device("/CPU:0"):
            _go()
    else:
        _go()


if __name__ == "__main__":
    main()

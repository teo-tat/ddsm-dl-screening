"""Re-ranks the localiser's candidate components: one score per candidate, the best
taken per image, against the fixed rule that keeps the highest-peak component.
Judged by downstream mean per-lesion box IoU, not by candidate AUROC."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from . import config, train
from .models import _get_backbone, unfreeze_for_finetuning

tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)

STORE = config.OUTPUTS_DIR / "candidate_store_rung3_unet_bundle"
# `full` includes cand_rank and is_rule_pick, which encode the peak rule itself, so a
# model can learn to copy it; `norule` drops them, leaving image-derived evidence only.
FEATURE_SETS = {
    "full": (
        "peak_prob",
        "mean_prob",
        "sum_prob",
        "area_px",
        "dist_to_breast_edge_norm",
        "member_support_mean",
        "n_members_supporting",
        "cand_rank",
        "is_rule_pick",
    ),
    "norule": (
        "peak_prob",
        "mean_prob",
        "sum_prob",
        "area_px",
        "dist_to_breast_edge_norm",
        "member_support_mean",
        "n_members_supporting",
    ),
}
FEATURES = FEATURE_SETS["norule"]
TAU_GRID = np.round(np.arange(0.0, 0.96, 0.05), 2)


def _bundle():
    """localiser_bundle as a module: its canvas geometry is the frame boxes are scored in."""
    p = Path(__file__).resolve().parent.parent / "notebooks" / "scripts" / "localiser_bundle.py"
    spec = importlib.util.spec_from_file_location("lb", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_split(split: str, limit: int | None = None, store: Path | None = None):
    """Candidate crops and their meta, in candidate-table order."""
    d = Path(store) if store else STORE
    x = np.ascontiguousarray(np.load(d / f"{split}_images.npy", mmap_mode="r"), dtype=np.float32)
    meta = pd.read_csv(d / f"{split}_meta.csv")
    if len(x) != len(meta):
        raise SystemExit(f"{split}: store misaligned, {len(x)} crops vs {len(meta)} rows")
    if limit is not None:
        x, meta = x[:limit], meta.iloc[:limit].reset_index(drop=True)
    return x, meta


def _feature_matrix(meta: pd.DataFrame, features=FEATURES) -> np.ndarray:
    f = meta[list(features)].astype(float)
    # Distance to the breast edge is NaN where no breast box was found; the median
    # fills it, taken from the split being transformed rather than across splits.
    return f.fillna(f.median()).to_numpy(np.float32)


def build_cnn(n_features: int, arch: str = "vgg16") -> keras.Model:
    """The backbone (VGG16 by default) on the crop, tabular features concatenated before
    the head."""
    shape = (config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)
    backbone = _get_backbone(arch, shape)
    backbone.trainable = False
    img_in = keras.Input(shape=shape, name="crop")
    feat_in = keras.Input(shape=(n_features,), name="features")
    x = backbone(img_in, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # The tabular features enter beside the pooled crop, so the head can weigh what the
    # map said about the component against the pixels instead of rediscovering it.
    x = layers.Concatenate()([x, layers.BatchNormalization()(feat_in)])
    x = layers.Dense(
        config._HEAD_UNITS if hasattr(config, "_HEAD_UNITS") else 256,
        activation="relu",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", name="candidate")(x)
    return keras.Model([img_in, feat_in], out, name=f"rerank_{arch}")


def _dataset(
    x: np.ndarray, f: np.ndarray, y: np.ndarray, *, batch: int, shuffle: bool
) -> tf.data.Dataset:
    x3 = np.repeat(x, 3, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(((x3, f), y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=config.SEED, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


class CandidateAUROC(keras.callbacks.Callback):
    """Validation candidate AUROC through the eager path, for early stopping: the
    compiled graph on tensorflow-metal does not apply ReLU."""

    def __init__(self, ds, y):
        super().__init__()
        self.ds, self.y = ds, y

    def on_epoch_end(self, epoch, logs=None):
        p = np.concatenate(
            [np.asarray(self.model(xb, training=False)).ravel() for xb, _ in self.ds]
        )
        logs = logs if logs is not None else {}
        logs["val_cand_auc"] = (
            float(roc_auc_score(self.y, p)) if len(np.unique(self.y)) > 1 else float("nan")
        )


def select_boxes(meta: pd.DataFrame, scores: np.ndarray, tau: float) -> pd.DataFrame:
    """Per image: the highest-scoring candidate, or the rule's pick when that score is
    below tau, so an unconfident re-ranker cannot replace the rule's pick.
    """
    m = meta.copy()
    m["score"] = scores
    best = m.loc[m.groupby("image_id")["score"].idxmax()].set_index("image_id")
    rule = m[m["cand_rank"] == 1].set_index("image_id")
    # Take the re-ranker's pick where it clears tau or where the rule has no
    # candidate to fall back to, and the rule's pick everywhere else.
    take_rr = (best["score"] >= tau) | (~best.index.isin(rule.index))
    fb = best.index[~take_rr]
    out = pd.concat([best[take_rr], rule.reindex(fb)])
    out["selection_source"] = np.where(
        out.index.isin(best.index[take_rr]), "reranker", "rule_fallback"
    )
    return out.loc[best.index]


def _iou(a, b) -> float:
    ih = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    iw = max(0, min(a[3], b[3]) - max(a[2], b[2]) + 1)
    inter = ih * iw
    union = (a[1] - a[0] + 1) * (a[3] - a[2] + 1) + (b[1] - b[0] + 1) * (b[3] - b[2] + 1) - inter
    return inter / union if union else 0.0


_SIDE_CACHE: dict = {}


def side_tables(split: str, lb):
    """`lb._load_split_side_tables`, computed once per split: it reloads the mask array
    and rebuilds every ground-truth box, which the tau sweep would otherwise repeat."""
    if split not in _SIDE_CACHE:
        _SIDE_CACHE[split] = lb._load_split_side_tables(split)
    return _SIDE_CACHE[split]


def lesion_table(split: str, chosen: pd.DataFrame, lb) -> pd.DataFrame:
    """Per-lesion box table, one prediction per image: box_iou (the column compared) and gt_*
    in the canvas frame, pred_* in the mammogram's; box_iou_mammogram 0 with no box, else empty."""
    meta, gts, _ = side_tables(split, lb)
    rows = []
    for i, r in enumerate(meta.itertuples()):
        img = str(r.image_id)
        gt = gts[i]
        rec = {
            "split": split,
            "lesion_id": r.lesion_id,
            "image_id": img,
            "patient_id": r.patient_id,
            "label": int(r.label),
            "orig_h": int(r.orig_h),
            "orig_w": int(r.orig_w),
        }
        if img not in chosen.index:
            rec.update(
                detected=False,
                status="miss" if gt is not None else "no_gt",
                box_iou=0.0,
                box_iou_mammogram=0.0,
                selection_source="no_candidate",
            )
            rows.append(rec)
            continue
        c = chosen.loc[img]
        cb = (int(c["canvas_y0"]), int(c["canvas_y1"]), int(c["canvas_x0"]), int(c["canvas_x1"]))
        mb = (int(c["y0"]), int(c["y1"]), int(c["x0"]), int(c["x1"]))
        rec.update(
            detected=True,
            pred_y0=mb[0],
            pred_y1=mb[1],
            pred_x0=mb[2],
            pred_x1=mb[3],
            rerank_score=float(c["score"]),
            selection_source=c["selection_source"],
        )
        if gt is None:
            rec.update(status="no_gt", box_iou=np.nan, box_iou_mammogram=np.nan)
        else:
            rec.update(
                status="ok",
                box_iou=_iou(gt, cb),
                gt_y0=gt[0],
                gt_y1=gt[1],
                gt_x0=gt[2],
                gt_x1=gt[3],
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def mean_box_iou(tbl: pd.DataFrame) -> float:
    """Miss-inclusive mean per-lesion box IoU — the downstream metric."""
    s = tbl[tbl["status"] != "no_gt"]["box_iou"].fillna(0.0)
    return float(s.mean()) if len(s) else float("nan")


def select_tau(meta: pd.DataFrame, scores: np.ndarray, split: str, lb) -> tuple[float, list]:
    """tau maximising train mean box IoU under the selection rule."""
    trace = []
    for tau in TAU_GRID:
        v = mean_box_iou(lesion_table(split, select_boxes(meta, scores, float(tau)), lb))
        trace.append({"tau": float(tau), "mean_box_iou": v})
        print(f"    tau {tau:.2f} train mean box IoU {v:.4f}", flush=True)
    best = max(trace, key=lambda r: r["mean_box_iou"])
    return best["tau"], trace


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=["features", "cnn"], default="features")
    ap.add_argument(
        "--store",
        default=None,
        help="candidate store; default outputs/candidate_store_rung3_unet_bundle",
    )
    ap.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS),
        default="norule",
        help="norule (default) drops cand_rank and is_rule_pick, which encode "
        "the peak rule; full keeps them",
    )
    ap.add_argument("--arch", default="vgg16")
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N candidates per split")
    ap.add_argument(
        "--write-boxes",
        default=None,
        help="folder for localiser_boxes_{split}.csv (a NEW folder, never artifacts/)",
    )
    ap.add_argument("--out-dir", default=str(config.RESULTS_DIR))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    args = ap.parse_args(argv)
    if args.parse_only:
        print("[parse-only] arguments valid; no data loaded")
        return

    tag = args.tag or f"rerank_{args.model}"
    lb = _bundle()
    t0 = time.time()
    dev = (
        tf.device("/CPU:0")
        if args.cpu
        else tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")
    )
    with dev:
        xtr, mtr = load_split("train", args.limit, args.store)
        xva, mva = load_split("val", args.limit, args.store)
        ytr, yva = mtr["label"].to_numpy(int), mva["label"].to_numpy(int)
        feats = FEATURE_SETS[args.feature_set]
        ftr_raw, fva_raw = _feature_matrix(mtr, feats), _feature_matrix(mva, feats)
        # Scaler fitted on train only, then applied to validation.
        scaler = StandardScaler().fit(ftr_raw)
        ftr, fva = scaler.transform(ftr_raw).astype(np.float32), scaler.transform(fva_raw).astype(
            np.float32
        )
        print(
            f"[rerank] train {len(mtr)} candidates ({ytr.sum()} positive), "
            f"val {len(mva)} ({yva.sum()} positive)"
        )

        history = None
        if args.model == "features":
            clf = LogisticRegression(max_iter=2000, random_state=config.SEED).fit(ftr, ytr)
            str_, sva = clf.predict_proba(ftr)[:, 1], clf.predict_proba(fva)[:, 1]
        else:
            model = build_cnn(ftr.shape[1], args.arch)
            tr = _dataset(xtr, ftr, ytr, batch=args.batch_size, shuffle=True)
            va = _dataset(xva, fva, yva, batch=args.batch_size, shuffle=False)
            cb = [
                CandidateAUROC(va, yva),
                keras.callbacks.EarlyStopping(
                    monitor="val_cand_auc",
                    mode="max",
                    patience=args.patience,
                    restore_best_weights=True,
                ),
            ]
            train._compile(model, lr=config.INITIAL_LR)
            print(f"\n--- Stage 1: frozen backbone ({tag}) ---")
            h1 = model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=cb, verbose=2)
            print(f"\n--- Stage 2: fine-tuning ({tag}) ---")
            # Pass the whole model: unfreeze_for_finetuning locates the backbone
            # sub-model itself and would not find one inside a bare backbone.
            unfreeze_for_finetuning(model, args.arch)
            train._compile(model, lr=config.FINETUNE_LR)
            h2 = model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=cb, verbose=2)
            history = {
                "frozen": {k: [float(x) for x in v] for k, v in h1.history.items()},
                "finetune": {k: [float(x) for x in v] for k, v in h2.history.items()},
            }
            # Save before scoring: tau selection reads a per-lesion store that need not
            # exist on the machine doing the training.
            wpath = Path(args.out_dir) / f"{tag}_best.weights.h5"
            wpath.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(wpath))
            print(f"-> {wpath}")
            score = lambda d: np.concatenate(
                [np.asarray(model(xb, training=False)).ravel() for xb, _ in d]
            )
            str_, sva = score(_dataset(xtr, ftr, ytr, batch=args.batch_size, shuffle=False)), score(
                va
            )

    auc_tr = float(roc_auc_score(ytr, str_)) if len(np.unique(ytr)) > 1 else float("nan")
    auc_va = float(roc_auc_score(yva, sva)) if len(np.unique(yva)) > 1 else float("nan")
    print(f"\n[rerank] candidate AUROC (DIAGNOSTIC ONLY): train {auc_tr:.4f} val {auc_va:.4f}")

    print("\n[rerank] selecting tau on TRAIN by downstream mean box IoU:")
    tau, trace = select_tau(mtr, str_, "train", lb)
    print(f"[rerank] tau = {tau:.2f}")

    result = {
        "tag": tag,
        "model": args.model,
        "feature_set": args.feature_set,
        "features": list(feats),
        "n_train": int(len(mtr)),
        "n_val": int(len(mva)),
        "candidate_auroc_train": round(auc_tr, 6),
        "candidate_auroc_val": round(auc_va, 6),
        "candidate_auroc_note": "diagnostic only; promotion is decided downstream",
        "tau": float(tau),
        "tau_selected_on": "train, by mean box IoU",
        "tau_trace": trace,
        "ceiling_val_images": 0.093,
        "ceiling_train_images": 0.117,
        "wall_time_s": round(time.time() - t0, 1),
    }
    if history:
        result["history"] = history

    for split, meta_s, sc in (("train", mtr, str_), ("val", mva, sva)):
        tbl = lesion_table(split, select_boxes(meta_s, sc, tau), lb)
        result[f"mean_box_iou_{split}"] = round(mean_box_iou(tbl), 6)
        n_rr = int((tbl["selection_source"] == "reranker").sum())
        result[f"n_reranker_picks_{split}"] = n_rr
        result[f"n_rule_fallback_{split}"] = int((tbl["selection_source"] == "rule_fallback").sum())
        print(
            f"[rerank] {split}: mean box IoU {result[f'mean_box_iou_{split}']:.4f} "
            f"({n_rr} re-ranker picks, {result[f'n_rule_fallback_{split}']} rule fallbacks)"
        )
        if args.write_boxes:
            out = Path(args.write_boxes)
            out.mkdir(parents=True, exist_ok=True)
            tbl.to_csv(out / f"localiser_boxes_{split}.csv", index=False)
            print(f"-> {out / f'localiser_boxes_{split}.csv'}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rp = Path(args.out_dir) / f"{tag}_result.json"
    rp.write_text(json.dumps(result, indent=2))
    print(f"-> {rp}")


if __name__ == "__main__":
    main()

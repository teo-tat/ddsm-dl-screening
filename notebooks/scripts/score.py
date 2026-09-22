#!/usr/bin/env python
"""score.py — exact validation AUC of a trained checkpoint, through the eager path.

The exact AUC is the reported number and the Keras history a monitoring signal
only. Validation only: no store holds test rows.

  store score a single-stream checkpoint against a crop store
  fusion score a paired checkpoint against its pair store, swap-averaged
  ablations exact validation AUC of the regularisation ablations
  rescore-cuda re-score every run trained off this machine, on the laptop CPU
  seed-spread one tag's base run and its replicates: exact AUC each, and the spread
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

CUDA = ROOT / "outputs" / "_cuda"
RESULTS = ROOT / "outputs" / "results"


def _discovered(what: str, n: int, where, *, required: bool = False) -> int:
    """Print the count and the search location, then refuse an empty required
    discovery: a tool that found nothing has checked nothing."""
    places = where if isinstance(where, (list, tuple)) else [where]
    loc = ", ".join(str(p) for p in places)
    print(f"[discovered] {n} {what} under {loc}")
    if required and n == 0:
        raise SystemExit(
            f"REFUSED: no {what} found under {loc}. An empty discovery is not a "
            f"pass — this run checked nothing."
        )
    return n


class _NullCtx:
    """Do-nothing context manager, so the device branch stays one code path."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _tf():
    """TensorFlow, imported on use, with the seeds fixed: `config` is framework-free,
    and a subcommand that only reads tables off disk needs no framework."""
    import tensorflow as tf

    config.set_seeds()
    return tf


def _device(cpu: bool):
    """(context manager, device string) for --cpu. `tf.config.set_visible_devices`
    cannot be called after import, so a scope chooses the device."""
    tf = _tf()
    dev = "/CPU:0" if cpu else None
    return (tf.device(dev) if dev else _NullCtx()), dev


def _environment(tf, device: str | None) -> dict:
    """The provenance block every score JSON carries."""
    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    env = {
        "device": device or ("GPU" if gpus else "CPU"),
        "gpus": gpus,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        from importlib.metadata import version

        env["tensorflow-metal"] = version("tensorflow-metal")
    except Exception:
        env["tensorflow-metal"] = "not installed"
    return env


def write_predictions(
    out_path: str | Path, meta: pd.DataFrame, y: np.ndarray, p: np.ndarray, auc: float
) -> None:
    """Write the val prediction table in `evaluate._prediction_table`'s schema, then
    re-read it and assert its AUC reproduces the scalar this run reported."""
    from src import data_loader, evaluate

    # The store's row key decides which frame the meta columns join to: a lesion or
    # a pair store joins to the crop frame, a whole-image store to the full frame.
    if "lesion_id_a" in meta.columns:
        meta_key, frame_key, source = "lesion_id_a", "lesion_id", "crop"
    elif "lesion_id" in meta.columns:
        meta_key, frame_key, source = "lesion_id", "lesion_id", "crop"
    elif "image file path" in meta.columns:
        meta_key, frame_key, source = "image file path", "image file path", "full"
    else:
        raise SystemExit(f"store meta has no row key: columns {list(meta.columns)}")

    frame = data_loader.split_frames(source=source)["val"]
    frame = frame.set_index(frame_key).loc[meta[meta_key].to_numpy()].reset_index()
    # Fitted on validation, exactly as evaluate does it; the table records the
    # operating point beside the probabilities so downstream analyses need no model.
    threshold = evaluate.optimal_threshold(y, p, target_sensitivity=config.TARGET_SENSITIVITY)
    tbl = evaluate._prediction_table(frame, y, p, partition="val", threshold=threshold)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_path, index=False)
    print(f"-> {out_path}")

    back = pd.read_csv(out_path)
    re_auc = float(roc_auc_score(back["y_true"].to_numpy(), back["y_prob"].to_numpy()))
    if abs(re_auc - auc) > 1e-6:
        raise SystemExit(
            f"table AUC {re_auc:.8f} != scored AUC {auc:.8f} "
            f"(delta {abs(re_auc - auc):.2e}); the table does not "
            f"reproduce the run, so it was not kept"
        )
    print(
        f"   table AUC {re_auc:.6f} reproduces the scored value "
        f"(delta {abs(re_auc - auc):.1e}); threshold {threshold:.4f}"
    )


def _finish(
    result: dict, args, meta: pd.DataFrame, y: np.ndarray, p: np.ndarray, auc: float
) -> None:
    """Print the score JSON, write the optional products, apply --expect."""
    print(json.dumps(result, indent=2))
    if args.out_predictions:
        write_predictions(args.out_predictions, meta, y, p, auc)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"-> {args.out}")
    if args.expect is not None:
        delta = abs(auc - args.expect)
        ok = delta <= args.tolerance
        print(
            f"\nexpected {args.expect:.6f} measured {auc:.6f} |delta| {delta:.6f} "
            f"tolerance {args.tolerance} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            sys.exit(1)


# Store: the single-stream scorer and the hardware-parity check
def load_val(store: Path, normalisation: str, intensity_norm: str, batch_size: int):
    """The store's validation split, standardised exactly as training does."""
    tf = _tf()
    from src import data_loader

    x = np.ascontiguousarray(np.load(store / "val_images.npy", mmap_mode="r"), dtype=np.float32)
    meta = pd.read_csv(store / "val_meta.csv")
    if len(x) != len(meta):
        raise SystemExit(f"store is misaligned: {len(x)} images, {len(meta)} meta rows")
    if normalisation == "imagenet":  # what _finalise does for 3-channel models
        x = np.repeat(x, 3, axis=-1)
    y = meta["label"].to_numpy(np.int64)
    std_fn = data_loader._make_batch_standardize_fn(
        normalisation, intensity_norm=intensity_norm, source="crop"
    )
    ds = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .batch(batch_size)
        .map(std_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, y, meta


def score_eager(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """Eager-path predictions over *ds*, batch by batch."""
    ys, ps = [], []
    for images, labels in ds:
        ps.append(np.asarray(model(images, training=False)).ravel())
        ys.append(np.asarray(labels).ravel())
    return np.concatenate(ys).astype(int), np.concatenate(ps).astype(float)


def cmd_store(args) -> None:
    """Score a single-stream checkpoint on its store's validation split; with --expect,
    the hardware-parity check that a known validation AUC reproduces on this machine."""
    tf = _tf()
    from src.models import MODEL_REGISTRY, build_model, get_normalisation_mode

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {args.model!r}; registry: {sorted(MODEL_REGISTRY)}")
    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    norm = get_normalisation_mode(args.model)
    ts = tuple(manifest["target_size"])
    channels = MODEL_REGISTRY[args.model]["channels"]

    ctx, device = _device(args.cpu)
    with ctx:
        ds, _, meta = load_val(store, norm, manifest["intensity_norm"], args.batch_size)
        model = build_model(args.model, input_shape=(ts[0], ts[1], channels))
        model.load_weights(str(args.weights))
        y, p = score_eager(model, ds)

    auc = float(roc_auc_score(y, p))
    result = {
        "store": str(store),
        "store_tag": manifest.get("store_tag"),
        "model": args.model,
        "weights": str(args.weights),
        "n": int(len(y)),
        "n_malignant": int(y.sum()),
        "exact_val_auc": round(auc, 6),
        "path": "eager model(x, training=False)",
        **_environment(tf, device),
    }
    _finish(result, args, meta, y, p, auc)


# Fusion: the paired scorer
def cmd_fusion(args) -> None:
    """Reuses `train_fusion_store`'s own load, dataset and score functions, so the
    input pipeline is identical to the one the model was trained with."""
    tf = _tf()
    from src.models import build_fusion, build_model
    from src.train_fusion_store import load_store, make_dataset, score

    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    pair = manifest["pair"]
    paired = bool(manifest["paired"])
    # Only the view mode is order-symmetric, so only it is swap-averaged. The
    # controls take crop A alone (or a fixed pair) and must not be.
    symmetric = pair == "view"

    ctx, device = _device(args.cpu)
    with ctx:
        xa, xb, y_true, meta = load_store(store, "val")
        ds = make_dataset(
            xa,
            xb,
            y_true,
            augment=False,
            shuffle=False,
            batch_size=args.batch_size,
            both_orderings=False,
        )
        shape = (config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3)
        model = (
            build_fusion(args.model, input_shape=shape)
            if paired
            else build_model(args.model, input_shape=shape)
        )
        model.load_weights(str(args.weights))
        y, p = score(model, ds, symmetric)

    auc = float(roc_auc_score(y, p))
    result = {
        "store": str(store),
        "fusion_tag": manifest.get("fusion_tag"),
        "pair": pair,
        "paired": paired,
        "symmetric": symmetric,
        "model": args.model,
        "weights": str(args.weights),
        "n": int(len(y)),
        "n_malignant": int(y.sum()),
        "exact_val_auc": round(auc, 6),
        "path": "eager model(x, training=False)" + (", swap-averaged" if symmetric else ""),
        **_environment(tf, device),
    }
    _finish(result, args, meta, y, p, auc)


# Ablations: exact validation AUC for the regularisation family
M020 = ROOT / "outputs" / "crop_store_m020"

OVERLAY = ROOT / "outputs" / "crop_store_m020_overlay"

WEIGHTS = ROOT / "outputs" / "weights"

# (task, tag, model, store, ablate). The last two rows are the LR-matched VGG16
# overlay pair, re-scored here so both arms are measured the same way.
ABLATION_RUNS: list[tuple[str, str, str, Path, str | None]] = [
    ("A4.0", "regularised_crop_m020_kg", "regularised", M020, None),
    ("A4.1", "regularised_crop_m020_no_dropout_kg", "regularised", M020, "dropout"),
    ("A4.2", "regularised_crop_m020_no_l2_kg", "regularised", M020, "l2"),
    ("A4.3", "regularised_crop_m020_no_se_kg", "regularised", M020, "se"),
    ("A4.4", "regularised_crop_m020_no_aug_kg", "regularised", M020, "aug"),
    ("A4.5", "regularised_crop_m020_lr2e-3_kg", "regularised", M020, None),
    ("A4.6", "regularised_crop_m020_overlay_lrmatched_kg", "regularised", OVERLAY, None),
    ("overlay", "vgg16_crop_m020_overlay", "vgg16", OVERLAY, None),
    ("overlay-control", "vgg16_crop_m020", "vgg16", M020, None),
]


def _rel(p: Path) -> str:
    """Path relative to the repo when it is inside it, absolute otherwise (temp runs)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _score_ablation(
    task: str,
    tag: str,
    model_name: str,
    store: Path,
    ablate: str | None,
    batch_size: int,
    out_dir: Path,
) -> dict:
    """Score one checkpoint on its store's validation split; write its table."""
    from src import train
    from src.models import MODEL_REGISTRY, build_model, get_normalisation_mode

    manifest = json.loads((store / "manifest.json").read_text())
    norm = get_normalisation_mode(model_name)
    ts = tuple(manifest["target_size"])
    ds, _, meta = load_val(store, norm, manifest["intensity_norm"], batch_size)
    # `aug` is a data setting, so its architecture is the unablated one; the other
    # three change the graph and must be rebuilt exactly as the run built it.
    m = build_model(
        model_name,
        input_shape=(ts[0], ts[1], MODEL_REGISTRY[model_name]["channels"]),
        **train._regularised_kwargs(model_name, "crop", ablate if ablate != "aug" else None),
    )
    weights = WEIGHTS / f"{tag}_best.weights.h5"
    m.load_weights(str(weights))
    y, p = score_eager(m, ds)
    auc = float(roc_auc_score(y, p))
    table = out_dir / f"{tag}_predictions_val.csv"
    write_predictions(table, meta, y, p, auc)
    return {
        "task": task,
        "tag": tag,
        "model": model_name,
        "ablate": ablate,
        "store": store.name,
        "weights": weights.name,
        "table": _rel(table),
        "n": int(len(y)),
        "n_malignant": int(y.sum()),
        "exact_val_auc": round(auc, 6),
        "path": "eager model(x, training=False)",
    }


def cmd_ablations(args) -> None:
    """Exact validation AUC and val table of every ablation run, scored from its weights
    and store; the Keras history is not read."""
    tf = _tf()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _discovered("ablation runs to score", len(ABLATION_RUNS), WEIGHTS, required=True)

    rows: list[dict] = []
    ctx, device = _device(args.cpu)
    with ctx:
        for task, tag, model_name, store, ablate in ABLATION_RUNS:
            print(f"\n--- {task} {tag}")
            try:
                rows.append(
                    _score_ablation(task, tag, model_name, store, ablate, args.batch_size, out_dir)
                )
            except Exception as exc:  # one attempt, then move on
                print(f"[ablations] {tag}: FAILED — {type(exc).__name__}: {exc}")
                rows.append(
                    {
                        "task": task,
                        "tag": tag,
                        "model": model_name,
                        "ablate": ablate,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    summary = out_dir / "exact_val_auc.json"
    summary.write_text(
        json.dumps(
            {
                "what": "exact validation AUC of the regularisation ablations and the "
                "LR-matched VGG16 overlay pair, scored from the surviving weights",
                "split": "val",
                "device": "CPU" if args.cpu else "default",
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "runs": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n-> {_rel(summary)}")
    ok = sum("error" not in r for r in rows)
    print(f"{ok} scored, {len(rows) - ok} failed")
    if ok < len(rows):
        raise SystemExit(
            f"REFUSED: {len(rows) - ok} of {len(rows)} ablation runs failed to score; "
            f"their error rows are in the summary {_rel(summary)}"
        )


# Rescore-cuda: every off-machine run, re-scored here
def _laptop_store(pod_store: str) -> Path:
    """/workspace/stores/crop_store_m020 -> outputs/crop_store_m020."""
    return ROOT / "outputs" / Path(pod_store).name


def _laptop_weights(pod_weights: str) -> Path | None:
    """/workspace/runs/<dir>/<file> -> outputs/_cuda/<dir>/<file>."""
    p = Path(pod_weights)
    if len(p.parts) < 2:
        return None
    cand = CUDA / p.parts[-2] / p.name
    return cand if cand.exists() else None


def _discover() -> list[dict]:
    """Every CUDA run with a canonical checkpoint, with its session-reported value."""
    jobs: list[dict] = []

    # Crop-store runs: the session wrote one score_<tag>.json per run.
    for f in sorted(CUDA.glob("score_*.json")):
        d = json.loads(f.read_text())
        jobs.append(
            {
                "kind": "crop",
                "tag": Path(d["weights"]).name.replace("_best.weights.h5", ""),
                "store": _laptop_store(d["store"]),
                "model": d["model"],
                "weights": _laptop_weights(d["weights"]),
                "pod_auc": d["exact_val_auc"],
            }
        )

    # Fusion runs: train_fusion_store reports its own exact AUC in the manifest.
    for man in sorted(CUDA.glob("*/*_manifest.json")):
        d = json.loads(man.read_text())
        if "pair" not in d:  # crop-store manifest, already covered
            continue
        w = man.parent / f"{d['tag']}_best.weights.h5"
        jobs.append(
            {
                "kind": "fusion",
                "tag": d["tag"],
                "store": _laptop_store(d["store"]),
                "model": d["arch"],
                "weights": w if w.exists() else None,
                "pod_auc": d["exact_val_auc"],
            }
        )
    return jobs


def _score_job(job: dict, tolerance: float, reuse: bool, predictions_dir: str | None) -> dict:
    """Run the right subcommand as a subprocess; return the comparison row. One
    subprocess per run keeps each model's weights out of the next run's memory."""
    out = RESULTS / f"rescore_{job['tag']}.json"
    # Scoring is deterministic — fixed weights, fixed row order, no dropout, the
    # eager path — so a cached JSON holds what re-scoring would produce again.
    if reuse and out.exists():
        return _row(job, json.loads(out.read_text()), tolerance, reused=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "store" if job["kind"] == "crop" else "fusion",
        "--store",
        str(job["store"]),
        "--model",
        job["model"],
        "--weights",
        str(job["weights"]),
        "--cpu",
        "--out",
        str(out),
    ]
    if predictions_dir:
        cmd += [
            "--out-predictions",
            str(Path(predictions_dir) / f"{job['tag']}_predictions_val.csv"),
        ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(ROOT),
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", ""),
        },
    )
    if proc.returncode != 0 or not out.exists():
        return {
            "tag": job["tag"],
            "kind": job["kind"],
            "pod_auc": job["pod_auc"],
            "laptop_auc": None,
            "delta": None,
            "n_pairs": None,
            "pair_flips": None,
            "agrees": False,
            "tolerance": tolerance,
            "table_auc": None,
            "table_matches": None,
            "source": "failed",
            "note": (proc.stderr.strip().splitlines() or ["failed"])[-1][:200],
        }
    return _row(job, json.loads(out.read_text()), tolerance)


def _row(job: dict, d: dict, tolerance: float, reused: bool = False) -> dict:
    """One comparison row from a scorer's JSON."""
    laptop = d["exact_val_auc"]
    delta = abs(laptop - job["pod_auc"])
    # One flipped pair = 1 / (n_pos * n_neg). Expressing the delta this way is what
    # makes it readable: an integer means ties swapped, not a different computation.
    n_pos, n = int(d["n_malignant"]), int(d["n"])
    unit = 1.0 / (n_pos * (n - n_pos)) if n_pos and n - n_pos else float("nan")
    # Every table on disk must reproduce the scalar its own run reported; re-checking
    # here covers cached rows and tables an earlier invocation wrote.
    tbl_path = RESULTS / f"{job['tag']}_predictions_val.csv"
    table_auc, table_ok = None, None
    if tbl_path.exists():
        t = pd.read_csv(tbl_path)
        table_auc = float(roc_auc_score(t["y_true"].to_numpy(), t["y_prob"].to_numpy()))
        table_ok = bool(abs(table_auc - laptop) <= 1e-6)
    return {
        "tag": job["tag"],
        "kind": job["kind"],
        "pod_auc": job["pod_auc"],
        "laptop_auc": laptop,
        "delta": round(delta, 8),
        "n_pairs": round(1 / unit),
        "pair_flips": round(delta / unit, 2),
        "agrees": bool(delta <= tolerance),
        "tolerance": tolerance,
        "table_auc": None if table_auc is None else round(table_auc, 6),
        "table_matches": table_ok,
        "source": "cached" if reused else "scored",
        "note": "",
    }


def cmd_rescore_cuda(args) -> None:
    """Re-score every off-machine run here, so every reported number comes from one
    machine, one environment and the eager path."""
    jobs = _discover()
    _discovered("runs with a score JSON or a fusion manifest", len(jobs), CUDA, required=True)
    if args.tag:
        jobs = [j for j in jobs if j["tag"] in set(args.tag)]
    missing = [j["tag"] for j in jobs if j["weights"] is None]
    jobs = [j for j in jobs if j["weights"] is not None]
    print(
        f"[rescore] {len(jobs)} runs to score"
        + (f"; {len(missing)} without a local checkpoint: {missing}" if missing else "")
    )

    rows = []
    for i, job in enumerate(jobs, 1):
        print(f"[rescore] {i}/{len(jobs)} {job['tag']} ({job['kind']})", flush=True)
        row = _score_job(job, args.tolerance, args.reuse, args.out_predictions)
        rows.append(row)
        state = "ok" if row["agrees"] else "DISAGREES"
        flips = row.get("pair_flips")
        print(
            f"    pod {row['pod_auc']} | laptop {row['laptop_auc']} | "
            f"delta {row['delta']}"
            + (f" ({flips} pair flips)" if flips is not None else "")
            + f" -> {state}{(' :: ' + row['note']) if row['note'] else ''}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["kind", "tag"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    bad = df[~df["agrees"]]
    print(
        f"[rescore] {len(df) - len(bad)}/{len(df)} agree within {args.tolerance}; "
        f"max delta {df['delta'].max()}"
    )
    if len(bad):
        print("[rescore] DISAGREEMENTS:")
        print(bad.to_string(index=False))
    have = df[df["table_auc"].notna()]
    bad_tbl = have[~have["table_matches"].astype(bool)]
    print(
        f"[rescore] prediction tables: {len(have)}/{len(df)} present, "
        f"{len(have) - len(bad_tbl)} reproduce their run's AUC to 1e-6"
    )
    if len(bad_tbl):
        print("[rescore] TABLES THAT DO NOT REPRODUCE THEIR RUN:")
        print(bad_tbl[["tag", "laptop_auc", "table_auc"]].to_string(index=False))
    sys.exit(1 if (len(bad) or len(bad_tbl) or missing) else 0)


# Seed-spread: one tag's replicates and their spread
def _exact(tag: str):
    """Exact validation AUC from the prediction table, or None if absent."""
    p = config.RESULTS_DIR / f"{tag}_predictions_val.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return float(roc_auc_score(d["y_true"].to_numpy(int), d["y_prob"].to_numpy(float)))


def _history(tag: str):
    from src.models import MODEL_REGISTRY

    arch = tag.split("_")[0]
    scratch = MODEL_REGISTRY.get(arch, {}).get("normalisation") == "scratch"
    p = config.RESULTS_DIR / (f"{tag}_history.json" if scratch else f"{tag}_finetune_history.json")
    return json.loads(p.read_text()) if p.exists() else None


def cmd_seed_spread(args) -> None:
    """Exact AUC from `{tag}_predictions_val.csv`, never the Keras history; the epochs
    and the history AUC come from the history, so a divergence stays visible."""
    rows = []
    ts = args.tag_suffix
    for seed, tag in [(config.SEED, f"{args.tag}{ts}")] + [
        (s, f"{args.tag}_s{s}{ts}") for s in args.seed
    ]:
        h = _history(tag)
        if h is None:
            rows.append(dict(tag=tag, seed=seed, status="missing"))
            continue
        va, ta = np.asarray(h["val_auc"]), np.asarray(h.get("auc", []))
        b = int(va.argmax())
        ex = _exact(tag)
        if ex is None:
            rows.append(dict(tag=tag, seed=seed, status="no prediction table"))
            continue
        rows.append(
            dict(
                tag=tag,
                seed=seed,
                status="ok",
                exact_val_auc=round(ex, 4),
                history_val_auc=round(float(va[b]), 4),
                history_minus_exact=round(float(va[b]) - ex, 4),
                epoch_best=b + 1,
                epochs=int(va.size),
                train_auc_at_best=(round(float(ta[b]), 4) if ta.size > b else np.nan),
            )
        )
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    out = Path(args.out) if args.out else config.RESULTS_DIR / f"seed_spread_{args.tag}{ts}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"-> {out}")
    print("\n| tag | seed | exact val AUC | history | history − exact | epoch / epochs |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in df.itertuples():
        if r.status != "ok":
            print(f"| {r.tag} | {r.seed} | missing | | | |")
            continue
        flag = " **>0.005**" if abs(r.history_minus_exact) > 0.005 else ""
        print(
            f"| {r.tag} | {r.seed} | {r.exact_val_auc:.4f} | {r.history_val_auc:.4f} | "
            f"{r.history_minus_exact:+.4f}{flag} | {int(r.epoch_best)} / {int(r.epochs)} |"
        )
    big = ok[ok["history_minus_exact"].abs() > 0.005] if len(ok) else ok
    if len(big):
        print(
            f"\n{len(big)} tag(s) where the history differs from the exact AUC by more than "
            f"0.005 — the history is a monitoring signal only:"
        )
        for r in big.itertuples():
            print(
                f"  {r.tag}: history {r.history_val_auc:.4f} vs exact {r.exact_val_auc:.4f} "
                f"({r.history_minus_exact:+.4f})"
            )
    if len(ok) >= 2:
        v = ok["exact_val_auc"].to_numpy()
        print(
            f"\nspread over {len(ok)} seeds: mean {v.mean():.4f} SD {v.std(ddof=1):.4f} "
            f"range {v.min():.4f}-{v.max():.4f} ({v.max() - v.min():.4f}) "
            f"noise floor {config.VAL_AUC_NOISE_FLOOR:.2f}"
        )
    else:
        print(f"\n{len(ok)} of {len(df)} runs present; spread needs at least two")


# CLI
def _add_scorer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument(
        "--expect",
        type=float,
        default=None,
        help="expected exact validation AUC; exits non-zero if missed",
    )
    p.add_argument("--tolerance", type=float, default=0.001)
    p.add_argument("--out", default=None, help="write the result as JSON")
    p.add_argument(
        "--out-predictions",
        default=None,
        help="also write the val prediction table in evaluate's schema, so "
        "compare.py, analysis.py and ladder.py can read this run",
    )
    p.add_argument("--cpu", action="store_true", help="force the CPU device")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("store", help="score a single-stream checkpoint against a crop store")
    # No choices= here: the registry lives in src.models, which imports TensorFlow,
    # and building the parser must not. cmd_store validates the name instead.
    p.add_argument("--model", default="vgg16", help="architecture name from src.models")
    _add_scorer_args(p)
    p.set_defaults(func=cmd_store)

    p = sub.add_parser("fusion", help="score a paired checkpoint against its pair store")
    p.add_argument("--model", default="vgg16")
    _add_scorer_args(p)
    p.set_defaults(func=cmd_fusion)

    p = sub.add_parser("ablations", help="exact validation AUC of the regularisation ablations")
    p.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "ablation_exact"),
        help="folder for the per-tag tables and exact_val_auc.json",
    )
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--cpu", action="store_true", help="force the CPU device")
    p.set_defaults(func=cmd_ablations)

    p = sub.add_parser("rescore-cuda", help="re-score every off-machine run on this laptop")
    p.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="pod-vs-laptop agreement bar. AUC on the 243 validation lesions moves in "
        "steps of 1/14672 = 6.8e-5, one per flipped pair, and a 1e-6 probability "
        "difference flips only near-tied pairs; 5e-3 is 73 pair flips.",
    )
    p.add_argument("--tag", nargs="+", default=None, help="score only these run tags")
    p.add_argument(
        "--out-predictions",
        default=None,
        help="directory to write {tag}_predictions_val.csv into, per run, in "
        "evaluate's schema. Needs real scoring: the per-row probabilities "
        "are not in the score JSON, so --reuse cannot serve it.",
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="read an existing outputs/results/rescore_<tag>.json instead of "
        "re-scoring that tag; scoring is deterministic, so the table comes "
        "out the same",
    )
    p.add_argument("--out", default=str(RESULTS / "cuda_rescore_table.csv"))
    p.set_defaults(func=cmd_rescore_cuda)

    p = sub.add_parser("seed-spread", help="a tag's replicates: exact AUC each, and the spread")
    p.add_argument("--tag", required=True, help="base tag, without the platform suffix")
    p.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[1, 2],
        help="replicate seeds; the base run carries config.SEED",
    )
    p.add_argument(
        "--tag-suffix",
        default="",
        help="platform suffix the tags carry; a replicate's tag is {tag}_s{seed}{suffix}, "
        "with the seed before the suffix",
    )
    p.add_argument(
        "--out", default=None, help="default outputs/results/seed_spread_{tag}{tag-suffix}.csv"
    )
    p.set_defaults(func=cmd_seed_spread)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

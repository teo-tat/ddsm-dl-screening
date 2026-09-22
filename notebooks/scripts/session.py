#!/usr/bin/env python
"""session.py — declare, assert and run a one-session measurement of the ledger.

A session is a fixed list of training rows launched off this machine; the rows and
the launch commands are declared separately and must agree before the first launch.

  declare: write the rows file from the declared table
  commands: write the launch lines from the family specification, not the rows file
  stage: record the code and store md5s of a session that stages
  assert: refuse unless rows, commands, recomputed tags and stores all agree. For a
    staging session it also compares the code and store files with the md5s `stage`
    recorded, so code edited after staging reports as a difference.
  run: execute the rows, recording each only after its post-conditions verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

# The declaration files live beside the tool that writes them; `assert` can be
# pointed at another copy with --rows/--commands.
SESSION_DIR = Path(__file__).resolve().parent


def _count_lines(path: Path) -> int:
    """Launch lines in `path` — non-empty, not the generated header — 0 if absent.
    The count comes from the file itself, so it names a set this run actually read."""
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _count_rows(path: Path) -> int:
    """Entries in the rows JSON, 0 if it does not exist or does not parse."""
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text()))
    except (ValueError, TypeError):
        return 0


def _discovered(what: str, n: int, where, *, required: bool = False) -> int:
    """Print the count and the search location, then refuse an empty required
    discovery: a session that asserts zero rows has asserted nothing."""
    places = where if isinstance(where, (list, tuple)) else [where]
    loc = ", ".join(str(p) for p in places)
    print(f"[discovered] {n} {what} under {loc}")
    if required and n == 0:
        raise SystemExit(
            f"REFUSED: no {what} found under {loc}. An empty discovery is not a "
            f"pass — this run checked nothing."
        )
    return n


SESSION = "cuda0917"  # rebound by _select() from --session

LEDGER = "_cuda0917"

_CHILD_ENV: dict = dict(os.environ)  # replaced in main() before any TensorFlow import

ROWS_JSON = SESSION_DIR / "session_cuda0917_rows.json"

COMMANDS_TXT = SESSION_DIR / "session_cuda0917_commands.txt"

OUT_DIR = "outputs/_cuda/s0917/weights"  # train_*_store --out-dir: weights, results and logs

PATIENCE = "25"  # config.EARLY_STOPPING_PATIENCE is 10: never defaulted

SEEDS = (28, 1, 2)

CODE_FILES = (
    "notebooks/scripts/stores.py",
    "src/data_loader.py",
    "src/boxes.py",
    "src/geometry.py",
    "src/localise.py",
    "src/config.py",
    "src/train_crop_store.py",
    "src/train_fusion_store.py",
    "src/train.py",
    "src/models.py",
)

# The context-fusion session

# Two rows, one seed, its own out-dir, one more code file, and two pre-flights the
# other session has no analogue for: the pod's card and a parity score.
CODE_FILES_0918 = CODE_FILES + ("src/test_pass_guard.py",)

CARD_0918 = "NVIDIA GeForce RTX 4090"

PARITY_0918 = Path("/workspace/runs/parity_vgg16_crop_m020_cuda.json")

PARITY_EXPECT, PARITY_TOL = 0.8843, 0.001

# (store, tag the trainer writes, what the store's manifest must say)
_R18 = [
    (
        "fusion_store_vgg16_fusion_context",
        "vgg16_fusion_context_cuda0918",
        dict(
            pair="context", rows="all", paired=True, box_source="oracle", margin_a=0.2, margin_b=1.0
        ),
    ),
    (
        "fusion_store_vgg16_fusion_control_all",
        "vgg16_fusion_control_all_cuda0918",
        dict(pair="control", rows="all", paired=False, box_source="oracle", margin_a=0.2),
    ),
]

SESSIONS: dict[str, dict] = {
    "cuda0917": dict(
        ledger="_cuda0917",
        out_dir="outputs/_cuda/s0917/weights",
        seeds=(28, 1, 2),
        code_files=CODE_FILES,
        card=None,
        parity=None,
        stores=None,
    ),
    "cuda0918": dict(
        ledger="_cuda0918",
        out_dir="outputs/_cuda/s0918",
        seeds=(28,),
        code_files=CODE_FILES_0918,
        card=CARD_0918,
        parity=PARITY_0918,
        stores=_R18,
    ),
}


def _select(name: str) -> None:
    """Rebind the module's session constants to the named session: every part reads
    them as globals, so one rebinding switches all of them together."""
    global SESSION, LEDGER, OUT_DIR, SEEDS, CODE_FILES, ROWS_JSON, COMMANDS_TXT
    if name not in SESSIONS:
        raise SystemExit(f"unknown session {name!r}; one of: {', '.join(SESSIONS)}")
    spec = SESSIONS[name]
    SESSION = name
    LEDGER = spec["ledger"]
    OUT_DIR = spec["out_dir"]
    SEEDS = spec["seeds"]
    CODE_FILES = spec["code_files"]
    ROWS_JSON = SESSION_DIR / f"session_{name}_rows.json"
    COMMANDS_TXT = SESSION_DIR / f"session_{name}_commands.txt"


def _spec() -> dict:
    return SESSIONS[SESSION]


_R = [
    (
        "ladder",
        "crop",
        "crop_store_rung1_oracle",
        "vgg16",
        None,
        "_rung1_oracle_cuda0917",
        SEEDS,
        "vgg16_crop_rung1_oracle_cuda0917",
    ),
    (
        "ladder",
        "crop",
        "crop_store_rung2_oracle",
        "vgg16",
        None,
        "_rung2_oracle_cuda0917",
        (28, 1, 2, 3, 4),
        "vgg16_crop_rung2_oracle_cuda0917",
    ),
    (
        "ladder",
        "crop",
        "crop_store_rung3_unet_promoted",
        "vgg16",
        None,
        "_rung3_unet_promoted_cuda0917",
        SEEDS,
        "vgg16_crop_rung3_unet_promoted_cuda0917",
    ),
    (
        "ladder",
        "crop",
        "crop_store_rung4_cam",
        "vgg16",
        None,
        "_rung4_cam_cuda0917",
        SEEDS,
        "vgg16_crop_rung4_cam_cuda0917",
    ),
    (
        "ladder",
        "crop",
        "crop_store_rung5_jitter_promoted",
        "vgg16",
        None,
        "_rung5_jitter_promoted_cuda0917",
        SEEDS,
        "vgg16_crop_rung5_jitter_promoted_cuda0917",
    ),
    (
        "ladder",
        "crop",
        "crop_store_rung6_shuffled",
        "vgg16",
        None,
        "_rung6_shuffled_cuda0917",
        SEEDS,
        "vgg16_crop_rung6_shuffled_cuda0917",
    ),
    (
        "ladder",
        "crop",
        "crop_store_rung7_whole",
        "vgg16",
        None,
        "_rung7_whole_cuda0917",
        SEEDS,
        "vgg16_crop_rung7_whole_cuda0917",
    ),
    (
        "recovery",
        "crop",
        "crop_store_rung3_unet_bundle",
        "vgg16",
        None,
        "_rung3_unet_bundle_cuda0917",
        SEEDS,
        "vgg16_crop_rung3_unet_bundle_cuda0917",
    ),
    (
        "familyA",
        "crop",
        "crop_store_m020",
        "vgg16",
        None,
        "_m020_cuda0917",
        SEEDS,
        "vgg16_crop_m020_cuda0917",
    ),
    (
        "familyA",
        "crop",
        "crop_store_m020",
        "resnet50",
        None,
        "_m020_cuda0917",
        SEEDS,
        "resnet50_crop_m020_cuda0917",
    ),
    (
        "familyA",
        "crop",
        "crop_store_m020",
        "densenet121",
        None,
        "_m020_cuda0917",
        SEEDS,
        "densenet121_crop_m020_cuda0917",
    ),
    (
        "familyA",
        "crop",
        "crop_store_m020",
        "efficientnet",
        None,
        "_m020_cuda0917",
        SEEDS,
        "efficientnet_crop_m020_cuda0917",
    ),
    (
        "scratch",
        "crop",
        "crop_store_m020",
        "baseline",
        None,
        "_m020_cuda0917",
        SEEDS,
        "baseline_crop_m020_cuda0917",
    ),
    (
        "scratch",
        "crop",
        "crop_store_m020",
        "scaled",
        None,
        "_m020_cuda0917",
        SEEDS,
        "scaled_crop_m020_cuda0917",
    ),
    (
        "scratch",
        "crop",
        "crop_store_m020",
        "regularised",
        None,
        "_m020_cuda0917",
        SEEDS,
        "regularised_crop_m020_cuda0917",
    ),
    (
        "full",
        "crop",
        "crop_store_full_image",
        "vgg16",
        "vgg16",
        "_full_cuda0917",
        SEEDS,
        "vgg16_full_cuda0917",
    ),
    (
        "fusion",
        "fusion",
        "fusion_store_vgg16_fusion_view_unet_promoted",
        "vgg16",
        None,
        "_cuda0917",
        SEEDS,
        "vgg16_fusion_view_unet_promoted_cuda0917",
    ),
    (
        "fusion",
        "fusion",
        "fusion_store_vgg16_fusion_view",
        "vgg16",
        None,
        "_cuda0917",
        SEEDS,
        "vgg16_fusion_view_cuda0917",
    ),
    (
        "fusion",
        "fusion",
        "fusion_store_vgg16_fusion_control",
        "vgg16",
        None,
        "_cuda0917",
        SEEDS,
        "vgg16_fusion_control_cuda0917",
    ),
]

STORE_EXPECT = {
    "crop_store_rung1_oracle": dict(box_source="oracle", margin=0.0, crop_sizing="box_provider"),
    "crop_store_rung2_oracle": dict(box_source="oracle", margin=0.2, crop_sizing="box_provider"),
    "crop_store_rung3_unet_promoted": dict(
        box_source="unet", margin=0.2, boxes_dir="outputs/results/localiser_bundle/boxes_A24"
    ),
    "crop_store_rung3_unet_bundle": dict(
        box_source="unet", margin=0.2, boxes_dir="outputs/results/localiser_bundle/boxes"
    ),
    "crop_store_rung4_cam": dict(box_source="cam", margin=0.2),
    "crop_store_rung5_jitter_promoted": dict(
        box_source="jitter", margin=0.2, boxes_dir="outputs/results/localiser_bundle/boxes_A24"
    ),
    "crop_store_rung6_shuffled": dict(box_source="shuffled", margin=0.2),
    "crop_store_rung7_whole": dict(box_source="whole", margin=0.2),
    "crop_store_m020": dict(box_source=None, margin=0.2, crop_sizing="mask_margin"),
    "crop_store_full_image": dict(box_source=None, source="full"),
    "fusion_store_vgg16_fusion_view_unet_promoted": dict(
        pair="view", box_source="unet", boxes_dir="outputs/results/localiser_bundle/boxes_A24"
    ),
    "fusion_store_vgg16_fusion_view": dict(pair="view", box_source="oracle"),
    "fusion_store_vgg16_fusion_control": dict(pair="control", box_source="oracle"),
}


def _seed_tag(base_tag: str, seed: int) -> str:
    return base_tag if seed == config.SEED else f"{base_tag}_s{seed}"


def declared_rows() -> list[dict]:
    if SESSION == "cuda0918":
        return [
            dict(
                family="fusion",
                trainer="fusion",
                store=store,
                model="vgg16",
                tag_base=None,
                tag_suffix=LEDGER,
                seed=SEEDS[0],
                patience=int(PATIENCE),
                tag=tag,
            )
            for store, tag, _ in _R18
        ]
    rows = []
    for fam, trainer, store, model, tag_base, suffix, seeds, tag28 in _R:
        for s in seeds:
            rows.append(
                dict(
                    family=fam,
                    trainer=trainer,
                    store=store,
                    model=model,
                    tag_base=tag_base,
                    tag_suffix=suffix,
                    seed=s,
                    patience=int(PATIENCE),
                    tag=_seed_tag(tag28, s),
                )
            )
    return rows


def command_lines() -> list[str]:
    if SESSION == "cuda0918":
        # The same flags, in the same order, as the other session's fusion line.
        fus18 = (
            "python -m src.train_fusion_store --store outputs/{store} --arch vgg16 "
            "--seed " + str(SEEDS[0]) + " --tag-suffix " + LEDGER + " --platform-suffix '' "
            "--patience " + PATIENCE + " --out-dir " + OUT_DIR
        )
        return [fus18.format(store=store) for store, _, _ in _R18]
    crop = (
        "python -m src.train_crop_store --store outputs/{store} --model {model} --seed {seed} "
        "--tag-suffix {suffix} --platform-suffix '' --patience "
        + PATIENCE
        + " --out-dir "
        + OUT_DIR
    )
    fus = (
        "python -m src.train_fusion_store --store outputs/{store} --arch vgg16 --seed {seed} "
        "--tag-suffix "
        + LEDGER
        + " --platform-suffix '' --patience "
        + PATIENCE
        + " --out-dir "
        + OUT_DIR
    )
    lines = []
    ladder = [
        (1, "oracle", ""),
        (2, "oracle", ""),
        (3, "unet", "_promoted"),
        (4, "cam", ""),
        (5, "jitter", "_promoted"),
        (6, "shuffled", ""),
        (7, "whole", ""),
    ]
    for k, src, box in ladder:
        cond = f"rung{k}_{src}{box}"
        for s in ((28, 1, 2, 3, 4) if k == 2 else SEEDS):
            lines.append(
                crop.format(
                    store=f"crop_store_{cond}", model="vgg16", seed=s, suffix=f"_{cond}{LEDGER}"
                )
            )
    for s in SEEDS:
        lines.append(
            crop.format(
                store="crop_store_rung3_unet_bundle",
                model="vgg16",
                seed=s,
                suffix=f"_rung3_unet_bundle{LEDGER}",
            )
        )
    for model in (
        "vgg16",
        "resnet50",
        "densenet121",
        "efficientnet",
        "baseline",
        "scaled",
        "regularised",
    ):
        for s in SEEDS:
            lines.append(
                crop.format(store="crop_store_m020", model=model, seed=s, suffix=f"_m020{LEDGER}")
            )
    for s in SEEDS:
        lines.append(
            crop.format(
                store="crop_store_full_image", model="vgg16", seed=s, suffix=f"_full{LEDGER}"
            )
            + " --tag-base vgg16"
        )
    for store in (
        "fusion_store_vgg16_fusion_view_unet_promoted",
        "fusion_store_vgg16_fusion_view",
        "fusion_store_vgg16_fusion_control",
    ):
        for s in SEEDS:
            lines.append(fus.format(store=store, seed=s))
    return lines


def _flags(argv: list[str]) -> dict[str, str]:
    """`--key value` pairs; a flag without a value maps to ''. Positional args refused."""
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--"):
            raise ValueError(f"unexpected positional argument {a!r}")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[a[2:]] = argv[i + 1]
            i += 2
        else:
            out[a[2:]] = ""
            i += 1
    return out


def _tag_for(trainer: str, f: dict[str, str], stores_root: Path) -> str:
    """The tag the trainer itself will write, by its own rule."""
    from src import train

    seed = int(f["seed"])
    if trainer == "crop":
        base = f.get("tag-base") or f"{f['model']}_crop"
        return (
            f"{base}{train._run_suffix(f['tag-suffix'], ablate=None, lr=None, seed=seed)}"
            f"{f['platform-suffix']}"
        )
    man = json.loads((stores_root / Path(f["store"]).name / "manifest.json").read_text())
    return (
        f"{man['fusion_tag']}{f.get('tag-suffix', '')}"
        f"{train._run_suffix('', ablate=None, lr=None, seed=seed)}{f['platform-suffix']}"
    )


def _digest(p: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECT_OVERRIDE: Path | None = None


def expect_path() -> Path:
    """Where the staged store/code digests live for a session that stages."""
    return EXPECT_OVERRIDE or SESSION_DIR / f"session_{SESSION}_expect.json"


def stage() -> int:
    """Record code and store md5s, after checking each store against its own manifest,
    so nothing on the pod can differ from what the laptop recorded."""
    spec = _spec()
    if not spec["stores"]:
        raise SystemExit(
            f"session {SESSION} does not stage: it declares rows and commands "
            f"instead (run `declare` and `commands`)"
        )
    bad, stores = [], {}
    for store, _, _ in spec["stores"]:
        d = ROOT / "outputs" / store
        man = json.loads((d / "manifest.json").read_text())
        for name, fp in man["files"].items():
            if _digest(d / name, "sha256")[:16] != fp:
                bad.append(f"{store}/{name}: differs from the fingerprint its manifest records")
        stores[store] = {p.name: _digest(p) for p in sorted(d.iterdir()) if p.is_file()}
    if bad:
        print("REFUSED:", *bad, sep="\n  ")
        return 1
    out = expect_path()
    out.write_text(
        json.dumps(
            {
                "staged": dt.datetime.now(dt.timezone.utc).isoformat(),
                "code_md5": {f: _digest(ROOT / f) for f in CODE_FILES},
                "store_md5": stores,
            },
            indent=2,
        )
    )
    print(f"-> {out}")
    return 0


def _store_expect() -> dict[str, dict]:
    """What each declared store's manifest must say, for the selected session."""
    spec = _spec()
    if spec["stores"]:
        return {store: want for store, _, want in spec["stores"]}
    return STORE_EXPECT


def _store_checks(stores_root: Path) -> list[str]:
    """A staging session's store, code and leakage checks. Empty for one that does not."""
    spec = _spec()
    if not spec["stores"]:
        return []
    exp_p = expect_path()
    if not exp_p.exists():
        return [f"{exp_p.name} missing: run `stage` on the laptop"]
    exp, bad = json.loads(exp_p.read_text()), []
    for f, m in exp["code_md5"].items():
        # A staged expectation can name a file that no longer exists; the staged copy
        # is what the session ran, so this notes it instead of refusing.
        if not (ROOT / f).exists():
            print(f"  note: {f} is retired; the staged copy is what this session ran")
            continue
        if _digest(ROOT / f) != m:
            bad.append(f"{f}: md5 differs from the staged code")
    for store, tag, want in spec["stores"]:
        d = stores_root / store
        if not d.is_dir():
            bad.append(f"{store}: missing under {stores_root}")
            continue
        man = json.loads((d / "manifest.json").read_text())
        bad += [
            f"{store}: manifest {k}={man.get(k)!r}, expected {v!r}"
            for k, v in want.items()
            if man.get(k) != v
        ]
        files = {p.name for p in d.iterdir() if p.is_file()}
        bad += [f"{store}/{n}: a test file" for n in sorted(files) if n.startswith("test_")]
        for n, m in exp["store_md5"][store].items():
            if n not in files or _digest(d / n) != m:
                bad.append(f"{store}/{n}: missing or md5 differs from the staged store")
    return bad


def parity(path: Path) -> str | None:
    """None if the pod's parity result is present and within tolerance, else the reason."""
    if not path.exists():
        return f"no parity result at {path}: run the parity check first"
    r = json.loads(path.read_text())
    auc = r.get("exact_val_auc")
    if (
        Path(str(r.get("store"))).name != "crop_store_m020"
        or r.get("n") != 243
        or Path(str(r.get("weights"))).name != "vgg16_crop_m020_best.weights.h5"
        or auc is None
    ):
        return f"{path} is not the parity check (store, n or weights differ)"
    if abs(auc - PARITY_EXPECT) > PARITY_TOL:
        return f"parity {auc:.6f} is outside {PARITY_EXPECT} ± {PARITY_TOL}"
    return None


def preflight(parity_path: Path | None) -> list[str]:
    """The pod pre-flights a session declares: its card, and its parity score."""
    spec = _spec()
    bad = []
    if spec["card"]:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
        if name != spec["card"]:
            bad.append(f"GPU is {name or 'unreadable'!r}, the session declares {spec['card']!r}")
    if spec["parity"]:
        why = parity(Path(parity_path) if parity_path else spec["parity"])
        if why:
            bad.append(why)
    return bad


def check(stores_root: Path, parser: bool = True) -> list[str]:
    """Every mismatch between ROWS, COMMANDS, the trainers and the stores. Empty = agree."""
    bad: list[str] = []
    # The guard is against the SELECTED session's ledger, not config.LEDGER_SUFFIX:
    # that constant names one session, so it would refuse every other one outright.
    off = [r["tag"] for r in declared_rows() if LEDGER not in r["tag"]]
    if off:
        bad.append(
            f"{len(off)} declared tag(s) do not carry the session's ledger "
            f"{LEDGER!r}: {off[:3]}"
        )
    if LEDGER != config.LEDGER_SUFFIX:
        print(
            f"[note] this session's ledger is {LEDGER!r}; config.LEDGER_SUFFIX is "
            f"{config.LEDGER_SUFFIX!r} (the one the report's tables resolve through)"
        )
    bad += _store_checks(stores_root)
    if not ROWS_JSON.exists() or not COMMANDS_TXT.exists():
        return bad + ["rows or commands file missing: run `declare` and `commands`"]
    rows = json.loads(ROWS_JSON.read_text())
    if rows != declared_rows():
        bad.append(f"{ROWS_JSON.name} differs from ROWS in this file (edited by hand, or stale)")
    by_tag = {}
    for r in rows:
        if r["tag"] in by_tag:
            bad.append(f"duplicate declared tag {r['tag']}")
        by_tag[r["tag"]] = r
    seen = set()
    lines = [
        l for l in COMMANDS_TXT.read_text().splitlines() if l.strip() and not l.startswith("#")
    ]
    for n, line in enumerate(lines, 1):
        argv = shlex.split(line)
        if argv[:2] != ["python", "-m"] or argv[2] not in (
            "src.train_crop_store",
            "src.train_fusion_store",
        ):
            bad.append(f"line {n}: not a store-trainer command")
            continue
        trainer = "crop" if argv[2] == "src.train_crop_store" else "fusion"
        try:
            f = _flags(argv[3:])
        except ValueError as e:
            bad.append(f"line {n}: {e}")
            continue
        need = ["store", "seed", "tag-suffix", "platform-suffix", "patience", "out-dir"] + (
            ["model"] if trainer == "crop" else ["arch"]
        )
        miss = [k for k in need if k not in f]
        if miss:
            bad.append(
                f"line {n}: critical flag(s) not explicit: {', '.join('--' + k for k in miss)}"
            )
            continue
        if f["patience"] != PATIENCE:
            bad.append(f"line {n}: --patience {f['patience']} (declared {PATIENCE})")
        if f["platform-suffix"] != "":
            bad.append(f"line {n}: --platform-suffix {f['platform-suffix']!r} (declared '')")
        if f["out-dir"] != OUT_DIR:
            bad.append(f"line {n}: --out-dir {f['out-dir']} (declared {OUT_DIR})")
        store = Path(f["store"]).name
        smf = stores_root / store / "manifest.json"
        if not smf.exists():
            bad.append(f"line {n}: store {store} missing at {smf.parent}")
            continue
        sm = json.loads(smf.read_text())
        for k, v in _store_expect().get(store, {}).items():
            if (sm.get(k) if k != "source" else sm.get("source", "crop")) != v:
                bad.append(f"line {n}: store {store} manifest {k}={sm.get(k)!r}, expected {v!r}")
        if store not in _store_expect():
            bad.append(f"line {n}: store {store} has no declared expectation")
        tag = _tag_for(trainer, f, stores_root)
        if tag in seen:
            bad.append(f"line {n}: tag {tag} launched twice")
        seen.add(tag)
        r = by_tag.get(tag)
        if r is None:
            bad.append(f"line {n}: produces tag {tag}, which no declared row expects")
            continue
        for k, got in (
            ("trainer", trainer),
            ("store", store),
            ("seed", int(f["seed"])),
            ("tag_suffix", f["tag-suffix"] if trainer == "crop" else r["tag_suffix"]),
            ("tag_base", f.get("tag-base")),
        ):
            if k == "tag_base" and trainer == "fusion":
                continue
            if r[k] != got:
                bad.append(f"line {n}: {tag} {k}={got!r}, declared {r[k]!r}")
        if trainer == "crop" and r["model"] != f["model"]:
            bad.append(f"line {n}: {tag} model={f['model']!r}, declared {r['model']!r}")
        if parser:
            pr = subprocess.run(
                [sys.executable, "-m", argv[2], *argv[3:], "--parse-only"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )
            if pr.returncode != 0:
                bad.append(
                    f"line {n}: the trainer's own parser rejects it: "
                    f"{(pr.stderr.strip().splitlines() or ['?'])[-1][:120]}"
                )
    for t in sorted(set(by_tag) - seen):
        bad.append(f"declared row {t} has no launch command")
    return bad


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        e["tag"]: e for e in (json.loads(l) for l in path.read_text().splitlines() if l.strip())
    }


def _write_completed(path: Path, entries: dict[str, dict]) -> None:
    """Atomic: a crash leaves either the old file or the new one, never half of one."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for e in entries.values():
            fh.write(json.dumps(e) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _owned_files(out: Path, tag: str, all_tags: set[str]) -> list[Path]:
    """Files belonging to exactly `tag`: name starts `{tag}_`, and not `{longer}_` for a
    longer declared tag that extends it (so seed 28's files never include seed 1's)."""
    longer = [t for t in all_tags if t != tag and t.startswith(tag + "_")]
    return [
        p
        for p in out.iterdir()
        if p.is_file()
        and p.name.startswith(tag + "_")
        and not any(p.name.startswith(u + "_") for u in longer)
    ]


def run(args) -> int:
    root = Path(args.root)
    # --out is for a laptop smoke only: the real session writes to OUT_DIR, where the
    # fetch lands and the resolver reads, so smoke weights must never go there.
    out = Path(args.out).resolve() if args.out else ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    bad = check(ROOT / "outputs", parser=not args.no_parser)
    if bad:
        print(
            "REFUSED: the session does not match its declaration:",
            *bad,
            sep="\n  ",
            file=sys.stderr,
        )
        return 1
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < args.min_free_gb:
        print(
            f"REFUSED: {free_gb:.1f} GB free under {out}, need {args.min_free_gb}", file=sys.stderr
        )
        return 1

    rows = {r["tag"]: r for r in declared_rows()}
    lines = [
        l for l in COMMANDS_TXT.read_text().splitlines() if l.strip() and not l.startswith("#")
    ]
    order = []
    for line in lines:
        argv = shlex.split(line)
        trainer = "crop" if argv[2] == "src.train_crop_store" else "fusion"
        order.append((_tag_for(trainer, _flags(argv[3:]), ROOT / "outputs"), argv))
    if args.tag:
        wanted = set(args.tag.split(","))
        order = [o for o in order if o[0] in wanted]

    sess = root / "session_manifest.json"
    if not sess.exists():
        sess.write_text(
            json.dumps(
                {
                    "started": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "code_md5": {f: _md5(ROOT / f) for f in CODE_FILES},
                    "rows_md5": _md5(ROWS_JSON),
                    "commands_md5": _md5(COMMANDS_TXT),
                    "extra_args": args.extra_args,
                },
                indent=2,
            )
        )

    comp_path = root / "completed.jsonl"
    completed = _load_completed(comp_path)
    all_tags = set(rows)
    for tag, argv in order:
        r = rows[tag]
        if tag in completed:
            e = completed[tag]
            best = out / f"{tag}_best.weights.h5"
            ok = (
                e.get("store") == r["store"]
                and e.get("seed") == r["seed"]
                and best.exists()
                and _md5(best) == e.get("best_md5")
            )
            if not ok:
                print(
                    f"REFUSED: {tag} is recorded complete but no longer matches its record "
                    f"(store, seed, weights md5). Restore the recorded files, or drop the row "
                    f"from the completed list.",
                    file=sys.stderr,
                )
                return 2
            print(f"skip   {tag} (complete, md5 {e['best_md5'][:12]} verified)", flush=True)
            continue
        owned = _owned_files(out, tag, all_tags)
        if owned:
            q = root / "partial" / f"{tag}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            q.mkdir(parents=True)
            for p in owned:
                shutil.move(str(p), q / p.name)
            print(f"quar   {tag} {len(owned)} partial file(s) -> {q}", flush=True)
        if shutil.disk_usage(out).free / 1e9 < args.min_row_free_gb:
            print(f"REFUSED before {tag}: under {args.min_row_free_gb} GB free", file=sys.stderr)
            return 1
        rest = list(argv[3:])
        rest[rest.index("--out-dir") + 1] = str(out)
        cmd = [sys.executable, "-m", argv[2], *rest, *shlex.split(args.extra_args)]
        log = root / "logs" / f"{tag}.log"
        print(f"run    {tag}", flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**_CHILD_ENV, "PYTHONPATH": str(ROOT)},
            ).returncode
        # Post-conditions, not the exit code alone.
        man_p, best = out / f"{tag}_manifest.json", out / f"{tag}_best.weights.h5"
        errs = []
        if rc != 0:
            errs.append(f"exit {rc}")
        man = json.loads(man_p.read_text()) if man_p.exists() else None
        if man is None:
            errs.append("no run manifest")
        else:
            if man.get("tag") != tag:
                errs.append(f"manifest tag {man.get('tag')}")
            if man.get("seed_effective") != r["seed"]:
                errs.append(f"seed_effective {man.get('seed_effective')}")
            if str(man.get("patience")) != PATIENCE:
                errs.append(f"patience {man.get('patience')}")
            if Path(str(man.get("store", ""))).name != r["store"]:
                errs.append(f"store {man.get('store')}")
            if not man.get("gpu_name") and not args.allow_no_gpu_name:
                errs.append("gpu_name not recorded")
        if not best.exists():
            errs.append("no best weights")
        if errs:
            print(f"FAILED {tag}: {'; '.join(errs)} (log {log}). Stopping.", file=sys.stderr)
            return 3
        completed[tag] = dict(
            tag=tag,
            store=r["store"],
            seed=r["seed"],
            best_md5=_md5(best),
            gpu_name=man.get("gpu_name"),
            finished=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        _write_completed(comp_path, completed)
        print(f"done   {tag} ({len(completed)}/{len(rows)} complete)", flush=True)
    print(f"session: {len(completed)}/{len(rows)} rows complete -> {comp_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _with_session(p):
        p.add_argument(
            "--session",
            choices=sorted(SESSIONS),
            default="cuda0917",
            help="which measurement session's declaration to act on",
        )
        return p

    _with_session(sub.add_parser("declare"))
    _with_session(sub.add_parser("commands"))
    _with_session(sub.add_parser("stage", help="record code and store md5s (a staging session)"))
    a = _with_session(sub.add_parser("assert"))
    a.add_argument("--store-root", default=str(ROOT / "outputs"))
    a.add_argument("--rows", default=None, help="check a different rows file")
    a.add_argument("--commands", default=None, help="check a different commands file")
    a.add_argument(
        "--expect", default=None, help="check a different staged-digest file (a staging session)"
    )
    a.add_argument("--no-parser", action="store_true", help="skip the --parse-only subprocesses")
    r = _with_session(sub.add_parser("run"))
    r.add_argument(
        "--parity",
        default=None,
        help="the parity result, for a session that declares one; run does not read it",
    )
    r.add_argument(
        "--root", required=True, help="session bookkeeping: completed.jsonl, logs, partial/"
    )
    r.add_argument("--tag", default=None, help="comma-separated tags (smoke)")
    r.add_argument(
        "--extra-args",
        default="",
        help="appended to every launch, e.g. '--cpu --limit 16 --epochs 1'",
    )
    r.add_argument("--min-free-gb", type=float, default=35.0)
    r.add_argument("--min-row-free-gb", type=float, default=2.0)
    r.add_argument("--allow-no-gpu-name", action="store_true", help="laptop smoke only")
    r.add_argument("--no-parser", action="store_true")
    r.add_argument("--out", default=None, help="laptop smoke only: write outside the real OUT_DIR")
    args = ap.parse_args(argv)
    _select(args.session)

    global ROWS_JSON, COMMANDS_TXT, _CHILD_ENV
    # Recomputing tags imports src.train, and that import takes a GPU context the
    # children need: hide the GPU from this process only, and pass theirs through.
    _CHILD_ENV = dict(os.environ)
    if args.cmd == "run":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.cmd == "declare":
        ROWS_JSON.write_text(json.dumps(declared_rows(), indent=1))
        print(f"{len(declared_rows())} rows -> {ROWS_JSON}")
        return 0
    if args.cmd == "stage":
        return stage()
    if args.cmd == "commands":
        lines = command_lines()
        COMMANDS_TXT.write_text(
            f"# generated by session.py --session {SESSION} commands; "
            f"asserted against the rows file\n" + "\n".join(lines) + "\n"
        )
        print(f"{len(lines)} launch lines -> {COMMANDS_TXT}")
        return 0
    if args.cmd == "assert":
        if args.rows:
            ROWS_JSON = Path(args.rows)
        if args.commands:
            COMMANDS_TXT = Path(args.commands)
        if args.expect:
            global EXPECT_OVERRIDE
            EXPECT_OVERRIDE = Path(args.expect)
        _discovered("declared rows", _count_rows(ROWS_JSON), ROWS_JSON, required=True)
        _discovered("launch lines", _count_lines(COMMANDS_TXT), COMMANDS_TXT, required=True)
        bad = check(Path(args.store_root), parser=not args.no_parser)
        if bad:
            print(f"ASSERTION FAILED: {len(bad)} mismatch(es)", *bad, sep="\n  ")
            return 1
        print(
            f"ASSERTION HOLDS: {len(declared_rows())} declared rows = launch lines; "
            "every flag explicit, every tag recomputed, every store as declared"
        )
        return 0
    return run(args)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())

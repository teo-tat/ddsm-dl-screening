#!/usr/bin/env python
"""test_test_pass_guard.py — the loader-level barrier admits exactly one caller.

A guard that only ever refuses is untested, so this checks both directions: an
unauthorised read of the test partition through a guarded loader is refused, and a
matching token is admitted. The guard covers the public loading functions (split_frames,
build_datasets and fusion.build_pairs), not every function that could return test rows.
With no token set, it also calls each command-line refusal of the test partition and the
refusals of birads_baseline and localise_eval to overwrite frozen records; a refusal
inside main() runs with its first read replaced by a sentinel, so no check reads data.
Each command line is also run with a test argv, its reads before the refusal answered in
memory, to show that it reaches the refusal.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import test_pass_guard as G  # noqa: E402
from src import data_loader  # noqa: E402

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


FROZEN = G.FREEZE_TOKEN_FILE.exists()

# Before the freeze there is no token file; after it, the partition stays shut because
# the caller holds no matching environment token. The checks are the same either way.
print(
    f"\n=== 1. {'after' if FROZEN else 'before'} the freeze: "
    "every guarded test read is refused ==="
)
if FROZEN:
    chk("freeze token file exists", True, str(G.FREEZE_TOKEN_FILE.relative_to(ROOT)))
    chk("no environment token is set", G.ENV_VAR not in os.environ, G.ENV_VAR)
else:
    chk(
        "freeze token file does not exist",
        not G.FREEZE_TOKEN_FILE.exists(),
        str(G.FREEZE_TOKEN_FILE.relative_to(ROOT)),
    )
chk("authorised() is False", not G.authorised())
try:
    G.require_test_pass("unit test")
    chk("require_test_pass raises", False, "it returned")
except PermissionError as e:
    chk("require_test_pass raises PermissionError", True, str(e)[:70] + "...")

frames = data_loader.split_frames(source="crop")
chk("train is readable", len(frames["train"]) > 0, f"{len(frames['train'])} rows")
chk("val is readable", len(frames["val"]) > 0, f"{len(frames['val'])} rows")
import pandas as pd  # noqa: E402

refused = data_loader._RefusedSplit("unit test")
for how, fn in (
    ("frames['test']", lambda: frames["test"]),
    ("frames.get('test')", lambda: frames.get("test")),
    ("frames.items()", lambda: list(frames.items())),
    ("frames.values()", lambda: list(frames.values())),
    ("dict(frames)", lambda: dict(frames)),
    ("{**frames}", lambda: {**frames}),
    ("frames.copy()", lambda: frames.copy()),
    ("pd.concat(frames)", lambda: pd.concat(frames)),
    ("build_datasets' test split: len()", lambda: len(refused)),
    ("build_datasets' test split: iteration", lambda: list(refused)),
    ("build_datasets' test split: attribute", lambda: refused.columns),
):
    try:
        fn()
        chk(f"{how} refused", False, "IT WAS ALLOWED")
    except PermissionError:
        chk(f"{how} refused", True)

print("\n=== 2. a token alone is not enough: it must MATCH ===")
tmp = Path(tempfile.mkdtemp()) / ".freeze_token"
real = G.FREEZE_TOKEN_FILE
try:
    G.FREEZE_TOKEN_FILE = tmp
    tmp.write_text("the-real-token\n")
    os.environ[G.ENV_VAR] = "a-guess"
    chk("wrong env token is refused", not G.authorised())
    try:
        G.require_test_pass("unit test")
        chk("wrong token raises", False, "it returned")
    except PermissionError as e:
        chk("wrong token raises, and says why", "does not match" in str(e))
    os.environ.pop(G.ENV_VAR, None)
    chk("token file present but env unset is refused", not G.authorised())

    print("\n=== 3. the authorised pass IS admitted ===")
    os.environ[G.ENV_VAR] = "the-real-token"
    chk("matching token authorises", G.authorised())
    try:
        G.require_test_pass("unit test")
        chk("require_test_pass returns cleanly", True)
    except PermissionError as e:
        chk("require_test_pass returns cleanly", False, str(e)[:70])

    # The token as freeze.py writes it, and the env token as the pass derives it.
    print("\n=== 3b. the real token format, carried the way the pass carries it ===")
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location("freeze", ROOT / "notebooks/scripts/freeze.py")
    fz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fz)
    tmp.write_text(fz.token_text("0" * 64, 48))
    chk("the written token spans several lines", tmp.read_text().count("\n") >= 3)
    derived = subprocess.run(
        ["bash", "-c", f"tr -d '[:space:]' < '{tmp}'"], capture_output=True, text=True, check=True
    ).stdout
    os.environ[G.ENV_VAR] = derived
    chk("run_test_pass.sh's derived token authorises", G.authorised(), repr(derived[:40]))
    os.environ[G.ENV_VAR] = derived + "x"
    chk("a derived token with one extra character is still refused", not G.authorised())
finally:
    G.FREEZE_TOKEN_FILE = real
    os.environ.pop(G.ENV_VAR, None)

print("\n=== 4. and refuses again once the token is withdrawn ===")
chk("back to refusing", not G.authorised())

# Sections 5-7 run with the environment token unset, so every refusal must fire.
print("\n=== 5. fusion.build_pairs builds no test pairs without the token ===")
from src import fusion  # noqa: E402

chk("environment token is unset for the refusal checks", G.ENV_VAR not in os.environ, G.ENV_VAR)
try:
    pairs, counts = fusion.build_pairs(frames, "view", "paired")
    chk(
        "build_pairs on split_frames skips test",
        "test" not in pairs and "test" not in counts,
        f"splits {sorted(pairs)}",
    )
except PermissionError as e:
    pairs = {}
    chk("build_pairs on split_frames skips test", False, f"it read test: {str(e)[:60]}")
chk(
    "build_pairs builds train and validation pairs",
    all(
        s in pairs and len(pairs[s]) > 0 and set(pairs[s]["split"]) == {s} for s in ("train", "val")
    ),
    ", ".join(f"{s} {len(pairs[s])}" for s in ("train", "val") if s in pairs),
)
stored = ROOT / "artifacts" / "fusion_pairs_view_val.csv"
chk(
    "validation pairs match the stored view pair file",
    "val" in pairs
    and stored.exists()
    and pd.read_csv(stored)[["lesion_id_a", "lesion_id_b"]].equals(
        pairs["val"][["lesion_id_a", "lesion_id_b"]]
    ),
    str(stored.relative_to(ROOT)),
)
try:
    given = {"val": frames["val"], "test": data_loader._RefusedSplit("unit test")}
    only, _ = fusion.build_pairs(given, "view", "paired")
    chk("build_pairs never uses a test frame it is given", list(only) == ["val"], str(list(only)))
except PermissionError:
    chk("build_pairs never uses a test frame it is given", False, "it used the test frame")

print("\n=== 6. each command-line refusal fires for test and passes for validation ===")
from src import analysis, compare, ensemble, explain, precompute_localiser  # noqa: E402
from src import uncertainty  # noqa: E402


def exit_message(fn) -> str | None:
    """The message of the SystemExit fn raises, or None when it returns."""
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


def refuses(name: str, test_call, val_call) -> None:
    """test_call must exit with a REFUSED message; val_call must return."""
    msg = exit_message(test_call)
    chk(f"{name}: test is refused", (msg or "").startswith("REFUSED"), (msg or "it returned")[:70])
    msg = exit_message(val_call)
    chk(f"{name}: validation passes", msg is None, (msg or "")[:70])


def script(name: str):
    """A module from notebooks/scripts, loaded by path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "notebooks/scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


af, au = script("analyse_fusion"), script("analyse_uncertainty")
# Tables built in memory: one row whose partition or split column names the split.
val_rows = pd.DataFrame({"lesion_id": ["x"], "partition": ["val"], "split": ["val"]})
test_part = pd.DataFrame({"lesion_id": ["x"], "partition": ["test"]})
test_split = pd.DataFrame({"lesion_id": ["x"], "split": ["test"]})

msg = exit_message(fusion._refuse_test_stage)
chk(
    "fusion._refuse_test_stage: eval is refused",
    (msg or "").startswith("REFUSED"),
    (msg or "it returned")[:70],
)
refuses(
    "analyse_fusion --split test",
    lambda: af._refuse_test_outside_pass("test", {"t": val_rows}),
    lambda: af._refuse_test_outside_pass("val", {"t": val_rows}),
)
refuses(
    "analyse_fusion table with a test partition column",
    lambda: af._refuse_test_outside_pass(None, {"t": test_part}),
    lambda: af._refuse_test_outside_pass(None, {"t": val_rows}),
)
refuses(
    "analyse_fusion table with a test split column",
    lambda: af._refuse_test_outside_pass("val", {"t": test_split}),
    lambda: af._refuse_test_outside_pass("val", {"t": val_rows}),
)
refuses(
    "analyse_uncertainty deferral table with test rows",
    lambda: au._refuse_test_outside_pass({"predictions": test_part}),
    lambda: au._refuse_test_outside_pass({"predictions": val_rows}),
)
for mod in (explain, compare, analysis):
    refuses(
        f"{mod.__name__} --split",
        lambda m=mod: m._refuse_test_outside_pass("test"),
        lambda m=mod: m._refuse_test_outside_pass("val"),
    )
for mod in (uncertainty, ensemble, precompute_localiser):
    refuses(
        f"{mod.__name__} --splits",
        lambda m=mod: m._refuse_test_outside_pass(["train", "val", "test"]),
        lambda m=mod: m._refuse_test_outside_pass(["train", "val"]),
    )

print("\n=== 7. refusals inside main() stop before the first read ===")
from src import birads_baseline  # noqa: E402


class Sentinel(Exception):
    """Raised in place of the first read after a refusal point."""


def run_main(main, target, attr: str, argv: list[str] | None = None) -> tuple[str, list[str]]:
    """Runs main from an empty temp working directory with target.attr replaced by a call
    that raises Sentinel. Returns the outcome ("REFUSED..." message, "sentinel",
    "returned" or another exit message) and the files left in that directory."""

    def stop(*args, **kwargs):
        raise Sentinel(attr)

    saved, cwd, work = getattr(target, attr), os.getcwd(), Path(tempfile.mkdtemp())
    setattr(target, attr, stop)
    os.chdir(work)
    try:
        main() if argv is None else main(argv)
        outcome = "returned"
    except SystemExit as e:
        outcome = str(e)
    except Sentinel:
        outcome = "sentinel"
    finally:
        os.chdir(cwd)
        setattr(target, attr, saved)
    return outcome, sorted(p.name for p in work.rglob("*"))


out, left = run_main(fusion.main, fusion, "FusionRun", ["--pair", "view", "--stage", "eval"])
chk(
    "fusion --stage eval: main refuses before its first read",
    out.startswith("REFUSED") and not left,
    f"{out[:60]}; files {left}",
)
out, left = run_main(fusion.main, fusion, "FusionRun", ["--pair", "view", "--stage", "val"])
chk(
    "fusion --stage val: main reaches its first read",
    out == "sentinel" and not left,
    f"{out[:60]}; files {left}",
)

frozen = [birads_baseline.ART / n for n in ("birads_comparison_set.csv", "birads_baseline.json")]
chk("birads_baseline's frozen records exist", all(p.exists() for p in frozen))
out, left = run_main(birads_baseline.main, birads_baseline, "split_frames")
chk(
    "birads_baseline: main refuses while its frozen records exist",
    out.startswith("REFUSED") and not left,
    f"{out[:60]}; files {left}",
)
real_art = birads_baseline.ART
try:
    birads_baseline.ART = Path(tempfile.mkdtemp())
    out, left = run_main(birads_baseline.main, birads_baseline, "split_frames")
finally:
    birads_baseline.ART = real_art
chk(
    "birads_baseline: with no frozen records main reaches its first read",
    out == "sentinel" and not left,
    f"{out[:60]}; files {left}",
)

# Each command runs with a test argv. Reads before the refusal return in-memory frames,
# and the first statement after it raises Sentinel, so a missing call reads nothing.
print("\n=== 8. each command line reaches its refusal with a test input ===")
import runpy  # noqa: E402
import warnings  # noqa: E402

_MISSING = object()


def stop(*args, **kwargs):
    """Stands in for the first statement after a refusal."""
    raise Sentinel()


class StopFrame(pd.DataFrame):
    """An in-memory table whose merge, the first step after deferral's refusal, stops."""

    def merge(self, *args, **kwargs):
        raise Sentinel()


class ReadsInMemory:
    """Stands in for a module's pandas: read_csv returns the given frame, and every other
    name is pandas' own."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def read_csv(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame

    def __getattr__(self, name: str):
        return getattr(pd, name)


def run_entry(call, patches, argv: list[str] | None = None) -> tuple[str, list[str]]:
    """Runs call from an empty temp working directory with each (target, name, value)
    patch applied and, when argv is given, sys.argv set to it. Returns the outcome
    ("REFUSED..." message, "sentinel", "returned", another exit message or the error)
    and the files left in that directory."""
    saved = [(t, n, getattr(t, n, _MISSING)) for t, n, _ in patches]
    saved_argv, cwd, work = sys.argv, os.getcwd(), Path(tempfile.mkdtemp())
    for t, n, v in patches:
        setattr(t, n, v)
    if argv is not None:
        sys.argv = argv
    os.chdir(work)
    try:
        call()
        outcome = "returned"
    except SystemExit as e:
        outcome = str(e)
    except Sentinel:
        outcome = "sentinel"
    except Exception as e:
        outcome = f"{type(e).__name__}: {e}"
    finally:
        os.chdir(cwd)
        sys.argv = saved_argv
        for t, n, v in reversed(saved):
            if v is _MISSING:
                delattr(t, n)
            else:
                setattr(t, n, v)
    return outcome, sorted(p.name for p in work.rglob("*"))


def run_precompute_cli() -> None:
    """precompute_localiser's command line, run as __main__."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("src.precompute_localiser", run_name="__main__")


pairs_table = pd.DataFrame(
    {"lesion_id": ["x"], "lesion_id_a": ["x"], "lesion_id_b": ["x"], "label": [0]}
)
deferral_table = StopFrame(
    {"lesion_id": ["x"], "partition": ["test"], "y_true": [0], "y_prob": [0.5]}
)
wiring = (
    (
        "analyse_fusion controls --split test",
        lambda: af.main(
            ["controls", "--pairs", "p.csv", "--rung3", "r.csv", "--tag", "t", "--out", "."]
            + ["--split", "test"]
        ),
        [(af, "pd", ReadsInMemory(pairs_table)), (af, "print", stop)],
        None,
    ),
    (
        "analyse_uncertainty deferral with test rows",
        lambda: au.main(
            ["deferral", "--uncertainty", "u.csv", "--predictions", "p.csv"] + ["--out", "o.csv"]
        ),
        [(au, "pd", ReadsInMemory(deferral_table))],
        None,
    ),
    (
        "src.explain --split test",
        explain.main,
        [(explain, "print", stop), (explain, "_DEVICE", explain._DEVICE)],
        ["explain", "--model", "vgg16", "--split", "test"],
    ),
    (
        "src.uncertainty --splits val test",
        uncertainty.main,
        [(uncertainty, "print", stop), (uncertainty, "_DEVICE", uncertainty._DEVICE)],
        ["uncertainty", "--model", "vgg16", "--splits", "val", "test"],
    ),
    (
        "src.compare --split test",
        compare.main,
        [(compare, "primary_endpoint", stop), (compare, "SPLIT", compare.SPLIT)],
        ["compare", "--primary", "p", "--split", "test"],
    ),
    (
        "src.analysis --split test",
        analysis.main,
        [(analysis, "run", stop), (analysis, "SPLIT", analysis.SPLIT)],
        ["analysis", "--tag", "t", "--split", "test"],
    ),
    (
        "src.ensemble --splits val test",
        lambda: ensemble.main(
            ["--tags", "a", "b", "--out-tag", "o", "--splits", "val", "test", "--allow-test"]
        ),
        [(ensemble, "ensemble_split", stop)],
        None,
    ),
    (
        "src.precompute_localiser --splits train val test",
        run_precompute_cli,
        [(data_loader, "_prepare_frame", stop)],
        ["precompute_localiser", "--splits", "train", "val", "test", "--out-dir", "out"],
    ),
)
for name, call, patches, argv in wiring:
    out, left = run_entry(call, patches, argv)
    chk(
        f"{name}: the command line reaches the refusal",
        out.startswith("REFUSED") and not left,
        f"runtime; {out[:60]}; files {left}",
    )

print("\n=== 9. localise_eval refuses to overwrite its frozen outputs ===")
from src import localise_eval  # noqa: E402

out, left = run_entry(
    localise_eval.main,
    [(localise_eval, "_load_model", stop)],
    ["localise_eval", "--weights", "w.weights.h5"],
)
chk(
    "src.localise_eval with no --out-dir: main refuses before its first read",
    out.startswith("REFUSED") and not left,
    f"{out[:60]}; files {left}",
)

print("\n=== 10. localiser_bundle and analyse_ladder refuse test outside the pass ===")
lb, al = script("localiser_bundle"), script("analyse_ladder")
refuses(
    "localiser_bundle --split",
    lambda: lb._refuse_test_outside_pass(["test"]),
    lambda: lb._refuse_test_outside_pass(["val", "train"]),
)
refuses(
    "analyse_ladder errors --split",
    lambda: al._refuse_test_outside_pass("test"),
    lambda: al._refuse_test_outside_pass("val"),
)
token_file = G.FREEZE_TOKEN_FILE
try:
    G.FREEZE_TOKEN_FILE = Path(tempfile.mkdtemp()) / ".freeze_token"
    G.FREEZE_TOKEN_FILE.write_text("the-real-token\n")
    os.environ[G.ENV_VAR] = "the-real-token"
    for name, call in (
        ("localiser_bundle", lambda: lb._refuse_test_outside_pass(["test"])),
        ("analyse_ladder errors", lambda: al._refuse_test_outside_pass("test")),
    ):
        msg = exit_message(call)
        chk(f"{name}: the authorised pass is admitted to test", msg is None, (msg or "")[:70])
finally:
    G.FREEZE_TOKEN_FILE = token_file
    os.environ.pop(G.ENV_VAR, None)


class StopPath:
    """Stands in for the bundle and maps folders: the first use after the refusal stops."""

    def __truediv__(self, other):
        raise Sentinel()

    def __getattr__(self, name: str):
        raise Sentinel()


# main() points the bundle folders at --bundle-dir; that step is replaced so they stay StopPath.
lb_patches = [(lb, "_set_bundle_dir", lambda path: None)]
lb_patches += [(lb, "BUNDLE", StopPath()), (lb, "MAPS", StopPath()), (lb, "_require_torch", stop)]
bundle_wiring = (
    (
        "localiser_bundle cache --split test --allow-test",
        lb.main,
        lb_patches,
        ["localiser_bundle", "cache", "--tag", "t", "--weights", "w.weights.h5"]
        + ["--split", "test", "--allow-test"],
    ),
    (
        "localiser_bundle apply --split test --allow-test",
        lb.main,
        lb_patches,
        ["localiser_bundle", "apply", "--split", "test", "--allow-test", "--boxes-dir", "boxes"],
    ),
    (
        "localiser_bundle write-torch-maps --split test --allow-test",
        lb.main,
        lb_patches,
        ["localiser_bundle", "write-torch-maps", "--weights", "w.pt", "--tag", "t"]
        + ["--split", "test", "--allow-test", "--tensors-dir", "tensors"],
    ),
    (
        "localiser_bundle recheck --split test",
        lb.main,
        lb_patches,
        ["localiser_bundle", "recheck", "--tag", "t", "--split", "test"],
    ),
    (
        "localiser_bundle torch-member-boxes --split test",
        lb.main,
        lb_patches,
        ["localiser_bundle", "torch-member-boxes", "--weights", "w.pt", "--encoder", "e"]
        + ["--tensors-dir", "tensors", "--split", "test"],
    ),
    (
        "analyse_ladder errors --split test --allow-test",
        lambda: al.main(
            ["errors", "--boxes-dir", "boxes", "--split", "test", "--allow-test"]
            + ["--out", "out", "--out-figure", "fig"]
        ),
        [(al, "build_frame", stop)],
        None,
    ),
)
for name, call, patches, argv in bundle_wiring:
    out, left = run_entry(call, patches, argv)
    chk(
        f"{name}: the command line reaches the refusal",
        out.startswith("REFUSED") and not left,
        f"runtime; {out[:60]}; files {left}",
    )

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)

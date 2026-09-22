#!/usr/bin/env python
"""The checks that refuse: ledger, weights, preprocessing, launchers.
Nothing here computes a result for the report; each subcommand catches one way a
reported number could be wrong, and refuses rather than reporting and continuing.

  ledger-snapshot record the digest of every prediction table in the ledger
  ledger-check refuse if any table changed since the snapshot
  verify-test-pass the test pass's own outputs against what the protocol requires
  results-manifest hash and size of every result file
  apply-amendments the pre-registration's amendment blocks, in their declared order
  weights-roundtrip rebuild each model, load its checkpoint, score it, and compare
  preproc-audit the preprocessing path, end to end, on real rows
  weights-boxes boxes derived from weights against the frozen box tables
  control-maps a candidate's cached maps against the live model's
  launcher-flags every launcher's flags parse, and name what they claim to
  cascade-handoff what the promotion cascade writes is what the test pass reads
  recache-diff stored maps against a re-cache through the current eager path
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import hashlib
import importlib.util
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

LAUNCHER_DIR = ROOT / "notebooks" / "scripts"


def _discovered(what: str, n: int, where, *, required: bool = False) -> int:
    """Print what this run found and where, then refuse an empty required discovery:
    zero inputs is the absence of a measurement, not a pass."""
    places = where if isinstance(where, (list, tuple)) else [where]
    loc = ", ".join(str(p) for p in places)
    print(f"[discovered] {n} {what} under {loc}")
    if required and n == 0:
        raise SystemExit(
            f"REFUSED: no {what} found under {loc}. An empty discovery is not a "
            f"pass — this run checked nothing."
        )
    return n


def _heavy() -> None:
    """Import TensorFlow, the model registry and the loader into this module, so the
    ledger, amendment, launcher and cascade checks never pay for them."""
    import h5py
    import pydicom
    import tensorflow as tf
    from tensorflow import keras

    from src import data_loader
    from src.localise import Letterbox
    from src.models import (
        MODEL_REGISTRY,
        build_fusion,
        build_model,
        get_input_shape,
        unfreeze_for_finetuning,
    )

    tf.random.set_seed(config.SEED)
    globals().update(
        h5py=h5py,
        pydicom=pydicom,
        tf=tf,
        keras=keras,
        data_loader=data_loader,
        Letterbox=Letterbox,
        MODEL_REGISTRY=MODEL_REGISTRY,
        build_fusion=build_fusion,
        build_model=build_model,
        get_input_shape=get_input_shape,
        unfreeze_for_finetuning=unfreeze_for_finetuning,
    )


# ledger_hash
RESULTS = ROOT / "outputs" / "results"

SNAP = ROOT / "outputs" / "results" / "ledger_hashes.json"

PATTERN = "*_predictions_val.csv"


def sha256(p: Path, cap: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while b := fh.read(cap):
            h.update(b)
    return h.hexdigest()


def current() -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(RESULTS.glob(PATTERN))}


def ledger_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("action", choices=["snapshot", "check"])
    ap.add_argument(
        "--strict-new",
        action="store_true",
        help="treat a NEW table as a failure too (a landing adds tables "
        "legitimately; the test pass should not)",
    )
    ap.add_argument(
        "--declared",
        nargs="*",
        default=None,
        metavar="TABLE",
        help="the tables this run says it will rewrite. A declared table is "
        "reported and permitted; an undeclared change still fails.",
    )
    ap.add_argument(
        "--written-since",
        type=float,
        default=None,
        metavar="EPOCH",
        help="unix time this run started. Each declared table must have been "
        "written at or after it; an older one means the step that declared it "
        "did not run.",
    )
    args = ap.parse_args(argv)
    now = current()
    # A snapshot of nothing would make every later check pass against nothing, and
    # that check is the last gate before any test number is quoted.
    _discovered(
        f"prediction tables matching {PATTERN}",
        len(now),
        RESULTS,
        required=args.action == "snapshot",
    )

    if args.action == "snapshot":
        SNAP.write_text(
            json.dumps(
                {
                    "taken_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "pattern": PATTERN,
                    "n": len(now),
                    "hashes": now,
                },
                indent=2,
            )
        )
        print(f"snapshot: {len(now)} tables -> {SNAP.relative_to(ROOT)}")
        return 0

    if not SNAP.exists():
        print(f"no snapshot at {SNAP.relative_to(ROOT)} — run `snapshot` first", file=sys.stderr)
        return 2
    d = json.loads(SNAP.read_text())
    was = d["hashes"]
    changed = sorted(k for k in was if k in now and now[k] != was[k])
    missing = sorted(k for k in was if k not in now)
    new = sorted(k for k in now if k not in was)
    print(f"snapshot {d['taken_utc']}: {d['n']} tables | now: {len(now)}")
    # Every name, never a slice: this is a gate before the test pass and NEW is the
    # category the test tables arrive in, so a reader must see which files appeared.
    for label, items in (("CHANGED", changed), ("MISSING", missing), ("NEW", new)):
        print(f"  {label:<8}: {len(items)}")
        for name in items:
            print(f"      {name}")
    # A declaration is a claim to be verified, not an exemption: each declared table's
    # fate is reported, and a change outside the set still fails.
    undeclared, unwritten = changed, []
    if args.declared is not None:
        declared = sorted(set(args.declared))
        undeclared = sorted(set(changed) - set(declared))
        print(f"\n  declared to change : {len(declared)}")
        for name in declared:
            f = RESULTS / name
            wrote = (
                args.written_since is not None
                and f.exists()
                and f.stat().st_mtime >= args.written_since
            )
            if args.written_since is None:
                mark = (
                    "changed"
                    if name in changed
                    else ("NEW" if name in new else "unchanged (write not verified)")
                )
            elif not f.exists():
                mark = "NOT WRITTEN (absent)"
            elif not wrote:
                mark = "NOT WRITTEN (older than this run)"
            else:
                mark = "rewritten" if name in changed or name in new else "rewritten, identical"
            if mark.startswith("NOT WRITTEN"):
                unwritten.append(name)
            print(f"      {name}  [{mark}]")
        print(f"  changed, undeclared: {len(undeclared)}")
        for name in undeclared:
            print(f"      {name}")

    bad = bool(undeclared or missing or unwritten) or (args.strict_new and bool(new))
    if bad:
        if unwritten and not undeclared:
            print(
                "\nDECLARATION NOT MET — a table this run declared was not written. "
                "The step that declared it did not run."
            )
        else:
            print(
                "\nLEDGER CHANGED — a reference table was rewritten, deleted, or "
                "(with --strict-new) added."
            )
        return 1
    if args.declared is not None:
        print(
            f"\nledger intact: {len(changed)} declared table(s) changed as stated; every "
            f"other snapshotted table is byte-identical."
        )
    else:
        print("\nledger intact: every snapshotted table is byte-identical.")
    return 0


# verify_test_pass
R = ROOT / "outputs" / "results"

ARCH = "vgg16"

# Tags are derived as run_test_pass.sh derives them: the ledger suffix from config,
# the rung-3 suffix from the promotion pointer.
LEDGER = config.LEDGER_SUFFIX

POINTER = ROOT / "refs" / "promoted_pointer.json"

R3 = json.loads(POINTER.read_text())["rung3_suffix"] if POINTER.exists() else "_bundle"

RUNGS = {
    1: f"rung1_oracle{LEDGER}",
    2: f"rung2_oracle{LEDGER}",
    3: f"rung3_unet{R3}",
    4: f"rung4_cam{LEDGER}",
    5: f"rung5_jitter{R3}",
    6: f"rung6_shuffled{LEDGER}",
    7: f"rung7_whole{LEDGER}",
}

FAMILY_A = ["vgg16", "resnet50", "densenet121", "efficientnet"]

CONTEXT = [
    f"{ARCH}_fusion_context{config.CONTEXT_FUSION_SUFFIX}",
    f"{ARCH}_fusion_control_all{config.CONTEXT_FUSION_SUFFIX}",
]  # every lesion

PASS_TAGS = ROOT / "outputs" / "logs" / ".pass_tags.txt"  # written by the pass's pre-flight

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def table(tag, split):
    p = R / f"{tag}_predictions_{split}.csv"
    return pd.read_csv(p) if p.exists() else None


def verify_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument(
        "--expect-n",
        type=int,
        default=None,
        help="expected lesion rows; default 243 for both partitions",
    )
    args = ap.parse_args(argv)
    s = args.split
    n_expect = args.expect_n or 243
    print(f"--- verifying the {s} pass ---")
    where = (
        "from " + str(POINTER.relative_to(ROOT))
        if POINTER.exists()
        else "no pointer: pre-promotion default"
    )
    print(f"  tags: ledger {LEDGER}, rung-3 suffix {R3} ({where})\n")

    # 1. every ladder rung produced a table, with the right number of rows
    print("1. ladder tables")
    seen = {}
    for k, suffix in RUNGS.items():
        t = table(f"{ARCH}_crop_{suffix}", s)
        if not chk(f"rung {k} ({suffix}) table exists", t is not None):
            continue
        seen[k] = t
        chk(f"rung {k} has {n_expect} rows", len(t) == n_expect, f"{len(t)}")

    # 2. schema and content, on every table found
    print("\n2. schema and content")
    need = {"y_true", "y_prob", "partition"}
    for k, t in seen.items():
        chk(
            f"rung {k} carries {sorted(need)}",
            need <= set(t.columns),
            (f"missing {sorted(need - set(t.columns))}" if not need <= set(t.columns) else ""),
        )
        if need <= set(t.columns):
            chk(
                f"rung {k} y_prob within [0,1]",
                bool(t["y_prob"].between(0, 1).all()),
                f"range [{t['y_prob'].min():.4f}, {t['y_prob'].max():.4f}]",
            )
            chk(f"rung {k} y_true is binary", set(t["y_true"].unique()) <= {0, 1})
            chk(
                f"rung {k} partition column is all '{s}'",
                bool((t["partition"].astype(str).str.lower() == s).all()),
                str(t["partition"].unique()[:3]),
            )
            chk(f"rung {k} no null probabilities", int(t["y_prob"].isna().sum()) == 0)

    # 3. the rungs are not accidentally the same model
    print("\n3. rungs are distinct")
    if 3 in seen and 7 in seen:
        same = seen[3]["y_prob"].round(6).equals(seen[7]["y_prob"].round(6))
        chk(
            "rung 3 and rung 7 differ",
            not same,
            "identical probabilities would mean the box source was ignored",
        )
    if 2 in seen and 3 in seen:
        chk(
            "rung 2 and rung 3 differ",
            not seen[2]["y_prob"].round(6).equals(seen[3]["y_prob"].round(6)),
        )

    # 4. Family A
    print("\n4. architecture family")
    for a in FAMILY_A:
        t = table(f"{a}_crop_m020{LEDGER}", s)
        chk(
            f"{a}_crop_m020{LEDGER} table exists",
            t is not None,
            "" if t is not None else "Family A incomplete",
        )

    # 4a. The context contrast is on every lesion, not the view pairs.
    print("\n4a. context fusion and its control")
    for c in CONTEXT:
        t = table(c, s)
        chk(
            f"{c} has {n_expect} rows",
            t is not None and len(t) == n_expect,
            "no table" if t is None else f"{len(t)}",
        )

    # 4b. Every tag in the pass's own list (written at its pre-flight) has a table.
    print("\n4b. every tag the pass declared produced a table")
    tags = (
        [x.strip() for x in PASS_TAGS.read_text().splitlines() if x.strip()]
        if PASS_TAGS.exists()
        else []
    )
    print(f"  [discovered] {len(tags)} tags in {PASS_TAGS.relative_to(ROOT)}")
    if chk(
        "the pass's tag list exists and is not empty",
        bool(tags),
        ("" if tags else "the pass writes it at pre-flight; without it nothing can be checked"),
    ):
        missing = [t for t in tags if table(t, s) is None]
        chk(
            f"all {len(tags)} declared tags have a {s} table",
            not missing,
            f"{len(missing)} missing: {missing[:4]}" if missing else "",
        )
        wrong = []
        for t in tags:
            tb = table(t, s)
            if (
                tb is not None
                and "partition" in tb.columns
                and set(tb["partition"].astype(str).str.lower()) != {s}
            ):
                wrong.append(t)
        chk(f"every declared table's partition column is '{s}'", not wrong, str(wrong[:4]))

    # 5. the comparison reports the pre-registration asks for
    print("\n5. comparison reports")
    # src.compare redirects its output to the smoke folder on the val split, so a
    # dry run's reports do not sit beside the real ones. Look in both places.
    VAL_OUT = ROOT / "outputs" / "results" / "validation"
    for f in (
        f"comparison_report_{s}.json",
        f"comparison_architectures_{s}.json",
        f"comparison_fusion_declared_{s}.json",
        f"comparison_fusion_context_{s}.json",
    ):
        cands = [R / f, VAL_OUT / f]
        p = next((c for c in cands if c.exists()), cands[0])
        ok = chk(
            f"{f} exists",
            p.exists(),
            (
                f"found at {p.relative_to(ROOT)}"
                if p.exists()
                else f"looked in {[str(c.relative_to(ROOT)) for c in cands]}"
            ),
        )
        if ok:
            d = json.loads(p.read_text())
            chk(f"{f} names the split", d.get("split") == s, str(d.get("split")))
            if "primary" in d:
                pr = d["primary"]
                chk(
                    "primary endpoint has both AUCs",
                    pr.get("auc_model") is not None and pr.get("auc_birads") is not None,
                    f"model {pr.get('auc_model')}, birads {pr.get('auc_birads')}",
                )
                chk(
                    "primary endpoint n matches n_expected",
                    pr.get("n") == pr.get("n_expected"),
                    f"{pr.get('n')} vs {pr.get('n_expected')}",
                )

    # 6. hygiene: a val run must not have written test tables
    print("\n6. hygiene")
    if s == "val":
        strays = sorted(p.name for p in R.glob("*_predictions_test.csv"))
        chk("no test prediction table exists", not strays, f"{len(strays)} found: {strays[:3]}")
    # A table whose tag names one split and whose suffix names the other.
    crossed = sorted(
        p.name
        for p in list(R.glob("*_test_predictions_val.csv"))
        + list(R.glob("*_val_predictions_test.csv"))
    )
    chk(
        "no table carries one split in its tag and the other in its suffix",
        not crossed,
        f"{len(crossed)} found: {crossed[:3]}",
    )

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL[:8]) + (" ..." if len(FAIL) > 8 else ""))
        return 1
    print(f"the {s} pass is complete and internally consistent.")
    return 0


# results_manifest
HASHED = [
    ROOT / "artifacts",
    config.RESULTS_DIR,
    config.FIGURES_DIR,
    ROOT / "docs",
    ROOT / "requirements.txt",
    ROOT / "src" / "config.py",
    ROOT / "notebooks" / "scripts" / "run_test_pass.sh",
]

LISTED = [config.WEIGHTS_DIR]

SKIP_DIRS = {"_superseded", "__pycache__", ".ipynb_checkpoints"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(base: Path):
    if base.is_file():
        yield base
        return
    for p in sorted(base.rglob("*")):
        if (
            p.is_file()
            and not (set(p.relative_to(base).parts[:-1]) & SKIP_DIRS)
            and not p.name.startswith(".")
        ):
            yield p


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unavailable"


def environment() -> dict:
    import numpy, scipy, sklearn, tensorflow as tf
    import keras

    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "git_uncommitted_paths": len(_git("status", "--porcelain").splitlines()),
        "seed": config.SEED,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def manifest_main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args(argv)
    rows = []
    for base in HASHED:
        if not base.exists():
            continue
        for p in _walk(base):
            if p.name.startswith("results_manifest"):
                continue
            st = p.stat()
            rows.append(
                dict(
                    path=str(p.relative_to(ROOT)),
                    bytes=st.st_size,
                    modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    sha256=_sha(p),
                )
            )
    for base in LISTED:
        for p in _walk(base):
            st = p.stat()
            rows.append(
                dict(
                    path=str(p.relative_to(ROOT)),
                    bytes=st.st_size,
                    modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    sha256="",
                )
            )
    _discovered("files", len(rows), [*HASHED, *LISTED], required=True)
    df = pd.DataFrame(rows)
    env = environment()
    out_json = config.OUTPUTS_DIR / "results_manifest.json"
    out_csv = config.OUTPUTS_DIR / "results_manifest.csv"
    out_json.write_text(
        json.dumps({"environment": env, "n_files": int(len(df)), "files": rows}, indent=2)
    )
    df.to_csv(out_csv, index=False)
    hashed = int((df["sha256"] != "").sum())
    print(
        f"[manifest] {len(df)} files ({hashed} hashed, {len(df) - hashed} listed) | "
        f"TF {env['tensorflow']} keras {env['keras']} numpy {env['numpy']} | git {env['git_head']} "
        f"({env['git_uncommitted_paths']} uncommitted paths)"
    )
    print(f"-> {out_json}")
    print(f"-> {out_csv}")


# apply_amendments
PENDING = ROOT / "docs" / "preregistration_amendments_pending.md"

AMEND_PREREG = ROOT / "docs" / "test_pass_preregistration.md"

MARK = "<!-- amendments applied at D3 -->"


def blocks(text: str) -> dict[str, tuple[str, str]]:
    """`{'7a': (title, body)}` for every `## N.` / `### Na.` / `#### Nx.` heading; the
    number must be followed by a period, so "#### 12c result" is not a block."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^#{2,4}\s+(\d+[a-z]?)\.\s+(.*)$", line)
        if m:
            if cur:
                out[cur[0]] = (cur[1], "\n".join(buf).strip())
            cur, buf = (m.group(1), m.group(2).strip()), []
            continue
        if line.startswith("# Application order"):
            if cur:
                out[cur[0]] = (cur[1], "\n".join(buf).strip())
            cur = None
        if cur:
            buf.append(line)
    if cur:
        out[cur[0]] = (cur[1], "\n".join(buf).strip())
    return out


def order(text: str) -> list[tuple[int, list[str], str]]:
    """The application-order table, as (position, [block numbers], rationale); a row may
    group several blocks, so every reference in it is collected, not just the first."""
    seg = text[text.find("# Application order") :]
    out = []
    for m in re.finditer(r"^\|\s*(\d+)\s*\|([^|]*)\|([^|]*)\|", seg, re.M):
        refs = re.findall(r"§(\d+[a-z]?)", m.group(2))
        if refs:
            out.append((int(m.group(1)), refs, m.group(3).strip()))
    return out


def assemble(B: dict[str, tuple[str, str]], O: list[tuple[int, list[str], str]]) -> str:
    """The appendix: the pending file's blocks in its order table's order. freeze.py
    shares this definition to check the pre-registration is byte-identical to it."""
    parts = [
        MARK,
        "",
        "# Appendix — amendments, applied in the order fixed before the freeze",
        "",
        "Assembled by `notebooks/scripts/apply_amendments.py` from",
        "`docs/preregistration_amendments_pending.md`. The order is the one that file's own",
        "table fixed; several blocks restate figures an earlier block establishes, so it is",
        "load-bearing rather than cosmetic. Blocks written after a result was seen say so in",
        "their first line and disclose the order of events.",
        "",
    ]
    for pos, refs, why in O:
        present = [n for n in refs if n in B]
        if not present:
            continue
        head = ", ".join("§" + n for n in present)
        parts += [
            f"## {pos}. {head} — {B[present[0]][0]}",
            "",
            (f"*Why here: {why}*" if why else ""),
            "",
        ]
        for n in present:
            title, body = B[n]
            if len(present) > 1:
                parts += [f"### §{n} — {title}", ""]
            parts += [body, ""]
    return "\n".join(parts)


def document(cur: str, appendix: str) -> str:
    """The pre-registration as --apply writes it: the protocol above the marker, then
    the appendix, replacing any appendix already present."""
    if MARK in cur:
        return cur[: cur.index(MARK)] + appendix
    return cur.rstrip() + "\n\n" + appendix


def amend_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true", help="write; requires the freeze")
    ap.add_argument("--out", default=None, help="write the assembled appendix here instead")
    args = ap.parse_args(argv)

    if not PENDING.exists():
        print(f"missing {PENDING}", file=sys.stderr)
        return 2
    text = PENDING.read_text()
    B, O = blocks(text), order(text)
    print(f"blocks found : {len(B)}  ({', '.join('§' + k for k in B)})")
    print(f"order entries: {len(O)}")

    # 1. consistency
    print("\n--- order table vs blocks ---")
    ok = True
    ordered = [b for _, refs, _ in O for b in refs]
    dupes = {b for b in ordered if ordered.count(b) > 1}
    missing_from_order = [k for k in B if k not in ordered]
    missing_blocks = [b for b in ordered if b not in B]
    for label, bad in (
        ("duplicated in the order", sorted(dupes)),
        ("block exists but is not ordered", sorted(missing_from_order)),
        ("ordered but no such block", sorted(missing_blocks)),
    ):
        if bad:
            ok = False
            print(f"  [FAIL] {label}: {['§'+x for x in bad]}")
        else:
            print(f"  [PASS] none {label}")
    positions = [p for p, _, _ in O]
    contiguous = positions == list(range(1, len(positions) + 1))
    print(f"  [{'PASS' if contiguous else 'FAIL'}] positions are 1..{len(positions)} with no gaps")
    ok &= contiguous

    # 2. assemble
    appendix = assemble(B, O)

    # 3. diff
    print("\n--- diff against the pre-registration ---")
    cur = AMEND_PREREG.read_text() if AMEND_PREREG.exists() else ""
    new = document(cur, appendix)
    note = (
        "an appendix is already present; it would be REPLACED"
        if MARK in cur
        else "the appendix would be APPENDED"
    )
    print(f"  {note}")
    d = list(
        difflib.unified_diff(
            cur.splitlines(),
            new.splitlines(),
            "test_pass_preregistration.md (current)",
            "test_pass_preregistration.md (after)",
            lineterm="",
            n=1,
        )
    )
    add = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
    print(f"  +{add} lines, -{rem} lines from {len(O)} amendment blocks")
    for line in d[:24]:
        print("   " + line[:118])
    if len(d) > 24:
        print(f"   ... {len(d) - 24} more diff lines not shown")

    if args.out:
        Path(args.out).write_text(appendix)
        print(f"\n-> {args.out}")

    # 4. write only at the freeze
    if not args.apply:
        print("\nDRY RUN: nothing written. Re-run with --apply at the freeze.")
        return 0 if ok else 1
    sys.path.insert(0, str(ROOT))
    from src.test_pass_guard import freeze_token

    if freeze_token() is not None:
        print(
            "\nREFUSING TO APPLY: the freeze token already exists, so the test partition "
            "is readable. The protocol is fixed before the partition opens, never after. "
            "Use freeze.py, which creates the token once this appendix is in place."
        )
        return 1
    if not ok:
        print("\nREFUSING TO APPLY: the order table and the blocks disagree.")
        return 1
    AMEND_PREREG.write_text(new)
    print(f"\napplied -> {AMEND_PREREG}")
    return 0


# check_weights_roundtrip


ROUNDTRIP_OUT_DIR: Path = config.RESULTS_DIR / "weights_roundtrip"

_TRANSFER_ARCHS: frozenset[str] = frozenset({"vgg16", "resnet50", "densenet121", "efficientnet"})


def arch_of(tag: str) -> str:
    """The architecture a tag was built from: the longest registry key that prefixes
    it, so a shorter key can never shadow a longer one."""
    for name in sorted(MODEL_REGISTRY, key=len, reverse=True):
        if tag == name or tag.startswith(name + "_"):
            return name
    raise ValueError(f"Cannot infer an architecture from tag '{tag}'")


def build_for_tag(tag: str, finetune_partition: bool = False) -> keras.Model:
    """Build the uncompiled model a tag's checkpoint belongs to; finetune_partition
    gives it the trainable partition the transfer runs saved their checkpoint under."""
    arch = arch_of(tag)
    shape = get_input_shape(arch)
    model = build_fusion(arch, shape) if is_fusion(tag) else build_model(arch, input_shape=shape)
    if finetune_partition and arch in _TRANSFER_ARCHS:
        unfreeze_for_finetuning(model, backbone_name=arch)
    return model


def is_fusion(tag: str) -> bool:
    """Whether a tag is a two-crop fusion model, decided from the head kernel: fusion
    doubles the input width, and the single-crop controls also carry `_fusion_`."""
    path = weights_path(tag)
    if not path.exists():
        return "_fusion_" in tag
    idx = file_variable_index(path)
    head = [v for k, v in idx.items() if "functional" not in k and v.ndim == 2 and v.shape[1] > 1]
    if not head:
        return "_fusion_" in tag
    pooled = int(head[0].shape[0])
    single = build_model(arch_of(tag), input_shape=get_input_shape(arch_of(tag)))
    expected = int(single.get_layer("head_dense").kernel.shape[0])
    return pooled == 2 * expected


def weights_path(tag: str) -> Path:
    return config.WEIGHTS_DIR / f"{tag}_best.weights.h5"


def known_tags() -> list[str]:
    """Every tag with both a canonical checkpoint and a validation table, which is the
    set whose logged metric can be checked against what the checkpoint delivers."""
    tags = []
    for w in sorted(config.WEIGHTS_DIR.glob("*_best.weights.h5")):
        tag = w.name[: -len("_best.weights.h5")]
        if (config.RESULTS_DIR / f"{tag}_predictions_val.csv").exists():
            try:
                arch_of(tag)
            except ValueError:
                continue
            tags.append(tag)
    return tags


def file_variable_index(path: str | Path) -> dict[str, np.ndarray]:
    """Every layer variable of a `.weights.h5` file, keyed by its h5 path. Only the
    `layers/` subtree is read; optimizer state never reaches an uncompiled model."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as f:

        def visit(name: str) -> None:
            if name.startswith("layers/") and isinstance(f[name], h5py.Dataset):
                out[name] = np.asarray(f[name])

        f.visit(lambda n: visit(n))
    return out


def saved_variable_index(model: keras.Model) -> dict[str, np.ndarray]:
    """Serialise a model through Keras and read its variables back, so the key space
    is by construction the one `load_weights` matches against."""
    tmp = Path(tempfile.mkdtemp(prefix="wroundtrip_")) / "probe.weights.h5"
    try:
        model.save_weights(str(tmp))
        return file_variable_index(tmp)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _layer_group(key: str) -> str:
    return key.rsplit("/vars/", 1)[0]


def _var_slot(key: str) -> int:
    return int(key.rsplit("/vars/", 1)[1])


def _key_comparison(
    model_idx: dict[str, np.ndarray], file_idx: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Compare key spaces and shapes between a model and a checkpoint."""
    mk, fk = set(model_idx), set(file_idx)
    shape_mismatch = {
        k: {"model": list(model_idx[k].shape), "file": list(file_idx[k].shape)}
        for k in sorted(mk & fk)
        if model_idx[k].shape != file_idx[k].shape
    }
    return {
        "n_model_variables": len(mk),
        "n_file_variables": len(fk),
        "missing_from_file": sorted(mk - fk),
        "extra_in_file": sorted(fk - mk),
        "shape_mismatch": shape_mismatch,
    }


def _mismatches(post_idx: dict[str, np.ndarray], file_idx: dict[str, np.ndarray]) -> dict[str, Any]:
    """Variables whose value after load differs from the file."""
    bad = {}
    for k in sorted(set(post_idx) & set(file_idx)):
        a, b = post_idx[k], file_idx[k]
        if a.shape != b.shape:
            bad[k] = {"reason": "shape", "model": list(a.shape), "file": list(b.shape)}
        elif not np.array_equal(a, b):
            bad[k] = {
                "reason": "value",
                "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
            }
    return bad


def _unloaded_variables(
    init_idx: dict[str, np.ndarray],
    post_idx: dict[str, np.ndarray],
    file_idx: dict[str, np.ndarray],
) -> list[str]:
    """Variables still holding their initialised value after load; one counts only if
    the file's value for that key differs from the initialisation."""
    stuck = []
    for k in sorted(set(init_idx) & set(post_idx) & set(file_idx)):
        if init_idx[k].shape != post_idx[k].shape:
            continue
        if not np.array_equal(init_idx[k], post_idx[k]):
            continue
        if file_idx[k].shape == init_idx[k].shape and np.array_equal(file_idx[k], init_idx[k]):
            continue
        stuck.append(k)
    return stuck


def _variable_order(
    model: keras.Model, post_idx: dict[str, np.ndarray], file_idx: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Per-layer variable order in the model against the file: every model slot's
    post-load value is searched for among the file's slots of the same group."""
    roles = _model_variable_roles(model)
    groups = sorted({_layer_group(k) for k in set(post_idx) & set(file_idx)})
    out: dict[str, Any] = {}
    for g in groups:
        m_slots = sorted((k for k in post_idx if _layer_group(k) == g), key=_var_slot)
        f_slots = sorted((k for k in file_idx if _layer_group(k) == g), key=_var_slot)
        entries = []
        reordered = False
        for k in m_slots:
            i = _var_slot(k)
            matches = [
                _var_slot(fk)
                for fk in f_slots
                if file_idx[fk].shape == post_idx[k].shape
                and np.array_equal(file_idx[fk], post_idx[k])
            ]
            if matches and i not in matches:
                reordered = True
            entries.append(
                {
                    "slot": i,
                    "role": roles.get(g, {}).get(i, "unknown"),
                    "shape": list(post_idx[k].shape),
                    "matches_file_slots": matches,
                }
            )
        out[g] = {"reordered": reordered, "slots": entries}
    return out


def _model_variable_roles(model: keras.Model) -> dict[str, dict[int, str]]:
    """Each layer group as `{slot index: variable path}`, in the order Keras
    serialises a layer, trainable variables first, so slots line up with `vars/<i>`."""
    roles: dict[str, dict[int, str]] = {}

    def walk(layer: keras.layers.Layer) -> None:
        trainable = list(getattr(layer, "_trainable_variables", []))
        non_trainable = list(getattr(layer, "_non_trainable_variables", []))
        ordered = trainable + non_trainable
        if ordered:
            roles[layer.name] = {
                i: getattr(v, "path", getattr(v, "name", f"var_{i}")) for i, v in enumerate(ordered)
            }
        for sub in getattr(layer, "layers", []):
            walk(sub)

    for lyr in model.layers:
        walk(lyr)
    return roles


def logged_vs_table(tag: str) -> dict[str, Any]:
    """The history's best `val_auc` against the exact validation AUC: Keras's binned
    metric at the best epoch, versus scikit-learn's AUC on the table's probabilities."""
    out: dict[str, Any] = {"tag": tag}
    hist = None
    for suffix in ("_finetune_history.json", "_history.json"):
        p = config.RESULTS_DIR / f"{tag}{suffix}"
        if p.exists():
            hist = json.loads(p.read_text())
            out["history_file"] = p.name
            break
    if hist and "val_auc" in hist:
        v = [float(x) for x in hist["val_auc"]]
        out["history_best_val_auc"] = round(max(v), 4)
        out["history_best_epoch"] = int(np.argmax(v)) + 1
        out["history_n_epochs"] = len(v)
    table = config.RESULTS_DIR / f"{tag}_predictions_val.csv"
    if table.exists():
        df = pd.read_csv(table)
        df = df[df["partition"] == "val"]
        out["table_exact_val_auc"] = round(float(roc_auc_score(df["y_true"], df["y_prob"])), 4)
        out["table_n_rows"] = int(len(df))
    if "history_best_val_auc" in out and "table_exact_val_auc" in out:
        d = out["table_exact_val_auc"] - out["history_best_val_auc"]
        out["table_minus_history"] = round(d, 4)
        out["reproduces"] = bool(abs(d) <= 0.005)
    return out


def roundtrip_report(tag: str, finetune_partition: bool = False) -> dict[str, Any]:
    """Build a tag's model, load its checkpoint and report what the load delivered.
    With finetune_partition the model carries the end-of-fine-tuning trainable split."""
    path = weights_path(tag)
    if not path.exists():
        raise FileNotFoundError(path)

    model = build_for_tag(tag, finetune_partition=finetune_partition)
    init_idx = saved_variable_index(model)
    file_idx = file_variable_index(path)
    keys = _key_comparison(init_idx, file_idx)

    model.load_weights(str(path))
    post_idx = saved_variable_index(model)

    order = _variable_order(model, post_idx, file_idx)
    mismatches = _mismatches(post_idx, file_idx)
    unloaded = _unloaded_variables(init_idx, post_idx, file_idx)

    return {
        "tag": tag,
        "weights_file": str(path),
        "arch": arch_of(tag),
        "finetune_partition": finetune_partition,
        "keys": keys,
        "mismatches": mismatches,
        "n_mismatches": len(mismatches),
        "unloaded": unloaded,
        "n_unloaded": len(unloaded),
        "reordered_groups": sorted(g for g, v in order.items() if v["reordered"]),
        "order": order,
        "metrics": logged_vs_table(tag),
    }


def diff_reports(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def summary(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "arch": r["arch"],
            "n_model_variables": r["keys"]["n_model_variables"],
            "n_file_variables": r["keys"]["n_file_variables"],
            "missing_from_file": r["keys"]["missing_from_file"],
            "extra_in_file": r["keys"]["extra_in_file"],
            "shape_mismatch": sorted(r["keys"]["shape_mismatch"]),
            "n_mismatches": r["n_mismatches"],
            "n_unloaded": r["n_unloaded"],
            "reordered_groups": r["reordered_groups"],
            "reproduces": r["metrics"].get("reproduces"),
            "table_exact_val_auc": r["metrics"].get("table_exact_val_auc"),
            "history_best_val_auc": r["metrics"].get("history_best_val_auc"),
            "table_minus_history": r["metrics"].get("table_minus_history"),
        }

    sa, sb = summary(a), summary(b)
    differing = {k: {a["tag"]: sa[k], b["tag"]: sb[k]} for k in sa if sa[k] != sb[k]}
    return {
        "tag_a": a["tag"],
        "tag_b": b["tag"],
        "identical_fields": sorted(k for k in sa if sa[k] == sb[k]),
        "differing_fields": differing,
        "structurally_identical": not differing,
    }


def compare_load_modes(tag: str) -> dict[str, Any]:
    """Load one checkpoint into two models differing only in trainable partition:
    Keras serialises trainable variables first, so a differing split could reorder them.
    """
    path = weights_path(tag)
    arch = arch_of(tag)
    if arch not in _TRANSFER_ARCHS:
        return {
            "tag": tag,
            "arch": arch,
            "applicable": False,
            "reason": "scratch model has no fine-tuning partition",
        }

    frozen = build_for_tag(tag, finetune_partition=False)
    frozen.load_weights(str(path))
    idx_frozen = saved_variable_index(frozen)

    unfrozen = build_for_tag(tag, finetune_partition=True)
    unfrozen.load_weights(str(path))
    idx_unfrozen = saved_variable_index(unfrozen)

    differing = []
    for k in sorted(set(idx_frozen) & set(idx_unfrozen)):
        a, b = idx_frozen[k], idx_unfrozen[k]
        if a.shape != b.shape or not np.array_equal(a, b):
            differing.append(k)

    # The eager call, never predict(): the compiled path is wrong on this machine.
    probe = (
        np.random.default_rng(config.SEED)
        .normal(size=(2, *get_input_shape(arch)))
        .astype("float32")
    )
    inputs = [probe, probe] if "_fusion_" in tag else probe
    p_frozen = np.asarray(frozen(inputs, training=False)).ravel()
    p_unfrozen = np.asarray(unfrozen(inputs, training=False)).ravel()

    return {
        "tag": tag,
        "arch": arch,
        "applicable": True,
        "n_variables": len(idx_frozen),
        "n_differing": len(differing),
        "differing_keys": differing[:50],
        "identical": not differing,
        "probe_pred_frozen_partition": [round(float(x), 6) for x in p_frozen],
        "probe_pred_finetune_partition": [round(float(x), 6) for x in p_unfrozen],
        "probe_max_abs_diff": round(float(np.max(np.abs(p_frozen - p_unfrozen))), 8),
    }


def check_device_consistency(
    archs: tuple[str, ...] = (
        "vgg16",
        "resnet50",
        "densenet121",
        "efficientnet",
        "baseline",
        "scaled",
        "regularised",
    ),
    batch: int = 8,
) -> dict[str, Any]:
    """The three inference routes per architecture, on freshly initialised models: the
    CPU eager call is the reference, and on Metal the compiled route disagrees with it.
    """
    rows = []
    for arch in archs:
        ch = MODEL_REGISTRY[arch]["channels"]
        model = build_model(arch, input_shape=(*config.TARGET_SIZE, ch))
        x = (
            np.random.default_rng(config.SEED)
            .normal(size=(batch, *config.TARGET_SIZE, ch))
            .astype("float32")
        )
        with tf.device("/CPU:0"):
            ref = np.asarray(model(x, training=False)).ravel()
        with tf.device("/GPU:0"):
            eager = np.asarray(model(x, training=False)).ravel()
            compiled = model.predict(x, verbose=0).ravel()
        rows.append(
            {
                "arch": arch,
                "gpu_eager_vs_cpu": float(np.max(np.abs(eager - ref))),
                "gpu_compiled_vs_cpu": float(np.max(np.abs(compiled - ref))),
                "compiled_route_broken": bool(np.max(np.abs(compiled - ref)) > 1e-4),
            }
        )
    return {
        "reference": "CPU eager call",
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "rows": rows,
        "n_broken": sum(r["compiled_route_broken"] for r in rows),
    }


def check_localiser_device_consistency() -> dict[str, Any]:
    """The same three-route comparison for the two U-Net localisers, which carry no
    dense head, so it decides whether the box numbers are touched."""
    from src import localise

    rows = []
    x = (
        np.random.default_rng(config.SEED)
        .uniform(0.0, 1.0, size=(2, config.LOC_INPUT_H, config.LOC_INPUT_W, 1))
        .astype("float32")
    )
    for name, builder in (
        ("build_unet", localise.build_unet),
        ("build_unet_pretrained", localise.build_unet_pretrained),
    ):
        model = builder()
        with tf.device("/CPU:0"):
            ref = np.asarray(model(x, training=False))
        with tf.device("/GPU:0"):
            eager = np.asarray(model(x, training=False))
            compiled = model.predict(x, verbose=0)
        rows.append(
            {
                "builder": name,
                "gpu_eager_vs_cpu": float(np.max(np.abs(eager - ref))),
                "gpu_compiled_vs_cpu": float(np.max(np.abs(compiled - ref))),
                "routes_agree_on_gpu": bool(np.allclose(eager, compiled, atol=1e-5)),
            }
        )
    return {"reference": "CPU eager call", "rows": rows}


_BOXES: str = "outputs/results/localiser_bundle/boxes"

ROUTE_CHECK_TAGS: dict[str, dict[str, Any]] = {
    "vgg16_crop_rung3_unet_bundle": {
        "box_source": "unet",
        "boxes_dir": _BOXES,
        "crop_margin": 0.20,
    },
    "vgg16_crop_rung5_jitter_bundle": {
        "box_source": "jitter",
        "boxes_dir": _BOXES,
        "crop_margin": 0.20,
    },
    "vgg16_crop_rung7_whole": {"box_source": "whole", "boxes_dir": None, "crop_margin": 0.20},
    "resnet50_crop_m020": {
        "box_source": None,
        "boxes_dir": None,
        "crop_margin": 0.20,
        "crop_sizing": "mask_margin",
    },
}


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Inverse sigmoid, so the deviation is reported on the model's own scale."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compare_inference_routes(tag: str) -> dict[str, Any]:
    """Score one checkpoint on validation through both routes: the eager call every
    prediction table holds, and `predict`, which is what fit's validation pass ran."""
    from src import data_loader, evaluate as ev
    from src.boxes import provider_for
    from src.models import get_normalisation_mode

    if tag not in ROUTE_CHECK_TAGS:
        raise ValueError(f"no dataset arguments recorded for '{tag}'")
    spec = ROUTE_CHECK_TAGS[tag]
    arch = arch_of(tag)
    config.set_seeds()

    provider, crop_sizing = None, spec.get("crop_sizing", config.CROP_SIZING)
    if spec["box_source"]:
        provider = provider_for(
            spec["box_source"], boxes_dir=spec["boxes_dir"], splits=("train", "val")
        )
        crop_sizing = "box_provider"

    _, val_ds, _, _, _ = data_loader.build_datasets(
        train_csv=config.TRAIN_CSV,
        test_csv=config.TEST_CSV,
        train_img_dir=config.TRAIN_IMG_DIR,
        test_img_dir=config.TEST_IMG_DIR,
        train_roi_dir=config.TRAIN_ROI_DIR,
        test_roi_dir=config.TEST_ROI_DIR,
        target_size=config.TARGET_SIZE,
        batch_size=config.BATCH_SIZE,
        normalisation=get_normalisation_mode(arch),
        augment=False,
        source="crop",
        crop_sizing=crop_sizing,
        crop_margin=spec["crop_margin"],
        box_provider=provider,
    )

    model = build_for_tag(tag)
    model.load_weights(str(weights_path(tag)))

    y, p_eager = ev.collect_predictions(model, val_ds, tta="none")
    p_compiled = model.predict(val_ds, verbose=0).ravel()
    # Parity of the eager path against the CPU on the real rows: this decides whether
    # eager-on-Metal is safe to score with.
    with tf.device("/CPU:0"):
        _, p_eager_cpu = ev.collect_predictions(model, val_ds, tta="none")

    d_prob = np.abs(p_eager - p_compiled)
    d_logit = np.abs(_logit(p_eager) - _logit(p_compiled))
    d_cpu = np.abs(p_eager - p_eager_cpu)
    return {
        "auc_eager_cpu": round(float(roc_auc_score(y, p_eager_cpu)), 4),
        "max_abs_prob_diff_eager_gpu_vs_cpu": float(d_cpu.max()),
        "eager_gpu_matches_cpu_1e5": bool(d_cpu.max() < 1e-5),
        "tag": tag,
        "arch": arch,
        "n": int(len(y)),
        "auc_eager": round(float(roc_auc_score(y, p_eager)), 4),
        "auc_compiled": round(float(roc_auc_score(y, p_compiled)), 4),
        "auc_compiled_minus_eager": round(
            float(roc_auc_score(y, p_compiled) - roc_auc_score(y, p_eager)), 4
        ),
        "max_abs_prob_diff": round(float(d_prob.max()), 6),
        "median_abs_prob_diff": round(float(np.median(d_prob)), 6),
        "max_abs_logit_diff": round(float(d_logit.max()), 6),
        "median_abs_logit_diff": round(float(np.median(d_logit)), 6),
        "table_exact_val_auc": logged_vs_table(tag).get("table_exact_val_auc"),
        "history_best_val_auc": logged_vs_table(tag).get("history_best_val_auc"),
    }


def check_op_specificity() -> dict[str, Any]:
    """Locate the defect by building minimal heads and testing each on both routes: a
    dense stack with no activation is clean, and any ReLU in the head reproduces it."""
    from tensorflow.keras import layers

    def head(spec: str) -> keras.Model:
        i = keras.Input((7, 7, 512), name="feat")
        x = layers.GlobalAveragePooling2D()(i)
        if spec == "gap_only":
            return keras.Model(i, x)
        if spec == "dense_linear":
            return keras.Model(i, layers.Dense(256)(x))
        if spec == "dense_relu":
            return keras.Model(i, layers.Dense(256, activation="relu")(x))
        if spec == "dense_split_relu":
            x = layers.Dense(256)(x)
            return keras.Model(i, layers.Activation("relu")(x))
        if spec == "two_dense_linear":
            x = layers.Dense(256)(x)
            return keras.Model(i, layers.Dense(1)(x))
        if spec == "project_head":
            x = layers.Dense(256, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            return keras.Model(i, layers.Dense(1, activation="sigmoid")(x))
        raise ValueError(spec)

    x = np.random.default_rng(config.SEED).normal(size=(8, 7, 7, 512)).astype("float32")
    rows = []
    for spec in (
        "gap_only",
        "dense_linear",
        "dense_relu",
        "dense_split_relu",
        "two_dense_linear",
        "project_head",
    ):
        m = head(spec)
        with tf.device("/CPU:0"):
            ref = np.asarray(m(x, training=False)).ravel()
        with tf.device("/GPU:0"):
            eager = np.asarray(m(x, training=False)).ravel()
            compiled = m.predict(x, verbose=0).ravel()
        rows.append(
            {
                "head": spec,
                "gpu_eager_vs_cpu": float(np.max(np.abs(eager - ref))),
                "gpu_compiled_vs_cpu": float(np.max(np.abs(compiled - ref))),
                "broken": bool(np.max(np.abs(compiled - ref)) > 1e-4),
            }
        )
    return {"reference": "CPU eager call", "rows": rows}


def framework_versions() -> dict[str, str]:
    """Record the exact stack the defect was measured on."""
    out = {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__}
    try:
        from importlib.metadata import version

        out["tensorflow-metal"] = version("tensorflow-metal")
    except Exception:  # not installed off-Mac
        out["tensorflow-metal"] = "not installed"
    out["gpu_devices"] = str([d.name for d in tf.config.list_physical_devices("GPU")])
    return out


def check_scaled_checkpoint(tag: str = "scaled_crop_m020") -> dict[str, Any]:
    """Whether a `_best` checkpoint is simply the last epoch: the stage runs without
    early stopping, so `_best` equal to `_final` would make the logged peak unreachable.
    """
    best = config.WEIGHTS_DIR / f"{tag}_best.weights.h5"
    final = config.WEIGHTS_DIR / f"{tag}_final.weights.h5"
    out: dict[str, Any] = {"tag": tag, "best_file": str(best), "final_file": str(final)}
    if not best.exists():
        out["error"] = "no _best checkpoint"
        return out

    b_idx = file_variable_index(best)
    if final.exists():
        f_idx = file_variable_index(final)
        same = [
            k
            for k in sorted(set(b_idx) & set(f_idx))
            if b_idx[k].shape == f_idx[k].shape and np.array_equal(b_idx[k], f_idx[k])
        ]
        out["n_variables"] = len(b_idx)
        out["n_identical_to_final"] = len(same)
        out["best_equals_final"] = len(same) == len(b_idx)

    log = config.LOGS_DIR / f"{tag}_training.csv"
    if log.exists():
        df = pd.read_csv(log)
        if "val_auc" in df.columns:
            v = df["val_auc"].to_numpy(dtype=float)
            out["log_n_epochs"] = int(len(v))
            out["log_best_val_auc"] = round(float(v.max()), 4)
            out["log_best_epoch"] = int(v.argmax()) + 1
            out["log_last_val_auc"] = round(float(v[-1]), 4)
            out["best_epoch_is_last"] = bool(v.argmax() + 1 == len(v))
    out["metrics"] = logged_vs_table(tag)
    return out


def check_localiser_input_graph(encoder: str = "vgg16") -> dict[str, Any]:
    """The tensor statistics entering the pretrained encoder's first convolution:
    grayscale replicated to three channels, x255 before the Caffe mean, BGR order."""
    from src import localise  # imported here so the other checks stay cheap

    model = localise.build_unet_pretrained(encoder=encoder, input_h=64, input_w=64)
    pre = model.get_layer("caffe_preprocess")
    probe = keras.Model(model.inputs, pre.output)

    rng = np.random.default_rng(config.SEED)
    x = rng.uniform(0.0, 1.0, size=(2, 64, 64, 1)).astype("float32")
    y = np.asarray(probe(x, training=False))

    mean_bgr = np.asarray(localise._CAFFE_BGR_MEAN, dtype=np.float32)
    expected = np.repeat(x * 255.0, 3, axis=-1)[..., ::-1] - mean_bgr
    ones = np.asarray(probe(np.ones((1, 64, 64, 1), "float32"), training=False))

    # After undoing the BGR reversal every channel must be the same replicated
    # grayscale plane, offset by its own mean; adding the mean back recovers it.
    restored = y[..., ::-1] + mean_bgr[::-1]
    replicated = bool(
        np.allclose(restored[..., 0], restored[..., 1], atol=1e-3)
        and np.allclose(restored[..., 0], restored[..., 2], atol=1e-3)
    )

    return {
        "encoder": encoder,
        "caffe_bgr_mean": [float(v) for v in mean_bgr],
        "input_channels": int(x.shape[-1]),
        "preprocessed_channels": int(y.shape[-1]),
        "channel_replication_ok": replicated,
        "matches_x255_then_mean_subtraction": bool(np.allclose(y, expected, atol=1e-3)),
        "max_abs_diff_vs_expected": float(np.max(np.abs(y - expected))),
        "unit_input_output_per_channel": [round(float(v), 3) for v in ones[0, 0, 0, :]],
        "expected_unit_input_output": [round(float(255.0 - m), 3) for m in mean_bgr],
        "input_stats": {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
        },
        "first_conv_input_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "std": float(y.std()),
        },
    }


def _write(obj: Any, path: Path) -> None:
    """Write *obj* as JSON and print the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"-> {path}")


def roundtrip_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tag", nargs="+", default=None, help="tags to round-trip (default: none unless --all)"
    )
    p.add_argument(
        "--all", action="store_true", help="round-trip every tag with a checkpoint and a val table"
    )
    p.add_argument(
        "--diff",
        nargs=2,
        metavar=("TAG_A", "TAG_B"),
        default=None,
        help="check 2: diff two tags' round-trip reports",
    )
    p.add_argument(
        "--load-modes",
        nargs="+",
        default=None,
        metavar="TAG",
        help="check 3: load one checkpoint into both trainable partitions",
    )
    p.add_argument(
        "--device-check",
        action="store_true",
        help="check 3b: eager versus compiled inference, GPU versus CPU",
    )
    p.add_argument(
        "--route-check",
        nargs="*",
        default=None,
        metavar="TAG",
        help="check 3c: score checkpoints on validation through both "
        "routes (no argument: the four recorded tags)",
    )
    p.add_argument(
        "--scaled-check",
        action="store_true",
        help="check 4: stage-6 _best versus _final and the CSV log",
    )
    p.add_argument("--localiser-graph", action="store_true", help="check 5: run 4's input graph")
    p.add_argument(
        "--summary",
        action="store_true",
        help="logged versus exact validation AUC for every known tag",
    )
    p.add_argument("--out", default=str(ROUNDTRIP_OUT_DIR))
    return p.parse_args(argv)


def roundtrip_main(argv: list[str] | None = None) -> None:
    args = roundtrip_parse_args(argv)
    out_dir = Path(args.out)

    if args.summary:
        _summary_tags = known_tags()
        _discovered(
            "tags with a checkpoint and a validation table",
            len(_summary_tags),
            [config.WEIGHTS_DIR, config.RESULTS_DIR],
            required=True,
        )
        rows = [logged_vs_table(t) for t in _summary_tags]
        rows.sort(key=lambda r: r.get("table_minus_history", 0.0))
        _write(rows, out_dir / "logged_vs_table.json")
        print(f"\n{'tag':44s} {'exact':>7s} {'history':>8s} {'delta':>7s} ok")
        for r in rows:
            print(
                f"{r['tag']:44s} {r.get('table_exact_val_auc', float('nan')):7.4f} "
                f"{r.get('history_best_val_auc', float('nan')):8.4f} "
                f"{r.get('table_minus_history', float('nan')):7.4f} "
                f"{'yes' if r.get('reproduces') else 'NO'}"
            )

    tags = list(args.tag or [])
    if args.all:
        tags = known_tags()
        # "0 tags round-tripped, exit 0" is how a checkpoint defect stays hidden.
        _discovered(
            "tags with a checkpoint and a validation table",
            len(tags),
            [config.WEIGHTS_DIR, config.RESULTS_DIR],
            required=True,
        )
    if args.diff:
        tags = list(dict.fromkeys(tags + list(args.diff)))

    reports: dict[str, dict[str, Any]] = {}
    for tag in tags:
        print(f"\n[roundtrip] {tag}")
        rep = roundtrip_report(tag)
        reports[tag] = rep
        _write(rep, out_dir / f"roundtrip_{tag}.json")
        print(
            f"    variables model/file : {rep['keys']['n_model_variables']}"
            f"/{rep['keys']['n_file_variables']}"
        )
        print(f"    missing from file    : {len(rep['keys']['missing_from_file'])}")
        print(f"    extra in file        : {len(rep['keys']['extra_in_file'])}")
        print(f"    value mismatches     : {rep['n_mismatches']}")
        print(f"    still at init        : {rep['n_unloaded']}")
        print(f"    reordered groups     : {rep['reordered_groups'] or 'none'}")
        m = rep["metrics"]
        print(
            f"    exact {m.get('table_exact_val_auc')} vs history "
            f"{m.get('history_best_val_auc')} -> "
            f"{'reproduces' if m.get('reproduces') else 'DOES NOT REPRODUCE'}"
        )

    if args.diff:
        a, b = args.diff
        d = diff_reports(reports[a], reports[b])
        _write(d, out_dir / f"diff_{a}__vs__{b}.json")
        print(
            f"\n[diff] {a} vs {b}: "
            f"{'structurally identical' if d['structurally_identical'] else 'DIFFERS'}"
        )
        for k, v in d["differing_fields"].items():
            print(f"    {k}: {v}")

    for tag in args.load_modes or []:
        r = compare_load_modes(tag)
        _write(r, out_dir / f"load_modes_{tag}.json")
        verdict = (
            "not applicable"
            if not r["applicable"]
            else ("IDENTICAL" if r["identical"] else "DIFFERS")
        )
        print(f"\n[load-modes] {tag}: {verdict}")
        if r.get("applicable"):
            print(f"    differing variables : {r['n_differing']}/{r['n_variables']}")
            print(f"    probe pred max diff : {r['probe_max_abs_diff']}")

    if args.device_check:
        r = check_device_consistency()
        _write(r, out_dir / "device_consistency.json")
        print(
            f"\n[device] reference {r['reference']}; tf {r['tensorflow']}, " f"keras {r['keras']}"
        )
        print(f"    {'arch':14s} {'gpu eager':>12s} {'gpu compiled':>14s} verdict")
        for row in r["rows"]:
            print(
                f"    {row['arch']:14s} {row['gpu_eager_vs_cpu']:12.2e} "
                f"{row['gpu_compiled_vs_cpu']:14.2e} "
                f"{'COMPILED ROUTE BROKEN' if row['compiled_route_broken'] else 'ok'}"
            )
        lr = check_localiser_device_consistency()
        _write(lr, out_dir / "device_consistency_localiser.json")
        for row in lr["rows"]:
            print(
                f"    {row['builder']:22s} eager {row['gpu_eager_vs_cpu']:.2e} "
                f"compiled {row['gpu_compiled_vs_cpu']:.2e} "
                f"routes agree on GPU: {row['routes_agree_on_gpu']}"
            )

    if args.route_check is not None:
        tags_rc = args.route_check or list(ROUTE_CHECK_TAGS)
        rows = [compare_inference_routes(t) for t in tags_rc]
        payload = {
            "versions": framework_versions(),
            "op_specificity": check_op_specificity(),
            "rows": rows,
        }
        _write(payload, out_dir / "inference_route_comparison.json")
        v = payload["versions"]
        print(
            f"\n[routes] tf {v['tensorflow']}, keras {v['keras']}, "
            f"tensorflow-metal {v['tensorflow-metal']}"
        )
        print(
            f"    {'tag':34s} {'AUC eager':>10s} {'AUC cpu':>9s} {'AUC compiled':>13s} "
            f"{'d(AUC)':>8s} {'max|dlogit|':>12s} {'eager==cpu':>11s}"
        )
        for r in rows:
            print(
                f"    {r['tag']:34s} {r['auc_eager']:10.4f} {r['auc_eager_cpu']:9.4f} "
                f"{r['auc_compiled']:13.4f} {r['auc_compiled_minus_eager']:8.4f} "
                f"{r['max_abs_logit_diff']:12.4f} "
                f"{str(r['eager_gpu_matches_cpu_1e5']):>11s}"
            )
        print(f"\n    {'minimal head':20s} {'gpu compiled vs cpu':>21s} verdict")
        for row in payload["op_specificity"]["rows"]:
            print(
                f"    {row['head']:20s} {row['gpu_compiled_vs_cpu']:21.2e} "
                f"{'BROKEN' if row['broken'] else 'ok'}"
            )

    if args.scaled_check:
        r = check_scaled_checkpoint()
        _write(r, out_dir / "scaled_checkpoint.json")
        print(
            f"\n[scaled] best == final: {r.get('best_equals_final')}; "
            f"best epoch {r.get('log_best_epoch')} of {r.get('log_n_epochs')}; "
            f"best epoch is last: {r.get('best_epoch_is_last')}"
        )

    if args.localiser_graph:
        r = check_localiser_input_graph()
        _write(r, out_dir / "localiser_input_graph.json")
        print(
            f"\n[localiser graph] channels {r['input_channels']} -> "
            f"{r['preprocessed_channels']}; replication ok "
            f"{r['channel_replication_ok']}; x255-then-mean "
            f"{r['matches_x255_then_mean_subtraction']}"
        )


# preproc_audit

PREPROC_OUT_DIR: Path = config.RESULTS_DIR / "preproc_audit"

OCCUPANCY_RULE: float = 0.70

CLIP_RULE: float = 0.05

MIN_MASK_PX: int = 16


def breast_box(arr: np.ndarray, thresh_frac: float = 0.10) -> tuple[int, int, int, int]:
    """`(y0, y1, x0, x1)` of the breast, inclusive, as the largest bright blob: the
    largest connected component drops the label tags, tape and scanner border."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return 0, arr.shape[0] - 1, 0, arr.shape[1] - 1
    mask = arr > (lo + thresh_frac * (hi - lo))
    labelled, n = ndimage.label(mask)
    if n == 0:
        return 0, arr.shape[0] - 1, 0, arr.shape[1] - 1
    sizes = ndimage.sum(mask, labelled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labelled == keep)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def side_of_film(box: tuple[int, int, int, int], width: int) -> str:
    """Return ``'left'`` or ``'right'`` for which half of the film the breast sits in."""
    _, _, x0, x1 = box
    return "left" if (x0 + x1) / 2.0 < width / 2.0 else "right"


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of a binary mask, or None when it is empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def measure_lesion(image_path: str, mask_path: str) -> dict[str, Any] | None:
    """One lesion at full resolution and under both letterbox schemes, as raw
    measurements: the rules are applied later, so the table can be re-stratified."""
    arr = data_loader._read_full(image_path)
    h, w = arr.shape[:2]

    ds = pydicom.dcmread(mask_path, force=True)
    m = ds.pixel_array
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        m = m.max() - m
    m = m > (m.max() * 0.5 if m.max() > 0 else 0)
    if m.shape[:2] != (h, w):
        return None  # mask and film disagree; reported as a count, not measured

    box = breast_box(arr)
    by0, by1, bx0, bx1 = box
    bh, bw = by1 - by0 + 1, bx1 - bx0 + 1

    mb = _mask_box(m)
    if mb is None:
        return None
    my0, my1, mx0, mx1 = mb
    lesion_side_px = float(max(my1 - my0 + 1, mx1 - mx0 + 1))

    # Current pipeline: percentiles over the whole film, then letterbox the film.
    lo, hi = np.percentile(arr, [config.PERCENTILE_LOW, config.PERCENTILE_HIGH])
    at_ceiling = float(np.mean(arr[m] >= hi)) if hi > lo else 1.0
    # Alternative: percentiles inside the breast box only.
    inside = arr[by0 : by1 + 1, bx0 : bx1 + 1]
    lo_b, hi_b = np.percentile(inside, [config.PERCENTILE_LOW, config.PERCENTILE_HIGH])
    at_ceiling_breast = float(np.mean(arr[m] >= hi_b)) if hi_b > lo_b else 1.0

    film_lb = Letterbox.for_shape(h, w)
    breast_lb = Letterbox.for_shape(bh, bw)

    return {
        "film_h": h,
        "film_w": w,
        "breast_y0": by0,
        "breast_y1": by1,
        "breast_x0": bx0,
        "breast_x1": bx1,
        "breast_w_frac": bw / w,
        "breast_h_frac": bh / h,
        "breast_side_measured": side_of_film(box, w),
        "lesion_side_px_native": lesion_side_px,
        "lesion_px_native": int(m.sum()),
        "film_scale": film_lb.scale,
        "breast_scale": breast_lb.scale,
        "lesion_side_canvas_film": lesion_side_px * film_lb.scale,
        "lesion_side_canvas_breast": lesion_side_px * breast_lb.scale,
        "lesion_px_canvas_film": float(m.sum()) * film_lb.scale**2,
        "resolution_gain": breast_lb.scale / film_lb.scale if film_lb.scale else np.nan,
        "mask_frac_at_clip_ceiling": at_ceiling,
        "mask_frac_at_clip_ceiling_breast_percentiles": at_ceiling_breast,
    }


def check_breast_occupancy(df: pd.DataFrame) -> dict[str, Any]:
    """Is the breast small enough on the film to be worth cropping?"""
    med_w = float(df["breast_w_frac"].median())
    med_h = float(df["breast_h_frac"].median())
    gain = float(df["resolution_gain"].median())
    return {
        "median_breast_width_frac": round(med_w, 4),
        "median_breast_height_frac": round(med_h, 4),
        "q1_breast_width_frac": round(float(df["breast_w_frac"].quantile(0.25)), 4),
        "median_resolution_gain": round(gain, 4),
        "rule": f"median breast width < {OCCUPANCY_RULE}",
        "triggers_rebuild": bool(med_w < OCCUPANCY_RULE),
    }


def check_lesion_size(df: pd.DataFrame) -> dict[str, Any]:
    """Lesion side in canvas pixels, current versus breast-crop letterbox."""
    out = {}
    for label, col in (
        ("current_film_letterbox", "lesion_side_canvas_film"),
        ("breast_crop_letterbox", "lesion_side_canvas_breast"),
    ):
        out[label] = {
            "median_px": round(float(df[col].median()), 2),
            "q1_px": round(float(df[col].quantile(0.25)), 2),
            "min_px": round(float(df[col].min()), 2),
        }
    out["median_gain_factor"] = round(
        out["breast_crop_letterbox"]["median_px"]
        / max(out["current_film_letterbox"]["median_px"], 1e-9),
        3,
    )
    return out


def check_percentile_clipping(df: pd.DataFrame) -> dict[str, Any]:
    """GT pixels lost at the clip ceiling, stratified by size and density."""
    df = df.copy()
    df["size_quartile"] = pd.qcut(df["lesion_side_px_native"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    strata: dict[str, Any] = {}
    worst = 0.0
    for key, group in [
        ("size_quartile", df.groupby("size_quartile", observed=True)),
        ("breast_density", df.groupby("breast_density", observed=True)),
    ]:
        rows = {}
        for name, g in group:
            frac = float(g["mask_frac_at_clip_ceiling"].mean())
            rows[str(name)] = {
                "n": int(len(g)),
                "mean_frac_at_ceiling": round(frac, 4),
                "mean_frac_at_ceiling_breast_percentiles": round(
                    float(g["mask_frac_at_clip_ceiling_breast_percentiles"].mean()), 4
                ),
                "exceeds_rule": bool(frac > CLIP_RULE),
            }
            worst = max(worst, frac)
        strata[key] = rows
    return {
        "overall_mean_frac_at_ceiling": round(float(df["mask_frac_at_clip_ceiling"].mean()), 4),
        "overall_mean_frac_at_ceiling_breast_percentiles": round(
            float(df["mask_frac_at_clip_ceiling_breast_percentiles"].mean()), 4
        ),
        "strata": strata,
        "rule": f"any stratum mean > {CLIP_RULE}",
        "worst_stratum_frac": round(worst, 4),
        "triggers_in_breast_percentiles": bool(worst > CLIP_RULE),
    }


def check_mask_survival(df: pd.DataFrame) -> dict[str, Any]:
    """How small the smallest lesions get on the current canvas."""
    px = df["lesion_px_canvas_film"]
    return {
        "median_mask_px_on_canvas": round(float(px.median()), 1),
        "min_mask_px_on_canvas": round(float(px.min()), 2),
        "n_below_16_px": int((px < MIN_MASK_PX).sum()),
        "n_below_64_px": int((px < 64).sum()),
        "rule": f"any lesion < {MIN_MASK_PX} px on the canvas",
        "triggers_area_resize": bool((px < MIN_MASK_PX).any()),
    }


def check_resize_antialiasing() -> dict[str, Any]:
    """How the cache resizes, read from the code rather than assumed."""
    import inspect
    from src import localise

    src = inspect.getsource(localise.letterbox)
    uses_tf = "tf.image.resize" in src
    uses_cv2 = "cv2.resize" in src
    antialias = "antialias=True" in src
    return {
        "letterbox_source_uses": (
            "tf.image.resize" if uses_tf else "cv2.resize" if uses_cv2 else "unknown"
        ),
        "antialias_requested": bool(antialias),
        "rule": "no antialias -> enable it in cache v2",
        "triggers_antialias": not antialias,
        "note": (
            "Read from the source of localise.letterbox; confirm the mask "
            "branch separately, masks must not be antialiased but "
            "area-resized then thresholded."
        ),
    }


def check_orientation(df: pd.DataFrame) -> dict[str, Any]:
    """How many films are right-breast, and does the metadata agree?"""
    meta = df["breast_side_meta"].str.upper().str[0]
    measured = df["breast_side_measured"].str.upper().str[0]
    # A right breast sits on the left of the film, so agreement is the mirrored
    # comparison; both conventions are reported rather than assumed.
    occupancy = float(df["breast_w_frac"].median())
    return {
        "n": int(len(df)),
        "frac_right_breast_metadata": round(float((meta == "R").mean()), 4),
        "frac_breast_on_left_of_film": round(float((measured == "L").mean()), 4),
        "agreement_mirrored": round(float((meta != measured).mean()), 4),
        "agreement_direct": round(float((meta == measured).mean()), 4),
        "rule": "canonicalise to one side in cache v2 (cost 0)",
        "triggers_canonicalise": True,
        "side_measure_informative": bool(occupancy < 0.95),
        "note": (
            "side_of_film compares the breast box centroid with the film "
            "centre, which only discriminates when the breast does not fill "
            "the film. At the measured occupancy it does not, so the "
            "measured column is uninformative and the metadata column is "
            "the one to use; canonicalising needs the chest-wall side, "
            "detected from which vertical edge carries tissue."
        ),
    }


CLIP_CANDIDATES: tuple[float, ...] = (99.0, 99.5, 99.8, 99.9, 99.95, 100.0)


def clip_sweep(
    frame: pd.DataFrame, fulls: dict, masks: dict, limit: int | None = None
) -> dict[str, Any]:
    """Fraction of GT mask pixels at the clip ceiling, per candidate upper percentile,
    with the lower percentile held: the rule is about saturation of bright tissue."""
    rows = []
    todo = frame if limit is None else frame.head(limit)
    for i, (_, r) in enumerate(todo.iterrows(), start=1):
        try:
            arr = data_loader._read_full(str(fulls[r["image_id"]]))
            ds = pydicom.dcmread(str(masks[r["lesion_id"]]), force=True)
            m = ds.pixel_array
            if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                m = m.max() - m
            m = m > (m.max() * 0.5 if m.max() > 0 else 0)
            if m.shape[:2] != arr.shape[:2] or not m.any():
                continue
            vals = arr[m]
            hi = np.percentile(arr, list(CLIP_CANDIDATES))
            row = {
                "lesion_id": r["lesion_id"],
                "lesion_px": int(m.sum()),
                "breast_density": r.get("breast_density"),
                "subtlety": r.get("subtlety"),
            }
            for pct, h in zip(CLIP_CANDIDATES, hi):
                row[f"frac_at_ceiling_p{pct}"] = float(np.mean(vals >= h))
            rows.append(row)
        except Exception as exc:
            print(f"    [skip] {r['lesion_id']}: {type(exc).__name__}")
        if i % 25 == 0:
            print(f"    swept {i}/{len(todo)}")

    df = pd.DataFrame(rows)
    df["size_quartile"] = pd.qcut(df["lesion_px"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out: dict[str, Any] = {
        "n": int(len(df)),
        "candidates": list(CLIP_CANDIDATES),
        "percentile_low_held_at": config.PERCENTILE_LOW,
        "rule": f"mean fraction at ceiling <= {CLIP_RULE} in every stratum",
        "overall": {},
        "worst_size_quartile": {},
        "passes": {},
    }
    for pct in CLIP_CANDIDATES:
        col = f"frac_at_ceiling_p{pct}"
        worst = float(df.groupby("size_quartile", observed=True)[col].mean().max())
        out["overall"][str(pct)] = round(float(df[col].mean()), 4)
        out["worst_size_quartile"][str(pct)] = round(worst, 4)
        out["passes"][str(pct)] = bool(worst <= CLIP_RULE)
    passing = [p for p in CLIP_CANDIDATES if out["passes"][str(p)]]
    out["lowest_passing_ceiling"] = float(min(passing)) if passing else None
    return out, df


def path_b_clip_sweep(
    frame: pd.DataFrame, fulls: dict, masks: dict, margin: float, limit: int | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The same clipping metric on the classifier crops rather than the film: the crop
    is normalised over itself, a field that is mostly lesion and peritumoral tissue."""
    rows = []
    todo = frame if limit is None else frame.head(limit)
    for i, (_, r) in enumerate(todo.iterrows(), start=1):
        try:
            full = data_loader._read_full(str(fulls[r["image_id"]]))
            ds = pydicom.dcmread(str(masks[r["lesion_id"]]), force=True)
            m = ds.pixel_array
            if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                m = m.max() - m
            if m.shape[:2] != full.shape[:2]:
                continue
            box = data_loader._derive_bbox(m, full.shape)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            h, w = (y1 - y0 + 1), (x1 - x0 + 1)
            my, mx = int(h * margin), int(w * margin)
            y0c, y1c = max(0, y0 - my), min(full.shape[0], y1 + 1 + my)
            x0c, x1c = max(0, x0 - mx), min(full.shape[1], x1 + 1 + mx)
            crop = full[y0c:y1c, x0c:x1c]
            cmask = np.asarray(m)[y0c:y1c, x0c:x1c] > (m.max() * 0.5 if m.max() > 0 else 0)
            if not cmask.any():
                continue
            vals = crop[cmask]
            hi_c = np.percentile(crop, list(CLIP_CANDIDATES))
            hi_f = np.percentile(full, list(CLIP_CANDIDATES))
            row = {
                "lesion_id": r["lesion_id"],
                "lesion_px": int(cmask.sum()),
                "crop_h": int(crop.shape[0]),
                "crop_w": int(crop.shape[1]),
                "lesion_frac_of_crop": float(cmask.mean()),
                "breast_density": r.get("breast_density"),
                "subtlety": r.get("subtlety"),
            }
            for pct, hc, hf in zip(CLIP_CANDIDATES, hi_c, hi_f):
                row[f"crop_frac_at_ceiling_p{pct}"] = float(np.mean(vals >= hc))
                row[f"film_frac_at_ceiling_p{pct}"] = float(np.mean(vals >= hf))
            rows.append(row)
        except Exception as exc:
            print(f"    [skip] {r['lesion_id']}: {type(exc).__name__}")
        if i % 25 == 0:
            print(f"    swept {i}/{len(todo)}")

    df = pd.DataFrame(rows)
    df["size_quartile"] = pd.qcut(df["lesion_px"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out: dict[str, Any] = {
        "n": int(len(df)),
        "margin": margin,
        "candidates": list(CLIP_CANDIDATES),
        "percentile_scope": "crop (data_loader._finalise runs on the cropped array)",
        "median_lesion_frac_of_crop": round(float(df["lesion_frac_of_crop"].median()), 4),
        "rule": f"mean fraction at ceiling <= {CLIP_RULE} in every size quartile",
        "crop_scope": {},
        "film_scope": {},
        "worst_quartile_crop": {},
        "passes_crop": {},
    }
    for pct in CLIP_CANDIDATES:
        cc, fc = f"crop_frac_at_ceiling_p{pct}", f"film_frac_at_ceiling_p{pct}"
        worst = float(df.groupby("size_quartile", observed=True)[cc].mean().max())
        out["crop_scope"][str(pct)] = round(float(df[cc].mean()), 4)
        out["film_scope"][str(pct)] = round(float(df[fc].mean()), 4)
        out["worst_quartile_crop"][str(pct)] = round(worst, 4)
        out["passes_crop"][str(pct)] = bool(worst <= CLIP_RULE)
    passing = [p for p in CLIP_CANDIDATES if out["passes_crop"][str(p)]]
    out["lowest_passing_ceiling_crop"] = float(min(passing)) if passing else None
    out["by_quartile_at_current"] = {
        str(k): round(float(v), 4)
        for k, v in df.groupby("size_quartile", observed=True)[
            f"crop_frac_at_ceiling_p{config.PERCENTILE_HIGH}"
        ]
        .mean()
        .items()
    }
    return out, df


def write_figure(df: pd.DataFrame, path: Path) -> None:
    """Four panels: occupancy, lesion size gain, clipping by stratum, mask survival."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(df["breast_w_frac"], bins=30, color="#4C72B0")
    ax[0, 0].axvline(OCCUPANCY_RULE, color="crimson", ls="--", label=f"rule {OCCUPANCY_RULE}")
    ax[0, 0].set_title("Breast width as a fraction of film width")
    ax[0, 0].legend()

    ax[0, 1].hist(
        [df["lesion_side_canvas_film"], df["lesion_side_canvas_breast"]],
        bins=30,
        label=["film letterbox", "breast-crop letterbox"],
        color=["#4C72B0", "#DD8452"],
    )
    ax[0, 1].set_title("Lesion side on the 1024x576 canvas (px)")
    ax[0, 1].legend()

    q = pd.qcut(df["lesion_side_px_native"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    means = df.groupby(q, observed=True)["mask_frac_at_clip_ceiling"].mean()
    ax[1, 0].bar(means.index.astype(str), means.to_numpy(), color="#4C72B0")
    ax[1, 0].axhline(CLIP_RULE, color="crimson", ls="--", label=f"rule {CLIP_RULE}")
    ax[1, 0].set_title("GT pixels at the clip ceiling, by lesion size quartile")
    ax[1, 0].legend()

    ax[1, 1].hist(df["lesion_px_canvas_film"], bins=40, color="#4C72B0")
    ax[1, 1].axvline(MIN_MASK_PX, color="crimson", ls="--", label=f"rule {MIN_MASK_PX} px")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_title("GT mask pixels surviving onto the canvas")
    ax[1, 1].legend()

    fig.suptitle("Localiser preprocessing audit (validation lesions)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"-> {path}")


def preproc_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "train"],
        help="partition to audit; test is not an option",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="measure only the first N lesions (smoke run)"
    )
    p.add_argument("--out", default=str(PREPROC_OUT_DIR))
    p.add_argument(
        "--path-b",
        action="store_true",
        help="run the clipping metric on the classifier crops at "
        "--margin instead of on the film",
    )
    p.add_argument(
        "--margin", type=float, default=config.CROP_MARGIN_FRAC, help="crop margin for --path-b"
    )
    p.add_argument(
        "--clip-sweep",
        action="store_true",
        help="measure the fraction of GT pixels at the ceiling for a range "
        "of upper percentiles, and report the lowest one that passes the "
        "clipping rule",
    )
    p.add_argument(
        "--out-figure",
        default=None,
        help="default: outputs/figures/; point a smoke run elsewhere "
        "so it cannot leave a partial figure among the real ones",
    )
    return p.parse_args(argv)


def preproc_main(argv: list[str] | None = None) -> None:
    args = preproc_parse_args(argv)
    if args.split == "test":  # unreachable, kept explicit
        raise SystemExit("the test partition is frozen")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-lesion rows: source="full" pools on the image path and silently drops the
    # second lesion of a multi-lesion image, which is 12.35 % of rows.
    frames = data_loader.split_frames(source="crop")
    frame = frames[args.split]
    masks = data_loader._build_combined_roi_lookup(
        config.TRAIN_ROI_DIR, config.TEST_ROI_DIR, pick="mask"
    )
    fulls = {
        Path(k).parts[0]: v
        for k, v in data_loader._build_combined_lookup(
            config.TRAIN_IMG_DIR, config.TEST_IMG_DIR
        ).items()
    }

    if args.path_b:
        report, table = path_b_clip_sweep(frame, fulls, masks, args.margin, args.limit)
        t = out_dir / f"path_b_clip_{args.split}_m{int(round(args.margin*100)):03d}.csv"
        table.to_csv(t, index=False)
        print(f"-> {t}")
        j = out_dir / f"path_b_clip_{args.split}_m{int(round(args.margin*100)):03d}.json"
        j.write_text(json.dumps(report, indent=2, default=str))
        print(f"-> {j}")
        print(f"\nPath B percentiles are computed on: {report['percentile_scope']}")
        print(f"median lesion share of the crop: {report['median_lesion_frac_of_crop']}")
        print(
            f"\n{'ceiling':>10s} {'crop scope':>11s} {'worst quartile':>15s} "
            f"{'film scope':>11s} rule (crop)"
        )
        for pct in CLIP_CANDIDATES:
            k = str(pct)
            print(
                f"{'p' + k:>10s} {report['crop_scope'][k]:11.4f} "
                f"{report['worst_quartile_crop'][k]:15.4f} "
                f"{report['film_scope'][k]:11.4f} "
                f"{'PASS' if report['passes_crop'][k] else 'fail'}"
            )
        print(
            f"\nby size quartile at the current p{config.PERCENTILE_HIGH}: "
            f"{report['by_quartile_at_current']}"
        )
        print(f"lowest passing ceiling on the crop: {report['lowest_passing_ceiling_crop']}")
        return

    if args.clip_sweep:
        report, table = clip_sweep(frame, fulls, masks, args.limit)
        t = out_dir / f"clip_sweep_{args.split}.csv"
        table.to_csv(t, index=False)
        print(f"-> {t}")
        j = out_dir / f"clip_sweep_{args.split}.json"
        j.write_text(json.dumps(report, indent=2, default=str))
        print(f"-> {j}")
        print(f"\n{'ceiling':>10s} {'mean frac':>10s} {'worst quartile':>15s} rule")
        for pct in CLIP_CANDIDATES:
            k = str(pct)
            print(
                f"{'p' + k:>10s} {report['overall'][k]:10.4f} "
                f"{report['worst_size_quartile'][k]:15.4f} "
                f"{'PASS' if report['passes'][k] else 'fail'}"
            )
        print(f"\nlowest passing ceiling: {report['lowest_passing_ceiling']}")
        return

    rows, skipped, skip_reasons = [], 0, []
    todo = frame if args.limit is None else frame.head(args.limit)
    for i, (_, r) in enumerate(todo.iterrows(), start=1):
        try:
            m = measure_lesion(str(fulls[r["image_id"]]), str(masks[r["lesion_id"]]))
        except Exception as exc:  # a bad DICOM is data, not a crash
            print(f"    [skip] {r['lesion_id']}: {type(exc).__name__}: {exc}")
            skip_reasons.append({"lesion_id": r["lesion_id"], "reason": type(exc).__name__})
            m = None
        if m is None:
            if not skip_reasons or skip_reasons[-1]["lesion_id"] != r["lesion_id"]:
                # measure_lesion returned None: mask/film shape disagreement or
                # an empty mask. Both are data facts worth counting, not crashes.
                skip_reasons.append(
                    {"lesion_id": r["lesion_id"], "reason": "shape_mismatch_or_empty_mask"}
                )
            skipped += 1
            continue
        m.update(
            {
                "lesion_id": r["lesion_id"],
                "image_id": r["image_id"],
                "patient_id": r["patient_id"],
                "breast_density": r.get("breast_density"),
                "subtlety": r.get("subtlety"),
                "breast_side_meta": str(r.get("left or right breast", "")),
            }
        )
        rows.append(m)
        if i % 25 == 0:
            print(f"    measured {i}/{len(todo)}")

    df = pd.DataFrame(rows)
    table = out_dir / f"preproc_audit_{args.split}_per_lesion.csv"
    df.to_csv(table, index=False)
    print(f"-> {table}")

    report = {
        "split": args.split,
        "n_measured": int(len(df)),
        "n_skipped": int(skipped),
        "skip_reasons": skip_reasons,
        "frame_source": "crop (per-lesion rows)",
        "canvas": [config.LOC_INPUT_H, config.LOC_INPUT_W],
        "percentiles": [config.PERCENTILE_LOW, config.PERCENTILE_HIGH],
        "breast_occupancy": check_breast_occupancy(df),
        "lesion_size": check_lesion_size(df),
        "percentile_clipping": check_percentile_clipping(df),
        "mask_survival": check_mask_survival(df),
        "resize_antialiasing": check_resize_antialiasing(),
        "orientation": check_orientation(df),
    }
    report["cache_v2_triggered"] = bool(
        report["breast_occupancy"]["triggers_rebuild"]
        or report["percentile_clipping"]["triggers_in_breast_percentiles"]
        or report["mask_survival"]["triggers_area_resize"]
        or report["resize_antialiasing"]["triggers_antialias"]
    )

    path = out_dir / f"preproc_audit_{args.split}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"-> {path}")
    fig_dir = Path(args.out_figure) if args.out_figure else config.FIGURES_DIR
    write_figure(df, fig_dir / f"preproc_audit_{args.split}.png")

    print(
        f"\nbreast width (median frac of film) : "
        f"{report['breast_occupancy']['median_breast_width_frac']} "
        f"-> rebuild {report['breast_occupancy']['triggers_rebuild']}"
    )
    print(
        f"lesion side on canvas (median px)  : "
        f"{report['lesion_size']['current_film_letterbox']['median_px']} now, "
        f"{report['lesion_size']['breast_crop_letterbox']['median_px']} under a breast crop "
        f"({report['lesion_size']['median_gain_factor']}x)"
    )
    print(
        f"GT pixels at clip ceiling (mean)   : "
        f"{report['percentile_clipping']['overall_mean_frac_at_ceiling']} film, "
        f"{report['percentile_clipping']['overall_mean_frac_at_ceiling_breast_percentiles']}"
        f" in-breast"
    )
    print(f"lesions under 16 canvas px         : " f"{report['mask_survival']['n_below_16_px']}")
    print(f"cache v2 triggered                 : {report['cache_v2_triggered']}")


# weights_boxes_equivalence
def _bundle():
    """The bundle module beside this one, loaded on use, so boxes are derived with the
    same `_select_box` and REFERENCE_RULE the promoted route uses."""
    global _lb, lb
    if globals().get("_lb") is None:
        spec = importlib.util.spec_from_file_location(
            "localiser_bundle", Path(__file__).resolve().parent / "localiser_bundle.py"
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("localiser_bundle", m)
        spec.loader.exec_module(m)
        _lb = lb = m
    return _lb


_lb = lb = None


EXPECT_DIFFER = ("bundle_weights",)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, list[str]]:
    """Element-wise equality over every column, NaN equal to NaN; EXPECT_DIFFER columns
    are reported as declared exceptions and do not fail the check."""
    notes = []
    if list(a.columns) != list(b.columns):
        return False, [f"column sets differ: {set(a.columns) ^ set(b.columns)}"]
    if len(a) != len(b):
        return False, [f"row counts differ: {len(a)} vs {len(b)}"]
    ok = True
    for c in a.columns:
        if c in EXPECT_DIFFER:
            va, vb = sorted(set(a[c].astype(str))), sorted(set(b[c].astype(str)))
            notes.append(f"  {c}: declared exception — provenance label, {va} vs {vb}")
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xn, yn = x.to_numpy(), y.to_numpy()
            both_nan = pd.isna(xn) & pd.isna(yn)
            same = (xn == yn) | both_nan
            if not same.all():
                d = np.abs(
                    np.where(pd.isna(xn) | pd.isna(yn), 0.0, xn.astype(float) - yn.astype(float))
                )
                notes.append(
                    f"  {c}: {int((~same).sum())} of {len(x)} rows differ, "
                    f"max |diff| {d.max():.3e}"
                )
                ok = False
        else:
            same = (x.astype(str) == y.astype(str)) | (pd.isna(x) & pd.isna(y))
            if not same.all():
                notes.append(f"  {c}: {int((~same).sum())} of {len(x)} rows differ (non-numeric)")
                ok = False
    return ok, notes


def through_apply(runs, cfg_base, tmp: Path) -> tuple[bool, list[str], str]:
    """Equivalence through `apply`, the function that writes the test boxes: it runs
    twice, once from a selection labelled `equal` and once from one labelled `ones`."""
    import hashlib

    sel_path = _lb.BUNDLE / "bundle_selection.json"
    archived = sel_path.read_bytes()
    before = hashlib.md5(archived).hexdigest()
    outs = {}
    try:
        for label in ("equal", "ones"):
            cfg = dict(cfg_base)
            cfg["weights"] = label
            sel_path.write_text(
                json.dumps(
                    {
                        "selected_cfg_id": -1,
                        "selected": cfg,
                        "runs": list(runs),
                        "selection_rule": "equivalence probe, not a selection",
                        "train": {"mean_iou": 0.0},
                        "val": {"mean_iou": 0.0},
                    },
                    indent=2,
                )
            )
            d = tmp / label
            d.mkdir(parents=True, exist_ok=True)
            _lb.apply(
                argparse.Namespace(
                    split="val", boxes_dir=str(d), allow_test=False, batch=8, cpu=False, limit=None
                )
            )
            outs[label] = pd.read_csv(d / "localiser_boxes_val.csv")
    finally:
        sel_path.write_bytes(archived)
        after = hashlib.md5(sel_path.read_bytes()).hexdigest()
    note = (
        f"live selection restored: md5 {before[:12]} -> {after[:12]} "
        f"{'OK' if before == after else 'RESTORE FAILED'}"
    )
    if before != after:
        return False, [note], note
    ok, notes = compare(outs["equal"], outs["ones"])
    return ok, notes, note


def through_sweep(runs, limit: int, tmp: Path) -> tuple[bool, list[str], str]:
    """Equivalence through `sweep`'s own export: every bundle file it can touch is
    archived by md5 first and restored afterwards, and the restoration is asserted."""
    import hashlib, shutil

    names = [
        "bundle_selection.json",
        "bundle_metrics.json",
        "bundle_boxes_val.csv",
        "bundle_boxes_train.csv",
        "sweep_configs.csv",
        "sweep_iou_train.npy",
        "sweep_iou_val.npy",
    ]
    arch = tmp / "_archive"
    arch.mkdir(parents=True, exist_ok=True)
    before = {}
    for n in names:
        f = _lb.BUNDLE / n
        if f.exists():
            shutil.copy2(f, arch / n)
            before[n] = hashlib.md5(f.read_bytes()).hexdigest()
    outs = {}
    try:
        for mode in ("none", "ones"):
            _lb.sweep(
                argparse.Namespace(
                    runs=list(runs),
                    limit=limit,
                    weights=mode,
                    workers=4,
                    reference=str(ROOT / "artifacts" / "localiser_boxes_val.csv"),
                )
            )
            outs[mode] = pd.read_csv(_lb.BUNDLE / "bundle_boxes_val.csv")
    finally:
        for n, h in before.items():
            shutil.copy2(arch / n, _lb.BUNDLE / n)
        after = {n: hashlib.md5((_lb.BUNDLE / n).read_bytes()).hexdigest() for n in before}
    # A probe that selected `equal` on both sides never entered the weighted path, so
    # it compared equal against equal and proves nothing.
    labels = {m: sorted(set(df["bundle_weights"].astype(str))) for m, df in outs.items()}
    if labels.get("ones") == ["equal"]:
        return (
            False,
            [
                f"  VACUOUS: --weights ones selected a configuration labelled "
                f"{labels['ones']}, i.e. a subset of the members, so the weighted path "
                f"was never entered. Compared equal against equal.",
                f"  none -> {labels.get('none')}, ones -> {labels.get('ones')}",
            ],
            "probe did not exercise the weighted path",
        )
    bad = [n for n in before if before[n] != after[n]]
    note = (
        f"{len(before)} bundle files archived and restored; "
        f"{'all md5s match' if not bad else 'RESTORE FAILED for ' + ', '.join(bad)}"
    )
    if bad:
        return False, [note], note
    ok, notes = compare(outs["none"], outs["ones"])
    return ok, notes, note


def wbe_main(argv=None) -> int:
    _bundle()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--through",
        choices=["function", "apply", "sweep"],
        default="function",
        help="function: _bundle_boxes directly. apply: the real entry point "
        "that writes the boxes folder and the test boxes.",
    )
    ap.add_argument("--out", default=None, help="scratch dir for --through apply|sweep")
    ap.add_argument("--limit", type=int, default=12, help="rows per split for --through sweep")
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "train"])
    ap.add_argument("--tta", default="1")
    ap.add_argument("--rule", default="peak")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min-px", type=int, default=16)
    ap.add_argument("--breast-filter", default="0")
    a = ap.parse_args(argv)

    bad = [s for s in a.split if s not in ("val", "train")]
    if bad:
        raise SystemExit(f"refusing to read {bad}: validation and train only")

    runs = tuple(a.tag)
    cfg = {
        "ensemble": "+".join(runs),
        "tta": a.tta not in ("0", "False", "false"),
        "breast_filter": a.breast_filter not in ("0", "False", "false"),
        "rule": a.rule,
        "threshold": a.threshold,
        "min_px": a.min_px,
    }
    ones = tuple([1.0] * len(runs))
    print(f"members    : {' '.join(runs)} (k = {len(runs)})")
    print(f"configuration: {cfg}")
    print(f"weights    : None (equal) vs {ones} (all ones, via the weighted path)\n")

    if a.through == "sweep":
        import tempfile

        tmp = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="wbe_sweep_"))
        ok, notes, note = through_sweep(runs, a.limit, tmp)
        print(
            f"through sweep export (val, limit {a.limit}): "
            f"{'IDENTICAL apart from the declared exception' if ok else 'DIFFERENCES FOUND'}"
        )
        for n in notes:
            print(n)
        print(f"  {note}")
        print()
        if ok:
            print(
                "EQUIVALENCE HOLDS THROUGH sweep: --weights ones exports the same table as "
                "--weights none, element-wise."
            )
            return 0
        print(
            "EQUIVALENCE NOT ESTABLISHED THROUGH sweep — see above. The probe compared "
            "nothing, so the code is untested rather than wrong."
        )
        return 1

    if a.through == "apply":
        import tempfile

        tmp = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="wbe_apply_"))
        ok, notes, note = through_apply(runs, cfg, tmp)
        print(
            f"through apply (val): "
            f"{'IDENTICAL apart from the declared exception' if ok else 'DIFFERENCES FOUND'}"
        )
        for n in notes:
            print(n)
        print(f"  {note}")
        print()
        if ok:
            print(
                "EQUIVALENCE HOLDS THROUGH apply: a selection labelled 'ones' writes the "
                "same boxes folder as one labelled 'equal', element-wise."
            )
            return 0
        print("EQUIVALENCE FAILS THROUGH apply — the test-pass path must not be used.")
        return 1

    all_ok = True
    for split in a.split:
        meta, gts, breasts = _lb._load_split_side_tables(split, None)
        none_df = _lb._bundle_boxes(split, runs, cfg, meta, gts, breasts, weights=None)
        ones_df = _lb._bundle_boxes(split, runs, cfg, meta, gts, breasts, weights=ones)
        ok, notes = compare(none_df, ones_df)
        iou_n = none_df[none_df["status"] != "no_gt"]["box_iou"].fillna(0).mean()
        iou_o = ones_df[ones_df["status"] != "no_gt"]["box_iou"].fillna(0).mean()
        print(
            f"{split}: {len(none_df)} rows x {len(none_df.columns)} columns "
            f"mean box IoU {iou_n:.10f} vs {iou_o:.10f} (|diff| {abs(iou_n - iou_o):.3e})"
        )
        print(f"  element-wise: {'IDENTICAL' if ok else 'DIFFERENCES FOUND'}")
        for n in notes:
            print(n)
        all_ok &= ok

    print()
    if all_ok:
        print(
            "EQUIVALENCE HOLDS: all-ones weights reproduce the unweighted exported table "
            "exactly, on every row and every column."
        )
        return 0
    print(
        "EQUIVALENCE FAILS: the weighted path does not reproduce the unweighted table. "
        "Do not use any weighted table until this is explained."
    )
    return 1


# control_compare_maps


def _refuse_if_live(candidate: Path, live: Path) -> None:
    """Refuse a candidate directory inside the live maps: paths are resolved, so a
    symlink or a `..` cannot make the control compare the stored maps with themselves.
    """
    c, l = candidate.resolve(), live.resolve()
    if c == l or l in c.parents or c in l.parents:
        raise SystemExit(
            f"REFUSED: the candidate maps-dir resolves inside the live maps directory.\n"
            f"  candidate: {c}\n  live     : {l}\n"
            f"It would compare the stored maps with themselves. Point --candidate at a "
            f"scratch directory."
        )


def control_main(argv: list[str] | None = None) -> int:
    _bundle()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--candidate", required=True, help="scratch maps-dir holding the new maps")
    ap.add_argument("--reference", default=None, help="default: the bundle's maps dir")
    ap.add_argument(
        "--tensors-dir",
        default="outputs/localiser_tensors_v1_notest",
        help="store holding {split}_masks.npy for the box comparison",
    )
    ap.add_argument(
        "--gate",
        choices=["boxes"],
        default="boxes",
        help="what decides the verdict. 'boxes': every derived box must match "
        "exactly. The map difference is reported as a number but gates "
        "nothing: once the boxes match, it changes nothing downstream.",
    )
    args = ap.parse_args(argv)

    lb._set_bundle_dir(str(lb.config.LOC_BUNDLE_DIR))
    live = Path(args.reference) if args.reference else lb.MAPS
    cand = Path(args.candidate)
    _refuse_if_live(cand, live)

    print("--- criterion, declared before the verdict ---")
    print("  GATE  : every derived box identical (REFERENCE_RULE, all scored lesions)")
    print("  report: max |diff| on plain and flip, element-wise — a number, not a bar")
    print(f"  live (stored) : {live}")
    print(f"  candidate     : {cand}\n")

    fails: list[str] = []
    arrays = {}
    for kind in ("plain", "flip"):
        a = np.load(live / f"{args.tag}_{args.split}_{kind}.npy", mmap_mode="r")
        b = np.load(cand / f"{args.tag}_{args.split}_{kind}.npy", mmap_mode="r")
        arrays[kind] = (a, b)
        if a.shape != b.shape or a.dtype != b.dtype:
            fails.append(f"{kind}: shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            print(f"  {kind:<5} SHAPE/DTYPE MISMATCH")
            continue
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        mx = int(d.max())
        print(
            f"  {kind:<5} shape {a.shape} {a.dtype} stored [{a.min()},{a.max()}] "
            f"cand [{b.min()},{b.max()}]"
        )
        print(
            f"        max |diff| {mx}/255 mean {d.mean():.3e} "
            f"rows differing {int((d.reshape(len(d), -1).max(1) > 0).sum())}/{len(d)}"
            f"   (reported, not gated)"
        )

    print("\n  --- derived boxes, REFERENCE_RULE ---")
    msks = np.load(Path(args.tensors_dir) / f"{args.split}_masks.npy", mmap_mode="r")
    pa, pb = arrays["plain"]
    moved, ia, ib = [], [], []
    for i in range(len(pa)):
        gt = lb._gt_box(msks[i])
        if gt is None:
            continue
        ba = lb._select_box(
            pa[i].astype(np.float32) / 255.0, gt=None, breast=None, **lb.REFERENCE_RULE
        )[0]
        bb = lb._select_box(
            pb[i].astype(np.float32) / 255.0, gt=None, breast=None, **lb.REFERENCE_RULE
        )[0]
        ia.append(lb._iou(gt, ba) if ba is not None else 0.0)
        ib.append(lb._iou(gt, bb) if bb is not None else 0.0)
        if ba != bb:
            moved.append((i, ba, bb, ia[-1], ib[-1]))
    ia, ib = np.array(ia), np.array(ib)
    print(f"  lesions scored        : {len(ia)}")
    print(f"  mean box IoU stored   : {ia.mean():.6f}")
    print(f"  mean box IoU candidate: {ib.mean():.6f} delta {ib.mean() - ia.mean():+.3e}")
    print(f"  boxes MOVED           : {len(moved)}")
    print(f"  max |per-lesion dIoU| : {np.abs(ib - ia).max():.6e}")
    for i, ba, bb, x, y in moved[:10]:
        print(f"    row {i}: {ba} IoU {x:.4f} -> {bb} IoU {y:.4f} ({y - x:+.4f})")
    if moved:
        fails.append(f"{len(moved)} derived box(es) moved — the gate")

    print()
    if fails:
        print("CONTROL FAILED against the declared criterion:")
        for f in fails:
            print(f"  - {f}")
        print("No test map may be written. Bring this back to the laptop.")
        return 1
    print("CONTROL PASSED against the declared criterion.")
    return 0


# check_launcher_flags
LAUNCHERS = sorted(
    # The launchers live in notebooks/scripts/ permanently, so they are found there
    # rather than beside this file.
    p.name
    for p in LAUNCHER_DIR.glob("*.sh")
    if p.name != "launcher_selfcopy.sh"  # sourced, never invoked
)

SUPPORTED = {
    "src.evaluate",
    "src.rerank",
    "src.localise_torch.train",
    "src.localise_torch.detector",
    # Every module a launcher invokes supports --parse-only; one missing from
    # this set is reported as skipped, which reads like a pass and is not one.
    "src.ladder",
    "src.compare",
    "src.explain",
    "src.uncertainty",
    "src.fusion",
    "src.ensemble",
    "src.analysis",
    "src.train",
    "src.localise_eval",
    "src.train_localiser_l4",
    "src.train_crop_store",
}

ENVS = {
    "src.localise_torch.train": ".torch-env/bin/python",
    "src.localise_torch.detector": ".torch-env/bin/python",
}


def _default_py() -> str:
    """The project venv, or the running interpreter when a checkout has no venv."""
    venv = ROOT / ".ddsm-env" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


FALLBACK = {
    "ARCH": "vgg16",
    "M": "0.20",
    "SPLIT": "val",
    "src": "unet",
    "k": "3",
    "m": "0.20",
    "enc": "resnet18",
    "name": "l7",
    "E": "resnet34",
    "RUN": "l6",
}


def loop_bindings(text: str, base: dict[str, str]) -> list[tuple[int, str, str]]:
    """(line, var, first value) for every `for VAR in ...`, in file order: positional,
    because one name can be bound by several loops in the same script."""
    out = []
    for i, line in enumerate(text.splitlines()):
        m = re.match(r"\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+?);?\s*do\s*$", line)
        if not m:
            continue
        var, items = m.group(1), m.group(2).split()
        first = items[0] if items else ""
        if first.startswith("$"):  # for s in $SEEDS
            first = base.get(first.strip("${}").split(":-")[0], "").split(" ")[0]
        if first and "$" not in first:
            out.append((i, var, first))
    return out


def env_at(base: dict[str, str], bindings, line: int) -> dict[str, str]:
    """base, plus the nearest preceding binding of each loop variable."""
    env = dict(base)
    for i, var, val in bindings:
        if i < line:
            env[var] = val
    return env


def launcher_assignments(text: str) -> dict[str, str]:
    """Literal assignments, then chained ones resolved (PATIENCE=$EPOCHS)."""
    raw = {}
    for m in re.finditer(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(\$\{[^}]+\}|\"[^\"]*\"|'[^']*'|\S+)", text, re.M
    ):
        v = m.group(2).strip("\"'").rstrip(";")
        if "$(" in v or "`" in v:  # a command substitution: unresolvable here
            continue
        raw[m.group(1)] = v
    out = {k: v for k, v in raw.items() if not v.startswith("$")}
    for _ in range(4):  # resolve chains, a few passes is plenty
        for k, v in raw.items():
            if k in out:
                continue
            r = re.sub(
                r"\$\{?([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}?",
                lambda m: out.get(m.group(1), m.group(2)[2:] if m.group(2) else ""),
                v,
            )
            if r and "$" not in r:
                out[k] = r
    return out


def launcher_expand(tok: str, env: dict[str, str]) -> str:
    # A command substitution has no value until the launcher runs, so it becomes a
    # neutral numeric literal and the flags around it are still checked.
    tok = re.sub(r"\\+\$", "$", tok)  # \$FRAC / \$(...) are escaped on purpose
    tok = re.sub(r"\$\((?:[^()]|\([^()]*\))*\)", "0", tok)

    def sub(m):
        raw = m.group(1) or m.group(2)
        name, _, default = raw.lstrip("{").rstrip("}").partition(":-")
        if name in env:
            return env[name]
        # ${VAR:-default} carries its own answer.
        if default:
            return default
        # Generic fallback is numeric: --parse-only loads nothing, so a path can
        # be any string, but an int-typed flag must still parse.
        return FALLBACK.get(name, "1")

    return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", sub, tok)


def fn_of(text: str, mod: str) -> str | None:
    """Name of the shell function whose body invokes `-m <mod>`, found by the module
    name: invocations() folds continuations, so the joined line is not in the file."""
    idx = text.find(f"-m {mod}")
    if idx < 0:
        return None
    best = None
    for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", text, re.M):
        if m.start() < idx:
            best = m.group(1)
        else:
            break
    return best


def call_sites(text: str, fn: str | None) -> list[str]:
    """Argument lists every call of `fn` passes, minus its own leading $1."""
    if not fn:
        return []
    out = []
    for m in re.finditer(rf"^\s*{re.escape(fn)}\s+(\S.*)$", text, re.M):
        site_line = text[: m.start()].count("\n")
        args = m.group(1).strip()
        # A wrapper consumes its own leading arguments before forwarding "$@"; how many
        # is read from its body, where `shift` is 1, `shift 3` is 3, and they add up.
        body = text.split(f"{fn}()", 1)[1][:400]
        n = sum(int(m or 1) for m in re.findall(r"\bshift\s*(\d*)", body))
        for _ in range(n):
            args = args.split(None, 1)[1] if " " in args else ""
        out.append((site_line, args))
    return [a for a in out if a[1]]


def invocations(text: str):
    """Join backslash continuations, then yield each `-m src.<mod> ...` line."""
    joined, buf, start = [], "", 0
    for i, line in enumerate(text.splitlines()):
        s = line.rstrip()
        if s.endswith("\\"):
            if not buf:
                start = i
            buf += s[:-1] + " "
            continue
        joined.append((buf + s, start if buf else i))
        buf = ""
    for line, lineno in joined:
        m = re.search(r"-m\s+(src\.[A-Za-z0-9_.]+)(.*)$", line)
        if m and not line.lstrip().startswith("#"):
            yield m.group(1), m.group(2), lineno


def run_one(job):
    name, mod, cmd, hist = job
    r = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    return name, mod, r.returncode, r.stderr, hist


def launcher_main(argv: list[str] | None = None) -> int:
    fails, skipped, ok, jobs, historical = [], [], 0, [], []
    for name in LAUNCHERS:
        p = ROOT / "notebooks" / "scripts" / name
        if not p.exists():
            print(f"  (absent) {name}")
            continue
        text = p.read_text()
        historical_reason = None
        hm = re.search(r"^#\s*HISTORICAL:\s*(.+)$", text, re.M)
        if hm:
            historical_reason = hm.group(1).strip()
        base_env = launcher_assignments(text)
        bindings = loop_bindings(text, base_env)
        for mod, rest, lineno in invocations(text):
            env = env_at(base_env, bindings, lineno)
            if mod not in SUPPORTED:
                skipped.append((name, mod))
                continue
            # A wrapper that forwards "$@" has no argv of its own, so it is checked at
            # its call sites, with the call's argument list substituted for "$@".
            variants = [(lineno, rest)]
            # "${@:N}" forwards the extra arguments the script was given: nothing to
            # substitute, so the fixed part is checked with the pass-through removed.
            if re.search(r'"?\$\{@:\d+\}"?', rest):
                rest = re.sub(r'"?\$\{@:\d+\}"?', "", rest)
                variants = [(lineno, rest)]
            elif re.search(r'"\$@"|\$@|\$\*', rest):
                sites = call_sites(text, fn_of(text, mod))
                variants = [
                    (ln, rest.replace('"$@"', c).replace("$@", c).replace("$*", c))
                    for ln, c in sites
                ]
                if not variants:
                    # No call site found: check the wrapper's own fixed flags with the
                    # forward removed, since that is the part the launcher writes.
                    variants = [(lineno, re.sub(r'"\$@"|\$@|\$\*', "", rest))]
            for site_ln, rest in variants:
                env = env_at(base_env, bindings, site_ln)
                try:
                    toks = shlex.split(launcher_expand(rest, env), posix=True)
                except ValueError:
                    try:
                        # A heredoc or an unbalanced quote defeats POSIX splitting;
                        # non-POSIX splitting still gives the flag names.
                        toks = shlex.split(launcher_expand(rest, env), posix=False)
                    except ValueError:
                        fails.append(
                            (
                                name,
                                mod,
                                "could not be split into arguments " "(unbalanced quoting?)",
                            )
                        )
                        continue
                # Cut at the first shell operator: the invocation ends there, and the
                # tail (redirections, `&& OK=1`, closing subshell parens) is not argv.
                cut = len(toks)
                for i, tk in enumerate(toks):
                    if (
                        tk in (")", "|", "&&", "||", ";", ">", ">>", "2>", "2>&1", "tee")
                        or tk.startswith(">")
                        or tk.startswith("2>")
                    ):
                        cut = i
                        break
                toks = [tk.rstrip(";") for tk in toks[:cut]]
                env_py = ENVS.get(mod)
                cmd = [
                    str(ROOT / env_py) if env_py else _default_py(),
                    "-m",
                    mod,
                    *toks,
                    "--parse-only",
                ]
                jobs.append((name, mod, cmd, historical_reason))
    # Each check spawns a full TensorFlow import, ~5 s apiece, and they are independent
    # subprocesses, so they run together.
    _discovered("launcher invocations", len(jobs), LAUNCHER_DIR, required=True)
    print(f"  checking {len(jobs)} launcher invocations in parallel...")
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for name, mod, rc, stderr, hist in sorted(
            pool.map(run_one, jobs), key=lambda x: (x[0], x[1])
        ):
            if rc == 0:
                ok += 1
                print(f"  OK    {name}: -m {mod}")
            elif hist:
                historical.append((name, mod, hist))
                print(f"  HIST  {name}: -m {mod} (declared historical in the file)")
            else:
                err = [l for l in stderr.splitlines() if l.strip()][-1:] or ["(no stderr)"]
                fails.append((name, mod, err[0]))
                print(f"  FAIL  {name}: -m {mod}\n          {err[0][:110]}")

    print(
        f"\n  {ok} invocations parse, {len(fails)} fail, {len(skipped)} skipped, "
        f"{len(historical)} historical"
    )
    if historical:
        print("  historical (the file declares itself not re-runnable, with the reason):")
        for n, m, why in sorted(set(historical)):
            print(f"    {n}: -m {m}\n        {why}")
    if skipped:
        print("  skipped (no --parse-only support):")
        for n, m in sorted(set(skipped)):
            print(f"    {n}: {m}")
    if fails:
        # Every failure is listed here, including the ones raised while resolving the
        # command line rather than while parsing it.
        print("\n  failures:")
        for n, m, e in fails:
            print(f"    {n}: -m {m}\n        {e[:140]}")
        print("\nLAUNCHER FLAGS ARE STALE — fix before any launch.")
        return 1
    print("\nEvery supported invocation parses against its module's current parser.")
    return 0


# check_cascade_handoff
CASCADE = ROOT / "notebooks" / "scripts" / "promotion_cascade.sh"

TESTPASS = ROOT / "notebooks" / "scripts" / "run_test_pass.sh"

CASCADE_PREREG = ROOT / "docs" / "test_pass_preregistration.md"

OUT = ROOT / "outputs" / "results" / "cascade_handoff"

FINDINGS: list[dict] = []


def flag(severity: str, what: str, detail: str) -> None:
    FINDINGS.append({"severity": severity, "what": what, "detail": detail})


def make_copy(work: Path) -> Path:
    """A working copy: real scripts and docs, symlinks for the heavy trees."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for d in ("notebooks", "docs"):
        shutil.copytree(
            ROOT / d,
            work / d,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.ipynb_checkpoints"),
        )
    # refs/ carries the pointer the cascade reads and the dry-run pointer it writes, so
    # it is copied rather than symlinked.
    shutil.copytree(ROOT / "refs", work / "refs")
    for d in ("src", "outputs", "artifacts", "data", ".ddsm-env"):
        src = ROOT / d
        if src.exists():
            (work / d).symlink_to(src)
    return work


def dry_run_cascade(work: Path) -> str:
    # The cascade requires a boxes folder and has no default, so it is taken from the
    # pointer run_test_pass.sh reads; without one, the pre-promotion default applies.
    pointer = ROOT / "refs" / "promoted_pointer.json"
    boxes = (
        json.loads(pointer.read_text())["boxes_dir"]
        if pointer.exists()
        else "outputs/results/localiser_bundle/boxes"
    )
    r = subprocess.run(
        ["bash", "notebooks/scripts/promotion_cascade.sh", "--dry-run", boxes],
        cwd=work,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        flag(
            "BLOCKER",
            "promotion_cascade.sh --dry-run exits non-zero",
            (
                (r.stderr.strip() or r.stdout.strip()).splitlines()[-1:][0]
                if (r.stderr.strip() or r.stdout.strip())
                else f"exit {r.returncode}"
            ),
        )
    return r.stdout + r.stderr


TAG_FLAGS = re.compile(r"--(?:out-)?tag(?:-suffix)?\s+(\S+)")

TAGS_FLAG = re.compile(r"--tags\s+((?:[\w./-]+\s*)+)")

BOXES_FLAG = re.compile(r"--boxes-dir\s+(\S+)")

SUFFIX_FLAG = re.compile(r"--rung3-suffix\s+(\S+)")

JUNK = re.compile(r"^[\W_]*$|\]|\"|;")


def cascade_assignments(text: str) -> dict[str, str]:
    """Literal shell assignments, then chained ones (BOXES=${1:-outputs/...})."""
    raw = {}
    for m in re.finditer(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(\$\{[^}]+\}|\"[^\"]*\"|'[^']*'|\S+)", text, re.M
    ):
        raw[m.group(1)] = m.group(2).strip("\"'").rstrip(";")
    out = {k: v for k, v in raw.items() if "$" not in v}
    for _ in range(4):
        for k, v in raw.items():
            if k in out:
                continue
            # ${1:-default} and ${VAR:-default} both resolve to the default here:
            # a dry run takes no arguments, so the default is what would be used.
            r = re.sub(
                r"\$\{?([A-Za-z0-9_]+)(:-([^}]*))?\}?",
                lambda m: out.get(m.group(1), m.group(3) or ""),
                v,
            )
            if r and "$" not in r:
                out[k] = r
    return out


def cascade_expand(tok: str, env: dict[str, str]) -> str:
    return re.sub(
        r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: env.get(m.group(1), m.group(0)), tok
    )


def scan(text: str, extra: str = "") -> dict[str, set[str]]:
    """Tags, box folders and suffixes a script names, with its own variables resolved;
    the dry-run log is scanned too, since a run that aborts prints only what it reached.
    """
    env = cascade_assignments(text)
    body = text + "\n" + extra
    tags = {cascade_expand(x, env) for x in TAG_FLAGS.findall(body)}
    for m in TAGS_FLAG.findall(body):
        tags.update(cascade_expand(x, env) for x in m.split())
    clean = lambda s: {x for x in s if x and not JUNK.search(x) and "$" not in x}
    return {
        "tags": clean(tags),
        "boxes": clean({cascade_expand(x, env) for x in BOXES_FLAG.findall(body)}),
        "rung3_suffix": clean({cascade_expand(x, env) for x in SUFFIX_FLAG.findall(body)}),
    }


def cascade_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="scratch copy dir (default: a temp dir)")
    a = ap.parse_args(argv)
    work = Path(a.out) if a.out else ROOT / ".handoff_copy"

    print(f"copy   : {work}")
    make_copy(work)
    print(
        "running: promotion_cascade.sh --dry-run  (nothing executes; run() is a no-op when DRY=1)"
    )
    cascade_out = dry_run_cascade(work)
    produced = scan(CASCADE.read_text(), cascade_out)
    consumed = scan(TESTPASS.read_text())
    prereg = CASCADE_PREREG.read_text()
    # A hand-off compared from an empty dry run compares nothing: the cascade's
    # own output is what this check is a verdict about.
    _discovered(
        "tags and box folders in the cascade's dry run",
        len(produced["tags"]) + len(produced["boxes"]),
        CASCADE,
        required=True,
    )
    _discovered(
        "tags and box folders the test pass reads",
        len(consumed["tags"]) + len(consumed["boxes"]),
        TESTPASS,
        required=True,
    )

    print(f"\ncascade would produce tags   : {' '.join(sorted(produced['tags'])) or '(none seen)'}")
    print(f"cascade would write boxes-dir: {' '.join(sorted(produced['boxes'])) or '(none seen)'}")
    print(f"test pass reads tags/suffixes: {' '.join(sorted(consumed['tags'])) or '(none)'}")
    print(f"test pass reads boxes-dir    : {' '.join(sorted(consumed['boxes'])) or '(none)'}")
    print(
        f"test pass rung3-suffix       : {' '.join(sorted(consumed['rung3_suffix'])) or '(none)'}"
    )

    # 1. suffix agreement, on the rung-3 family only
    # _full and _m020 are the whole-image reference and the architecture family, which
    # the cascade never retrains, so comparing them would be a false alarm.
    rung3 = lambda ss: {s for s in ss if "rung3" in s or s in ("_bundle", "_promoted")}
    prod_suffixes = rung3(
        {s for s in produced["tags"] if s.startswith("_")} | produced["rung3_suffix"]
    )
    cons_suffixes = rung3(
        set(consumed["rung3_suffix"]) | {s for s in consumed["tags"] if s.startswith("_")}
    )
    for s in sorted(cons_suffixes):
        if s not in prod_suffixes and prod_suffixes:
            flag(
                "MISMATCH",
                f"test pass reads rung-3 suffix '{s}', cascade writes " f"{sorted(prod_suffixes)}",
                "the test pass would evaluate the pre-promotion models under their old tags",
            )
    # 2. boxes-dir: does the test pass follow a promotion?
    tp_literal = [b for b in sorted(consumed["boxes"]) if not b.startswith("$")]
    for b in tp_literal:
        flag(
            "MISMATCH",
            f"test pass boxes-dir is the fixed literal '{b}'",
            "the cascade's own default is the same folder, so this agrees today. But a promotion "
            "writes its boxes to a NEW folder (localiser_bundle.apply refuses to overwrite), and "
            "nothing propagates that name into the test pass, which would then crop on the "
            "superseded boxes with no error",
        )
    # 3. named tags exist as weights
    for t in sorted(consumed["tags"] | produced["tags"]):
        if t.startswith("-") or t.startswith("_") or "_" not in t:
            continue
        w = ROOT / "outputs" / "weights" / f"{t}_best.weights.h5"
        if not w.exists():
            flag(
                "INFO",
                f"no weights yet for tag '{t}'",
                "expected before the cascade runs; listed so the set is visible",
            )
    # 4. the pre-registration tag table
    for t in sorted(produced["tags"]):
        if t.startswith("_") or t.startswith("-"):
            continue
        if t not in prereg:
            flag(
                "MISMATCH",
                f"cascade tag '{t}' is not in the pre-registration tag table",
                "the test pass evaluates the tags that table names; a tag it does not name "
                "cannot enter the test pass without amending the pre-registration",
            )
    for s in sorted(prod_suffixes):
        if s not in prereg:
            flag(
                "MISMATCH",
                f"cascade suffix '{s}' does not appear in the pre-registration",
                "same reason: the tag table is the authority on what is evaluated",
            )

    print(f"\n--- findings: {len(FINDINGS)} (flagged, NOT fixed) ---")
    for f in FINDINGS:
        print(f"  [{f['severity']}] {f['what']}")
        print(f"      {f['detail']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "handoff_findings.json").write_text(
        json.dumps(
            {
                "produced": {k: sorted(v) for k, v in produced.items()},
                "consumed": {k: sorted(v) for k, v in consumed.items()},
                "findings": FINDINGS,
            },
            indent=2,
        )
    )
    (OUT / "cascade_dry_run.log").write_text(cascade_out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n-> {(OUT / 'handoff_findings.json').relative_to(ROOT)}")
    print(f"-> {(OUT / 'cascade_dry_run.log').relative_to(ROOT)}")
    print("copy removed; the cascade ran on the copy, not on the repo")
    return 0


# CLI
# subcommand -> (entry point, whether it needs TensorFlow and the loader)


# recache-diff
KERAS_MEMBERS = ("run1", "run3", "run4")


def corner_shift(a, b) -> float:
    """Largest corner movement between two boxes, in canvas pixels."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float(max(abs(x - y) for x, y in zip(a, b)))


def recache_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eager-maps", required=True, help="maps folder of the re-cache")
    ap.add_argument("--stored-maps", default=str(config.LOC_BUNDLE_DIR / "maps"))
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "eager_vs_compiled_maps.json"))
    args = ap.parse_args(argv)

    lb = _bundle()
    eager, stored = Path(args.eager_maps), Path(args.stored_maps)
    sel = json.loads((config.LOC_BUNDLE_DIR / "bundle_selection.json").read_text())
    cfg = sel["selected"]
    ens = tuple(cfg["ensemble"].split("+"))
    print(
        f"[discovered] eager maps under {eager}; stored under {stored}; "
        f"promoted ensemble {'+'.join(ens)} (cfg {sel.get('selected_cfg_id')})"
    )

    meta, gts, breasts = lb._load_split_side_tables(args.split, None)
    n = len(meta)
    res: dict = {
        "split": args.split,
        "n": n,
        "members": {},
        "ensemble": {"members": list(ens), "cfg_id": sel.get("selected_cfg_id")},
    }

    # 1 and 2: per member
    rule = dict(
        tta=cfg["tta"],
        breast_filter=cfg["breast_filter"],
        rule=cfg["rule"],
        threshold=cfg["threshold"],
        min_px=cfg["min_px"],
    )
    for m in KERAS_MEMBERS:
        pe = np.load(eager / f"{m}_{args.split}_plain.npy", mmap_mode="r")
        ps = np.load(stored / f"{m}_{args.split}_plain.npy", mmap_mode="r")
        fe = np.load(eager / f"{m}_{args.split}_flip.npy", mmap_mode="r")
        fs = np.load(stored / f"{m}_{args.split}_flip.npy", mmap_mode="r")
        max_abs = 0
        for i in range(0, n, 32):
            a = np.asarray(pe[i : i + 32], np.int16)
            b = np.asarray(ps[i : i + 32], np.int16)
            max_abs = max(max_abs, int(np.abs(a - b).max()))
            a = np.asarray(fe[i : i + 32], np.int16)
            b = np.asarray(fs[i : i + 32], np.int16)
            max_abs = max(max_abs, int(np.abs(a - b).max()))
        moved, shifts, ious_e, ious_s = 0, [], [], []
        for i in range(n):
            be, _ = lb._select_box(
                np.asarray(pe[i], np.float32) / 255.0, gt=None, breast=breasts[i], **rule
            )
            bs, _ = lb._select_box(
                np.asarray(ps[i], np.float32) / 255.0, gt=None, breast=breasts[i], **rule
            )
            d = corner_shift(be, bs)
            if d > 0:
                moved += 1
                shifts.append(d)
            if gts[i] is not None:
                ious_e.append(lb._iou(gts[i], be) if be is not None else 0.0)
                ious_s.append(lb._iou(gts[i], bs) if bs is not None else 0.0)
        rec = {
            "max_abs_map_diff_8bit": max_abs,
            "boxes_moved": moved,
            "n": n,
            "max_corner_shift_px": (max(shifts) if shifts else 0.0),
            "median_corner_shift_px": (float(np.median(shifts)) if shifts else 0.0),
            "mean_box_iou_eager": float(np.mean(ious_e)),
            "mean_box_iou_stored": float(np.mean(ious_s)),
        }
        rec["mean_box_iou_delta"] = rec["mean_box_iou_eager"] - rec["mean_box_iou_stored"]
        res["members"][m] = rec
        print(
            f"  {m:5} map max|diff| {max_abs:3d}/255 boxes moved {moved:3d}/{n} "
            f"max shift {rec['max_corner_shift_px']:.0f}px mean box IoU "
            f"{rec['mean_box_iou_stored']:.4f} -> {rec['mean_box_iou_eager']:.4f} "
            f"({rec['mean_box_iou_delta']:+.4f})"
        )

    # 3: the promoted ensemble, with the re-cached members substituted
    wvec = lb._weights_for_label(ens, cfg.get("weights", "equal"))
    maps_stored = {
        r: (
            np.load(stored / f"{r}_{args.split}_plain.npy", mmap_mode="r"),
            np.load(stored / f"{r}_{args.split}_flip.npy", mmap_mode="r"),
        )
        for r in ens
    }
    maps_mixed = dict(maps_stored)
    for m in KERAS_MEMBERS:
        maps_mixed[m] = (
            np.load(eager / f"{m}_{args.split}_plain.npy", mmap_mode="r"),
            np.load(eager / f"{m}_{args.split}_flip.npy", mmap_mode="r"),
        )
    out = {}
    for label, src in (("as_selected", maps_stored), ("with_eager_keras", maps_mixed)):
        ious, moved_boxes = [], []
        for i in range(n):
            plain, flip = lb._load_row_maps(src, i)
            em = lb._ensemble_map(plain, flip, ens, cfg["tta"], weights=wvec)
            b, _ = lb._select_box(em, gt=None, breast=breasts[i], **rule)
            moved_boxes.append(b)
            if gts[i] is not None:
                ious.append(lb._iou(gts[i], b) if b is not None else 0.0)
        out[label] = {"mean_box_iou": float(np.mean(ious)), "boxes": moved_boxes}
    shifts = [
        corner_shift(a, b)
        for a, b in zip(out["as_selected"]["boxes"], out["with_eager_keras"]["boxes"])
    ]
    n_moved = sum(1 for s in shifts if s > 0)
    res["ensemble"].update(
        {
            "mean_box_iou_as_selected": out["as_selected"]["mean_box_iou"],
            "mean_box_iou_with_eager_keras": out["with_eager_keras"]["mean_box_iou"],
            "delta": out["with_eager_keras"]["mean_box_iou"] - out["as_selected"]["mean_box_iou"],
            "boxes_moved": n_moved,
            "max_corner_shift_px": max(shifts) if shifts else 0.0,
        }
    )
    e = res["ensemble"]
    print(
        f"\n  promoted ensemble box IoU {e['mean_box_iou_as_selected']:.6f} -> "
        f"{e['mean_box_iou_with_eager_keras']:.6f} ({e['delta']:+.6f}); "
        f"{n_moved}/{n} boxes move, max shift {e['max_corner_shift_px']:.0f}px"
    )
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"-> {args.out}")
    return 0


SUBCOMMANDS = {
    "ledger-snapshot": (lambda a: ledger_main(["snapshot"] + a), False),
    "ledger-check": (lambda a: ledger_main(["check"] + a), False),
    "verify-test-pass": (verify_main, False),
    "results-manifest": (manifest_main, False),
    "apply-amendments": (amend_main, False),
    "weights-roundtrip": (roundtrip_main, True),
    "preproc-audit": (preproc_main, True),
    "weights-boxes": (wbe_main, True),
    "control-maps": (control_main, True),
    "launcher-flags": (launcher_main, False),
    "cascade-handoff": (cascade_main, False),
    "recache-diff": (recache_main, False),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("subcommands: " + ", ".join(SUBCOMMANDS))
        return 0
    sub, rest = argv[0], argv[1:]
    if sub not in SUBCOMMANDS:
        print(f"unknown subcommand {sub!r}; one of: {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        return 2
    fn, heavy = SUBCOMMANDS[sub]
    if heavy:
        _heavy()
    rc = fn(rest)
    return 0 if rc is None else int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

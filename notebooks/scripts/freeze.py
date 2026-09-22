#!/usr/bin/env python
"""Creates `docs/.freeze_token`, the file that makes the test partition readable.
Refuses unless the pre-registration already carries the appendix apply-amendments
would write now and its protocol text matches the hash recorded in `refs/`.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PENDING = ROOT / "docs" / "preregistration_amendments_pending.md"
PREREG = ROOT / "docs" / "test_pass_preregistration.md"
TOKEN = ROOT / "docs" / ".freeze_token"
MARK = "<!-- amendments applied at D3 -->"
HEAD_HASH = ROOT / "refs" / "prereg_head.sha256"  # refs/ is never cleaned


def _applier():
    """integrity.py loaded as a module, so the appendix is compared against what its
    own assembler would write."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "integrity", ROOT / "notebooks" / "scripts" / "integrity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def head_sha256(text: str) -> str:
    """SHA-256 of the protocol above the appendix marker (the whole text if absent)."""
    head = text[: text.index(MARK)] if MARK in text else text
    return hashlib.sha256(head.encode()).hexdigest()


def token_text(digest: str, n_blocks: int, when: datetime | None = None) -> str:
    """The token file's contents; the guard and the shell compare it as a
    whitespace-free normal form, so the line layout carries no meaning."""
    when = when or datetime.now(timezone.utc)
    return (
        f"frozen {when.isoformat(timespec='seconds')}\n"
        f"protocol {PREREG.relative_to(ROOT)}\n"
        f"protocol_sha256 {digest}\n"
        f"amendment_blocks {n_blocks}\n"
    )


def recorded_head() -> str | None:
    """The protocol-text hash recorded in refs/, or None if not yet recorded."""
    if not HEAD_HASH.exists():
        return None
    for line in HEAD_HASH.read_text().splitlines():
        if line.startswith("sha256 "):
            return line.split()[1]
    return None


def record_head() -> int:
    """Record the SHA-256 of the protocol text above the appendix marker, once."""
    if TOKEN.exists():
        print("REFUSED: the freeze has happened; the protocol text is already fixed.")
        return 1
    if HEAD_HASH.exists():
        print(
            f"REFUSED: {HEAD_HASH.relative_to(ROOT)} already exists. Delete it deliberately "
            "to record a new hash."
        )
        return 1
    digest = head_sha256(PREREG.read_text())
    HEAD_HASH.write_text(
        f"sha256 {digest}\n"
        f"protocol {PREREG.relative_to(ROOT)} (text above the appendix marker)\n"
        f"recorded {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    )
    print(f"recorded protocol-text sha256 {digest[:16]}")
    print(f"-> {HEAD_HASH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--freeze", action="store_true", help="create the token; without it this only reports"
    )
    ap.add_argument(
        "--record-head",
        action="store_true",
        help="record the protocol text's hash in refs/, once, after it has been read",
    )
    args = ap.parse_args(argv)
    if args.record_head:
        return record_head()

    checks: list[tuple[str, bool, str]] = []

    exists = TOKEN.exists()
    checks.append(
        (
            "the freeze has not already happened",
            not exists,
            f"{TOKEN.relative_to(ROOT)} exists" if exists else "no token",
        )
    )

    applied = PREREG.exists() and MARK in PREREG.read_text()
    checks.append(
        (
            "the pre-registration carries the applied appendix",
            applied,
            (
                "marker present"
                if applied
                else "run integrity.py apply-amendments --apply FIRST — the protocol is "
                "fixed before the partition opens"
            ),
        )
    )

    r = subprocess.run(
        [sys.executable, str(ROOT / "notebooks/scripts/integrity.py"), "apply-amendments"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    consistent = r.returncode == 0
    checks.append(
        (
            "the amendment order table and blocks agree",
            consistent,
            (
                "integrity.py apply-amendments exits 0"
                if consistent
                else (r.stdout.strip().splitlines() or ["failed"])[-1]
            ),
        )
    )

    aa = _applier()
    pending = PENDING.read_text() if PENDING.exists() else ""
    pend = aa.blocks(pending)
    cur = PREREG.read_text() if PREREG.exists() else ""
    same = bool(cur) and aa.document(cur, aa.assemble(pend, aa.order(pending))) == cur
    checks.append(
        (
            "the appendix is exactly what integrity.py apply-amendments would write now",
            same,
            (
                "byte-identical"
                if same
                else "differs: a block changed or was added since --apply, or the appendix was "
                "edited by hand. Re-run integrity.py apply-amendments --apply"
            ),
        )
    )

    want, have = recorded_head(), head_sha256(cur)
    # Before the freeze a missing hash is a state, not a fault: the protocol text is
    # read and recorded once, and --freeze still refuses until it is. Once the token
    # exists the hash must be there and must match.
    unrecorded = want is None and not exists
    checks.append(
        (
            "the protocol text matches its recorded hash",
            want == have or unrecorded,
            (
                f"sha256 {have[:16]}"
                if want == have
                else (
                    "not recorded yet: read the protocol text, then run --record-head"
                    if want is None
                    else f"recorded {want[:16]}, now {have[:16]}"
                )
            ),
        )
    )

    print(f"[freeze] token   : {TOKEN.relative_to(ROOT)}")
    print(f"[freeze] protocol: {PREREG.relative_to(ROOT)}")
    print(f"[freeze] pending : {PENDING.relative_to(ROOT)} ({len(pend)} blocks)\n")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    ok = all(c[1] for c in checks)

    if not args.freeze:
        print("\nREPORT ONLY: no token created. Re-run with --freeze when every check passes.")
        return 0 if ok else 1
    if ok and recorded_head() is None:
        print(
            "\nREFUSING TO FREEZE: the protocol text has no recorded hash. Read it, then "
            "run --record-head."
        )
        return 1
    if not ok:
        print("\nREFUSING TO FREEZE: the protocol is not fixed, so the partition stays closed.")
        return 1

    digest = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    TOKEN.write_text(token_text(digest, len(pend)))
    print("\nFROZEN. The test partition is now readable by the pre-registered pass only.")
    print(f"  protocol sha256 {digest[:16]} ({len(pend)} amendment blocks applied)")
    print(f"-> {TOKEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

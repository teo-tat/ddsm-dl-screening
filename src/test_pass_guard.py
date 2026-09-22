"""Loader-level barrier on the test partition: the loaders that hand out test rows
check here first. `data_loader.split_frames` and `fusion.build_pairs` raise through
`require_test_pass()`; `data_loader.build_datasets` returns a placeholder that refuses
every use. Only a process whose environment token matches the freeze token file is
admitted. `patient_partition` and `_three_way_split` return case-file metadata and are
not guarded.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: Written by notebooks/scripts/freeze.py. It authorises nothing on its own: the pass
#: also exports its contents as ENV_VAR, and authorised() compares the two.
FREEZE_TOKEN_FILE = _ROOT / "docs" / ".freeze_token"
ENV_VAR = "DDSM_TEST_PASS_TOKEN"


def freeze_token() -> str | None:
    """The token, or None if the freeze has not happened."""
    if not FREEZE_TOKEN_FILE.exists():
        return None
    t = FREEZE_TOKEN_FILE.read_text().strip()
    return t or None


def _normalised(token: str) -> str:
    """The token with all whitespace removed: the file spans several lines and
    run_test_pass.sh strips whitespace on export, so both sides compare in this form.
    """
    return "".join(token.split())


def authorised() -> bool:
    tok = freeze_token()
    return bool(tok) and _normalised(os.environ.get(ENV_VAR, "")) == _normalised(tok)


def require_test_pass(caller: str = "") -> None:
    if authorised():
        return
    tok = freeze_token()
    where = f" ({caller})" if caller else ""
    if tok is None:
        raise PermissionError(
            f"refusing to read the test partition{where}: the freeze has not happened. "
            f"{FREEZE_TOKEN_FILE.relative_to(_ROOT)} does not exist, so no code path may "
            f"return test rows. Use --split val. This is the loader-level barrier: it does "
            f"not depend on any launcher passing the right flag."
        )
    raise PermissionError(
        f"refusing to read the test partition{where}: {ENV_VAR} is unset or does not match "
        f"the freeze token. Only notebooks/scripts/run_test_pass.sh in real mode sets it. "
        f"A validation dry run of that script does not, by design."
    )

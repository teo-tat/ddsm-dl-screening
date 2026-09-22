#!/usr/bin/env python
"""test_member_frames.py — the frame assertion refuses a member cached in the wrong frame.

`_assert_member_frames` stops the ensemble averaging a member predicted in a frame it was
never trained in; `POD_MOUNT_ALIASES` maps each pod mount to its laptop store.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "lb", ROOT / "notebooks" / "scripts" / "localiser_bundle.py"
)
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")


def write_manifest(maps: Path, run: str, tensors_dir: str | None) -> None:
    maps.mkdir(parents=True, exist_ok=True)
    body = {"run": run}
    if tensors_dir is not None:
        body["tensors_dir"] = tensors_dir
    (maps / f"{run}_manifest.json").write_text(json.dumps(body))


def asserts_ok(tmp: Path, runs: tuple[str, ...]) -> bool:
    """True if _assert_member_frames accepts this member set."""
    lb._set_bundle_dir(tmp)
    try:
        lb._assert_member_frames(runs)
        return True
    except SystemExit:
        return False


def main() -> int:
    v2 = str(lb.config.OUTPUTS_DIR / "localiser_tensors_v2")
    v3 = str(lb.config.OUTPUTS_DIR / "localiser_tensors_v3")

    print("\n=== _resolve_frame rewrites pod mounts and nothing else ===")
    got, alias = lb._resolve_frame("/workspace/cache_v3_perlesion")
    check(
        "pod mount resolves to its laptop store",
        str(got) == v3 and alias == "/workspace/cache_v3_perlesion",
        str(got),
    )
    got, alias = lb._resolve_frame("/workspace/cache_v3_perlesion/")
    check("trailing slash resolves too", str(got) == v3, str(got))
    got, alias = lb._resolve_frame(v2)
    check(
        "a laptop path is returned unchanged, no alias", str(got) == v2 and alias is None, str(got)
    )
    got, alias = lb._resolve_frame("/workspace/some_other_cache")
    check(
        "an unknown pod path is NOT rewritten",
        str(got) == "/workspace/some_other_cache" and alias is None,
        str(got),
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        maps = tmp / "maps"

        print("\n=== the alias lets a pod-cached member through ===")
        write_manifest(maps, "l6", "/workspace/cache_v3_perlesion")
        check("l6 cached at the pod's v3 mount passes", asserts_ok(tmp, ("l6",)))
        write_manifest(maps, "l6b", "/workspace/cache_v3_perlesion")
        check("l6b cached at the pod's v3 mount passes", asserts_ok(tmp, ("l6b",)))

        print("\n=== a wrong laptop store is refused ===")
        write_manifest(maps, "l6", v2)
        check("l6 cached from v2 (expected v3) is REFUSED", not asserts_ok(tmp, ("l6",)))
        write_manifest(maps, "l5", v3)
        check("l5 cached from v3 (expected v2) is REFUSED", not asserts_ok(tmp, ("l5",)))
        write_manifest(maps, "run4", v2)
        check("run4 cached from v2 (expected v1) is REFUSED", not asserts_ok(tmp, ("run4",)))

        print("\n=== a wrong POD store fails too: the alias is not a blanket pass ===")
        write_manifest(maps, "l6", "/workspace/cache_v2_perlesion")
        check(
            "l6 cached at the pod's v2 mount is REFUSED",
            not asserts_ok(tmp, ("l6",)),
            "alias resolves it to v2, which is still not l6's frame",
        )

        print("\n=== an unlisted member is refused, not swept unchecked ===")
        write_manifest(maps, "l99", v3)
        check("a run absent from MEMBER_TENSORS is REFUSED", not asserts_ok(tmp, ("l99",)))

        print("\n=== a relative path names the same store as the absolute default ===")
        write_manifest(maps, "run1", "outputs/localiser_tensors")
        check("run1 at a relative v1 path passes", asserts_ok(tmp, ("run1",)))
        (maps / "run3_manifest.json").unlink(missing_ok=True)
        write_manifest(maps, "run3", None)
        check("a manifest with no tensors_dir reads as v1 and passes", asserts_ok(tmp, ("run3",)))

    print("\n=== l7 and l8 point at cache v2 ===")
    check("l7 -> cache v2", str(lb.MEMBER_TENSORS["l7"]) == v2, str(lb.MEMBER_TENSORS["l7"]))
    check("l8 -> cache v2", str(lb.MEMBER_TENSORS["l8"]) == v2, str(lb.MEMBER_TENSORS["l8"]))

    # These members train and validate on cache v1, so each must resolve to the
    # test-free v1 export.
    print("\n=== the members trained on cache v1 and their seed replicates point at v1-notest ===")
    v1n = str(lb.config.OUTPUTS_DIR / "localiser_tensors_v1_notest")
    for m in ("l7v", "l7a", "l8a", "l9", "l7v_s1", "l7v_s2", "run4_s1", "run4_s2"):
        got = str(lb.MEMBER_TENSORS.get(m, "MISSING"))
        check(f"{m} -> v1 notest", got == v1n, got)
    # The pod mount these members record resolves to the same store.
    got, alias = lb._resolve_frame("/workspace/cache_v1_perlesion")
    check(
        "pod mount /workspace/cache_v1_perlesion resolves to v1 notest",
        str(got) == v1n and alias == "/workspace/cache_v1_perlesion",
        f"{got} via {alias}",
    )

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

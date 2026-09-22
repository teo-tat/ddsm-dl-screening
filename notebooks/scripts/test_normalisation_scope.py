#!/usr/bin/env python
"""test_normalisation_scope.py — the localiser's clip override stays out of the classifier path.

Both paths share `data_loader._normalise_intensity`. The localiser's upper clip is its own
constant, equal to the crop path's 99.0, so the override is exercised with 99.8. Checks the
constants, the default and override calls, which modules read the localiser constant, and
that decoded validation crops equal the stored crop store.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src import config, data_loader

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one assertion and print its outcome."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def test_constants() -> None:
    """1. The shared clip constants, and the localiser's own."""
    check(
        "config.PERCENTILE_HIGH is 99.0",
        config.PERCENTILE_HIGH == 99.0,
        f"got {config.PERCENTILE_HIGH}",
    )
    check(
        "config.PERCENTILE_LOW is 1.0",
        config.PERCENTILE_LOW == 1.0,
        f"got {config.PERCENTILE_LOW}",
    )
    check(
        "config.LOC_PERCENTILE_HIGH is 99.0, the clip of the localiser tensor cache",
        getattr(config, "LOC_PERCENTILE_HIGH", None) == 99.0,
        f"got {getattr(config, 'LOC_PERCENTILE_HIGH', None)}",
    )


def test_default_unchanged() -> None:
    """2 and 3. The default call is p1/p99; the override is p1/p99.8."""
    rng = np.random.default_rng(config.SEED)
    arr = rng.gamma(2.0, 3000.0, size=(512, 512)).astype(np.float32)

    def reference(a: np.ndarray, high: float) -> np.ndarray:
        """p1/p-high normalisation, including the clip and cast the real function
        ends with."""
        lo, hi = np.percentile(a, [1.0, high])
        out = (a - lo) / (hi - lo + 1e-8)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    expected = reference(arr, 99.0)
    got = data_loader._normalise_intensity(arr, "percentile")
    check(
        "default percentile call == p1/p99",
        np.array_equal(got, expected),
        f"max|d| {np.max(np.abs(got - expected)):.3e}",
    )

    expected8 = reference(arr, 99.8)
    got8 = data_loader._normalise_intensity(arr, "percentile", percentile_high=99.8)
    check(
        "override call == p1/p99.8",
        np.array_equal(got8, expected8),
        f"max|d| {np.max(np.abs(got8 - expected8)):.3e}",
    )
    check(
        "override actually differs from the default",
        not np.allclose(got, got8),
        f"max|d| {np.max(np.abs(got - got8)):.3e}",
    )

    # The other strategies ignore the override argument.
    a = data_loader._normalise_intensity(arr, "minmax")
    b = data_loader._normalise_intensity(arr, "minmax", percentile_high=99.8)
    check("non-percentile strategies ignore the override", np.array_equal(a, b))


def _reads_loc_constant(path: Path) -> bool:
    """True when the module reads config.LOC_PERCENTILE_HIGH. Parsed from the AST, so
    naming it in a docstring or comment does not count."""
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "LOC_PERCENTILE_HIGH":
            return True
        if isinstance(node, ast.Name) and node.id == "LOC_PERCENTILE_HIGH":
            return True
    return False


def test_scope() -> None:
    """4. Only the localiser reads the localiser constant."""
    allowed = {"src/config.py", "src/precompute_localiser.py"}
    offenders = [
        p.as_posix()
        for p in sorted(Path(".").glob("src/**/*.py"))
        if p.as_posix() not in allowed and _reads_loc_constant(p)
    ]
    check(
        "LOC_PERCENTILE_HIGH is read only by config/precompute",
        not offenders,
        f"offenders: {offenders}" if offenders else "",
    )

    for name in (
        "src/train.py",
        "src/train_crop_store.py",
        "src/evaluate.py",
        "src/fusion.py",
        "src/data_loader.py",
    ):
        f = Path(name)
        if f.exists():
            check(f"{name} does not read LOC_PERCENTILE_HIGH", not _reads_loc_constant(f))


def test_crop_store_identical(rows: int) -> None:
    """5. The classifier crop path reproduces the stored validation crops."""
    store = config.OUTPUTS_DIR / "crop_store_m020"
    if not (store / "val_images.npy").exists():
        check("crop store present for the byte-equality check", False, f"missing {store}")
        return

    ref = np.load(store / "val_images.npy", mmap_mode="r")
    n = min(rows, len(ref))
    print(f"  decoding {n} validation crops through the real pipeline")
    _, val_ds, _, _, _ = data_loader.build_datasets(
        train_csv=config.TRAIN_CSV,
        test_csv=config.TEST_CSV,
        train_img_dir=config.TRAIN_IMG_DIR,
        test_img_dir=config.TEST_IMG_DIR,
        train_roi_dir=config.TRAIN_ROI_DIR,
        test_roi_dir=config.TEST_ROI_DIR,
        target_size=config.TARGET_SIZE,
        batch_size=config.BATCH_SIZE,
        normalisation="scratch",
        augment=False,
        source="crop",
        crop_sizing="mask_margin",
        crop_margin=0.20,
        shuffle_train=False,
        standardize=False,
    )

    got, seen = [], 0
    for x, _ in val_ds:
        got.append(x.numpy())
        seen += len(x)
        if seen >= n:
            break
    got = np.concatenate(got)[:n]
    expected = np.asarray(ref[:n])
    same = np.array_equal(got, expected)
    check(
        "classifier crops are byte-identical to the crop store",
        same,
        (f"n={n}, max|d| {np.max(np.abs(got - expected)):.3e}" if not same else f"n={n}, exact"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=32,
        help="validation crops to re-decode for the byte-equality check",
    )
    ap.add_argument("--skip-decode", action="store_true", help="skip check 5 (the only slow one)")
    args = ap.parse_args()

    print("1. constants")
    test_constants()
    print("2/3. normalisation call")
    test_default_unchanged()
    print("4. scope")
    test_scope()
    if not args.skip_decode:
        print("5. crop-store byte equality")
        test_crop_store_identical(args.rows)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED — the classifier path keeps its p1/p99 normalisation.")


if __name__ == "__main__":
    main()

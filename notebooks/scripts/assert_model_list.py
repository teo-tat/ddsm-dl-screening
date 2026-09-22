#!/usr/bin/env python
"""Matches the tags the test pass will score against the pre-registration's table,
in both directions. The match is on the stem: a trailing `_s<N>`, the promotion
suffix and the ledger suffix are stripped, and everything else must match literally.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs" / "test_pass_preregistration.md"


def prereg_tags(text: str) -> dict[str, list[str]]:
    """{tag: [alternatives]} for every tag in the table headed `| tag | role |`; a
    `{a|b}` placeholder expands to one alternative per branch."""
    out: dict[str, list[str]] = {}
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("| tag | role |"):
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                # The document carries more than one tag table, so the scan resumes
                # rather than stopping at the end of the first.
                in_table = False
                continue
            if set(line.strip()) <= set("|-: "):
                continue
            # An escaped pipe inside a placeholder (`vgg16_fusion_{view\|context}`)
            # is not a column separator.
            row = line.replace("\\|", "\x00")
            cell = row.split("|")[1].strip().replace("\x00", "|")
            # A cell may join two tags with the word "and", not only a comma.
            prev_stem = None
            for raw in re.split(r",|\s+and\s+", cell):
                tag = raw.strip()
                # `vgg16_crop_rung3_unet_bundle_s1, _s2` — a fragment starting with
                # `_` continues the previous tag's stem rather than being one.
                if tag.startswith("_") and prev_stem:
                    tag = prev_stem + tag
                elif tag and not tag.startswith("("):
                    prev_stem = re.sub(r"_s\d+$", "", tag)
                if not tag or tag.startswith("("):
                    continue
                tag = re.sub(r"\s*\(PLACEHOLDER[^)]*\)", "", tag).strip()
                # Prose shorthand for a family: `baseline/scaled/regularised _crop_m020`
                fam = re.match(r"^([\w/]+)\s+(_\S+)$", tag)
                if fam and "/" in fam.group(1):
                    for part in fam.group(1).split("/"):
                        out[f"{part}{fam.group(2)}"] = [f"{part}{fam.group(2)}"]
                    continue
                m = re.search(r"\{([^}]+)\}", tag)
                if m:
                    out[tag] = [tag.replace(m.group(0), alt) for alt in m.group(1).split("|")]
                else:
                    out[tag] = [tag]
    return out


def substitutions(text: str) -> dict[str, str]:
    """{pre-registered tag: tag as measured}, read from the table headed
    `| pre-registered tag | tag as measured |` so the document names both literals."""
    out: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("| pre-registered tag | tag as measured |"):
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                in_table = False
                continue
            if set(line.strip()) <= set("|-: "):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and cells[1]:
                out[cells[0]] = cells[1]
    return out


def stem_of(tag: str, ledger: str, rung3_suffix: str) -> str:
    """The pre-registered system a measured tag belongs to: a trailing `_ens3`, a
    trailing `_s<N>`, the promotion suffix and the ledger suffix are stripped."""
    s = tag
    if s.endswith("_ens3"):
        s = s[: -len("_ens3")]
    s = re.sub(r"_s\d+$", "", s)
    for suffix in (rung3_suffix, ledger):
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tags-file", required=True, help="the model list the pass built, one tag per line"
    )
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--rung3-suffix", required=True)
    ap.add_argument("--prereg", default=str(PREREG))
    args = ap.parse_args(argv)

    prereg_path = Path(args.prereg)
    declared = prereg_tags(prereg_path.read_text())
    scored = [t.strip() for t in Path(args.tags_file).read_text().splitlines() if t.strip()]
    # The protocol may be given as a path outside the repo, so the printed name
    # falls back to the path as given.
    try:
        shown = prereg_path.resolve().relative_to(ROOT)
    except ValueError:
        shown = prereg_path
    print(
        f"[discovered] {len(declared)} tag rows in {shown}; "
        f"{len(scored)} tags the pass will score"
    )
    if not declared or not scored:
        raise SystemExit("REFUSED: one of the two lists is empty, so there is nothing to compare.")

    # A pre-registered row is satisfied by the tag the substitution table, where the
    # document carries one, says it is now measured as.
    subs = substitutions(prereg_path.read_text())
    if subs:
        print(f"[discovered] {len(subs)} substitution rows (the tag as measured)")
    alt_to_row, excused = {}, set()
    for row, alts in declared.items():
        for alt in alts:
            alt_to_row[alt] = row
            measured = subs.get(alt)
            if measured == "not scored":
                excused.add(row)
                continue
            if measured:
                # The exact measured name identifies its row; the stem is a fallback
                # and the first row to claim it keeps it.
                alt_to_row[measured] = row
                alt_to_row.setdefault(stem_of(measured, args.ledger, args.rung3_suffix), row)
    unmatched, mapping = [], {}
    for tag in scored:
        stem = stem_of(tag, args.ledger, args.rung3_suffix)
        row = alt_to_row.get(tag) or alt_to_row.get(stem)
        if row is None:
            unmatched.append((tag, stem))
        else:
            mapping.setdefault(row, []).append(tag)

    print("\n  pre-registered row                                    scored as")
    for row in declared:
        got = mapping.get(row, [])
        shown = (
            ", ".join(got) if got else ("not scored (declared)" if row in excused else "— NOTHING")
        )
        print(f"  {row:52} {shown[:110]}")

    uncovered = [r for r in declared if r not in mapping and r not in excused]
    if excused:
        print(
            f"\n  declared 'not scored' by the document ({len(excused)}): "
            + ", ".join(sorted(excused))
        )
    bad = []
    if uncovered:
        bad.append(
            f"{len(uncovered)} pre-registered tag(s) no scored model covers: "
            + ", ".join(uncovered)
        )
    if unmatched:
        bad.append(
            f"{len(unmatched)} scored tag(s) the pre-registration does not name: "
            + ", ".join(f"{t} (stem {s})" for t, s in unmatched)
        )
    if bad:
        print(
            "\nMODEL LIST DOES NOT MATCH THE PRE-REGISTRATION:", *bad, sep="\n  ", file=sys.stderr
        )
        print(
            "  The test partition is scored once; a mismatch here means scoring the "
            "wrong models with no second attempt.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\n  model list matches: {len(declared)} pre-registered rows, all covered; "
        f"{len(scored)} scored tags, all named."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

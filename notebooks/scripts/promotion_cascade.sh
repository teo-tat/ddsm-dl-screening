#!/usr/bin/env bash
# promotion_cascade.sh — everything that must be redone when a localiser is promoted:
# the crop store rung 3 trains on, the fusion pair store and every number derived from
# them, so a promotion cannot land with half the ladder on the previous localiser.
#
# bash notebooks/scripts/promotion_cascade.sh --dry-run <boxes-dir>
# bash notebooks/scripts/promotion_cascade.sh <boxes-dir>
#
# Family A, the uncertainty/deferral thread and the re-ranker are NOT recomputed; they
# are labelled "measured on the runs 1+3+4 localiser" wherever they appear.
#
# The training steps are CUDA jobs: tensorflow-metal 1.2.0 does not apply ReLU in the
# compiled graph, so anything model.fit produces on this laptop is inadmissible.
set -u
# The self-copy is sourced HERE, before a single argument is consumed. It re-execs with
# "$@", so a `shift` above this line is lost from the run that actually happens.
OUTDIR=${OUTDIR:-outputs/logs/cascade_$(date +%Y%m%d_%H%M%S)}
. notebooks/scripts/launcher_selfcopy.sh || { echo "launcher_selfcopy.sh missing from this checkout — stale overlay, refusing to run without launch provenance" >&2; exit 1; }
export PYTHONPATH=. # the scripts below import src

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
# The boxes folder is required and has no default: a default would name the promotion
# being superseded, so a bare invocation would cascade and arm the test pass on it.
if [ -z "${1:-}" ]; then
    echo "usage: promotion_cascade.sh [--dry-run] <boxes-dir>"
    echo
    echo "  The boxes folder is REQUIRED and has no default. Pass the promotion"
    echo "  candidate's folder explicitly; a default here would point at the"
    echo "  promotion being superseded."
    exit 2
fi
BOXES=$1
PY=.ddsm-env/bin/python
M=0.20
# The base run is seed 28 and is built by src.ladder; the replicates are seeds 1 and 2
# and go through src.train --seed, because src.ladder has no --seed.
REPLICATE_SEEDS="1 2"
# One tag convention, defined here and threaded below. run_test_pass.sh reads it back
# out of the pointer rather than keeping a second copy that could name other runs.
SUFFIX=${SUFFIX:-_promoted}
# SUFFIX names the STORES, which are reused across ledgers and never renamed;
# TAG_SUFFIX names the RUNS, the store suffix plus the ledger the pointer must carry.
LEDGER=$(PYTHONPATH=. .ddsm-env/bin/python -c "from src import config; print(config.LEDGER_SUFFIX)")
TAG_SUFFIX=${SUFFIX}${LEDGER}
RUNG3_TAG=vgg16_crop_rung3_unet${TAG_SUFFIX}
# stores.py crop names the folder outputs/crop_store_<tag>, so the tag is the
# condition, not the folder: passing crop_store_... would double the prefix.
CROP_STORE=outputs/crop_store_rung3_unet${SUFFIX}
# src.fusion builds its tag as {arch}_fusion_{pair}[_unet]{suffix}, so with
# --box-source unet the runs land as vgg16_fusion_view_unet<suffix>.
FUSION_TAG=vgg16_fusion_view_unet${TAG_SUFFIX}
FUSION_STORE_TAG=vgg16_fusion_view_unet${SUFFIX} # -> outputs/fusion_store_<tag>
POINTER=refs/promoted_pointer.json # refs/ is never cleaned (refs/README.md)
FAIL=0

# What this run states it will rewrite in the reference ledger
# Rewriting these validation tables is the cascade's purpose, so step 8 is given this
# declared set and reports the difference in both directions, not "nothing moved".
DECLARED_TABLES=""
declare_table() { DECLARED_TABLES="$DECLARED_TABLES $1"; }
[ "${SKIP_RUNG3_HALF:-0}" = "1" ] || declare_table "${RUNG3_TAG}_ens3_predictions_val.csv"
if [ "${SKIP_FUSION_HALF:-0}" != "1" ]; then
  declare_table "${FUSION_TAG}_predictions_val.csv"
  declare_table "${FUSION_TAG}_s1_predictions_val.csv"
  declare_table "${FUSION_TAG}_s2_predictions_val.csv"
  declare_table "${FUSION_TAG}_ens3_predictions_val.csv"
fi
declare_table "rung3_on_pairs${TAG_SUFFIX}_predictions_val.csv"
# A declared table must have been WRITTEN by this run. Identical bytes are a correct
# outcome of a deterministic re-run; a file older than the run means it did not run.
RUN_START=$(date +%s)

say() { echo; echo "── $* ─────────────────────────────────────────────"; }
need() { if [ -e "$1" ]; then echo "  ok       $1"; else echo "  MISSING  $1"; FAIL=1; fi; }
run() {
    echo "  \$ $*"
    if [ "$DRY" = "0" ]; then "$@" || { echo "  FAILED"; exit 1; }; fi
}

echo "promotion cascade | boxes: $BOXES | suffix: $SUFFIX | mode: $([ $DRY = 1 ] && echo DRY-RUN || echo EXECUTE)"

say "0a. ledger snapshot"
# Steps 3, 5 and 6 write {tag}_predictions_val.csv, and some of those filenames are the
# reference ledger's own, so this run snapshots before and checks after.
run $PY notebooks/scripts/integrity.py ledger-snapshot

say "0b. §5c — resolved paths, printed beside what they are intended to be"
# A script that consumes a promotion prints where it actually resolved to and asserts
# it is not the promotion being replaced; printing alone would not be a guard.
BOXES_ABS=$(cd "$BOXES" 2>/dev/null && pwd -P || echo "MISSING:$BOXES")
PROMOTED_ABS=$(cd outputs/results/localiser_bundle/boxes 2>/dev/null && pwd -P || echo "-")
printf "  %-24s %s\n" "boxes-dir given" "$BOXES"
printf "  %-24s %s\n" "resolves to" "$BOXES_ABS"
printf "  %-24s %s\n" "current promotion" "$PROMOTED_ABS"
printf "  %-24s %s\n" "rung-3 suffix" "$TAG_SUFFIX  (store suffix $SUFFIX + ledger $LEDGER)"
printf "  %-24s %s\n" "pointer to be written" "$POINTER"
case "$BOXES_ABS" in
  MISSING:*) echo "  the boxes folder does not exist; nothing read"; exit 2;;
esac
if [ "$BOXES_ABS" = "$PROMOTED_ABS" ]; then
    echo
    echo "REFUSING: the boxes folder IS the current promotion"
    echo "  ($PROMOTED_ABS)."
    echo "  A cascade re-runs the pipeline onto a NEW promotion candidate. Pointing it at"
    echo "  the incumbent would retrain everything on the boxes already in use and then"
    echo "  arm the test pass on them, which is not a promotion and is not a no-op either."
    exit 2
fi
for s in val train; do
    f="$BOXES/localiser_boxes_${s}.csv"
    [ -f "$f" ] && printf "  %-24s %s  %s\n" "md5 ${s}" "$(md5 -q "$f" 2>/dev/null || md5sum "$f" | cut -d" " -f1)" "$f"
done
echo

# The selection swap, and the trap that makes it a transaction
# Step 9 records the LIVE bundle selection, so the cascade runs with the candidate
# swapped in, and the swap and the check that validates it are one transaction:
#
# swap in -> run -> floor holds -> pointer written -> the candidate STAYS live
# -> anything else -> incumbent restored, NO pointer
#
# The trap covers EXIT, INT and TERM: an interrupt between the swap and the floor gate
# must not leave a promotion half-made.
LIVE_SEL=outputs/results/localiser_bundle/bundle_selection.json
INCUMBENT_SEL=outputs/results/localiser_bundle/selected_runs134/bundle_selection.json
SWAPPED=0
PROMOTED=0
RESTORE_DONE=0
restore_incumbent () {
    # TERM fires the handler and then EXIT fires it again, and a verdict printed
    # twice reads as a second failure, so the handler runs once.
    [ "$RESTORE_DONE" = "1" ] && return 0
    RESTORE_DONE=1
    [ "$SWAPPED" = "1" ] || return 0
    if [ "$PROMOTED" = "1" ]; then
        echo
        echo "── the promotion completed: the candidate selection stays live ──"
        echo "  incumbent preserved at $INCUMBENT_SEL"
        return 0
    fi
    echo
    echo "── restoring the incumbent selection (the promotion did not complete) ──"
    cp "$INCUMBENT_SEL" "$LIVE_SEL"
    $PY - "$LIVE_SEL" "$INCUMBENT_SEL" <<'PYR'
import json, sys
live = json.loads(open(sys.argv[1]).read()); inc = json.loads(open(sys.argv[2]).read())
ok = live.get("selected_cfg_id") == inc.get("selected_cfg_id")
print(f"  live cfg {live.get('selected_cfg_id')} == incumbent cfg {inc.get('selected_cfg_id')}: "
      f"{'RESTORED' if ok else 'RESTORE FAILED'}")
sys.exit(0 if ok else 1)
PYR
    echo "  no pointer was written, so the test pass is not armed."
}
trap restore_incumbent EXIT INT TERM

say "0c. live selection set to the candidate"
if [ ! -f "$BOXES/bundle_selection.json" ]; then
    echo "  $BOXES/bundle_selection.json is missing."
    echo "  A boxes folder must carry the selection that produced it, so the cascade can set"
    echo "  it live without a second source of truth. Copy it in beside the box tables."
    exit 2
fi
$PY - "$BOXES" <<'PYS' || exit 2
import json, pathlib, sys
b = pathlib.Path(sys.argv[1])
sel = json.loads((b / "bundle_selection.json").read_text())
man = json.loads((b / "bundle_manifest.json").read_text())
if sel.get("selected") != man.get("selected"):
    print("  the folder's selection and its manifest disagree:")
    print(f"    selection: {sel.get('selected')}")
    print(f"    manifest : {man.get('selected')}")
    sys.exit(1)
print(f"  cfg {sel.get('selected_cfg_id')}  {sel['selected'].get('ensemble')}")
print(f"  weights {sel['selected'].get('weights')}  "
      f"rule {sel['selected'].get('rule')} threshold {sel['selected'].get('threshold')} "
      f"min_px {sel['selected'].get('min_px')} tta {sel['selected'].get('tta')}")
print("  folder selection and manifest agree")
PYS
if [ "$DRY" = "1" ]; then
    echo "  DRY RUN: the live selection is NOT touched."
else
    cp "$BOXES/bundle_selection.json" "$LIVE_SEL"
    SWAPPED=1
    echo "  live selection is now the candidate; the incumbent is restored on any"
    echo "  failure, interrupt or abort before the pointer is written."
fi
echo

say "0. inputs the cascade reads"
for s in val train; do need "$BOXES/localiser_boxes_${s}.csv"; done
need "outputs/results/localiser_bundle/bundle_selection.json"
need "outputs/results/localiser_bundle/selected_runs134/bundle_selection.json"
need "docs/run4_recipe.json"
$PY - "$BOXES" <<'PY'
import json, pathlib, sys
lb = pathlib.Path("outputs/results/localiser_bundle")
try:
    live = json.loads((lb / "bundle_selection.json").read_text())["selected"]
    frozen = json.loads((lb / "selected_runs134" / "bundle_selection.json").read_text())["selected"]
    print(f"  live selection   : {live.get('ensemble')} rule={live.get('rule')} "
          f"thr={live.get('threshold')} min_px={live.get('min_px')}")
    print(f"  frozen (runs134) : {frozen.get('ensemble')}")
    print(f"  {'MATCH — nothing promoted yet' if live == frozen else 'DIFFER — a promotion is in effect'}")
except Exception as e:
    print("  could not read the selection:", e)
PY

# 1, 2 and 3: the rung-3 half
# SKIP_RUNG3_HALF=1 asserts the artefacts instead of rebuilding them: step 2 calls
# model.fit, which is banned here, and would overwrite the runs the floor gate reads.
if [ "${SKIP_RUNG3_HALF:-0}" = "1" ]; then
  say "1+2+3. rung-3 half: PRE-COMPUTED, asserted not rebuilt"
  echo "  reason 1 (binding): step 2 calls model.fit; laptop-GPU training banned 9 Sep."
  echo "  reason 2: it would overwrite the very runs the 8b floor gate reads."
  echo "  reason 3: src.ladder/src.train is a third trainer in a comparison where the"
  echo "  0.814022 reference and these runs both came from train_crop_store."
  if $PY - "$CROP_STORE" "$BOXES" "$RUNG3_TAG" "$REPLICATE_SEEDS" <<'PYSKIP3'
import hashlib, json, pathlib, sys
store, boxes, tag, seeds = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4].split()
res = pathlib.Path("outputs/results")
bad, md5s = [], {}
tags = [tag] + [f"{tag}_s{s}" for s in seeds]

# A1. the crop store exists and its manifest records THESE boxes
mf = store / "manifest.json"
if not mf.exists():
    bad.append(f"A1: no crop store manifest at {mf}")
else:
    m = json.loads(mf.read_text())
    if str(pathlib.Path(m.get("boxes_dir", ""))) != str(pathlib.Path(boxes)):
        bad.append(f"A1: store boxes_dir {m.get('boxes_dir')} != cascade BOXES {boxes}")
    else:
        print(f"  ok   A1 store built from {m['boxes_dir']} (margin {m.get('margin')})")

# A2. all four prediction tables exist (the three seeds and ens3)
for t in tags + [f"{tag}_ens3"]:
    pt = res / f"{t}_predictions_val.csv"
    if not pt.exists():
        bad.append(f"A2: missing prediction table {pt.name}")
    else:
        md5s[t] = hashlib.md5(pt.read_bytes()).hexdigest()
print(f"  {'ok  ' if not any(b.startswith('A2') for b in bad) else 'FAIL'} A2 "
      f"{len(md5s)}/{len(tags) + 1} prediction tables present")

# A3. every seed run has a train manifest
mans = {}
for t in tags:
    rm = res / f"{t}_train_manifest.json"
    if not rm.exists():
        bad.append(f"A3: missing train manifest for {t}")
    else:
        mans[t] = json.loads(rm.read_text())
print(f"  {'ok  ' if len(mans) == len(tags) else 'FAIL'} A3 {len(mans)}/{len(tags)} train manifests")

# A4. every train manifest records THESE boxes, and the seed set is exactly the
#     one the floor gate averages over
got = set()
for t, m in mans.items():
    if str(pathlib.Path(m.get("boxes_dir") or "")) != str(pathlib.Path(boxes)):
        bad.append(f"A4: {t} records boxes_dir {m.get('boxes_dir')}, not {boxes}")
    got.add(m.get("seed"))
want = {28} | {int(s) for s in seeds}
if mans and got != want:
    bad.append(f"A4: seeds on disk {sorted(got)} != the floor gate's {sorted(want)}")
elif mans:
    print(f"  ok   A4 all {len(mans)} runs record {boxes}, seeds {sorted(got)}")

# A5. every run came from THIS crop store — the like-for-like assertion
for t, m in mans.items():
    st = str(m.get("trained_from_store") or "")
    if not st:
        bad.append(f"A5: {t} records no trained_from_store")
    elif pathlib.Path(st).name != store.name:
        bad.append(f"A5: {t} was trained from {st}, not a {store.name}")
if mans and not any(b.startswith("A5") for b in bad):
    print(f"  ok   A5 all {len(mans)} runs trained from {store.name} "
          f"(train_crop_store, not the ladder's DICOM path)")

if bad:
    print("  FAILED:"); [print("    -", b) for b in bad]; sys.exit(1)
pathlib.Path("outputs/results/.rung3_half_skip.json").write_text(json.dumps(
    {"skipped": True, "store": str(store), "store_boxes_dir": boxes,
     "tags": tags, "ens3_tag": f"{tag}_ens3", "prediction_table_md5": md5s,
     "seeds": sorted(int(s) for s in got if s is not None),
     "reasons": [
        "step 2 calls model.fit, and this project trains nothing on the laptop GPU",
        "it would overwrite the runs the 8b floor gate reads",
        "src.ladder/src.train is a third trainer in a comparison where the 0.814022 "
        "reference and these runs both came from train_crop_store"]}, indent=2))
print("  all five assertions hold; -> outputs/results/.rung3_half_skip.json")
PYSKIP3
  then
    :
  else
    echo "RUNG3-HALF ASSERTIONS FAILED — refusing to skip. Either the artefacts are"
    echo "  absent or they were not built from these boxes; rerun without SKIP_RUNG3_HALF=1."
    exit 1
  fi
else
say "1. crop store on the promoted boxes  (rung 3 trains on this)"
run $PY notebooks/scripts/stores.py crop --box-source unet --boxes-dir "$BOXES" \
        --margin $M --tag rung3_unet${SUFFIX}

say "2. rung 3 — 3 seeds  [CUDA]"
# Base run, seed 28: src.ladder builds the rung-3 condition and tags it
# vgg16_crop_rung3_unet${TAG_SUFFIX}.
run $PY -m src.ladder --stage train --rungs 3 --arch vgg16 --margin $M \
        --boxes-dir "$BOXES" --rung3-suffix "$TAG_SUFFIX"
# Replicates, seeds 1 and 2: src.train is what seeds a run. It appends _s<N> to
# the tag itself, so these land as vgg16_crop_rung3_unet${TAG_SUFFIX}_s1 / _s2.
for s in $REPLICATE_SEEDS; do
  run $PY -m src.train --model vgg16 --source crop --box-source unet \
          --boxes-dir "$BOXES" --crop-margin $M \
          --tag-suffix "_rung3_unet${TAG_SUFFIX}" --seed "$s"
done

say "3. rung 3 seed ensemble (ens3)"
# This step READS the three rung-3 seed prediction tables and never writes them:
# training produces weights, histories and manifests, not scores.
run $PY -m src.ensemble --tags "$RUNG3_TAG" "${RUNG3_TAG}_s1" "${RUNG3_TAG}_s2" \
        --out-tag "${RUNG3_TAG}_ens3" --method mean --splits val
fi

# 4 and 5: the fusion half
# SKIP_FUSION_HALF=1 asserts the artefacts instead of rebuilding them: step 5 trains
# through src.fusion, while the existing runs came from train_fusion_store, and a
# second trainer inside one comparison is not like-for-like.
if [ "${SKIP_FUSION_HALF:-0}" = "1" ]; then
  say "4+5. fusion half: PRE-COMPUTED, asserted not rebuilt"
  echo "  reason: step 5 trains via src.fusion --stage train (DICOM path); the CUDA"
  echo "  fusion ledger and the 11 Sep runs both came from train_fusion_store. Running"
  echo "  it would introduce a third trainer into a like-for-like comparison."
  FUSION_STORE_DIR=outputs/fusion_store_${FUSION_STORE_TAG}
  if $PY - "$FUSION_STORE_DIR" "$BOXES" "$FUSION_TAG" <<'PYSKIP'
import hashlib, json, pathlib, sys
store, boxes, tag = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
res = pathlib.Path("outputs/results")
bad, md5s = [], {}
# 1. the store exists and was built from THESE boxes
mf = store / "manifest.json"
if not mf.exists():
    bad.append(f"no fusion store manifest at {mf}")
else:
    m = json.loads(mf.read_text())
    if str(pathlib.Path(m.get("boxes_dir", ""))) != str(pathlib.Path(boxes)):
        bad.append(f"store boxes_dir {m.get('boxes_dir')} != cascade BOXES {boxes}")
    else:
        print(f"  ok   store built from {m['boxes_dir']}")
# 2. the three prediction tables and ens3 exist; 3. each ties to that store
for t in (tag, f"{tag}_s1", f"{tag}_s2", f"{tag}_ens3"):
    pt = res / f"{t}_predictions_val.csv"
    if not pt.exists():
        bad.append(f"missing prediction table {pt.name}"); continue
    md5s[t] = hashlib.md5(pt.read_bytes()).hexdigest()
    if t.endswith("_ens3"):
        print(f"  ok   {t} ({md5s[t][:12]})"); continue
    rm = res / f"{t}_manifest.json"
    if not rm.exists():
        bad.append(f"missing run manifest for {t}"); continue
    st = str(json.loads(rm.read_text()).get("store", ""))
    if store.name not in st:
        bad.append(f"{t} was trained from {st}, not {store.name}")
    else:
        print(f"  ok   {t} trained from {store.name} ({md5s[t][:12]})")
if bad:
    print("  FAILED:"); [print("    -", b) for b in bad]; sys.exit(1)
pathlib.Path("outputs/results/.fusion_half_skip.json").write_text(json.dumps(
    {"skipped": True, "store": str(store), "store_boxes_dir": boxes,
     "prediction_table_md5": md5s,
     "reason": ("step 5 trains via src.fusion --stage train; the CUDA fusion ledger and these "
                "runs came from train_fusion_store. Rebuilding would introduce a third trainer "
                "into a like-for-like comparison.")}, indent=2))
print("  all assertions hold; -> outputs/results/.fusion_half_skip.json")
PYSKIP
  then
    :
  else
    echo "FUSION-HALF ASSERTIONS FAILED — refusing to skip. Either the artefacts are"
    echo "  absent or they were not built from these boxes; rerun without SKIP_FUSION_HALF=1."
    exit 1
  fi
else
say "4. fusion pair store on the promoted boxes"
run $PY notebooks/scripts/stores.py fusion --model vgg16 --pair view \
        --box-source unet --boxes-dir "$BOXES" --margin $M --tag "$FUSION_STORE_TAG"

say "5. view fusion — 3 seeds, then ens3  [CUDA]   (S4 job (e) rerun)"
run $PY -m src.fusion --arch vgg16 --pair view --stage train --margin $M \
        --box-source unet --boxes-dir "$BOXES" --tag-suffix "$TAG_SUFFIX"
for s in $REPLICATE_SEEDS; do
  run $PY -m src.fusion --arch vgg16 --pair view --stage train --margin $M \
          --box-source unet --boxes-dir "$BOXES" --tag-suffix "$TAG_SUFFIX" --seed "$s"
done
run $PY -m src.ensemble --tags "$FUSION_TAG" "${FUSION_TAG}_s1" "${FUSION_TAG}_s2" \
        --out-tag "${FUSION_TAG}_ens3" --method mean --splits val
fi

say "6. recomputed from PREDICTION TABLES only — no retraining, laptop, validation"
# Family B, the ladder as pre-registered: rungs 2, 3, 4, 6 and 7 give 10 pairwise
# DeLong tests. Rungs 1 and 5 stay out; adding them would change the Holm family.
#
# Every comparator comes from the ONE ledger, and rung 3 is the base seed-28 run —
# not _ens3, because every other slot is a single run.
run $PY -m src.compare --split val --models \
        vgg16_crop_rung2_oracle${LEDGER} \
        ${RUNG3_TAG} \
        vgg16_crop_rung4_cam${LEDGER} \
        vgg16_crop_rung6_shuffled${LEDGER} \
        vgg16_crop_rung7_whole${LEDGER}
# analyse_ladder.py errors and src.explain belong to step 7: both need an artefact this
# cascade does not produce, and neither is on the gate or the pointer path.
run $PY notebooks/scripts/analyse_fusion.py controls \
        --pairs "outputs/fusion_store_${FUSION_STORE_TAG}/val_meta.csv" \
        --rung3 "outputs/results/${RUNG3_TAG}_ens3_predictions_val.csv" \
        --tag "rung3_on_pairs${TAG_SUFFIX}"
run $PY notebooks/scripts/analyse_fusion.py compare --split val \
        --fusion "outputs/results/${FUSION_TAG}_ens3_predictions_val.csv" \
        --controls "outputs/results/rung3_on_pairs${TAG_SUFFIX}_predictions_val.csv" \
        --oracle-fusion "outputs/results/vgg16_fusion_view${LEDGER}_ens3_predictions_val.csv" \
        --out "outputs/results/fusion_pair_comparisons_val${TAG_SUFFIX}.json"

say "7. NOT recomputed — labelled 'measured on the runs 1+3+4 localiser' (§7b)"
echo "  Family A (architecture comparison at the rung-3 condition)"
echo "  §12 localiser uncertainty, deferral curve, §12b/§12c mechanism tests"
echo "  the component re-ranker (§15/§15a)"
echo "  analyse_ladder.py errors — the rung-2/3/5 error frame."
echo "    It needs rung 5 on these boxes, which does not exist and which this"
echo "    cascade does not build (step 2 trains rung 3 only). Rung 5 is"
echo "    DESCRIPTIVE BY DECLARATION — outside Family B, whose five members the"
echo "    pre-registration fixes on a power argument — and it sits on neither the"
echo "    8b gate nor the step-9 pointer, so it cannot block a promotion"
echo "    transaction. Building it would be training a model to satisfy a step"
echo "    that does not need it; --tag5 _bundle would read rung 5 off the OLD box"
echo "    path inside a frame whose other rungs are on the new one, which is the"
echo "    mixing A24 replaced. Measured on the runs 1+3+4 localiser."
echo "  src.explain — Grad-CAM on the promoted rung 3."
echo "    PENDING, not declined. The weights ARE on this laptop — all three seeds"
echo "    are in outputs/_cuda/s8_weights/ — but src.explain and run_test_pass.sh"
echo "    both look in outputs/weights/, where no _promoted file exists."
echo "    outputs/_cuda/ is the deliberate CUDA landing area, so this"
echo "    is a path-resolution decision, not a missing transfer: either the files"
echo "    are copied across, or both readers learn the _cuda path. It returns to"
echo "    the recomputed set once that is settled, which must happen before the"
echo "    test pass regardless (run_test_pass.sh:171 refuses without it)."

say "8. ledger check: the declared tables changed, and nothing else did"
# Not "nothing moved": the declared tables move on purpose (DECLARED_TABLES above).
# The check verifies the claim in both directions.
echo "  declared:$DECLARED_TABLES"
run $PY notebooks/scripts/integrity.py ledger-check --declared $DECLARED_TABLES \
        --written-since "$RUN_START"

say "8b. FLOOR GATE — the pointer is not written unless rung 3 holds"
# Promotion requires a rung-3 retrain on the promoted boxes that does not fall.
# One-sided: a higher score passes, and only a fall of more than 0.02 fails.
#
# Both sides are three-seed means, so seed-to-seed spread does not enter a threshold
# only 0.02 wide as it would through a single-seed reference.
#
# The AUC is computed from the prediction tables, not the history metric: a history is
# the monitoring signal for epoch selection, never a result.
FLOOR_REF=0.814022 # CUDA three-seed mean: 0.814477 / 0.817816 / 0.809774
FLOOR_DROP=0.02
FLOOR=0.794022 # FLOOR_REF - FLOOR_DROP
if [ "$DRY" = "1" ]; then
  echo "  DRY RUN: would require mean(exact val AUC) over"
  echo "    ${RUNG3_TAG}, ${RUNG3_TAG}_s1, ${RUNG3_TAG}_s2"
  echo "  to be >= $FLOOR  (reference $FLOOR_REF, CUDA three-seed mean; drop allowed $FLOOR_DROP)"
  echo "  A dry run writes no real pointer, so nothing is armed either way."
else
  $PY - "$RUNG3_TAG" "$FLOOR_REF" "$FLOOR" <<'PYGATE'
import sys, pathlib
import pandas as pd
from sklearn.metrics import roc_auc_score
tag, ref, floor = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tags = [tag, f"{tag}_s1", f"{tag}_s2"]
res = pathlib.Path("outputs/results")
aucs, missing = [], []
for t in tags:
    f = res / f"{t}_predictions_val.csv"
    if not f.exists():
        missing.append(f.name); continue
    d = pd.read_csv(f)
    aucs.append(float(roc_auc_score(d["y_true"], d["y_prob"])))
    print(f"  {t:52} exact val AUC {aucs[-1]:.6f}")
if missing:
    print(f"  MISSING prediction tables: {missing}")
    print("  The floor cannot be evaluated, so it is NOT treated as passed.")
    sys.exit(1)
mean = sum(aucs) / len(aucs)
print(f"  three-seed mean {mean:.6f}  |  reference {ref:.6f}  |  floor {floor:.6f}  "
      f"|  margin {mean - floor:+.6f}")
if mean >= floor:
    print(f"  FLOOR HELD: {mean:.6f} >= {floor:.6f}. The promotion may be recorded.")
    sys.exit(0)
print(f"  FLOOR FAILED: {mean:.6f} < {floor:.6f} (a fall of {ref - mean:.6f} against the "
      f"0.02 allowed).")
sys.exit(1)
PYGATE
  if [ $? -ne 0 ]; then
    echo
    echo "PROMOTION REFUSED. The rung-3 floor did not hold on the promoted boxes."
    echo "  - no pointer is written, so the test pass is NOT armed;"
    echo "  - the promoted three-run selection stands unchanged;"
    echo "  - everything this run trained is on disk and is reported, not discarded."
    exit 1
  fi
fi

say "9. the pointer the test pass reads"
# BOXES and SUFFIX are written down once, here, with the live bundle selection they
# came from; run_test_pass.sh reads them from here instead of keeping its own copy.
#
# A dry run writes the pointer under .dryrun.json: the intent stays reviewable, and the
# test pass reads only the real name, so a dry run never arms it.
[ "$DRY" = "1" ] && POINTER="${POINTER%.json}.dryrun.json"
$PY - "$BOXES" "$TAG_SUFFIX" "$POINTER" "$RUNG3_TAG" "$FUSION_TAG" "$DRY" <<'PY'
import hashlib, json, pathlib, sys, time
boxes, suffix, pointer, rung3, fusion, dry = sys.argv[1:7]
lb = pathlib.Path("outputs/results/localiser_bundle")
sel = json.loads((lb / "bundle_selection.json").read_text())
md5 = {s: hashlib.md5((pathlib.Path(boxes) / f"localiser_boxes_{s}.csv").read_bytes()).hexdigest()
       for s in ("train", "val")
       if (pathlib.Path(boxes) / f"localiser_boxes_{s}.csv").exists()}
out = {"boxes_dir": boxes, "rung3_suffix": suffix,
       "primary_tag": rung3, "rung3_ens3_tag": f"{rung3}_ens3",
       "fusion_tag": fusion, "fusion_ens3_tag": f"{fusion}_ens3",
       "boxes_md5": md5,
       "bundle_selection": sel.get("selected"),
       "bundle_selected_cfg_id": sel.get("selected_cfg_id"),
       # A pointer that does not say what the run SKIPPED describes a run that did
       # not happen. One field covers both halves, so neither can be read alone.
       "precomputed_halves": {
           "rung3": (json.loads(pathlib.Path("outputs/results/.rung3_half_skip.json").read_text())
                     if pathlib.Path("outputs/results/.rung3_half_skip.json").exists()
                     else {"skipped": False, "note": "steps 1, 2 and 3 ran in this cascade"}),
           "fusion": (json.loads(pathlib.Path("outputs/results/.fusion_half_skip.json").read_text())
                      if pathlib.Path("outputs/results/.fusion_half_skip.json").exists()
                      else {"skipped": False, "note": "steps 4 and 5 ran in this cascade"})},
       "written_by": "promotion_cascade.sh", "dry_run": dry == "1",
       "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "why": ("run_test_pass.sh reads boxes_dir and rung3_suffix from here rather than "
               "hard-coding them, and asserts bundle_selection still matches the live "
               "selection and that the rung-3 weights' train manifest records these boxes")}
pathlib.Path(pointer).write_text(json.dumps(out, indent=2))
print(f"  wrote {pointer}: boxes={boxes} suffix={suffix} cfg={out['bundle_selected_cfg_id']}")
PY

PROMOTED=1

say "summary"
if [ "$FAIL" != "0" ]; then
    echo "  INPUTS MISSING — the cascade would fail. Fix the paths above."; exit 1
fi
echo "  inputs present."
[ "$DRY" = "1" ] && echo "  DRY RUN: nothing executed. Steps 2 and 5 are CUDA jobs; run them on the pod." \
                 || echo "  cascade complete."

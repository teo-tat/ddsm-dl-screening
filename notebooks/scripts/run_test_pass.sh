#!/usr/bin/env bash
# The single pre-registered test pass. It may be run once: if it fails before any test
# output exists it is fixed and re-run, but the moment any test output exists it is
# final. Run from the repo root.
#
# bash notebooks/scripts/run_test_pass.sh # the real pass
# bash notebooks/scripts/run_test_pass.sh --split val # DRY RUN, no test rows
#
# The dry run executes the identical sequence against the validation partition, so a
# missing weight file or a moved flag surfaces without a test row being read.
#
# It cannot check `apply --split test`, the only reader of test rows; it substitutes
# `--split val` there and says so.
#
# `--split val` writes the same {tag}_predictions_val.csv filenames as the reference
# val tables, so it copies every one aside first and restores them on exit.
set -euo pipefail
# The self-copy is sourced HERE, before a single argument is consumed. It re-execs with
# "$@", so a `shift` above this line is lost and a dry run could become the real pass.
OUTDIR=${OUTDIR:-outputs/logs/launch_$(date +%Y%m%d_%H%M%S)}
. notebooks/scripts/launcher_selfcopy.sh || { echo "launcher_selfcopy.sh missing from this checkout — stale overlay, refusing to run without launch provenance" >&2; exit 1; }

SPLIT=test
DRY=0
RESUME=""
if [ "${1:-}" = "--split" ] && [ "${2:-}" = "val" ]; then SPLIT=val; DRY=1; shift 2; fi
if [ "${1:-}" = "--split" ] && [ "${2:-}" = "test" ]; then shift 2; fi
# After the pass has started, a stage that reads tables only may be resumed.
if [ "${1:-}" = "--resume-from" ] && [ -n "${2:-}" ]; then RESUME="$2"; shift 2; fi
if [ -n "$RESUME" ] && [ "$DRY" = "1" ]; then echo "REFUSED: --resume-from is for the real pass only" >&2; exit 1; fi
ARCH=vgg16; M=0.20
python -c "import sys; sys.exit(0 if '.ddsm-env' in sys.prefix else 1)" \
  || { echo "REFUSED: python is not the project venv (.ddsm-env); activate it first" >&2; exit 1; }

# Stages that score a model on test, or come before scoring, never run twice. Every
# stage after them reads tables only, and those are the only ones --resume-from takes.
RESUMABLE=("primary endpoint + Family B" "Family A (architectures)"
           "A5: three-seed ensembles (descriptive; replace nothing, enter no family)"
           "fusion (a): declared contrast vs vgg16_fusion_control"
           "fusion (b): same-pairs controls, one view and naive averaging"
           "fusion (c): image level vs the BI-RADS anchor, descriptive"
           "fusion (d): declared context contrast vs vgg16_fusion_control_all"
           "ladder" "ladder error analysis (§4)" "analysis" "Grad-CAM grounding"
           "MC-dropout uncertainty" "data-flow figure" "ledger check")
if [ -n "$RESUME" ]; then
  ok=0; for s in "${RESUMABLE[@]}"; do [ "$s" = "$RESUME" ] && ok=1; done
  if [ "$ok" = "0" ]; then
    echo "REFUSED: '$RESUME' is not a stage that may be resumed. Resumable stages:" >&2
    printf '  %s\n' "${RESUMABLE[@]}" >&2
    exit 1
  fi
fi
MARKER=refs/test_pass_started.json # refs/ is never cleaned (refs/README.md)
# The loader-level barrier: every code path that returns test rows checks this token.
# A dry run leaves it unset, so a stage that omits --split is refused by the loader.
if [ "$DRY" = "0" ]; then
  if [ ! -f docs/.freeze_token ]; then
    echo "docs/.freeze_token does not exist - the freeze has not happened."
    echo "The test partition is unreadable until notebooks/scripts/freeze.py --freeze creates it."
    exit 1
  fi
  # The token must name this protocol, and the protocol text must match its recorded hash.
  PYTHONPATH=. python - <<'PYFROZEN' || exit 1
import importlib.util, hashlib
spec = importlib.util.spec_from_file_location("freeze", "notebooks/scripts/freeze.py")
fz = importlib.util.module_from_spec(spec); spec.loader.exec_module(fz)
tok = dict(l.split(" ", 1) for l in fz.TOKEN.read_text().splitlines() if " " in l)
now = hashlib.sha256(fz.PREREG.read_bytes()).hexdigest()
bad = []
if tok.get("protocol_sha256") != now:
    bad.append(f"the protocol changed after the freeze: token {tok.get('protocol_sha256', '?')[:16]}, now {now[:16]}")
if fz.recorded_head() != fz.head_sha256(fz.PREREG.read_text()):
    bad.append("the protocol text does not match refs/prereg_head.sha256")
if bad:
    raise SystemExit("REFUSED: " + "; ".join(bad))
print(f"[test pass] protocol sha256 {now[:16]} matches the freeze token and the recorded protocol text")
PYFROZEN
  export DDSM_TEST_PASS_TOKEN="$(tr -d '[:space:]' < docs/.freeze_token)"
  echo "[test pass] freeze token loaded; the loader will admit test reads"
else
  unset DDSM_TEST_PASS_TOKEN
  echo "[dry run] DDSM_TEST_PASS_TOKEN deliberately unset: any stage that omits --split"
  echo "[dry run] will be refused by the loader rather than reading test rows"
fi
if [ "$DRY" = "1" ]; then
  BACKUP=outputs/_superseded/dryrun_val_ledger_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$BACKUP"
  n=0
  for f in outputs/results/*_predictions_val.csv; do
    [ -e "$f" ] || continue
    cp -p "$f" "$BACKUP/"; n=$((n+1))
  done
  echo "[dry run] protected $n val prediction tables -> $BACKUP"
  restore_ledger() {
    c=0
    for f in "$BACKUP"/*.csv; do
      [ -e "$f" ] || continue
      cp -p "$f" outputs/results/; c=$((c+1))
    done
    echo "[dry run] restored $c val prediction tables from $BACKUP"
      echo "[dry run] the backup is KEPT under outputs/_superseded/. Do NOT delete it"
    echo "[dry run] until after the real test pass: it is the only proof the reference"
    echo "[dry run] ledger was unchanged by any dry run."
  }
  trap restore_ledger EXIT INT TERM
fi
# BOXES and the rung-3 suffix come from the promotion pointer when one exists, so a
# promotion cannot retrain on new boxes while this pass still scores the old ones.
BOXES=outputs/results/localiser_bundle/boxes
RUNG3_SUFFIX=_bundle
POINTER=refs/promoted_pointer.json # refs/ is never cleaned (refs/README.md)
if [ -f "$POINTER" ]; then
  BOXES=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['boxes_dir'])" "$POINTER")
  RUNG3_SUFFIX=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['rung3_suffix'])" "$POINTER")
  echo "promotion pointer: boxes=$BOXES suffix=$RUNG3_SUFFIX  ($POINTER)"
else
  echo "no promotion pointer: boxes=$BOXES suffix=$RUNG3_SUFFIX  (pre-promotion defaults)"
fi
PRIMARY=${ARCH}_crop_rung3_unet${RUNG3_SUFFIX}

# one ledger for every reported classifier row
# Every row carries config.LEDGER_SUFFIX, the one CUDA re-measurement session, and
# rungs 3 and 5 carry it inside RUNG3_SUFFIX; a pointer without it is refused below.
LEDGER=$(PYTHONPATH=. python -c "from src import config; print(config.LEDGER_SUFFIX)")
# The context-fusion pair was re-measured in its own session, outside the ledger.
CTX=$(PYTHONPATH=. python -c "from src import config; print(config.CONTEXT_FUSION_SUFFIX)") || exit 1
case "$RUNG3_SUFFIX" in
  *"$LEDGER") echo "ledger: $LEDGER (rung-3 suffix $RUNG3_SUFFIX carries it)" ;;
  *) echo "REFUSED: the pointer's rung-3 suffix '$RUNG3_SUFFIX' does not end in the ledger" \
          "suffix '$LEDGER'. The pointer predates the one-session re-measurement; re-run the" \
          "cascade on the ${LEDGER} runs so rung 3 and its neighbours come from one ledger." >&2
     exit 1 ;;
esac

# The weights this pass scores must have been trained on the boxes it crops with. A
# tag is a name, not evidence, so each run's train manifest records the folder and md5.
python - "$BOXES" "$PRIMARY" "$POINTER" <<'PYCHECK' || exit 1
import hashlib, json, pathlib, sys
boxes, primary, pointer = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
res, bad = pathlib.Path("outputs/results"), []

live = boxes / "localiser_boxes_val.csv"
if not live.exists():
    bad.append(f"the boxes folder has no localiser_boxes_val.csv: {boxes}")
md5 = {s: hashlib.md5((boxes / f"localiser_boxes_{s}.csv").read_bytes()).hexdigest()
       for s in ("train", "val") if (boxes / f"localiser_boxes_{s}.csv").exists()}

if pointer.exists():
    ptr = json.loads(pointer.read_text())
    if ptr.get("boxes_md5") and ptr["boxes_md5"] != md5:
        bad.append("the boxes folder has changed since the cascade wrote the pointer: "
                   f"pointer {ptr['boxes_md5']} vs live {md5}")
    sel_p = pathlib.Path("outputs/results/localiser_bundle/bundle_selection.json")
    if sel_p.exists():
        live_sel = json.loads(sel_p.read_text()).get("selected")
        if ptr.get("bundle_selection") and live_sel != ptr["bundle_selection"]:
            bad.append("the live bundle selection is not the one the cascade promoted: "
                       f"live {live_sel} vs pointer {ptr['bundle_selection']}")
    if ptr.get("dry_run"):
        bad.append("the pointer was written by a DRY RUN of the cascade: no promoted "
                   "weights exist for it. Re-run the cascade for real, or delete the pointer.")

man = res / f"{primary}_train_manifest.json"
if not man.exists():
    bad.append(f"no train manifest for the primary tag: {man}. The weights cannot be shown "
               "to have been trained on these boxes, and a matching filename is not evidence.")
else:
    m = json.loads(man.read_text())
    recorded = m.get("boxes_md5")
    # An absent md5 is reported as unmeasured, not as a mismatch: those are different
    # findings, and the pass needs the field measured either way.
    if recorded is None or (isinstance(recorded, dict) and "unavailable" in recorded):
        bad.append(
            f"{primary}'s manifest records no boxes_md5 (it is "
            f"{'null' if recorded is None else 'marked unavailable'}). The md5 was never "
            f"computed because src.train._save_train_manifest ran on a pod without the "
            f"boxes directory. This is NOT evidence that the boxes differ and it is NOT "
            f"evidence that they match — it is an unmeasured field, and the pass needs it "
            f"measured. Backfill it on a machine that holds {boxes} and re-run."
            + (f" Recorded reason: {recorded['unavailable']}"
               if isinstance(recorded, dict) else ""))
    elif recorded != md5:
        bad.append(f"{primary} was trained on different boxes: manifest {m.get('boxes_md5')} "
                   f"vs the folder this pass will crop with {md5}")
    elif m.get("boxes_dir") and str(pathlib.Path(m['boxes_dir'])) != str(boxes):
        print(f"  note: {primary} records boxes_dir {m['boxes_dir']}, this pass uses {boxes}; "
              "the table contents are identical (md5 match), so this is a moved folder, not "
              "a different box set.")
if bad:
    print("BOX/WEIGHT PROVENANCE CHECK FAILED:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print(f"  provenance ok: {primary}'s train manifest records {boxes}, and the folder still has "
      f"the md5 recorded there (md5 {md5.get('val','?')[:8]}…)")
PYCHECK
LOG=outputs/logs/test_pass_${SPLIT}_$(date +%Y%m%d_%H%M).stdout
mkdir -p outputs/logs

# Every weights file the pass will load, resolved up front through
# config.resolve_weights, which refuses a laptop file that has a CUDA twin.
PYTHONPATH=. python - "$ARCH" "$LEDGER" "$RUNG3_SUFFIX" "$CTX" <<'PYW' || exit 1
import sys
from src import config
arch, ledger, r3, ctx = sys.argv[1:5]
tags = [f"{arch}_crop_rung{k}_{src}{ledger}" for k, src in
        ((1, "oracle"), (2, "oracle"), (4, "cam"), (6, "shuffled"), (7, "whole"))]
tags += [f"{arch}_crop_rung3_unet{r3}", f"{arch}_crop_rung3_unet{r3}_s1",
         f"{arch}_crop_rung3_unet{r3}_s2", f"{arch}_crop_rung5_jitter{r3}",
         f"{arch}_crop_rung5_jitter{r3}_s1", f"{arch}_crop_rung5_jitter{r3}_s2",
         f"{arch}_full{ledger}"]
tags += [f"{m}_crop_m020{ledger}" for m in ("baseline", "scaled", "regularised",
                                            "vgg16", "resnet50", "densenet121", "efficientnet")]
fus, orc = f"{arch}_fusion_view_unet{r3}", f"{arch}_fusion_view{ledger}"
tags += [fus, f"{fus}_s1", f"{fus}_s2", orc, f"{orc}_s1", f"{orc}_s2",
         f"{arch}_fusion_control{ledger}", f"{arch}_fusion_context{ctx}", f"{arch}_fusion_control_all{ctx}"]
bad = []
for t in tags:
    try:
        p = config.resolve_weights(t)
        print(f"  weights ok  {t:48} {p.parent.name}/")
    except FileNotFoundError as e:          # WeightsRefused is a subclass
        kind = "REFUSED" if isinstance(e, config.WeightsRefused) else "MISSING"
        bad.append(f"  {kind:8} {t}")
if bad:
    print(f"PRE-FLIGHT FAILED: {len(bad)} of {len(tags)} weight tags cannot be loaded:",
          *bad, sep="\n", file=sys.stderr)
    raise SystemExit(1)
print(f"  pre-flight: all {len(tags)} weight tags resolve to one file each")
# Written down so the model-list assertion below reads THIS list rather than
# constructing a second one that could drift from it.
import pathlib
# The ens3 tables are averages of prediction tables, not weights, so they belong in
# the model list the tag assertion checks and not in the weight resolution above.
derived = [f"{arch}_crop_rung3_unet{r3}_ens3", f"{fus}_ens3", f"{orc}_ens3"]
pathlib.Path("outputs/logs/.pass_tags.txt").write_text("\n".join(tags + derived) + "\n")
PYW

# The models scored must be the models the frozen protocol names: a pointer can be
# self-consistent and still name a system never declared. Checked in both directions.
PYTHONPATH=. python notebooks/scripts/assert_model_list.py \
    --tags-file outputs/logs/.pass_tags.txt \
    --ledger "$LEDGER" --rung3-suffix "$RUNG3_SUFFIX" || exit 1
# A second pass is refused on two independent grounds: the pass-started marker, which
# nothing cleans, and the test tables themselves.
if [ "$DRY" = "0" ] && [ -z "$RESUME" ]; then
  if [ -e "$MARKER" ]; then
    echo "REFUSED: $MARKER exists - the pass has started, and it runs once (§36)."
    echo "A stage after scoring that failed may be resumed: --resume-from STAGE."
    exit 1
  fi
  if ls outputs/results/${ARCH}_crop_rung*_predictions_test.csv >/dev/null 2>&1 \
     || ls outputs/results/*_crop_m020_predictions_test.csv >/dev/null 2>&1; then
    echo "REFUSED: test prediction tables already exist - the pass has been run"; exit 1
  fi
fi
if [ -n "$RESUME" ] && [ ! -e "$MARKER" ]; then
  echo "REFUSED: --resume-from needs $MARKER; the pass has not started"; exit 1
fi
# Family A (architectures) needs all four crop-margin weight files selected on val.
ARCHS="vgg16 resnet50 densenet121 efficientnet"
for a in $ARCHS; do
  if [ ! -f outputs/weights/${a}_crop_m020_best.weights.h5 ]; then
    echo "${a}_crop_m020 weights missing - architecture family incomplete"; exit 1
  fi
done

{
echo "=== test pass $(date) ==="
echo "=== SPLIT=$SPLIT  (dry run: $DRY)${RESUME:+  (resuming from: $RESUME)}  — every stage below is run with --split $SPLIT ==="
# The reference ledger: snapshot before, verify after. A resumed pass keeps the
# snapshot taken when the pass started.
if [ -z "$RESUME" ]; then
  PYTHONPATH=. python notebooks/scripts/integrity.py ledger-snapshot
fi
ACTIVE=1
if [ -n "$RESUME" ]; then ACTIVE=0; fi
# run_stage NAME: print the stage header and say whether the stage runs. When
# resuming, every stage before the named one is skipped.
run_stage() {
  if [ "$ACTIVE" = "0" ] && [ "$1" = "$RESUME" ]; then ACTIVE=1; fi
  if [ "$ACTIVE" = "1" ]; then echo; echo "--- [$SPLIT] $1 ---"; return 0; fi
  echo "[resume] skipped: $1"; return 1
}

FUSION=view
FUSION_TAG=${ARCH}_fusion_${FUSION}_unet${RUNG3_SUFFIX}
# Stores are reused across ledgers, not renamed, so the store directory is named
# by the promotion suffix WITHOUT the ledger.
FUSION_STORE=outputs/fusion_store_${ARCH}_fusion_${FUSION}_unet${RUNG3_SUFFIX%"$LEDGER"}
ORACLE_FUSION_TAG=${ARCH}_fusion_${FUSION}${LEDGER}
# src.fusion takes no --split: `--stage val` writes the validation table from saved
# weights; `--stage eval` additionally scores TEST.
FUSION_STAGE=eval
if [ "$SPLIT" = "val" ]; then FUSION_STAGE=val; fi
# src.ensemble needs --allow-test for the test split, and val alongside it because the
# ensemble's threshold is fitted on validation.
ENS_SPLITS="--splits val"
if [ "$DRY" = "0" ]; then ENS_SPLITS="--splits val test --allow-test"; fi
# Derived tables (ens3, the per-pair controls) are not rewritten by a resumed pass.
exists_on_resume() {
  if [ -n "$RESUME" ] && [ -e "$1" ]; then echo "[resume] $1 exists - not regenerated"; return 0; fi
  return 1
}
ensemble() {
  local out=$1; shift
  exists_on_resume "outputs/results/${out}_predictions_${SPLIT}.csv" \
    || python -m src.ensemble --tags "$@" --out-tag "$out" --method mean $ENS_SPLITS
}

# The promoted localiser's TEST boxes, written by the same apply step that wrote val
# and train. The test maps it reads must be the ones recorded in refs/.
if run_stage "apply bundle boxes"; then
  if [ "$DRY" = "1" ]; then
    echo "[dry run] apply --split val (the real pass uses --split $SPLIT --allow-test)"
    PYTHONPATH=. python notebooks/scripts/localiser_bundle.py apply --split val --boxes-dir $BOXES
  else
    PYTHONPATH=. python - <<'PYMAPS' || exit 1
import hashlib, json, pathlib
man = json.loads(pathlib.Path("refs/test_maps_manifest.json").read_text())
maps, bad = pathlib.Path("outputs/results/localiser_bundle/maps"), []
for name, rec in man["arrays"].items():
    h = hashlib.sha256()
    with open(maps / name, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != rec["sha256"]:
        bad.append(name)
meta = pathlib.Path("outputs/localiser_tensors/test_meta.csv")
if hashlib.sha256(meta.read_bytes()).hexdigest() != man["test_meta_csv_sha256"]:
    bad.append(str(meta))
print(f"  [discovered] {len(man['arrays'])} test map arrays in refs/test_maps_manifest.json")
if bad:
    raise SystemExit(f"REFUSED: {len(bad)} file(s) differ from their recorded sha256: {bad}")
print("  test maps and the metadata they are aligned with match their recorded sha256")
PYMAPS
    PYTHONPATH=. python notebooks/scripts/localiser_bundle.py apply --split $SPLIT --allow-test --boxes-dir $BOXES
  fi
fi
# Every lesion of the split must have a U-Net box and a jitter draw before anything is
# scored, so a gap in the new box table stops the pass before the first test table.
if run_stage "box coverage"; then
  PYTHONPATH=. python - "$BOXES" "$SPLIT" <<'PYBOX' || exit 1
import sys
import pandas as pd
from pathlib import Path
from src.boxes import provider_for
from src.data_loader import split_frames
boxes, split = sys.argv[1], sys.argv[2]
splits = ("train", "val", "test") if split == "test" else ("train", "val")
for src in ("unet", "jitter"):
    provider_for(src, boxes_dir=boxes, splits=splits)
want = set(split_frames("crop")[split]["lesion_id"])
have = set(pd.read_csv(Path(boxes) / f"localiser_boxes_{split}.csv")["lesion_id"])
if want != have:
    raise SystemExit(f"REFUSED: the {split} box table covers {len(have & want)} of {len(want)} lesions")
print(f"  box coverage: all {len(want)} {split} lesions have a U-Net box and a jitter draw")
PYBOX
fi
# From here on the pass has started: it writes its marker before the first test table.
if [ "$DRY" = "0" ] && [ -z "$RESUME" ]; then
  PYTHONPATH=. python - "$MARKER" <<'PYMARK'
import datetime, hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_text(json.dumps({
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "protocol_sha256": hashlib.sha256(pathlib.Path("docs/test_pass_preregistration.md").read_bytes()).hexdigest(),
    "note": "written before the first test prediction table; the pass refuses to start again while it exists (§36)"},
    indent=2) + "\n")
print(f"-> {p}")
PYMARK
fi

# scoring: every model is scored on test here, once
# Rungs 3 and 5 consume the promoted localiser and carry its suffix; rungs 1, 2, 4, 6
# and 7 touch no localiser and carry the ledger suffix alone.
if run_stage "scoring: ladder rungs, seeds, whole image, scratch and Family A"; then
  for k in 1 2 3 4 5 6 7; do
    case $k in 1|2) src=oracle;; 3) src=unet;; 4) src=cam;; 5) src=jitter;; 6) src=shuffled;; 7) src=whole;; esac
    m=$M; if [ $k -eq 1 ]; then m=0.00; fi
    extra=""; suffix=_rung${k}_${src}${LEDGER}
    if [ $k -eq 3 ] || [ $k -eq 5 ]; then extra="--boxes-dir $BOXES"; suffix=_rung${k}_${src}${RUNG3_SUFFIX}; fi
    python -m src.evaluate --model $ARCH --source crop --box-source $src \
        --crop-margin $m --tag-suffix $suffix $extra --split $SPLIT
  done
  # Seed replicates of rungs 3 and 5, for the seed spread and the ensembles.
  for s in 1 2; do
    python -m src.evaluate --model $ARCH --source crop --box-source unet \
        --crop-margin $M --tag-suffix _rung3_unet${RUNG3_SUFFIX}_s${s} \
        --boxes-dir $BOXES --split $SPLIT
    python -m src.evaluate --model $ARCH --source crop --box-source jitter \
        --crop-margin $M --tag-suffix _rung5_jitter${RUNG3_SUFFIX}_s${s} \
        --boxes-dir $BOXES --split $SPLIT
  done
  # Whole-image reference, the scratch family and Family A.
  python -m src.evaluate --model $ARCH --source full --tag-suffix _full${LEDGER} --split $SPLIT
  for s in baseline scaled regularised $ARCHS; do
    python -m src.evaluate --model $s --source crop --crop-sizing mask_margin \
        --crop-margin $M --tag-suffix _m020${LEDGER} --split $SPLIT
  done
fi
# View fusion: the declared system on U-Net boxes, its declared control, the U-Net
# seed replicates, and the oracle-box fusion with its seeds.
if run_stage "scoring: fusion"; then
  python -m src.fusion --arch $ARCH --pair $FUSION --margin $M --box-source unet \
      --boxes-dir $BOXES --tag-suffix $RUNG3_SUFFIX --stage $FUSION_STAGE
  python -m src.fusion --arch $ARCH --pair control --margin $M --tag-suffix $LEDGER --stage $FUSION_STAGE
  for s in 1 2; do
    python -m src.fusion --arch $ARCH --pair $FUSION --margin $M --box-source unet \
        --boxes-dir $BOXES --tag-suffix $RUNG3_SUFFIX --seed $s --stage $FUSION_STAGE
  done
  python -m src.fusion --arch $ARCH --pair $FUSION --margin $M --box-source oracle \
      --tag-suffix $LEDGER --stage $FUSION_STAGE
  for s in 1 2; do
    python -m src.fusion --arch $ARCH --pair $FUSION --margin $M --box-source oracle \
        --tag-suffix $LEDGER --seed $s --stage $FUSION_STAGE
  done
  # Context fusion and its control on every lesion, for fusion (d).
  python -m src.fusion --arch $ARCH --pair context --margin $M --tag-suffix $CTX --stage $FUSION_STAGE
  python -m src.fusion --arch $ARCH --pair control --rows all --margin $M --tag-suffix $CTX \
      --stage $FUSION_STAGE
fi

# from here every stage reads tables only, and may be resumed
# Primary endpoint + Family B (ladder, 10 Holm-corrected DeLong tests).
if run_stage "primary endpoint + Family B"; then
  python -m src.compare --primary $PRIMARY \
      --models ${ARCH}_crop_rung2_oracle${LEDGER} $PRIMARY ${ARCH}_crop_rung4_cam${LEDGER} \
               ${ARCH}_crop_rung6_shuffled${LEDGER} ${ARCH}_crop_rung7_whole${LEDGER} \
      --out comparison_report_${SPLIT}.json --split $SPLIT
fi
# Family A (architectures, 6 Holm-corrected DeLong tests), separate family.
if run_stage "Family A (architectures)"; then
  python -m src.compare \
      --models vgg16_crop_m020${LEDGER} resnet50_crop_m020${LEDGER} \
               densenet121_crop_m020${LEDGER} efficientnet_crop_m020${LEDGER} \
      --out comparison_architectures_${SPLIT}.json --split $SPLIT
fi
# Mean-probability ensembles of the three seeds, averaged from prediction tables.
# TTA is formally omitted.
if run_stage "A5: three-seed ensembles (descriptive; replace nothing, enter no family)"; then
  ensemble ${PRIMARY}_ens3 $PRIMARY ${PRIMARY}_s1 ${PRIMARY}_s2
  ensemble ${FUSION_TAG}_ens3 $FUSION_TAG ${FUSION_TAG}_s1 ${FUSION_TAG}_s2
  ensemble ${ORACLE_FUSION_TAG}_ens3 $ORACLE_FUSION_TAG ${ORACLE_FUSION_TAG}_s1 ${ORACLE_FUSION_TAG}_s2
fi
# (a) The declared contrast, run because the frozen protocol names it.
if run_stage "fusion (a): declared contrast vs vgg16_fusion_control"; then
  python -m src.compare --models $FUSION_TAG ${ARCH}_fusion_control${LEDGER} \
      --out comparison_fusion_declared_${SPLIT}.json --split $SPLIT
fi
# (b) The same-pairs controls. The pairs come from the frozen pair artefact; where
# the fusion store holds the split, the two must agree.
if run_stage "fusion (b): same-pairs controls, one view and naive averaging"; then
  PAIRS=artifacts/fusion_pairs_${FUSION}_${SPLIT}.csv
  PYTHONPATH=. python - "$PAIRS" "${FUSION_STORE}/${SPLIT}_meta.csv" <<'PYPAIRS' || exit 1
import sys, pathlib
import pandas as pd
art, store = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if not art.exists():
    raise SystemExit(f"REFUSED: frozen pair artefact {art} is missing")
if store.exists() and not pd.read_csv(art).equals(pd.read_csv(store)):
    raise SystemExit(f"REFUSED: {art} and {store} disagree; the pairs fusion was trained on are not "
                     f"the pairs this pass would score")
print(f"  pairs: {art}" + (" (== store meta)" if store.exists() else " (store holds no such split)"))
PYPAIRS
  CONTROLS=outputs/results/rung3_on_pairs${RUNG3_SUFFIX}_${SPLIT}_predictions_${SPLIT}.csv
  exists_on_resume "$CONTROLS" || PYTHONPATH=. python notebooks/scripts/analyse_fusion.py controls \
      --pairs $PAIRS --rung3 outputs/results/${PRIMARY}_ens3_predictions_${SPLIT}.csv \
      --tag rung3_on_pairs${RUNG3_SUFFIX}_${SPLIT} --split $SPLIT
  PYTHONPATH=. python notebooks/scripts/analyse_fusion.py compare --split $SPLIT \
      --fusion outputs/results/${FUSION_TAG}_ens3_predictions_${SPLIT}.csv \
      --controls $CONTROLS \
      --oracle-fusion outputs/results/${ORACLE_FUSION_TAG}_ens3_predictions_${SPLIT}.csv \
      --out outputs/results/fusion_pair_comparisons_${SPLIT}.json
fi
# (c) Image level against the anchor, descriptive; the primary endpoint stays rung 3.
if run_stage "fusion (c): image level vs the BI-RADS anchor, descriptive"; then
  python -m src.compare --primary $FUSION_TAG \
      --out comparison_fusion_imagelevel_${SPLIT}.json --split $SPLIT
fi
# (d) The declared context contrast on the 243 lesions, uncorrected, in no family.
if run_stage "fusion (d): declared context contrast vs vgg16_fusion_control_all"; then
  python -m src.compare --models ${ARCH}_fusion_context${CTX} ${ARCH}_fusion_control_all${CTX} \
      --out comparison_fusion_context_${SPLIT}.json --split $SPLIT
fi
# The ladder figure across the seven rungs.
if run_stage "ladder"; then
  python -m src.ladder --arch $ARCH --margin $M --stage $SPLIT --boxes-dir $BOXES \
      --rung3-suffix $RUNG3_SUFFIX --ledger-suffix $LEDGER
fi
# The rung error frame. A dry run writes under its own suffix so it cannot overwrite
# the validation file already reported.
if run_stage "ladder error analysis (§4)"; then
  LEA_ARGS="--out-suffix ${LEDGER}_dryrun"
  if [ "$DRY" = "0" ]; then LEA_ARGS="--out-suffix ${LEDGER} --allow-test"; fi
  PYTHONPATH=. python notebooks/scripts/analyse_ladder.py errors --boxes-dir $BOXES \
      --tag-suffix $RUNG3_SUFFIX --tag rung2=${ARCH}_crop_rung2_oracle${LEDGER} --split $SPLIT $LEA_ARGS
fi
# Analysis of the primary tag beside the oracle rung and the whole-image reference.
if run_stage "analysis"; then
  python -m src.analysis --tag $PRIMARY ${ARCH}_crop_rung2_oracle${LEDGER} ${ARCH}_full${LEDGER} --split $SPLIT
fi
# Grad-CAM heatmaps for the primary rung-3 model, against the annotated lesion boxes.
if run_stage "Grad-CAM grounding"; then
  python -m src.explain --model $ARCH --tag-suffix _rung3_unet${RUNG3_SUFFIX} --box-source unet --boxes-dir $BOXES --margin $M --split $SPLIT
fi
# MC-dropout uncertainty for the primary rung-3 model.
if run_stage "MC-dropout uncertainty"; then
  python -m src.uncertainty --model $ARCH --tag-suffix _rung3_unet${RUNG3_SUFFIX} --box-source unet --boxes-dir $BOXES --margin $M --splits $SPLIT
fi
# The data-flow figure counts every partition, test labels included, so it runs here,
# under the pass's token, and nowhere else.
if run_stage "data-flow figure"; then
  if [ "$DRY" = "0" ]; then
    PYTHONPATH=. python notebooks/scripts/figures.py data-flow
  else
    echo "[dry run] the data-flow figure counts test labels; it runs in the real pass only"
  fi
fi
# The last gate: no reference table moved while the pass produced its numbers.
# --strict-new on the real pass: it adds *_predictions_test.csv, never a val table.
if run_stage "ledger check"; then
  STRICT=""
  if [ "$DRY" = "0" ]; then STRICT="--strict-new"; fi
  PYTHONPATH=. python notebooks/scripts/integrity.py ledger-check $STRICT \
    || { echo "LEDGER CHANGED during the pass - investigate before quoting any number (§36)"; exit 1; }
fi
echo "=== done $(date) ==="
} 2>&1 | tee "$LOG"

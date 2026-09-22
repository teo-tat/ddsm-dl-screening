#!/usr/bin/env bash
# combined_gate.sh — the one combined gate of the multi-passer rule. It forms every
# declared configuration, gates each against the promoted box table and picks the
# promotion candidate by the declared rule.
#
# bash notebooks/scripts/combined_gate.sh --dry-run --members l7v l7a l8a
# bash notebooks/scripts/combined_gate.sh --apply --members l7v l7a l8a
#
# Discovery is a check, not the authority: the member set is proposed from the recorded
# gate results and then required to match --members, so a stale file cannot change it.
#
# Nothing is written to the promoted boxes folder and every configuration gets its own.
# Once anything has run, the promoted three-run selection is restored and asserted on
# every exit path.
set -u
# The snapshot is taken before any argument is consumed: the re-exec passes "$@".
OUTDIR=${OUTDIR:-outputs/logs/combined_gate_$(date +%Y%m%d_%H%M%S)}
. notebooks/scripts/launcher_selfcopy.sh || { echo "launcher_selfcopy.sh missing" >&2; exit 1; }

# Measuring is not promoting, and the two do not share a switch: the default computes
# everything and writes the gate JSONs, --apply also runs the ledger check and the
# pointer step, and --dry-run prints the commands and executes none. No mode writes a
# pointer; the one the test pass reads, refs/promoted_pointer.json, is promotion_cascade.sh's.
APPLY=0; DRY=0; WEIGHTS=none; MEMBERS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift;;
        --dry-run) DRY=1; shift;;
        --weights) shift; WEIGHTS=$1; shift;;
        --members) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do MEMBERS="$MEMBERS $1"; shift; done;;
        *) echo "unknown argument: $1"; exit 1;;
    esac
done
MEMBERS=$(echo $MEMBERS)
[ -n "$MEMBERS" ] || { echo "usage: combined_gate.sh [--apply] [--dry-run] [--weights none|group] --members <m> [<m> ...]"; exit 1; }
case "$WEIGHTS" in none|group) ;; *) echo "--weights must be none or group"; exit 1;; esac

PY=.ddsm-env/bin/python
LB=outputs/results/localiser_bundle
E3=$LB/boxes/localiser_boxes_val.csv
POINTER=$LB/promoted_pointer.json
PROMOTED_IOU=0.4700
mkdir -p "$OUTDIR"
exec > >(tee -a "$OUTDIR/combined_gate.log") 2>&1
run () { echo "  \$ $*"; [ "$DRY" = "1" ] && return 0; "$@" || { echo "  FAILED"; exit 1; }; }

# restore on EVERY exit path, including failure and interrupt
LEDGER_OK=0
restore_and_assert () {
    # Restores whenever anything RAN, not only under --apply: the step-3 sweep
    # overwrites bundle_selection.json, so a measuring run displaces it too.
    [ "$DRY" = "1" ] && { echo; echo "restore: dry run touched nothing"; return 0; }
    echo; echo "── restoring the promoted three-run selection ──"
    # The safety action must not consume the product: this restore overwrites every
    # bundle file, sweep_configs.csv included, so the run's own copies go aside first.
    mkdir -p "$OUTDIR/run_artefacts"
    for f in sweep_configs.csv bundle_selection.json bundle_metrics.json \
             bundle_boxes_val.csv bundle_boxes_train.csv; do
        [ -f "$LB/$f" ] && cp "$LB/$f" "$OUTDIR/run_artefacts/$f" 2>/dev/null
    done
    echo "  run artefacts archived -> $OUTDIR/run_artefacts/ (before the restore overwrites them)"
    cp "$LB"/selected_runs134/* "$LB/" 2>/dev/null
    PYTHONPATH=. $PY - <<'EOF'
import json, pathlib, sys
lb = pathlib.Path("outputs/results/localiser_bundle")
live = json.loads((lb / "bundle_selection.json").read_text())
frozen = json.loads((lb / "selected_runs134" / "bundle_selection.json").read_text())
if live.get("selected_cfg_id") != frozen.get("selected_cfg_id"):
    print(f"  RESTORE FAILED: live {live.get('selected_cfg_id')} != frozen {frozen.get('selected_cfg_id')}")
    sys.exit(1)
print(f"  restored and asserted: cfg {live['selected_cfg_id']} {live['selected']['ensemble']}")
EOF
}
trap restore_and_assert EXIT INT TERM

echo "=== combined gate | mode: $([ $DRY = 1 ] && echo DRY-RUN || { [ $APPLY = 1 ] && echo "MEASURE+APPLY" || echo MEASURE; }) | weights: $WEIGHTS ==="

# resolved defaults, printed beside what they are meant to be
echo
echo "── §5c resolved defaults ──"
printf "  %-22s %s\n" "promoted boxes" "$E3  (must be the CURRENT promotion)"
printf "  %-22s %s\n" "promoted box IoU" "$PROMOTED_IOU  (the reference every gate uses)"
printf "  %-22s %s\n" "frozen selection" "$LB/selected_runs134/bundle_selection.json"
printf "  %-22s %s\n" "pointer" "$POINTER  (not written by this script)"
for f in "$E3" "$LB/selected_runs134/bundle_selection.json"; do
    [ -f "$f" ] || { echo "  MISSING $f"; exit 1; }
done
PYTHONPATH=. $PY - "$LB" "$PROMOTED_IOU" <<'EOF' || exit 1
import json, pathlib, sys
import pandas as pd
lb, want = pathlib.Path(sys.argv[1]), float(sys.argv[2])
d = pd.read_csv(lb / "boxes" / "localiser_boxes_val.csv")
got = d["box_iou"].fillna(0).mean()
if abs(got - want) > 5e-4:
    print(f"  the promoted box table scores {got:.4f}, not {want}. The reference has moved.")
    sys.exit(1)
sel = json.loads((lb / "selected_runs134" / "bundle_selection.json").read_text())
print(f"  verified: promoted boxes score {got:.4f}, frozen cfg {sel['selected_cfg_id']} "
      f"{sel['selected']['ensemble']}")
EOF

echo
echo "── 0. ledger snapshot ──"
run $PY notebooks/scripts/integrity.py ledger-snapshot

# 1. discovery, as a CHECK against the explicit list
echo
echo "── 1. member set: discovered, then required to match --members ──"
PYTHONPATH=. $PY - "$LB" "$MEMBERS" <<'EOF' || exit 1
import json, pathlib, sys
lb, given = pathlib.Path(sys.argv[1]), sys.argv[2].split()
# Declaration order, which decides eligibility.
ORDER = ["l7v", "l7a", "l8a", "l9", "l10", "l7v_s1", "l7v_s2", "run4_s1", "run4_s2"]
found = {}
for f in sorted(lb.glob("gate_ens4*_vs_ens3.json")) + sorted(lb.glob("gate_boxfusion4*_vs_ens3.json")):
    name = f.stem.replace("gate_ens4", "").replace("gate_boxfusion4", "").replace("_vs_ens3", "")
    try:
        g = json.loads(f.read_text())
    except Exception as e:
        print(f"  UNREADABLE {f.name}: {e}"); sys.exit(1)
    found[name] = (bool(g.get("passes")), g.get("mean_diff"), f.name)
passers = [m for m in ORDER if found.get(m, (False,))[0]]
unknown = [m for m in found if m not in ORDER]
print(f"  declaration order : {' '.join(ORDER)}")
print(f"  gate files read   : {len(found)}")
for m in ORDER:
    if m in found:
        p, d, fn = found[m]
        print(f"    {m:<10} {'PASS' if p else 'fail'}  {d:+.4f}  ({fn})")
# A gate file for a member outside the declaration order is reported and never
# proposed: the order is the authority on eligibility.
if unknown:
    print("  ineligible, recorded for completeness (not in §7b's declaration order):")
    for m in sorted(unknown):
        p_, d_, fn = found[m]
        d_s = f"{d_:+.4f}" if isinstance(d_, (int, float)) else "  n/a  "
        print(f"    {m or '(unnamed)':<10} {'PASS' if p_ else 'fail'}  {d_s}  ({fn})")
print(f"  proposed (c)-passers, in declaration order : {' '.join(passers) or '(none)'}")
print(f"  supplied on the command line               : {' '.join(given)}")
if [m for m in ORDER if m in given] != passers or sorted(given) != sorted(passers):
    print()
    print("  MISMATCH. Discovery is a check, not the authority: the recorded gates and the")
    print("  list you supplied disagree, so the run stops rather than choosing one of them.")
    print("  Either a gate file is stale, renamed or hand-edited, or the list is wrong.")
    sys.exit(1)
print("  MATCH — the member set is fixed by the rule, not by this run.")
EOF

MAP_MEMBERS=$(echo "$MEMBERS" | tr ' ' '\n' | grep -v '^l10$' | tr '\n' ' ')
HAS_L10=$(echo " $MEMBERS " | grep -c ' l10 ' || true)
echo "  map members: $MAP_MEMBERS"
echo "  detector   : $([ "$HAS_L10" = "1" ] && echo 'l10 passed (c) — configurations B and C will be formed' || echo 'l10 absent — configuration A stands alone')"

# 2. each member's single-box recheck in the frame its maps were stored in, pod mounts
# resolved through the alias table; the sweep in step 3 asserts the frames
echo
echo "── 2. single-box recheck in each member's stored frame (pod mounts resolved) ──"
for m in $MAP_MEMBERS; do
    run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py recheck --tag "$m" --split val train
done

# 3. configuration A: the combined map ensemble
echo
echo "── 3. configuration A: combined map ensemble, rule selected on TRAIN ──"
run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py sweep --tag run1 run3 run4 $MAP_MEMBERS
run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py select --ensemble all
A_DIR=$LB/combined_A
run mkdir -p "$A_DIR"
run cp "$LB/bundle_selection.json" "$LB/bundle_metrics.json" "$LB/bundle_boxes_val.csv" "$LB/bundle_boxes_train.csv" "$LB/sweep_configs.csv" "$A_DIR/"
run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py gate \
    --candidate "$A_DIR/bundle_boxes_val.csv" --reference "$E3" \
    --label "combinedA_vs_ens3" --out "$LB/gate_combinedA_vs_ens3.json"

# 4. weighted map averaging, searched on TRAIN in the same step
echo
echo "── 4. §24 weighted map averaging (train-only search) ──"
# Weighting is implemented inside the sweep, over per-GROUP vectors so one pass
# evaluates many, and all-ones reproduces the unweighted sweep bit-identically.
if [ "$WEIGHTS" = "group" ]; then
  echo "  §24a group weighting, searched on TRAIN in the same pass (~3 h on 7 members)."
  run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py sweep \
      --tag run1 run3 run4 $MAP_MEMBERS --weighting group
  run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py select --ensemble all
  # A separate folder and a separate gate name: overwriting configuration A would
  # destroy the equal-weight number the weighted winner is compared against.
  A24_DIR=$LB/combined_A24
  run mkdir -p "$A24_DIR"
  run cp "$LB/bundle_selection.json" "$LB/bundle_metrics.json" "$LB/bundle_boxes_val.csv" "$LB/bundle_boxes_train.csv" "$LB/sweep_configs.csv" "$A24_DIR/"
  run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py gate \
      --candidate "$A24_DIR/bundle_boxes_val.csv" --reference "$E3" \
      --label "combinedA24_vs_ens3" --out "$LB/gate_combinedA24_vs_ens3.json"
  S24_READY=1
else
  echo "  NOT RUN in this invocation (--weights none). Configuration A above is the"
  echo "  EQUAL-WEIGHT ensemble. §24/§24a declare that the train winner of the"
  echo "  weighting search enters the gate, so A as gated here is §24's equal-weight"
  echo "  family only, and the result must be reported as such — not as §24's winner."
  S24_READY=0
fi

# 5. configurations B and C, only if l10 passed (c)
if [ "$HAS_L10" = "1" ]; then
  echo
  echo "── 5. configuration B: box fusion of l10 with A (weights on TRAIN) ──"
  B_DIR=outputs/results/combined_B_boxfusion_l10
  run env PYTHONPATH=. $PY notebooks/scripts/box_rules.py --rule fusion \
      --tag A="$A_DIR/bundle_boxes_{split}.csv" \
                l10=outputs/_cuda/localiser_l10_fasterrcnn_cachev1/localiser_boxes_{split}.csv \
      --out "$B_DIR"
  run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py gate \
      --candidate "$B_DIR/localiser_boxes_val.csv" --reference "$E3" \
      --label "combinedB_vs_ens3" --out "$LB/gate_combinedB_vs_ens3.json"
  echo "  NOTE: B carries the measured -0.0339 box-fusion handicap (§18). A loss to A"
  echo "  does NOT establish that the detector adds nothing — C is what separates them."

  echo
  echo "── 5c. configuration C: §19b hybrid, A's box except where A has none ──"
  C_DIR=outputs/results/combined_C_hybrid_l10
  run env PYTHONPATH=. $PY notebooks/scripts/box_rules.py --rule hybrid \
      --base "$A_DIR" --donor outputs/_cuda/localiser_l10_fasterrcnn_cachev1 \
      --out "$C_DIR" --split val train
  run env PYTHONPATH=. $PY notebooks/scripts/localiser_bundle.py gate \
      --candidate "$C_DIR/localiser_boxes_val.csv" --reference "$E3" \
      --label "combinedC_vs_ens3" --out "$LB/gate_combinedC_vs_ens3.json"
  echo "  C is one-directional: it cannot lose IoU on rows A already handles, so it"
  echo "  carries no handicap."
fi

# 6. the declared rule picks the promotion candidate
echo
echo "── 6. promotion candidate: highest validation box IoU among the declared configurations ──"
PYTHONPATH=. $PY - "$LB" "$APPLY" <<'EOF'
import json, pathlib, sys
lb, apply_ = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
rows = []
for name, label, handicap in (("combinedA", "A  combined map ensemble (equal weights)", ""),
                              ("combinedA24", "A24  §24a group-weighted winner", ""),
                              ("combinedB", "B  box fusion with l10", " (-0.0339 route handicap)"),
                              ("combinedC", "C  §19b hybrid with l10", "")):
    f = lb / f"gate_{name}_vs_ens3.json"
    if not f.exists():
        print(f"  {label:<28} not formed"); continue
    g = json.loads(f.read_text())
    rows.append((label, g["mean_iou_candidate"], g["mean_diff"], g["ci_lower"], g["ci_upper"],
                 bool(g["passes"])))
    print(f"  {label:<28}{g['mean_iou_candidate']:.4f}  {g['mean_diff']:+.4f} "
          f"[{g['ci_lower']:+.4f}, {g['ci_upper']:+.4f}]  "
          f"{'PASS' if g['passes'] else 'FAIL'}{handicap}")
if not rows:
    print("  nothing gated (dry run)"); sys.exit(0)
best = max(rows, key=lambda r: r[1])
print(f"\n  declared rule -> promotion candidate: {best[0]} at {best[1]:.4f}")
if not best[5]:
    print("  but it does NOT pass its gate against 0.4700 — nothing is promoted.")
EOF

echo
echo "── 7. ledger check: nothing in the reference ledger moved ──"
if [ "$APPLY" = "1" ]; then
    $PY notebooks/scripts/integrity.py ledger-check && LEDGER_OK=1 || LEDGER_OK=0
else
    echo "  \$ $PY notebooks/scripts/integrity.py ledger-check"
fi

# 8. the pointer step, reached only under --apply and only if clean; it writes nothing
echo
echo "── 8. promotion pointer ──"
if [ "$APPLY" != "1" ]; then
    echo "  without --apply: nothing promoted."
elif [ "$LEDGER_OK" != "1" ]; then
    echo "  LEDGER CHECK FAILED — refusing to write $POINTER."
    exit 1
elif [ "$S24_READY" != "1" ]; then
    echo "  §24 has not run — refusing to write $POINTER."
    echo "  The pre-registration makes the weighted search part of this gate; applying"
    echo "  without it would silently drop a declared step."
    exit 1
else
    echo "  clean; promotion_cascade.sh, not this script, writes refs/promoted_pointer.json"
fi

echo
echo "combined gate complete."

# Pre-registered test pass (D1)

Written before any declared test evaluation. Every threshold, model, and
comparison below is fixed from the validation partition; nothing is changed
after the pass. The only prior contact with the test partition is the
`vgg16_crop_m000` plumbing check (lesion AUC 0.816), which is not reported.

> **Superseded by §33.** The last sentence above is not accurate: every prior contact with the test partition is listed in §33, which carries the statement that replaces it.

## Partition
Single patient-level partition (`data_loader.patient_partition`, SGKF seed 28):
test = 119 patients, 221 mammograms (106 malignant), 243 lesions (113 malignant).
BI-RADS comparison set: 203 mammograms (BI-RADS 0 excluded).

## Models entering the pass (m* = 0.20, selected on val AUC, single seed)
| tag | role |
|---|---|
| vgg16_crop_rung3_unet_bundle | deployable system — PRIMARY |
| vgg16_crop_rung1_oracle, vgg16_crop_rung2_oracle | oracle ceiling (development condition) |
| vgg16_crop_rung4_cam, vgg16_crop_rung5_jitter, vgg16_crop_rung6_shuffled, vgg16_crop_rung7_whole | ladder |
| vgg16_crop_m020, resnet50_crop_m020, densenet121_crop_m020, efficientnet_crop_m020 | architecture family (oracle mask_margin crops, m = 0.20; Wang 2024 Table 3 comparison) |
| vgg16_fusion_{view\|context} (PLACEHOLDER — set by A3 on val) and vgg16_fusion_control | second declared system, if selected (see Fusion below) |
| vgg16_full | Pipeline A whole-image reference |
| baseline/scaled/regularised _crop_m020 | scratch stages 5–7 (descriptive only) |

> **Restated by §28:** every tag in this table is scored under the name §28 gives it; §5 and §37 add rows, and §11 adds a device column.

> **Superseded by §28:** the fusion row's placeholder was never filled in here; view was resolved (§27.1), and §28 names the tags the pass scores.

Rung 3 uses localiser run 3 iff its validation `box_iou_inclusive_mean`
exceeds run 1's 0.3853; otherwise run 1 (frozen artefacts). Decided from
`localise_eval` on val only, before the pass.

> **Superseded by §39:** replaced on 9 September by the section below; rung 3 uses the promoted localiser (§34).

## Localiser selection (amended 9 Sep)
Rung 3 uses the mean probability map of every trained localiser (scratch runs
1 and 3, VGG16-encoder patch-trained run 4), with threshold, minimum component
size, component-selection rule and flip test-time averaging chosen on the
TRAINING split only. Gate: patient-cluster bootstrap 95% CI of the per-lesion
box-IoU difference against run 1 on the 243 validation lesions excludes zero.
Order of events, disclosed: a procedure selecting freely among single runs and
post-processing on train was specified first and gave validation box IoU 0.431
(run 4 + flip TTA + summed-probability rule); the fixed-ensemble framing was
adopted after both validation values were known and gave 0.470. Both are
reported. Run 4 alone (0.419) does not pass the gate; its contribution is
through complementary errors in the ensemble. Rung 5's jitter error model is
resampled from the selected localiser's validation errors. A whole-image
fine-tune of run 4 (run 4b) is in progress; if it finishes by 10 Sep and either
run 4b alone or a four-member ensemble passes the same paired gate against the
three-run ensemble, rungs 3 and 5 are retrained from it. Nothing enters after
10 Sep. Test-split boxes are written by the same apply step and are not read
until the pass.

> **Corrected by §39:** the promoted localiser is not the mean of every trained localiser, and it was not gated against run 1.

> **Superseded by §7:** the sentences from "A whole-image fine-tune" to "Nothing enters after 10 Sep." are replaced there.

Selected configuration (`outputs/results/localiser_bundle/bundle_selection.json`,
9 Sep): runs 1+3+4, flip TTA, peak-probability component, threshold 0.15,
minimum component 256 px; train 0.5456, validation 0.4700, detection 0.955;
gate +0.0848 [+0.0510, +0.1186]. Rung tags `vgg16_crop_rung3_unet_bundle` and
`vgg16_crop_rung5_jitter_bundle`; boxes and error model under
`outputs/results/localiser_bundle/boxes/`; `artifacts/` unchanged (run 1).

> **Superseded by §34:** this paragraph describes the three-run selection (cfg 2814); §34 replaces it with the promoted A24 (cfg 54485).

The architecture family reuses the C2 weights selected on val
(`{arch}_crop_m020_best.weights.h5`; val AUC VGG16 0.8837, EfficientNetB0
0.8503, DenseNet121 0.8478, ResNet50 0.8054, single seed, matched schedule).
Canonical `_best` weights are the fine-tune-stage best for every member,
even where the frozen stage peaked higher (EfficientNetB0: 0.8524 frozen vs
0.8503 fine-tune); the rule is fixed here so no member is re-picked.
Their ranking is declared with the measured single-seed noise floor of
≈ 0.02 AUC: differences below it are not narrated as real.

> **Restated by §32:** these are Metal values; on the reference ledger (§11, §26a) the four read VGG16 0.8777, EfficientNetB0 0.8552, DenseNet121 0.8465 and ResNet50 0.7967, and the weights are the `_cuda0917` ones §28 names.

Fusion placeholder: task A3 trains two fusion modes, each against a
single-crop VGG16 control on identical rows, all on oracle mask_margin crops,
i.e. at the rung-2 development condition.

- **Context** (`vgg16_fusion_context` vs `vgg16_fusion_control_all`): the same
  lesion at m = 0.20 and m = 1.00; needs no pairing, so it uses every lesion
  (1210 / 243 / 243). Declared the original-model contribution if it beats
  its control by > 0.02 val AUC on 243 lesions (the measured noise floor).
  The control is also a third replicate of the rung-2 condition and tightens
  the noise-floor estimate.
- **View** (`vgg16_fusion_view` vs `vgg16_fusion_control`): CC and MLO of the
  same breast at m = 0.20, on the paired subset fixed by
  `artifacts/fusion_pairs_view_{split}.csv` (written 7 Sep, before any fusion
  training): 491 train / 100 val / 101 test pairs, 81–83 % of lesions; CC is
  crop A. Pairs on single-mass breasts: 447 / 87 / 86; pairs matched on the
  per-image `abnormality id` (same breast, not verifiably the same lesion):
  44 / 13 / 15; label-disagreeing pairs dropped: 1 / 0 / 2. Training uses
  both orderings, (CC, MLO) and (MLO, CC), and evaluation averages them, so
  the model is order-symmetric. Because n is smaller, the rule scales with
  it: declared if it beats its control by ≥ 0.04 on the 100 val pairs
  (0.02 × √(243/100)); 0 to 0.04 is **inconclusive at this sample size**, not
  negative; below 0 is negative. Whatever the rule says, view fusion gets its
  pre-registered paired DeLong on the 101 test pairs; the rule only decides
  whether it is a declared system.
- View-fusion and its control are reported as a pair with their Δ and CI.
  Neither is compared with the full-data single-crop models (0.88 on 1210
  rows): a different row set, about 40 % of the crops. This is also why the
  promised CC+MLO experiment can only be a subset study — about 30 % of
  breasts are single-view in the mass subset. The `_single` variant
  (single-mass breasts only) is a sensitivity check, not a selection option. A fusion
mode is declared the original-model contribution only if it beats
`vgg16_fusion_control` by > 0.02 val AUC (the noise floor). It is **not** a
second deployable system: the deployable claim belongs to rung 3 alone. If a
mode wins, it is re-run once with U-Net boxes (`--box-source unet`) before
the freeze, and that run enters the pass as `vgg16_fusion_{mode}_unet`,
reported beside rung 3 descriptively. The winning tags are written into this
table and into the runbook before the freeze. If neither mode clears the
bar, no fusion model enters the pass, the result is reported as negative
from validation only, and rung 3 remains the sole system.

> **Corrected by §39:** the per-mode rules above are the ones applied, not the single > 0.02 rule against `vgg16_fusion_control` in this paragraph.

> **Restated by §28 and §35:** the U-Net re-run enters the pass as `vgg16_fusion_view_unet_promoted_cuda0917`, trained on A24's boxes with three seeds (§35); the tag table above was not updated, and §28 names the tags.

Localiser test-vs-val gap (already declared from `localise_eval`): detection
0.864 on test vs 0.901 on val, miss-inclusive box IoU 0.353 vs 0.385. Two
consequences are fixed in advance: rung 3 is expected to lose more on test
than the oracle rungs, and rung 5's error model is resampled from the
promoted ensemble's 232 observed validation error tuples with its validation
fallback rate of 0.0453 (`outputs/results/localiser_bundle/boxes/localiser_error_model.json`),
so rung 5 mimics the validation error, not the test fallback (13.6 % for
run 1; the ensemble's test rate is not known until the pass). Rung 5 is
therefore read as "val-calibrated random error", not as a test control.

> **Superseded by §34:** this paragraph names the three-run ensemble's error model and states an expectation formed after run 1's test metrics had been seen; §34 replaces the paragraph and withdraws the expectation as a prediction, and §33 lists the 7 September contact the test figures come from.

## Primary endpoint
H1: the deployable system's image-level AUC (lesion scores aggregated by max,
`config.IMAGE_SCORE_AGGREGATION`) on the 203-image comparison set differs from
the BI-RADS anchor (AUC 0.809 [0.747, 0.872]). Paired DeLong, two-sided,
α = 0.05. `compare.primary_endpoint("vgg16_crop_rung3_unet_bundle")`.
Reported with its patient-bootstrap 95 % CI regardless of significance.

> **Restated by §28:** the primary model is scored as `vgg16_crop_rung3_unet_promoted_cuda0917`.

## Secondary families (lesion level, n = 243, Holm-corrected within family)
Two families, each Holm-corrected on its own; no correction across families
because they answer different questions (Wang 2024 Table 3 architecture
comparison vs the localisation-degradation ladder).

**Family A — architectures** (`compare --models ... --out comparison_architectures.json`):
`vgg16_crop_m020`, `resnet50_crop_m020`, `densenet121_crop_m020`,
`efficientnet_crop_m020` → 6 pairwise DeLong tests. All four use the same
oracle mask_margin crops at m = 0.20, the same schedule and the same rows,
so the only varied factor is the backbone.

> **Restated by §28 and §39:** the four models are scored under §28's tags, and the report is `comparison_architectures_test.json`.

**Family B — ladder** (`compare --models ... --out comparison_report.json`):
rung2, rung3, rung4, rung6, rung7 → 10 pairwise DeLong tests.
Rungs 1 and 5 are reported descriptively (rung 1 duplicates rung 2's
question; rung 5 is a control on rung 3's error model, read against rung 3
only). Reason for the restricted family: power at n = 243 does not support
21 corrected tests, and these five carry the design's questions —
value of localisation (3 vs 7), cost of imperfect localisation (3 vs 2),
supervised vs weak (3 vs 4), localisation vs context (4 vs 7, 6 vs 7).

> **Restated by §28 and §39:** the rungs are scored under §28's tags, and the report is `comparison_report_test.json`.

## Fusion (outside both families)
View fusion is declared a system only if exact validation AUC (sklearn, from
--stage val prediction tables) of vgg16_fusion_view exceeds vgg16_fusion_control
by >= 0.04 on the paired validation subset; context fusion likewise by >= 0.02
on 243 lesions against its own control. A declared view-fusion system is
re-trained once with U-Net boxes (selected ensemble, single box, the rung-3
condition, so the fusion result is like-for-like with rung 3) and that
re-run is what enters the pass; the declaration is not revisited.
Applied 9 Sep: view declared (exact val AUC 0.8964 vs control 0.8551,
+0.0413); re-run `vgg16_fusion_view_unet_bundle` on the promoted three-run
ensemble's boxes, exact val AUC 0.8872 on the 100 pairs (control 0.8551,
+0.0321; oracle-box view 0.8964).

> **Restated by §13 and §32:** these are Metal values, and the three-run ensemble is no longer the promoted localiser. On CUDA the declared view gap is +0.0139 on three-seed means (§13); the U-Net re-run was repeated on A24's boxes (§35), with the one-session figures in §32.

Two paired DeLong tests, two-sided, α = 0.05, each uncorrected because each
is a single pre-declared contrast on its own row set: context fusion vs
`vgg16_fusion_control_all` on the 243 test lesions, and view fusion vs
`vgg16_fusion_control` on the 101 test pairs (label of crop A). Neither is
added to a family. A declared fusion mode (per the val rules above) is also
evaluated at image level against the BI-RADS anchor descriptively (with its
bootstrap CI); the primary endpoint remains rung 3 and is not re-assigned
after the pass.

> **Run by the pass from 18 September; see §37.** Until then the pass ran only the view test; the context test is scored on `_cuda0918` re-measurements of both models.

## Ensemble, TTA and seeds (A5; descriptive)
The single-seed model selected on val is the model under test for every
endpoint above. If extra seeds of the declared system(s) are trained (A5.1),
the seed spread is reported as the variance estimate the ladder figure
lacks, and the mean-probability ensemble (`src.ensemble`) and the TTA
variant (`evaluate --tta`, identity / h-flip / 90-180-270° rotations) are
evaluated once on test and reported with CIs as like-for-like literature
comparisons (Shen 2019 quotes 0.88 single / 0.91 ensemble). They do not
replace the primary model, enter no Holm family, and are not used to
re-select anything. Any ensemble/TTA lines are appended to the runbook
before the freeze, or omitted; none are run ad hoc afterwards.

> **Superseded by §27.4:** no TTA variant is run. §38 corrects §27.4's reason: TTA was evaluated once, on validation on 8 September, and was negative.

## Power and reading rules (fixed before the pass)
Approximate paired-DeLong power at the declared sample sizes, assuming the
model-to-model correlation typical of classifiers on the same crops:

| endpoint | n | tests | smallest difference detectable at 80 % power |
|---|---:|---:|---:|
| Family A (architectures, Holm) | 243 lesions | 6 | ≈ 0.07 |
| Family B (ladder, Holm) | 243 lesions | 10 | ≈ 0.08 |
| Primary (system vs BI-RADS) | 203 images | 1 | ≈ 0.10 |

Validation gaps among VGG16, EfficientNetB0 and DenseNet121 are 0.03–0.04,
so Family A is expected to return no significant pair other than VGG16 vs
ResNet50 (0.08 on val); the family is reported as estimates with intervals
and the ranking is not narrated beyond the noise floor. The ladder's large
gaps (≈ 0.14 localisation vs none; ≈ 0.33 localisation vs excluded) are
expected to survive correction. A non-significant primary endpoint is not
read as equivalence with the BI-RADS anchor: the report quotes the
difference with its interval and states what the interval excludes.

> **Restated by §39:** on the reference ledger the Family A gaps are 0.0225, 0.0312 and 0.0810, and the ladder gaps +0.1437 and +0.3014 (§32); the expectations above are unchanged.

DeLong treats lesions as independent; 243 lesions come from 119 patients.
Each AUC difference is also reported with a patient-cluster bootstrap 95 %
CI (`compare.py`, same resampling unit as every other interval in the
report). The bootstrap interval is robustness, not a second test; where the
two disagree, the report says so and the DeLong p stands as pre-registered.

## Fixed operating points (from val)
- screening threshold: smallest score giving sens ≥ 0.90 on val
- BI-RADS-matched point: specificity matched to 0.427 (`analysis.birads_matched_operating_point`)
- temperature: fitted on val, applied unchanged
- deferral: 30 % by MC-Dropout std (50 passes), threshold from val

> **Corrected by §38:** the BI-RADS-matched point matches sensitivity on validation to the anchor's 0.940 and compares specificity with its 0.427.

## Also run on test, once, descriptively
calibration (ECE, Brier), PPV/NPV at cbs_test and screening prevalence,
decision curve, subgroups, `explain` summary statistics (no per-case
selection), `uncertainty` deferral curve, ladder figure with CIs.

## Not permitted after the pass
Retraining, reselecting margin/architecture/threshold, adding models,
changing the aggregation rule, or re-running any test command with altered
arguments. A bug found after the pass is fixed and the whole pass re-run,
and the report says so.

> **Superseded by §36:** the "whole pass re-run" sentence is withdrawn (rule 5); a pass that fails after its first test table is resumed from the failed stage, never re-scored (rules 2 and 3).

<!-- amendments applied at D3 -->

# Appendix — amendments, applied in the order fixed before the freeze

Assembled by `notebooks/scripts/apply_amendments.py` from
`docs/preregistration_amendments_pending.md`. The order is the one that file's own
table fixed; several blocks restate figures an earlier block establishes, so it is
load-bearing rather than cosmetic. Blocks written after a result was seen say so in
their first line and disclose the order of events.

## 1. §33 — Prior contact with the test partition, complete, and a correction of the record (18 Sep, before the freeze)

*Why here: First, because it corrects the opening paragraph's statement about test contact and supersedes the closing sentences of §7 and §11, each of which carries a pointer to it; a reader meets it immediately after the protocol. Written 18 September, before the freeze.*

**What this corrects.** Three sentences in this document describe contact with the test partition:
the opening paragraph ("The only prior contact with the test partition is the `vgg16_crop_m000`
plumbing check"), the close of §7 ("The test partition has not been read at any point") and the
close of §11 ("The test partition has not been read"). The two later sentences were written on
9 September. None of the three is accurate at the freeze, and this block replaces all three.

In the opening paragraph, replace

> The only prior contact with the test partition is the `vgg16_crop_m000` plumbing check (lesion AUC
> 0.816), which is not reported.

with

> Every prior contact with the test partition is listed in §33. No test result has informed any
> selection, gate, threshold, promotion or decision.

and read the closing sentences of §7 and §11 as superseded by the same statement.

**The complete list.**

| when | what touched the test partition | what it produced | what was done with it |
| :--- | :--- | :--- | :--- |
| before 7 Sep | the `vgg16_crop_m000` plumbing check | one lesion AUC, 0.816 | quoted in the opening paragraph and not reported. Its prediction table was deleted on 10 September; its results file, model card and analysis files are kept in `outputs/_superseded/plumbing_pass_m000/`. |
| 7 Sep | `src/localise_eval.py` on localiser run 1, whose default then scored all three splits | run 1's test localisation metrics (`artifacts/localiser_metrics.json`) | detection 0.864, box IoU 0.353 and fallback 13.6 % are quoted under "Localiser test-vs-val gap". The default was changed to train and val on 8 September. |
| 7 Sep | `src/cam_localise.py`, run over all three splits | the CAM box source's test localisation metrics (`artifacts/cam_metrics.json`) | quoted nowhere and used for nothing. The CAM threshold was chosen by a validation sweep (`artifacts/cam_manifest.json`: "validation sweep, mid-plateau"). |
| 10 Sep | a validation dry run of `run_test_pass.sh`, launched before `--split` had been threaded through every stage; the stages without it defaulted to the test split | eight models scored on test — `vgg16_full`, `baseline_crop_m020`, `scaled_crop_m020`, `regularised_crop_m020`, `vgg16_crop_m020`, `resnet50_crop_m020`, `densenet121_crop_m020`, `efficientnet_crop_m020` — as prediction tables, a run log carrying seven test AUC lines, and one results file and one model card per model | the tables, the log and one stray file were deleted unopened on 10 September, and test AUC lines were removed from four other logs without being printed. The sixteen results files and model cards, which that deletion missed, were moved unopened to `outputs/_superseded/test_scoring_20260910/` on 18 September, with sha256 recorded before and after. The loader-level guard (`src/test_pass_guard.py`) was added on 10 September in response. |
| 18 Sep | the localiser test-map job: `write_torch_maps.py` for the four torch members on an RTX 5090 pod, and `localiser_bundle.py cache --images-only` for the three Keras members on the laptop | test probability maps for the seven promoted members, plain and flipped: fourteen arrays of 243 × 1024 × 576, uint8 | computed from test images only. The pod's data directory held no mask array, asserted before inference; the label column was removed from the metadata before upload; no metric was computed. Two validation controls ran first, each on the device that made its reference — l7a (torch, RTX 5090) and run4 (Keras, laptop) — and both reproduced every derived box of the stored validation maps, with the maps themselves bit-identical. The test boxes are written inside the pass. |
| 7 Sep onward | fusion's pair construction, in every validation dry run and fusion validation run | the test view-pair counts | only the counts the fusion placeholder states (101 test pairs, 2 label-disagreeing pairs dropped); closed on 18 September (§38) |
| throughout | `build_datasets`' test pipeline, in every run that called it | test image sizes, read from DICOM headers; no pixel decoded | closed on 18 September (§38) |
| 18 Sep | run4's test maps regenerated on the laptop (`localiser_bundle.py cache --images-only`, the identical command) | two arrays | test images only; bit-identical to the finished copy (below) |

**Also recorded, because they are computed from test rows though they are not model results.** The
partition counts under "Partition" and the BI-RADS anchor's test AUC under "Primary endpoint" come
from the test labels and the recorded BI-RADS assessments; both are part of the protocol's
specification. The frozen artefacts `artifacts/lesion_geometry.csv`, `artifacts/lesion_bbox_lcc.csv`
and `artifacts/localiser_boxes_test.csv`, and the localiser tensor cache, were built over all three
partitions. The view-pair tables used test labels to drop label-disagreeing pairs (two on test), as
the fusion placeholder above declares. During the pre-freeze review on 18 September, run 1's full
test row in `artifacts/localiser_metrics.json` was displayed, beyond the three figures quoted above,
and was displayed again on 19 September while checking whether any gate, selection, threshold or
promotion references a test number; it was not used either time. `config.PREVALENCE["cbis_test"]`
(0.4796 = 106/221) is the Partition line's test prevalence, used only for the declared PPV/NPV.

**Two contacts through gaps in the guard, found by the second pre-freeze review (18 Sep).** Fusion's
pair construction iterated the split dictionary, which the guard did not cover, so every validation
dry run and fusion validation run rebuilt the test view pairs from the test labels. It printed only
the counts the fusion placeholder states. `build_datasets` built the test pipeline in every run that
called it, walking the test rows, resolving their image paths and reading their image sizes, without
decoding any pixel. Both are closed (§38).

**run4's test maps were first copied into the bundle unfinished.** A watcher waited until six map
files existed rather than for the writer to finish, and run4's arrays were hashed and copied while
still being written. The pre-freeze manifest check caught it before any use; no run had read them,
because dry runs apply to validation and the rehearsal links validation maps in. run4's test maps
were regenerated with the identical command, and the regenerated arrays were bit-identical to the
finished scratch copy, which replaced the bundle's. The partial copy is kept in
`outputs/_superseded/run4_test_maps_partial_copy_20260918/`. The fourteen arrays' sha256 are recorded
in `refs/test_maps_manifest.json` with the card, dates and control verdicts, and the pass checks them
before the apply step reads the maps. Their row order rests on the controls, in which the same
writers reproduced the stored validation maps row for row.

**A reviewer's claim, checked and found wrong.** The second review stated that
`outputs/results/data_flow.json`, dated 17 September, was written through a guard bypass. The session
record shows that run was refused at the guard; the file's date comes from restoring a backup with
`cp`, and its counts predate the guard and equal the Partition line.

**One reading rule rests on a listed contact.** The expectation under "Localiser test-vs-val gap"
that rung 3 will lose more on test than the oracle rungs was formed after run 1's test metrics had
been seen. §34 withdraws it as a prediction.

**Order of events, and a correction of the author's own record.** On 10 September the author ruled
that the dry run's outputs be deleted and the event not disclosed, on the view that nothing had been
read. On 18 September, in the pre-freeze review, the author at first held that the partition had
never been touched. **Both positions were wrong.** The files that survived the deletion are results
files that `src.evaluate` writes only after scoring the test split. They carry 221 image rows for
`vgg16_full` against validation's 225, and 203 BI-RADS comparison images against validation's 209.
They show that the scoring occurred, and a protocol asserting no prior contact would have been false.

The claim that matters is narrower, and the record supports it: **no test result has informed any
selection, gate, threshold, promotion or decision.** It was checked on 18 September in two places:

- **Every JSON under `outputs/results/` and `refs/`** (563 files, training histories excluded): no
  gate, selection, pointer or promotion field carries a test-derived value.
- **Every session transcript:** no ruling cites a test number, and the one continuation decision (§7)
  cites the validation preview.

This block is written before the freeze so that the frozen document carries the correction, rather
than an erratum written after it.

## 2. §34 — The promoted localiser in the protocol's own text (executes §6; 18 Sep)

*Why here: Right after §33, because it corrects the protocol's own "Localiser selection" text, which a reader meets first. Executes §6 for the two paragraphs §28 did not reach, and withdraws one expectation as a prediction.*

**What this corrects.** A24 was promoted on 12 September (§24a, with the floor gate of §35) and is
recorded in `refs/promoted_pointer.json`. §28 applied §6's substitution to the tag table. Two
paragraphs under "Localiser selection (amended 9 Sep)" still describe the superseded three-run
selection, and this block applies §6 to them.

In "## Localiser selection (amended 9 Sep)", replace

> Selected configuration (`outputs/results/localiser_bundle/bundle_selection.json`,
> 9 Sep): runs 1+3+4, flip TTA, peak-probability component, threshold 0.15,
> minimum component 256 px; train 0.5456, validation 0.4700, detection 0.955;
> gate +0.0848 [+0.0510, +0.1186]. Rung tags `vgg16_crop_rung3_unet_bundle` and
> `vgg16_crop_rung5_jitter_bundle`; boxes and error model under
> `outputs/results/localiser_bundle/boxes/`; `artifacts/` unchanged (run 1).

with

> Selected configuration (`outputs/results/localiser_bundle/bundle_selection.json`, cfg 54485,
> promoted 12 Sep; §24a): runs 1+3+4 with l7v, l7a, l8a and l9, group-weighted (the three Keras
> members 0.368, the four torch members 1.474; label `g:keras=0.5,torch=2`), flip TTA,
> peak-probability component, threshold 0.40, minimum component 16 px, all selected on train;
> train 0.8051, validation 0.5300; gate against the three-run ensemble +0.0599 [+0.0328, +0.0869].
> Rung tags carry `_promoted_cuda0917` (§28); boxes and error model under
> `outputs/results/localiser_bundle/boxes_A24/`; `refs/promoted_pointer.json` records the md5s of its
> train and validation box tables;
> `artifacts/` unchanged (run 1). The superseded three-run selection (cfg 2814: runs 1+3+4,
> threshold 0.15, minimum component 256 px; train 0.5456, validation 0.4700, detection 0.955;
> gate against run 1 +0.0848 [+0.0510, +0.1186]) is kept with its numbers.

and replace

> Localiser test-vs-val gap (already declared from `localise_eval`): detection
> 0.864 on test vs 0.901 on val, miss-inclusive box IoU 0.353 vs 0.385. Two
> consequences are fixed in advance: rung 3 is expected to lose more on test
> than the oracle rungs, and rung 5's error model is resampled from the
> promoted ensemble's 232 observed validation error tuples with its validation
> fallback rate of 0.0453 (`outputs/results/localiser_bundle/boxes/localiser_error_model.json`),
> so rung 5 mimics the validation error, not the test fallback (13.6 % for
> run 1; the ensemble's test rate is not known until the pass). Rung 5 is
> therefore read as "val-calibrated random error", not as a test control.

with

> Localiser test-vs-val gap (`localise_eval` on run 1, 7 Sep; §33): detection 0.864 on test vs
> 0.901 on val, miss-inclusive box IoU 0.353 vs 0.385. Rung 5's error model is resampled from the
> promoted ensemble's 237 observed validation error tuples with its validation fallback rate of
> 0.0247 (`outputs/results/localiser_bundle/boxes_A24/localiser_error_model.json`), so rung 5
> mimics the validation error, not the test fallback (13.6 % for run 1; the promoted ensemble's
> test rate is not known until the pass). Rung 5 is therefore read as "val-calibrated random
> error", not as a test control.

**Withdrawn as a prediction: "rung 3 is expected to lose more on test than the oracle rungs".** It
was written as a consequence "fixed in advance", but it was formed after run 1's test metrics had
been seen (§33), so it is not a blind prediction. Nothing in the pass tests it. If rung 3 does lose
more on test than the oracle rungs, that is reported as an observation, with this history beside it.

## 3. §36 — What happens if the pass fails (18 Sep, before the freeze)

*Why here: Early, because it corrects the protocol's own "Not permitted after the pass" and resolves its conflict with §7b's "The test pass", which carries a pointer here.*

**Two rules contradicted each other.** "Not permitted after the pass" says "A bug found after the
pass is fixed and the whole pass re-run, and the report says so." §7b's "The test pass" (10 Sep)
says "Once any test output exists, the pass is final — no second pass, whatever the failure." §7
declared the first section unchanged, and `run_test_pass.sh` enforces the second: it refuses to
start once test prediction tables exist.

**Order of events.** The contradiction was found on 18 September, in a review of the pass's
real-mode code paths, which the validation dry run cannot execute. The same review found defects
that would have stopped the pass after test tables had been written; they are fixed (§37). A second
review the same day found that the rule as first written here could not be carried out: the script
had no way to resume, and the scoring stages compute their metrics in the same call that writes
their tables. The marker, the new stage order and `--resume-from` were added then, and the
rehearsal exercises them. This rule was written before the freeze, and no test data was read in
writing it.

**The rule.**

1. **Before any test prediction table exists** — the pre-flight, the check that the token names
   this protocol, the model-list assertion, the step that writes the test boxes, and the
   box-coverage check — a failure is fixed and the pass is re-run from the start. The failure is
   logged. The test-box step prints the localiser's test metrics, so a re-run under this rule can
   follow a test number having been seen; if it happens, it is disclosed.
2. **No test prediction table is regenerated.** After a model's test table exists, the model is not
   re-scored for that table, retrained, re-selected or re-thresholded. This rewording is a
   correction, not a relaxation: the first wording, "no model is re-scored on test", forbade
   Grad-CAM and MC-dropout, which run their own forward passes on test by design.
3. **The pass writes `refs/test_pass_started.json` before its first test prediction table, and
   refuses to start again while it exists.** Every model is scored before any comparison runs, so
   every later stage reads tables only. If one of those stages fails because of a defect in code,
   the defect is fixed and the pass is resumed once from that stage with `--resume-from`, which
   requires the marker and refuses the scoring stages. A derived table already written (an ensemble,
   the per-pair controls) is not rewritten. The failure, the fix and its diff are reported beside
   the results.
4. **If a gate fails after test tables exist** — the ledger check is the last — nothing from the pass
   is quoted until the change the gate reports is explained in writing beside the results. Nothing
   is re-scored.
5. **The "whole pass re-run" sentence is withdrawn.** Re-scoring the test partition after its
   numbers exist is what a single pass exists to prevent. §7b's "no second pass" stands as rules 2
   and 3.

## 4. §38 — Corrections from the second pre-freeze review (18 Sep)

*Why here: Early, because it corrects the protocol's own "Fixed operating points" sentence; it also corrects §27.4's reason for omitting TTA and §23.3's statement about the guard, each of which carries a pointer here.*

**The BI-RADS-matched operating point.** In "## Fixed operating points", replace

> - BI-RADS-matched point: specificity matched to 0.427 (`analysis.birads_matched_operating_point`)

with

> - BI-RADS-matched point: sensitivity matched on validation to the BI-RADS anchor's 0.940, and
>   specificity compared with its 0.427 (`analysis.birads_matched_operating_point`)

The code was the correct behaviour and the protocol sentence was wrong. Specificity at the
radiologists' own sensitivity is the clinically meaningful comparison, and it is what the function's
docstring describes. Found before the freeze, with nothing computed on test.

**§27.4's reason for omitting TTA.** "No TTA variant has been trained or evaluated on any ledger in
this project" is false. TTA was evaluated once, on validation, on 8 September on the Metal ledger,
and was negative (no TTA 0.8843, flip 0.8813, flip with rotations 0.8645); §2 records it as "negative
on validation". It has never been evaluated on the CUDA ledgers. The omission stands, for that
reason.

**§23.3's statement about the guard.** "The loader guard means the test split cannot be read without
it in any case" was not true until 18 September. The guard covered only `frames["test"]` and
`.get("test")`; iterating the split dictionary, and `build_datasets`, reached the test split without
it. The contacts this allowed are listed in §33. From 18 September every route to the test frame
refuses without the pass's token (indexing, `items()`, `values()`, copies, `dict()`, `pd.concat`),
`build_datasets` builds the test split only under the token, and the guard's own test covers each
route.

## 5. §39 — The protocol's own statements that no block corrected (18 Sep, before the freeze)

*Why here: Right after §38, because it corrects five statements in the protocol's own text that no block had corrected; each carries a pointer here.*

**What this corrects.** A read of the protocol against this appendix on 18 September found five
statements in the protocol's own text that the last two weeks made false and that no block corrected:
each was superseded only by implication, which is not a correction. They are corrected here, and each
carries a pointer to this block. Every other stale statement in the protocol carries a pointer to the
block that corrects it.

1. **"Rung 3 uses localiser run 3 iff its validation `box_iou_inclusive_mean` exceeds run 1's 0.3853;
   otherwise run 1 (frozen artefacts)."** Replaced on 9 September by the section below it, "Localiser
   selection (amended 9 Sep)", which selected an ensemble. Rung 3 uses the promoted localiser, A24
   (§34); `artifacts/` is unchanged.
2. **"Rung 3 uses the mean probability map of every trained localiser (scratch runs 1 and 3,
   VGG16-encoder patch-trained run 4) … Gate: … against run 1 …"** True of the three-run selection of 9
   September, not of the promoted localiser. A24 is a group-weighted mean (§24a: the three Keras members
   0.368 each, the four torch members 1.474 each) over the seven members of the set §7b closed — runs 1,
   3 and 4, l7v, l7a, l8a and l9. Localisers trained but not selected as members, among them runs 4b and
   4c, L6, L6b and L10b, are reported where they were gated. Threshold, minimum component,
   component rule and flip TTA were chosen on the training split only, as the section says. A24 was
   promoted by the combined gate (§7b, §24a): its per-lesion box IoU against the three-run ensemble's,
   +0.0599 [+0.0328, +0.0869] on the 243 validation lesions, and the rung-3 floor gate (§35).
3. **"A fusion mode is declared the original-model contribution only if it beats
   `vgg16_fusion_control` by > 0.02 val AUC (the noise floor)."** Written before the per-mode rules and
   contradicted by them. The rules applied on 9 September are the per-mode ones: view by ≥ 0.04 against
   `vgg16_fusion_control` on the 100 validation pairs, context by 0.02 against
   `vgg16_fusion_control_all` on the 243 lesions (the context paragraph writes "> 0.02", "Fusion (outside
   both families)" writes ">= 0.02"; context was negative under both).
4. **The output names `comparison_architectures.json` and `comparison_report.json`.** The pass writes
   `comparison_architectures_test.json` and `comparison_report_test.json`: every comparison output
   carries its split, so a validation dry run cannot write over a test result.
5. **"Validation gaps among VGG16, EfficientNetB0 and DenseNet121 are 0.03–0.04 … VGG16 vs ResNet50
   (0.08 on val) … ≈ 0.14 localisation vs none; ≈ 0.33 localisation vs excluded."** Metal-ledger
   figures. On the reference ledger (§32): VGG16 − EfficientNetB0 0.0225, VGG16 − DenseNet121 0.0312,
   VGG16 − ResNet50 0.0810; rung 3 − rung 7 +0.1437, rung 3 − rung 6 +0.3014. The expectations drawn
   from them are unchanged: the first two gaps lie below Family A's ≈ 0.07 detectable difference and
   VGG16 vs ResNet50 above it, and both ladder gaps lie far above Family B's ≈ 0.08.

No endpoint, family, threshold or aggregation rule changes.

## 6. §7 — Development continues on validation past 10 / 13 September (supersession, drafted 9 Sep)

*Why here: Supersedes the calendar rule everything else assumes. Discloses that it was decided after the validation preview.*

In "## Localiser selection (amended 9 Sep)", replace

> A whole-image
> fine-tune of run 4 (run 4b) is in progress; if it finishes by 10 Sep and either
> run 4b alone or a four-member ensemble passes the same paired gate against the
> three-run ensemble, rungs 3 and 5 are retrained from it. Nothing enters after
> 10 Sep.

with

> A whole-image fine-tune of run 4 (run 4b) and a regularised patch re-train (run 4c) were evaluated
> after this document was first written; both are gated exactly as above. **Localiser and classifier
> development continue on validation beyond the original 10 September candidate deadline and 13 September
> freeze, under unchanged gates and an unchanged selection rule**, following `docs/work_plan_auc090.md`.
> The gates remain: for a localiser candidate, the paired patient-cluster bootstrap 95 % CI (1,000 draws)
> of its per-lesion box IoU minus the promoted localiser's, misses scored 0, on the 243 validation
> lesions, must exclude zero, **and** a rung-3 retrain on the candidate's boxes must not lower exact
> validation AUC. For every classifier run, selection is on validation `val_auc` (max), single seed.
> **No freeze date is asserted here.** The freeze date is set when Gate C of the work plan is reached or
> abandoned (promoted-ensemble validation box IoU ≥ 0.55, detection at IoU 0.5 ≥ 0.75, rung-3 exact
> validation AUC ≥ 0.84), and is written into this document at that point, before the pass.
>
> **Order of events, disclosed.** This extension was decided on 9 September, *after* the validation
> preview of the primary endpoint had been read: rung 3 scored image-level 0.806 against the validation
> BI-RADS anchor 0.863, a difference of −0.057 [−0.122, +0.009], p = 0.07. The extension is therefore an
> amendment made in knowledge of a validation result that fell short of the anchor, and it is reported as
> such. It changes no endpoint, no family, no threshold rule and no aggregation rule. The test partition
> has not been read at any point, and no file matching `outputs/results/*test*` exists before the pass.

> **Superseded by §33.** The last sentence above was written on 9 September and is not accurate at the freeze. §33 lists every prior contact with the test partition and carries the statement that replaces it.

Nothing else in "## Not permitted after the pass" changes: once the pass is run, the stopping rule stands
as written.

> **Superseded by §36**, which changed "## Not permitted after the pass" on 18 September: its "whole pass re-run" sentence is withdrawn, and a pass that fails after its first test table is resumed from the failed stage, not re-run.

## 7. §8 — Exact AUC governs every number (drafted 9 Sep)

*Why here: Every figure in later blocks is an exact AUC. Must land before anything quotes one.*

Append to "## Power and reading rules (fixed before the pass)":

> Every AUC quoted as a result, and every selection or gate decision made on one, is the **exact**
> value computed by scikit-learn from a `{tag}_predictions_{split}.csv` table. Keras's binned `val_auc`
> from a training history is a monitoring metric only: it is used to pick the best epoch inside a run
> and never as a reported number or a comparison. The two differ by up to ≈ 0.02 on this data, always
> in the same direction (the history reads high), so a contrast computed from histories is not
> comparable with one computed from tables. Where an earlier draft quoted a history value, the exact
> value replaces it and the history is given in brackets.

This rule is why the seed and gap figures in §2 above are 0.8128 / 0.8156 / 0.8175 and 0.079 rather than
the 0.8123 / 0.8196 / 0.8166 and 0.061 of the first draft.

## 8. §9 — The scoring path is declared (drafted 9 Sep)

*Why here: Explains why the histories in §10 are a monitoring metric and the tables are not.*

Append to "## Power and reading rules (fixed before the pass)":

> **Every reported metric is scored through the eager inference path**
> (`model(x, training=False)`, which is what `evaluate.collect_predictions` calls) **or on CUDA
> hardware. The compiled TensorFlow graph path on tensorflow-metal is not used for any reported
> number.** On the project's development machine (Apple M3 Pro, tensorflow 2.17.0, keras 3.14.0,
> tensorflow-metal 1.2.0) the compiled path — `model.predict`, `model.evaluate`, and the validation
> pass inside `model.fit` — **does not apply ReLU at all: it returns the pre-activation values
> unchanged.** The behaviour is documented on the Apple Developer Forums as *"tensorflow-metal ReLU
> activation fails to clip negative values"* (reported against tensorflow-metal 1.2.0 with TF-macOS
> 2.16.2); this project's stack is TensorFlow 2.17.0 with the same tensorflow-metal 1.2.0, and the
> op-level reproduction matches the report exactly. Isolated on a two-layer head with random weights:
> 49.6 % of the pre-activation values are negative, the compiled ReLU output reaches −0.604 where the
> correct minimum is 0, and **the compiled ReLU output equals the same graph with the activation removed
> to 2.3e-07**, i.e. ReLU is a no-op in the compiled graph. The eager call on the same GPU agrees with
> the CPU to 2.3e-07.
>
> Measured on the 243 validation lesions, the two paths differ by a median of 2.6 to 2.9 in absolute
> logit and by up to 0.031 in AUC, always with the compiled path reading higher except at rung 3. A
> dense stack without activations is exact to 1e-7, and adding a ReLU reproduces the fault whether the
> activation is fused into the Dense layer or applied separately. It is present in training mode as well
> as inference mode, so the Keras training histories of every classifier in this project are affected and
> are quoted only as the monitoring signal that drove epoch selection, never as a result. The two U-Net
> localisers carry no dense head and agree between the two paths, so the localisation metrics, the
> ensemble selection and the promoted boxes are unaffected; this was verified rather than assumed.
>
> **The eager path on the Metal GPU is the reference and is safe to score with.** On the same 243 rows
> it agrees with the same computation forced onto the CPU to a maximum absolute probability difference of
> 4.4e-06 across the four tags checked, and the exact AUCs are identical to four decimals, so validation
> scoring is not forced onto the CPU. Detected 9 September by
> `notebooks/scripts/check_weights_roundtrip.py`, which compares both paths against a NumPy
> recomputation from the saved weight matrices.

Consequence for the runbook: no line changes, because `src.evaluate` already scores through the eager
path. The declaration exists so the report can state which numbers were produced how, and so that any
future run on this machine is read correctly.

## 9. §11 — The CUDA ledger is the reference; Metal is labelled history (decision B, 9 Sep)

*Why here: Decides *which* exact AUC every later block means. §2, §13, §14 all restate numbers to CUDA, so this must precede them.*

Append to "## Power and reading rules (fixed before the pass)":

> **Every validation number that a gate or a selection compares against is the CUDA value.** The
> project's development ledger was built on the Apple M3 Pro, on which tensorflow-metal 1.2.0 does not
> apply ReLU in the compiled graph (§9). The prediction tables produced there are correct, because they
> were scored through the eager path, but the training histories that chose each run's epoch are not,
> so a Metal run and a CUDA run of the same configuration are not replicates of one another: they
> differ in how long they trained. On 9 September the whole validation ledger was retrained on an
> RTX 4090 under the corrected default (early-stopping patience 25), and **those runs are the
> reference every later comparison is made against** — the rung-3 gate for any promoted localiser, the
> rung-2 ceiling, the architecture family and the fusion contrasts. The Metal numbers remain in the
> report as labelled history, with the device named wherever they appear, because they are the runs the
> ladder was discovered on and the reader is entitled to see both.
>
> **Device is part of a tag's identity.** Every result table in the report carries a device column, and
> no comparison mixes devices without saying so. Where a Metal and a CUDA value of the same condition
> are both quoted, they are quoted side by side and the CUDA value is the one that carries the decision.
>
> **Every reported number is scored on the laptop**, on one machine in one environment, through the
> eager path, whatever machine trained the weights. Pod-side scores exist in the run manifests as the
> in-session read that branched the work and are not reported. The two agree to a small number of
> flipped ranked pairs: across the eighteen runs of the CUDA ledger the pod-versus-laptop difference is
> an **exact integer multiple of 1/(n_pos x n_neg)** in every case — 0 to 6 pairs for seventeen of them
> and 26 pairs (0.00177) for ResNet50 — which is near-tied pairs swapping order under probability
> differences of order 1e-6, not a difference in the computed function. AUC is a rank statistic
> quantised at one pair (6.8e-05 on the 243-row splits, 4.0e-04 on the 100 fusion pairs), so exact
> agreement of the statistic is not the right check and is not claimed; agreement of the probabilities
> is, and it holds. The largest disagreement is 8.9 % of the single-seed noise floor and no ordering
> changes: the laptop-scored ladder is 2 > 1 > 3 > 5 > 4 > 7 > 6, as on the pod, and rung 3 minus
> rung 5 is +0.0792 against the pod's +0.0787 and Metal's +0.0792.

Tags table: add a **device** column, `Metal` or `CUDA`, to "## Models entering the pass". Every row
retrained on 9 September reads `CUDA`; any row still quoted from the original ledger reads `Metal` and
is named as history in the text.

**Order of events, disclosed.** This rule was written on 9 September, after the CUDA ledger had been
read on validation. It changes no endpoint, no family, no threshold rule and no aggregation rule; it
fixes which of two measurements of the same pre-registered quantity is authoritative, and requires
both to be shown. The test partition has not been read.

> **Superseded by §33.** The last sentence above was written on 9 September and is not accurate at the freeze. §33 lists every prior contact with the test partition and carries the statement that replaces it.

## 10. §10 — The noise floor for multi-seed comparisons (drafted 9 Sep, adopted after the result)

*Why here: Needs §11 first: the seed SDs it quotes are CUDA seed SDs.*

Append to "## Power and reading rules (fixed before the pass)":

> The ±0.02 AUC noise floor quoted throughout is a **single-seed** floor. It was measured from three
> code-path replicates of the rung-2 condition on the Metal device (`vgg16_crop_m020` 0.8843,
> `vgg16_crop_rung2_oracle` 0.8608, `vgg16_fusion_control_all` 0.8747), which vary by pipeline rather
> than by seed and were all produced on hardware since shown to compute the classification head
> incorrectly. It therefore bounds the wrong quantity for a comparison of two *k*-seed means, where the
> relevant scale is the seed spread and the mean's standard error falls with k.
>
> **For a comparison of means over k seeds, the threshold is `max(0.01, 3 × pooled seed SD)`.** The floor
> of 0.01 keeps the rule from becoming arbitrarily permissive when seeds happen to agree closely. The
> single-seed ±0.02 floor continues to govern every single-seed comparison, including every localiser
> gate, which is unchanged.

**Order of events, disclosed.** This rule was adopted **after** the patience result was known, and both
readings are reported:

| comparison | value | original rule (0.02, single-seed) | adopted rule (max(0.01, 3 × pooled SD)) |
| :--- | ---: | :--- | :--- |
| patience 25 vs patience 10, rung 2, three seeds each | **+0.0107** | does not fire | **fires** (threshold 0.01; pooled seed SD ≈ 0.002, so 3 SD ≈ 0.006) |

The patience-25 seeds are 0.8754 / 0.8745 / 0.8739, SD 0.0008; the patience-10 seeds are 0.8649 / 0.8613
/ 0.8654, SD 0.0022. The gain is positive on every seed and roughly five times the pooled seed SD, which
is what motivated re-examining a floor built from single-seed, cross-pipeline, defective-hardware
variation. The original rule's verdict, that the change does not fire, is reported beside the adopted
one wherever the patience result appears.

**Consequences, applied from 9 September:**

- Patience 25 is the default for every classifier run: the CUDA ledger, the resolution comparison, the
  rung-3 retrains used for localiser gating, and the work plan's §10 and §13 runs. The epoch cap is
  unchanged at 100 per stage.
- The rung-2 ceiling reference becomes the patience-25 three-seed mean **0.8746**, replacing 0.8639.
- The 448-px comparison is read against the patience-25 seed-28 value **0.8754**, not 0.8649.
- Localiser gates are single-seed and keep the ±0.02 floor.

## 11. §14, §14a — The rung-2 ceiling reference (decision D, 9 Sep)

*Why here: Sets the ceiling every §10-family comparison is read against. §14a supersedes the figure §14 declares and must be read with it: the appendix may not carry two live ceiling references. The correction sits here, where the rule is read, and points forward to the block that redeclared it.*

### §14 — The rung-2 ceiling reference (decision D, 9 Sep)

Append to "## Power and reading rules (fixed before the pass)":

> **The rung-2 ceiling reference is the ladder route's three-seed CUDA mean, 0.8731.** Two tags reach
> nominally the same condition — an oracle box at m = 0.20 — by different code paths, and the ladder's
> is the one every rung is measured through, so it is the one the ceiling is quoted from:
>
> | route | tag | seed 28 | seed 1 | seed 2 | mean | SD |
> | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
> | ladder (box provider) | `vgg16_crop_rung2_oracle` | 0.886110 | 0.866889 | 0.866344 | **0.873114** | 0.0113 |
> | family (mask margin) | `vgg16_crop_m020_p25` | 0.875341 | 0.874387 | 0.873841 | 0.874523 | 0.0008 |
>
> The family route's value is the architecture-comparison number and is not the ceiling reference.
>
> **Why the two routes differ, and by how much.** The ladder takes its oracle box from the localiser's
> box table, where the ground-truth box is measured on the 1024 x 576 letterbox canvas and mapped back
> through the inverse, quantising it at a median **5.9 mammogram pixels per canvas pixel** (up to 8.1);
> the family route takes the bounding box straight from the full-resolution mask DICOM. So the two
> crops differ slightly at their edges. **That difference is smaller than seed noise:** the three-seed
> means differ by **+0.0014**, while the apparent 0.0108 gap at seed 28 alone is an artefact of the
> ladder route's seed spread, which is fifteen times the family route's (SD 0.0113 against 0.0008).
> Neither route's seed-28 value should be quoted alone, and the ladder route in particular should
> always be quoted as a mean over seeds.

### §14a — The rung-2 ceiling reference is the five-seed mean; §14 defers to §26 (erratum with disclosure, 12 Sep)

**Read this block with §14 above it. §14's declaration of the ceiling reference is superseded here and
the figure it names must not be quoted.** This block is placed at §14's position, not §26's, because the
correction belongs where the wrong rule is read: §14 appends to "Power and reading rules (fixed before
the pass)", so a reader who stops at §14 would otherwise carry the superseded figure into the pass.

**Order of events, and why this is an erratum rather than something stronger.**

1. **§14 fixed the ceiling at the three-seed mean on 9 September, before seeds 3 and 4 existed.** It was
   correct on the seeds then available: `vgg16_crop_rung2_oracle` at seeds 28, 1 and 2 gave 0.886110,
   0.866889 and 0.866344, mean **0.873114**, seed SD **0.011258**.
2. **§26(b) added seeds 3 and 4 on 10 September and redeclared the ceiling**, at mean **0.872478**, seed
   SD **0.008405**, stating in terms that "wherever the ceiling appears, 0.8725 (five seeds) is the
   figure". **§26 governs.** §26 did not name §14, and §14 was left standing; the two references have
   been live together since, resolvable only by comparing their positions in this table. That is the
   defect this block closes.
3. **The refinement comes from additional seeds on the same condition.** Seeds 3 and 4 are further draws
   of the identical recipe — vgg16, patience 25, CUDA, margin 0.20, `crop_store_rung2_oracle`, exact
   validation AUC through the eager path. **It does not come from a result**, and **no test data was
   read**: every figure here is validation, 243 lesions.
4. **No gate verdict moves.** In particular the **§7b floor gate is unaffected**: it derives from rung 3's
   **old-boxes** three-seed mean (`FLOOR_REF=0.814022`, `FLOOR_DROP=0.02`, floor **0.794022**) and does
   not read the ceiling at any point. The rung-3 retrain cleared it at 0.861005 before this correction
   and clears it by the identical margin after.

**The corrected figures. These are the ones to cite.**

| | value |
| :--- | ---: |
| rung 2, oracle ceiling — **five-seed CUDA mean** | **0.872478** (seeds 0.886110 / 0.866889 / 0.866344 / 0.875136 / 0.867912; SD **0.008405**) |
| rung 3, promoted three-run boxes | 0.814022 (three-seed; SD 0.004040) |
| rung 3, A24 boxes | 0.861005 (three-seed; SD 0.004742) |
| measured localisation cost | **0.058456** |
| recovered | 0.046983 — **80.4 %** of the cost |
| residual to the ceiling | **0.011473** — **1.36 ×** the ceiling's own seed SD |

**Superseded by the above**, wherever they appear: ceiling 0.873114, seed SD 0.0113, cost 0.059092,
recovery 79.5 %, residual 0.012109 at 1.07 × SD. The affected passage is **§13c's "Prediction 2
sharpened"**, whose table and closing paragraph carry all six; §13c is applied at a later position than
this block and is corrected by it.

**0.0113 and 0.0084 are the same statistic on nested seed sets**, not two different statistics: the
sample standard deviation (ddof = 1) of exact validation AUC across seeds for `vgg16_crop_rung2_oracle`,
over {28, 1, 2} and over {28, 1, 2, 3, 4} respectively. Seed 28's 0.886110 is **+0.013632** above the
five-seed mean and is **not** an outlier at +1.62 SD, which is why §26 forbids quoting it alone.

**Withdrawn: "localisation has very nearly stopped being the binding constraint at single view"** (§13c).
It was written against a residual of 1.07 × the ceiling's seed SD and does not survive at **1.36 ×**. The
arithmetic that moves it is disclosed in full: **the ceiling falls** from 0.873114 to 0.872478, so the
residual falls with it from 0.012109 to 0.011473 — and the SD falls faster, from 0.011258 to 0.008405.
The ratio quoted by holding the old residual against the new SD, **1.44 ×**, pairs a three-seed numerator
with a five-seed denominator and is **not a coherent figure**; it is recorded here only so that it is not
arrived at independently.

> **Superseded by §32.** These figures were measured on the pre-`_cuda0917` ledger. §32 re-measures this quantity on the one-session ledger and carries the replacing value; the figures here stand as what was measured when this block was written.

**Replaces it as the load-bearing statement:** **80.4 % of the measured localisation cost is recovered,
and the residual is 1.36 × the ceiling's own seed standard deviation.** No claim is made that
localisation has ceased to bind, nor that the residual is inside seed noise. The ceiling remains the
noisier of the two conditions — §26's unexplained rung-2-versus-rung-3 seed-noise inversion stands at
**0.008405 against 0.004040**, a factor of **2.08** on five seeds against three, and every rung-2 claim
continues to require multi-seed backing.

## 12. §2 — Seed replicates and the seed ensemble of rung 3 (A5.1; descriptive)

*Why here: Restated under §11; the Metal values stay visible as superseded.*

Append to "## Ensemble, TTA and seeds":

> Two seed replicates of the primary model exist (`vgg16_crop_rung3_unet_bundle_s1`, `_s2`; exact
> validation AUC **0.8178 / 0.8098 against the primary's 0.8145, mean 0.8140, SD 0.0040 — CUDA,
> laptop-scored, the reference ledger per §11**; the superseded Metal values were 0.8128 / 0.8156
> against 0.8175, mean 0.8153, SD 0.0024). Each is evaluated
> once on test with the threshold fitted on
> its own validation predictions, and their mean-probability ensemble
> (`vgg16_crop_rung3_unet_bundle_ens3`, exact validation **0.8201 [0.7523, 0.8794] on CUDA**;
> superseded Metal value 0.8215; threshold fitted on the ensemble's
> validation probabilities) is
> reported with its patient-bootstrap CI. Two seed replicates of rung 5 (`vgg16_crop_rung5_jitter_bundle_s1`,
> `_s2`; exact validation 0.7535 / 0.7506 against the primary rung-5 run's 0.7383, mean 0.7475, SD 0.0081)
> are likewise evaluated once, so the rung-3-versus-rung-5 contrast carries a seed interval on both
> sides. The rung-3 minus rung-5 gap at the selected seed is **0.079** on exact validation AUC (0.061 on
> the Keras history metric). TTA is not run (negative on validation).

Runbook (after the rung loop, before `src.ladder`):

```bash
for s in 1 2; do
  python -m src.evaluate --model $ARCH --source crop --box-source unet --crop-margin $M --boxes-dir $BOXES --tag-suffix _rung3_unet_bundle_s$s
  python -m src.evaluate --model $ARCH --source crop --box-source jitter --crop-margin $M --boxes-dir $BOXES --tag-suffix _rung5_jitter_bundle_s$s
done
python -m src.ensemble --tags $PRIMARY ${PRIMARY}_s1 ${PRIMARY}_s2 --out-tag ${PRIMARY}_ens3 --splits val test --allow-test
```

## 13. §3 — Architecture family at the rung-3 condition (P4; descriptive)

*Why here: Descriptive rows; unaffected by the above but quoted in the same table.*

Append to "## Ensemble, TTA and seeds":

> The four C2 backbones are also fine-tuned once each on the promoted localiser's boxes at m = 0.20
> under the identical schedule (`{arch}_crop_rung3_unet_bundle`; VGG16 is the primary model itself),
> answering subsidiary question 5 under one fixed localiser. They and their four-backbone mean-probability
> ensemble (`rung3_arch_ens4`) are evaluated once on test and reported with patient-bootstrap CIs as
> descriptive rows beside rung 3: no Holm family (Family A remains the oracle-condition comparison), no
> re-selection, and no replacement of the primary model. Selection of these models is by the same
> validation rule as every other run (best `val_auc`, single seed).

Runbook (after the seed lines):

```bash
for a in resnet50 densenet121 efficientnet; do
  python -m src.evaluate --model $a --source crop --box-source unet --crop-margin $M --boxes-dir $BOXES --tag-suffix _rung3_unet_bundle
done
python -m src.ensemble --tags vgg16_crop_rung3_unet_bundle resnet50_crop_rung3_unet_bundle densenet121_crop_rung3_unet_bundle efficientnet_crop_rung3_unet_bundle --out-tag rung3_arch_ens4 --splits val test --allow-test
```

## 14. §13 — The declared view-fusion system does not replicate on correct hardware (decision A, 9 Sep)

*Why here: Depends on §10 (the k-seed rule it is judged under) and §11 (the ledger).*

The declaration **stands as made**. This document's fusion section says the declaration "is not
revisited", and it is not revisited here: the system that enters the pass is still
`vgg16_fusion_view_unet_bundle`, chosen under the pre-registered >= 0.04 rule on the evidence available
when the choice was made. What follows is a disclosed failure to replicate, reported beside it.

Append to "## Fusion (outside both families)":

> **The declaration rests on a Metal number that does not reproduce on CUDA.** View fusion was declared
> the original-model contribution because it beat its single-crop control by **+0.0413** on the
> development machine, clearing the pre-registered >= 0.04 rule. That machine has since been shown not
> to apply ReLU in the compiled graph (§9), which did not corrupt the prediction tables but did corrupt
> the training histories that selected each run's epoch, so a Metal run and a CUDA run of the same
> configuration are not replicates. Both conditions were therefore retrained on an RTX 4090 at the
> corrected default of patience 25, three seeds each, and scored on the laptop through the eager path:
>
> | | seed 28 | seed 1 | seed 2 | mean | SD | seed ensemble |
> | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
> | view | 0.894018 | 0.904456 | 0.904857 | **0.901110** | 0.006145 | **0.9057** [0.8428, 0.9589] |
> | control | 0.894821 | 0.884384 | 0.882377 | **0.887194** | 0.006681 | **0.8896** [0.8125, 0.9483] |
>
> **The gap is +0.0139 on three-seed means and +0.0161 on the seed ensembles, against the +0.0413 that
> justified the declaration.** It clears neither the pre-registered >= 0.04 rule nor the k-seed rule of
> §10, whose threshold here is max(0.01, 3 x pooled seed SD 0.00642) = 0.0193.
>
> **The mechanism is the control, not the view.** Against the Metal values, the view condition barely
> moved (0.8964 -> 0.9011, **+0.0047**) while its control gained **+0.0322** (0.8551 -> 0.8872). The
> declared advantage was largely the control being stopped early under a corrupted metric at patience
> 10, not the second view carrying information the first lacks.
>
> **What this does and does not establish.** It does not establish that two views are worthless: with
> 100 validation pairs the seed-ensemble intervals span roughly 0.12, so a difference of the size that
> remains is not resolvable by this design, and the honest statement is that the declared effect does
> not reproduce rather than that the effect is absent. Nor is the >= 0.90 reached by the view seed
> ensemble (0.9057) evidence for the claim: its control reaches 0.8896 on the same pairs, so the level
> reflects the pair subset being easier, not the fusion.
>
> **The open test.** Both conditions above use oracle boxes. Whether two views help at the
> **deployable** condition is a different question and is not answered here: it requires view fusion on
> the promoted localiser's single box per view, against rung 3 restricted to the same pairs
> (the like-for-like control of §1 above). That comparison is specified, is not yet run, and is what
> the report should point to as the live question.

**Order of events, disclosed.** The Metal declaration was made first. The tensorflow-metal defect was
found afterwards, the CUDA re-baseline was run because of it, and the three-seed replication was run
specifically to test whether the non-replication was itself a single-seed artefact. It was not. Both
readings appear wherever the fusion result is quoted.

## 15. §13a — The like-for-like control for the deployable fusion test (computed before the fusion run)

*Why here: The control §13's "open test" points at; recorded before the run it judges.*

§13 leaves open whether two views help at the **deployable** condition. Half of that comparison is the
control, and it is computed here **before** the fusion run it will judge, so the bar is set in advance.

`notebooks/scripts/rung3_on_pairs.py`, CUDA rung 3 restricted to the 100 validation pairs the fusion
store defines, scored per pair, patient-cluster bootstrap CIs:

| control | AUC [95 % CI] | what it answers |
| :--- | :--- | :--- |
| `view_a_only` | **0.8406** [0.7472, 0.9127] | one view against two — the declaration's question |
| `mean_a_b` | **0.8739** [0.7941, 0.9382] | learned fusion against naive two-view averaging |
| rung 3, all 243 lesions | 0.8145 | context only; different rows, not a control |

Three consequences, recorded now:

1. **The pair subset is easier than the full validation set** — 0.8406 against 0.8145 — so any fusion
   number computed on pairs and compared with rung 3's headline is flattered by roughly 0.026 before
   any modelling. Every fusion comparison in the report uses a control computed on the same pairs.
2. **Two views help by averaging alone.** Taking the mean of rung 3's two per-view probabilities
   reaches 0.8739, +0.033 over a single view, with no fusion model involved.
3. **The bar for the declared fusion system is therefore 0.8739, not 0.8406.** A learned fusion model
   that scores near 0.87 would beat the one-view control while doing nothing that a two-line averaging
   rule does not already do, and would not support a claim of an original modelling contribution. This
   bar is fixed here, before `vgg16_fusion_view_unet` is trained.

All 100 pairs carry the same label on both views, so the pair label is a property of the breast rather
than of crop A.

## 16. §13b, §13c — Where a learned combiner could beat averaging (hypothesis recorded before running)

*Why here: Reads §13a's sign reversal; must follow it. §13c shares this position because it is read entirely through §13b's mechanism: it predicts what a lower box-failure rate should do to the stratified advantage §13b measured. Written after configuration A's gate and before the cascade, the rung-3 retrain and the fusion re-run; §24a's A24 was not yet computed. It gates nothing.*

### §13b — Where a learned combiner could beat averaging (hypothesis recorded before running)

§13a finds no evidence that learned view fusion beats naive two-view averaging, and finds the sign of
the difference reversing between conditions (+0.018 on U-Net boxes, −0.034 on oracle boxes). If that
reversal is not simply noise, there is a mechanism that would produce it, and it is testable.

> **Hypothesis.** Averaging weights both views equally and cannot do otherwise. A learned combiner can
> down-weight a view whose crop missed the lesion. Its advantage should therefore **concentrate in
> pairs where at least one view's box failed**, and be absent or negative where both boxes are good —
> which is also why averaging should look near-optimal at the oracle condition, where by construction
> both boxes are correct and there is nothing to down-weight.
>
> **Split, fixed before running.** The 100 validation pairs are divided by the promoted ensemble's
> per-lesion box IoU on **both** views: `both_ok` (both views have IoU > 0) against `any_failed` (at
> least one view is a miss-inclusive failure, IoU = 0, fallbacks included). The same definition of
> failure as §12c and the localiser gate.
>
> **Measured, descriptively.** The exact AUC of fusion `ens3` and of naive averaging within each
> stratum, and their difference, on the U-Net-box pairs. Reported with stratum sizes and malignant
> counts.
>
> **What would support the hypothesis.** The fusion-minus-averaging difference is positive and larger
> in `any_failed` than in `both_ok`. **What would refute it.** The difference is flat across strata, or
> larger where both boxes are good — in which case the sign reversal in §13a is unexplained and is
> reported as noise.
>
> **The `any_failed` stratum will be small.** Its size is stated with the result, no inference is drawn
> from a stratum too small to support one, and no confidence interval is presented as though it
> resolved anything. This enters no family, corrects nothing and selects nothing.

### 13a result — the deployable paired system, measured

All numbers below are on the **same 100 validation pairs** (91 patients, 47 malignant), paired at the
pair level. Scripts: `notebooks/scripts/rung3_on_pairs.py`, `notebooks/scripts/fusion_pair_comparisons.py`.
Result files: `fusion_pair_comparisons_val_cuda.json`, `fusion_stratified_by_box_outcome_val.json`.

| system, on the 100 pairs | exact val AUC |
| :--- | ---: |
| one view, rung 3 (U-Net boxes) | 0.8406 |
| naive two-view averaging (U-Net boxes) | 0.8739 |
| **learned fusion `ens3` (U-Net boxes) — the deployable system** | **0.8920** |
| learned fusion `ens3` (oracle boxes) | 0.9057 |

**The two pre-specified tests, Holm-corrected across the two. Neither is significant.**

| test | difference | 95 % CI | DeLong p | Holm p |
| :--- | ---: | :--- | ---: | ---: |
| (i) learned fusion vs naive averaging | +0.0181 | [−0.0205, +0.0559] | 0.344 | 0.365 |
| (ii) learned fusion vs one view | +0.0514 | [−0.0219, +0.1294] | 0.183 | 0.365 |

Descriptive, outside the family: oracle-box fusion minus U-Net-box fusion is **+0.0136**
[−0.0458, +0.0823] — the localiser's residual cost to the paired system.

**Point 1 — the second view absorbs most of the localiser's cost.** Measured like for like, on
identical rows, with rung 2 restricted to the same pairs:

| | oracle boxes | U-Net boxes | localiser cost |
| :--- | ---: | ---: | ---: |
| one view | 0.9153 | 0.8406 | **+0.0747** |
| two views (`ens3`) | 0.9057 | 0.8920 | **+0.0136** |

The cost falls by **+0.0610, 82 % of the one-view cost**, CI [−0.0186, +0.1426], with 93.8 % of
bootstrap draws positive. Directionally strong; not resolvable at this n. This is the paired system's
argument: not that it is more accurate in the abstract, but that it is far less damaged by the
localiser, which is the component this project cannot fix.

**Point 2 — there is no evidence the learned combiner beats averaging, and one condition where it is
worse.** The increment is **+0.0181 on U-Net boxes and −0.0337 on oracle boxes** (0.9057 against
averaging's 0.9394). A quantity that reverses sign between conditions at similar magnitude is not an
under-powered positive effect; it is what noise looks like. The report must not claim learned fusion as
an original modelling contribution on this evidence. Most of the two-view gain is available from
averaging two per-view scores, which is a two-line rule.

**Point 3 — the caveats, stated wherever these numbers appear.** The pair subset is easier than the
full validation set: the same model scores **0.8406 on the 100 pairs against 0.8145 on all 243
lesions**, a difference of **0.026**, so no pair-level number may be compared with a lesion-level one.
And at 100 pairs and 91 patients nothing here resolves: every interval above spans zero, including the
one for the effect the design was built to detect.

**Point 4 (§13b) — where the learned combiner does earn its place.** Split by whether both views' boxes
survived, on the U-Net pairs:

| stratum | n | malignant | patients | fusion | averaging | difference | 95 % CI | draws > 0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| both boxes ok | 59 | 33 | 59 | 0.9126 | 0.9068 | **+0.0058** | [−0.0399, +0.0528] | 0.601 |
| at least one box failed | 41 | 14 | 35 | 0.8439 | 0.7804 | **+0.0635** | [−0.0349, +0.1708] | 0.890 |

**The §13b hypothesis is supported.** The learned combiner's advantage is roughly eleven times larger
where a box failed than where both boxes are good, and where both are good it is +0.006 — nothing.
That is the mechanism the hypothesis named: a learned combiner can down-weight a view whose crop
missed the lesion; averaging cannot. It also explains the oracle-box sign reversal without appeal to
noise alone — with both boxes correct by construction there is nothing to down-weight, averaging is
near-optimal there, and the −0.034 is variation around zero.

**Stated with its uncertainty, as §13b required.** The `any_failed` stratum is 41 pairs with 14
malignant across 35 patients; its interval spans zero and 89 % of bootstrap draws are positive. This is
a direction, not a demonstration. It is reported because it is the only account so far that predicts
the sign reversal in advance rather than explaining it after the fact, and because **41 % of pairs have
at least one failed box**, so the stratum where the combiner helps is not a rare corner of the data —
it is where a deployed system would spend much of its time.

### §13c — Two predictions for the cascade and the fusion re-run, recorded before either runs

**Order of events.** Written **11 Sep**, after configuration A's gate against the promoted ensemble
(+0.0515 [+0.0266, +0.0769], PASS) and **before** the promotion cascade, the rung-3 retrain and the
fusion re-run. §24a's group-weighted A24 had **not** been computed when these were written, so the
promotion candidate was not yet known. Both predictions are the author's, stated so that the
cascade's result reads as a test of them rather than as a description written afterwards.

#### Prediction 1 — rung 3 has real room to move

**The reasoning.** +0.0515 is not a marginal box-IoU gain when set against the only comparable
improvement this project has measured:

| step | val box IoU | gain |
| :--- | ---: | ---: |
| run 1 (frozen reference) | 0.3853 | — |
| promoted three-run ensemble | 0.4700 | **+0.0847** |
| configuration A, seven members | 0.5216 | **+0.0515** |

The seven-member gain is **61 % as large as the entire improvement** that promoting the three-run
ensemble bought over run 1, and it is applied on top of it. Against the ladder, the localisation cost is quoted from the **CUDA multi-seed** figures, for
consistency with §26's ruling and because the single-seed Metal pairing that gave 0.048 is not the
reference set:

| route | seed 28 | seed 1 | seed 2 | mean | SD |
| :--- | ---: | ---: | ---: | ---: | ---: |
| rung 2, oracle ceiling | 0.886110 | 0.866889 | 0.866344 | **0.873114** | 0.0113 |
| rung 3, promoted boxes | 0.814477 | 0.817816 | 0.809774 | **0.814022** | 0.0040 |

**The measured localisation cost is 0.059 on three-seed means** (0.873114 − 0.814022 = 0.059092),
with the **seed-28 contrast of +0.072** (0.886110 − 0.814477 = 0.071633) quoted beside it as the
pre-registered single-seed test. A gain of the size configuration A shows plausibly recovers a
visible share of that 0.059. Note the ceiling's own three-seed SD is 0.0113, nearly three times rung
3's 0.0040, so the gap's upper end is the noisier of the two.

**The prediction.** The rung-3 retrain on the promoted boxes will move **upward and measurably** —
beyond the ±0.008 localiser seed scale of §20a.2 and the rung-3 classifier seed SD of 0.0037.

**What would refute it.** Rung 3 retrained on the new boxes lands within seed noise of 0.8145/0.8166,
or falls. That would mean box IoU and downstream AUC have decoupled at this end of the range — which
is itself a reportable result, and one the ladder is built to detect.

**The floor still governs.** §7b's promotion condition is unchanged: a rung-3 retrain within 0.02 of
0.8145 is required for promotion. A prediction of improvement does not relax it.

> **Floor restated by §35:** one-sided, against the three-seed mean 0.814022 (floor 0.794022), not seed 28's 0.8145.

#### Prediction 2 — the fusion combiner's advantage should shrink

**The mechanism is already on record.** §13b predicted, before running, that a learned combiner's
advantage over naive averaging would concentrate in pairs where at least one view's box failed. The
measurement supports it (`outputs/results/fusion_stratified_by_box_outcome_val.json`):

| stratum | n | malignant | fusion | averaging | difference | 95 % CI |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `both_ok` | 59 | 33 | 0.9126 | 0.9068 | **+0.0058** | [−0.0399, +0.0528] |
| `any_failed` | 41 | 14 | 0.8439 | 0.7804 | **+0.0635** | [−0.0349, +0.1708] |

The advantage is **eleven times larger** where a box failed. Neither interval excludes zero — §13b
declared in advance that the `any_failed` stratum would be small and that no inference would rest on
it — but the direction and the ratio are as predicted.

**What the new ensemble does to that stratum, measured now.** Comparing the promoted ensemble's
validation box table against configuration A's, on the same 243 lesions:

| | promoted | configuration A |
| :--- | ---: | ---: |
| failures (box IoU = 0) | **68** (28.0 %) | **55** (22.6 %) |
| rescued (0 → > 0) | — | **20** |
| lost (> 0 → 0) | — | **7** |
| failed under both | — | 48 |

**A net 13 fewer failures, a 19 % relative reduction in the failure rate.**

**The prediction.** If the combiner's advantage lives in failed-box pairs, and the ensemble that
enters the fusion re-run fails on fewer of them, then **the combiner's advantage over naive averaging
should shrink** — the `any_failed` stratum gets smaller and easier, and the stratum where the
combiner has nothing to down-weight grows. The declared view-fusion system's already non-replicating
margin (§13, +0.0139 on three seeds against the ≥ 0.04 rule) should therefore **narrow further, not
widen**.

**What would refute it.** The fusion-minus-averaging difference holds or grows on the new boxes, or
the `any_failed` advantage stays at roughly +0.06 on a smaller stratum. Either would mean the
combiner is doing something other than down-weighting failed crops, and §13b's mechanism would need
re-examining rather than confirming.

**The consequence, stated in advance so that it cannot read as a rescue.** If the advantage narrows
as predicted, the reading is that **better localisation and view fusion are partly substitutes**:
both act on the same failed-box pairs, and a pair whose crop the ensemble has already rescued is a
pair the combiner no longer has anything to down-weight. It follows that **the deployable paired
system may gain less from the new ensemble than the +0.0515 box-IoU improvement suggests** — the two
interventions overlap rather than compounding.

That is a **coherent finding, not a disappointment**, and it is written here before the run precisely
so that it cannot be produced afterwards as an explanation for a smaller-than-hoped fusion number.
Two mechanisms competing for the same failures is a result about where the remaining headroom is: it
says the pairs that are still hard after the ensemble improves are hard for reasons neither a better
box nor a learned combiner addresses, which is a more useful statement for the write-up than either
intervention's isolated effect size.

**What this does not license.** It is not a reason to prefer the smaller number, nor to stop
reporting the fusion result. Both systems are measured and both are reported; the substitution
reading explains a narrowing if one occurs and is withdrawn if one does not.

#### Prediction 2 sharpened — localisation has nearly stopped being the binding constraint

**Recorded before the fusion session runs**, as a consequence of the rung-3 result that is now
measured and the fusion result that is not.

**The rung-3 retrain on A24's boxes landed at 0.861005** (three-seed exact mean; base 0.866412, s1
0.859051, s2 0.857552). Against the oracle ceiling and the old boxes:

| | value |
| :--- | ---: |
| rung 2, oracle ceiling (three-seed CUDA mean) | 0.873114 (seed SD 0.0113) |
| rung 3, promoted three-run boxes | 0.814022 (seed SD 0.0040) |
| **rung 3, A24 boxes** | **0.861005** |
| measured localisation cost | 0.059092 |
| **recovered** | **0.046983 — 79.5 % of the cost** |
| residual to the ceiling | 0.012109 — **1.07 ×** the ceiling's own seed SD (0.0113) |

**All three come from the same code path.** Every one is `train_crop_store` on CUDA from an exported
store at margin 0.20, percentile normalisation, no mask overlay — rung 2 from
`crop_store_rung2_oracle`, old rung 3 from `crop_store_rung3_unet_bundle`, new rung 3 from
`crop_store_rung3_unet_promoted`. The only variable is which boxes made the crops, which is what a
controlled comparison requires and what §14 and §26 were about when the routes differed.

**The consequence, in the exact figures, which are the ones to cite.** **79.5 %** of the
**0.059092** localisation cost is recovered. The residual, **0.012109**, is **comparable to the
ceiling's own seed noise at 1.07 × its SD of 0.0113 — comparable to, not inside it**; the ceiling is
the noisier of the two conditions by nearly three times. **Localisation has very nearly stopped
being the binding constraint at single view.**

Wherever this is cited: 0.861005, 0.873114, 0.814022, cost 0.059092, recovery 79.5 %, residual
0.012109 at 1.07 × SD. Not "about 80 %", not "inside the seed SD" — the residual and the SD are
within 7 % of each other and the rounding decides the sentence.

**What that predicts for the fusion session.** If the single-view system is close to its oracle
ceiling, the remaining headroom in the paired system is whatever fusion buys on **well-localised**
pairs — and §13b measured that directly at **+0.0058** in the `both_ok` stratum, against +0.0635 in
`any_failed`. So:

> **A fusion result near 0.90 on A24's boxes would come mostly from the boxes, not from the
> combiner.**

This sharpens prediction 2 rather than replacing it. Prediction 2 said the combiner's advantage
should narrow because the stratum it lives in shrinks (68 → 54 failures). This says where the
paired system's level will come from if it rises: the boxes. The two are separable in the result —
the combiner's advantage is fusion minus naive averaging **within** a stratum, while the level is
the absolute AUC — and the report must not read a higher paired AUC as evidence for the combiner.

**What would refute it.** The fusion-minus-averaging difference on A24's boxes holds near +0.06 in
`any_failed` *and* the `both_ok` difference grows materially above +0.006. That would mean the
combiner is contributing something the box quality does not explain, and the substitution reading
above is wrong.

**The oracle ceiling is not re-measured, and the reason is recorded.** A24 changes the localiser's
boxes. Rung 2's crops come from the ground-truth masks at the same margin, which A24 does not touch,
and its runs share the trainer, device and store pipeline with both rung-3 measurements. Re-running
it would resample seed noise on an unchanged condition and invite the ceiling to drift for reasons
that have nothing to do with the localiser.

#### 13c RESULT — both predictions tested, and both land

**Order of events.** Predictions 1 and 2 and the sharpening were recorded before the rung-3 retrain
and before the fusion session. Everything below was measured afterwards, on A24's boxes, and is
reported against what was written in advance.

**Prediction 1 — rung 3 has real room to move. LANDS.** Three-seed exact mean **0.861005** against
0.814022; **79.5 %** of the 0.059092 localisation cost recovered; residual 0.012109, **1.07 ×** the
ceiling's own seed SD of 0.0113. Refutation was "within seed noise or falls"; the move is an order
of magnitude larger than the threshold that would have refuted it.

> **Superseded by §32.** These figures were measured on the pre-`_cuda0917` ledger. §32 re-measures this quantity on the one-session ledger and carries the replacing value; the figures here stand as what was measured when this block was written.

**Prediction 2 — the combiner's advantage should narrow. LANDS, and reverses.**

| 100 validation pairs | old boxes | **A24 boxes** |
| :--- | ---: | ---: |
| one view, rung 3 | 0.8406 | **0.9073** |
| naive two-view averaging | 0.8739 | **0.9342** |
| learned view fusion (ens3) | 0.8920 | **0.9261** |
| **fusion − averaging** | **+0.0181** | **−0.0080** [−0.0354, +0.0169] |
| fusion − one view | +0.0514 | +0.0189 [−0.0282, +0.0666] |

DeLong p 0.5284, Holm 0.8514 — not significant, which is the point: on well-localised pairs there
is nothing for a combiner to do.

**The stratified test — §13b's declared mechanism.** Split as §13b fixed it: both views' box IoU > 0
(`both_ok`) against at least one miss-inclusive failure (`any_failed`), computed by
`notebooks/scripts/fusion_stratify_by_box.py`, written for this because no script produced the
earlier ad-hoc JSON.

| stratum | n | fusion | averaging | difference | old boxes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `both_ok` | 63 | 0.9605 | 0.9574 | **+0.0031** [−0.0241, +0.0322] | +0.0058 |
| `any_failed` | 37 | 0.8148 | 0.8519 | **−0.0370** [−0.1305, +0.0338] | **+0.0635** |

The eleven-fold concentration in `any_failed` **is gone and has inverted**, and the stratum shrank
from 41 pairs to 37 as the 68 → 54 lesion-failure count predicted.

**The sharpening — the level comes from the boxes, not the combiner. LANDS.** One view alone on
A24's boxes is **0.9073**, higher than the *fused* system on the old boxes (0.8920). The entire rise
in the paired system is attributable to localisation. §13c's warning applies exactly: **a higher
paired AUC is not evidence for the combiner.**

#### The primary endpoint's condition, recomputed on A24's boxes

The endpoint is **image-level on the BI-RADS comparison set**, not the 100 pairs.

| | image-level AUC, 209 validation images | vs anchor | p |
| :--- | ---: | ---: | ---: |
| BI-RADS anchor (validation) | 0.8627 | — | — |
| rung 3, old boxes | 0.8084 | −0.0543 [−0.1277, +0.0117] | 0.093 |
| **rung 3, A24 boxes** | **0.8589** | **−0.0038** [−0.0590, +0.0557] | **0.8919** |

**+0.0505 at image level.** The system has gone from trailing the radiologists' recorded assessment
by a margin that was approaching significance, to **statistically indistinguishable from it**. That
is not "beating the anchor" — p 0.89 means the data cannot separate them — and it must not be
written as one.

**Which number is which.** 0.90 is cleared **only on the 100-pair subset**, which §13 already
records as an easier row set. The three conditions are different and are quoted separately:

| condition | rows | A24 boxes |
| :--- | ---: | ---: |
| **primary endpoint (image level)** | 209 images | **0.8589** |
| rung 3, lesion level | 243 lesions | 0.861005 |
| paired subset, one view | 100 pairs | 0.9073 |

**These are validation numbers, and they are optimistic.** The localiser was selected on this set,
so the boxes are fitted to the rows they are scored on. The anchor gives the scale of the
partition's own difficulty: BI-RADS falls **0.8627 → 0.8090** from validation to test, **−0.0537**.
A model dropping the same amount lands near **0.805** — and the model carries an additional source
of optimism the anchor does not, because the anchor was never selected on anything. **No claim about
test performance follows from any number above.**

#### The oracle comparison: suggestive, and what it limits

Descriptively, A24-box fusion reaches **0.9261** against oracle-box fusion's **0.9057**, a gap of
−0.0205 [−0.0739, +0.0401]. **This is cross-session** — the oracle-box runs are from the earlier
CUDA re-baseline, the A24 runs from 11 September — and is **not quoted as a measurement**.

**But it raises a limitation that is recorded rather than measured away.** The rung-2 oracle ceiling,
0.873114, was measured on the old crop path. If A24's boxes match or beat oracle boxes on the paired
subset, then **that ceiling is not a ceiling for the current system**: a "ceiling" derived from one
box condition bounds only systems whose boxes are worse than it. The residual of 0.012109 is
therefore a distance to a *reference*, not to a limit, and the report must not describe rung 2 as an
upper bound on what localisation can achieve. **It is not re-measured** — see the reason recorded
above — and the limitation stands in its place.

**Neither prediction gates anything.** No promotion, no selection and no test-pass decision depends on
either. They are recorded so that what the cascade shows can be read against what was expected of it.

## 17. §1 — Fusion re-run condition (D2; why "matched k=3" was replaced)

*Why here: A note: records why the fusion paragraph's re-run wording changed on 9 September. The protocol already carries the change, so there is nothing to replace.*

**A note, not an instruction.** This block was first written as an instruction to replace "(selected
ensemble, matched k=3)" in "## Fusion (outside both families)". That wording was changed in the
protocol itself on 9 September, when the fusion re-run was specified, and the section has read "(selected
ensemble, single box, the rung-3 condition, so the fusion result is like-for-like with rung 3)" since.
There is nothing left to replace; the block stays to record why the wording changed. Matched top-k
selection picks the candidate by IoU with the ground truth and is therefore an upper bound, not a
deployable rule, so the declared view-fusion system is re-trained on the single box per view, the rung-3
condition. The re-run is `vgg16_fusion_view_unet_bundle`, measured as §28 names it.

Runbook (uncomment the six fusion placeholder lines and add, after them):

```bash
python -m src.fusion --arch $ARCH --pair view --margin $M --box-source unet --boxes-dir $BOXES --tag-suffix _bundle --stage eval
```

Tags table row: `vgg16_fusion_view_unet_bundle | declared view-fusion system at the rung-3 condition, reported beside rung 3 descriptively`.

## 18. §12, §12a, §12b, §12c — Localiser uncertainty as a predictor of localisation failure (work plan §12; specified before running)

*Why here: Self-contained and descriptive; §12c withdraws a claim in §12b, so keep them adjacent and in order.*

### §12 — Localiser uncertainty as a predictor of localisation failure (work plan §12; specified before running)

Append to "## Also run on test, once, descriptively" — or, if the analysis is validation-only at the
freeze, to "## Ensemble, TTA and seeds (A5; descriptive)":

> **Localiser disagreement is tested as a predictor of localisation failure, on validation only, and
> reported descriptively.** The promoted localiser is an ensemble of three U-Net runs whose probability
> maps are averaged before a single box is chosen. Where the members disagree, the averaged map is a
> compromise between competing hypotheses, and the resulting box may be wrong; the question is whether
> that disagreement is measurable in advance of knowing the answer, because a system that can say which
> of its own localisations to distrust is more useful clinically than one that cannot.
>
> **Score (fixed before the analysis was run).** Two disagreement measures per lesion, both computed
> from the same cached member maps the ensemble is built from, with the promoted configuration's own
> threshold, minimum component size and breast filter:
> 1. **Box disagreement** — one minus the mean pairwise box IoU of the three members' individually
>    selected boxes. A member that produces no box contributes IoU 0 to its pairs, so a member failing
>    to fire counts as disagreement rather than being dropped.
> 2. **Map disagreement** — the mean per-pixel standard deviation across the three members'
>    probability maps, taken inside the ensemble's own selected box (or, where the ensemble produces no
>    box, inside the breast box).
>
> **Outcome (fixed before the analysis was run).** The binary event is a **miss-inclusive
> localisation failure**: the promoted ensemble's per-lesion box IoU with the ground truth is 0,
> which includes the fallback cases where no box was produced. This is the same quantity, scored the
> same way, as the localiser gate.
>
> **Statistic.** AUROC of each disagreement score against that outcome over the 243 validation
> lesions, with a patient-cluster bootstrap 95 % CI (1,000 draws, 129 patients), reported beside the
> rate of the outcome itself. Both measures are reported whatever they show, and a null result is
> reported as a null result.
>
> **Status: descriptive.** It enters no family, corrects no hypothesis, is Holm-corrected against
> nothing, and changes no model, threshold or selection. It is computed on validation and is not run on
> test unless the freeze explicitly adds it.
>
> **What it replaces.** The pre-registration's Monte Carlo Dropout deferral analysis is at chance at
> rung 3, so it carries no information about which cases to defer. This analysis asks the same clinical
> question — can the system flag its own unreliable cases — of the component that actually fails, the
> localiser, rather than of the classifier.

**Order of events, disclosed.** The metric, the outcome and the statistic above were written into this
document **before the analysis was run**, precisely because it is the kind of analysis where the
choice of score can be tuned to the result. The cached member maps existed at the time of writing; no
disagreement score had been computed against any outcome.

### §12a — Deferral curve on the localiser-disagreement score (specified before running)

Append to §12 above:

> **The disagreement score is also read as a deferral rule, descriptively.** An AUROC says the score
> ranks failures above successes; it does not say whether acting on the ranking would help. The
> deferral curve asks the operational question directly: if the system referred its most uncertain
> localisations to a radiologist instead of reporting them, how much would what remains improve, and at
> what referral cost?
>
> **Procedure, fixed before running.** Validation lesions are ranked by **box disagreement** — the
> measure §12 pre-specified, and the only one of the two that proved informative there. For each
> deferral rate d from 0 to 0.5 in steps of 0.05, the d most-disagreeing lesions are removed and three
> quantities are recomputed on the retained set: (i) the miss-inclusive localisation failure rate,
> (ii) the mean per-lesion box IoU, and (iii) the exact AUC of the promoted rung-3 classifier on the
> retained lesions, read from its CUDA prediction table. Each is reported against a **random-deferral
> baseline**, the mean over 200 random subsets of the same size, so the curve shows the gain
> attributable to the ranking rather than to shrinking the set.
>
> **Reading rule.** The curve is descriptive and enters no family. A deferral rule is worth reporting
> as useful only if the retained-set curve lies outside the random-deferral band; a curve inside the
> band is reported as a null result. No deferral rate is selected, and nothing downstream is
> conditioned on this analysis.

**Order of events, disclosed.** Written before the curve was computed. The §12 AUROC result (box
disagreement 0.692 [0.565, 0.791]; map disagreement 0.570 [0.490, 0.669]) was known, and is the reason
only the box measure is carried forward here.

### §12b — Why the deferral curve is null on classification (hypothesis recorded before running)

Append to §12a:

> **The classifier arm of the deferral curve is null, and there is a mechanism that would explain it.**
> Deferring on localiser disagreement improves the retained set's localisation — failure rate and mean
> box IoU both leave the random band — but leaves the rung-3 classifier's AUC inside it at every
> deferral rate. The following explanation is **recorded before the analysis that tests it was run**:
>
> **Hypothesis.** A bad box produces a crop containing little or no lesion, and the classifier scores
> such a crop low. A low score on a **benign** lesion is the correct answer, and a low score on a
> **malignant** lesion is the wrong one. Localiser failure therefore costs the classifier only on
> malignant lesions, while on benign lesions it is silently rewarded. Rung 3's AUC is in that sense
> partly insensitive to localisation failure: the two error directions partly cancel inside the
> ranking, so removing the badly localised cases removes wrong malignant calls and accidentally
> correct benign ones together, and the AUC barely moves.
>
> **What is measured, on validation, descriptively.** At the 25 % deferral point of §12a: the label
> composition and exact classifier AUC of the deferred and retained sets, and the mean classifier score
> of deferred benign against deferred malignant lesions.
>
> **What would support the hypothesis.** Deferred benign and deferred malignant lesions carry similar,
> low mean scores, so the deferred set separates poorly and its AUC is near or below chance, while the
> retained set's AUC is not correspondingly higher. **What would refute it.** The deferred set
> separates as well as the retained one, or deferred malignant lesions score no lower than deferred
> benign ones — in which case the null is not explained by this mechanism and is reported as
> unexplained.
>
> The result is added to the §12 findings **either way**. It selects nothing, changes no model and
> enters no family.

#### 12b result — the hypothesis is refuted, and the null has a different cause

Run after §12b was recorded, on validation, at the 25 % deferral point:

| set | n | malignant | prevalence | AUC [95 % CI] | mean p given benign | mean p given malignant | score gap | localisation failure rate |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| deferred | 61 | 20 | 0.328 | **0.8220** [0.6712, 0.9309] | 0.3511 | 0.7094 | 0.3582 | 0.541 |
| retained | 182 | 92 | 0.505 | **0.7982** [0.7224, 0.8659] | 0.4717 | 0.7583 | 0.2866 | 0.192 |
| full set | 243 | 112 | 0.461 | 0.8145 | | | | |

**Half of the hypothesis holds and the conclusion does not.** Badly localised crops do score lower:
the deferred set's mean probability is lower in both classes (benign 0.351 against 0.472, malignant
0.709 against 0.758). But the two error directions do **not** cancel inside the ranking. The deferred
set separates *better* than the retained one, not worse — AUC 0.8220 against 0.7982, with a wider
score gap (0.358 against 0.287). The predicted signature of the mechanism, benign and malignant alike
scoring low so that the deferred set collapses toward chance, does not appear.

**The real reason the classifier arm is null.** The deferred set is the one where localisation fails
(failure rate 0.541 against 0.192, which is §12a working as intended) and is at the same time the set
the classifier handles *better*. Removing well-classified lesions cannot improve the AUC of what
remains, so no deferral rate helps. Two features of the deferred set explain it: it is
disproportionately **benign** (prevalence 0.328 against the retained set's 0.505), and its benign
lesions are scored confidently low, which separates them cleanly from its twenty malignant ones.

**What this says about the pipeline, and it is the point worth reporting.** Localiser-uncertain
lesions are not classifier-hard lesions. The two stages fail on largely different cases, which is why
an uncertainty signal that is genuinely informative about localisation (§12, AUROC 0.692) carries
none about diagnosis. A deferral rule built on the localiser would refer the wrong cases: it would
send the radiologist a set the classifier was already handling well.

**Stated with its uncertainty.** The deferred set is 61 lesions with 20 malignant, so its interval
[0.671, 0.931] overlaps the retained set's [0.722, 0.866] substantially. The claim supported here is
the negative one — that the deferred set is *not* worse separated, which is what the hypothesis
required — and not that it is reliably better.

### §12c — The same question on the ground-truth split (recorded before running)

§12b tested the mechanism through the **disagreement score**, which is only a 0.692-AUROC proxy for
localisation failure, so a null there could be the proxy's weakness rather than the mechanism's
absence. This repeats it on the outcome itself.

> **Split.** The 243 validation lesions are divided by the promoted ensemble's actual per-lesion box
> IoU with the ground truth: **68 miss-inclusive failures** (IoU = 0, fallbacks included) against
> **175 successes**. This is the ground-truth split, not the disagreement proxy.
>
> **Measured, descriptively.** Rung 3's exact AUC on each side with a patient-cluster bootstrap 95 %
> CI, and the mean classifier score within each label on each side.
>
> **Decision rule, fixed before running.**
> - If the **failure-set AUC is near chance** and both classes are scored low there, the benign half of
>   the failures is being counted correct for the wrong reason — a badly cropped benign lesion gets a
>   low score, which the metric rewards — and rung 3's headline AUC is partly insensitive to
>   localisation failure in exactly the way §12b hypothesised.
> - If the **failure-set AUC is around 0.8**, the hypothesis is refuted outright: the classifier
>   discriminates just as well on lesions the localiser missed, and localisation quality and
>   classification difficulty are close to independent.
>
> Reported either way, as part of the §12 findings. Selects nothing, enters no family.

#### 12c result — neither branch fires; the mechanism is real but partial, and §12b's reading was too strong

| set | n | malignant | prevalence | AUC [95 % CI] | mean p given benign | mean p given malignant | score gap |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| localisation failures | 68 | 22 | 0.324 | **0.7223** [0.5512, 0.8512] | 0.3840 | 0.5985 | 0.2145 |
| localisation successes | 175 | 90 | 0.514 | **0.8289** [0.7621, 0.8912] | 0.4609 | 0.7865 | 0.3255 |
| all | 243 | 112 | 0.461 | 0.8145 [0.7505, 0.8740] | 0.4339 | 0.7495 | 0.3156 |

**Neither pre-registered branch fires.** The failure-set AUC is 0.7223: not near chance (its interval
excludes 0.5) and not around 0.8. The rule was written as a dichotomy and the data are in between, so
both branches are reported as not met and the result is described as it is.

**The mechanism is present, and asymmetric as predicted.** Localisation failure costs rung 3 about
**0.107 of AUC** (0.8289 -> 0.7223) and compresses the score gap from 0.326 to 0.215. Both classes
score lower when the box is wrong, but **malignant lesions lose far more than benign ones** (−0.188
against −0.077), which is exactly the asymmetry §12b predicted: a badly cropped benign lesion still
gets the low score the metric wants, while a badly cropped malignant lesion loses the evidence that
would have earned a high one. What does not happen is the collapse to chance — roughly 0.72 of signal
survives, presumably because a wrong box often still contains peritumoral tissue or overlaps the lesion
partially.

**This narrows the conclusion drawn in §12b, and that sentence should be read with this one.** §12b
concluded from the disagreement-ranked split that "localiser-uncertain lesions are not classifier-hard
lesions" and that the two stages "fail on largely different cases". The first clause stands and the
second is too strong. Measured on the ground truth rather than the proxy, the lesions the localiser
actually fails on **are** harder for the classifier, by 0.107 of AUC. The two statements are
compatible because the disagreement score is only a 0.692-AUROC predictor of failure: the set it
defers is a mixture, roughly half failures and half successes, and that mixture happens to be well
classified. So the correct statement is:

> Localisation failure does hurt classification, asymmetrically and by about 0.107 of AUC, but the
> ensemble's disagreement score is too weak a proxy for failure to act on: deferring by it removes a
> set the classifier was handling well. The deferral rule fails not because the underlying quantity is
> irrelevant but because the available signal for it is not sharp enough.

**Stated with its uncertainty.** The failure set is 68 lesions with 22 malignant and its interval
[0.551, 0.851] overlaps the success set's [0.762, 0.891]. The direction is consistent across every
measure reported here — lower AUC, narrower gap, larger drop on malignant — but the difference is not
individually resolvable at this n.

## 19. §15, §15a — The component re-ranker (work plan §6; bound and rules recorded before training)

*Why here: Self-contained. §15a is a specification change made after a result and says so.*

### §15 — The component re-ranker (work plan §6; bound and rules recorded before training)

The promoted localiser picks one component per image by a fixed rule (peak probability, threshold 0.15,
minimum 256 px). It often picks the wrong one. The re-ranker learns the choice instead. Everything
below is fixed before any re-ranker is trained.

> **The ceiling is measured and is the pre-registered bound.** On validation the current rule selects a
> box reaching IoU ≥ 0.3 with a ground-truth lesion on **165 of 225 images (73.3 %)**, and at least one
> surviving candidate does so on **186 of 225 (82.7 %)**. **A perfect re-ranker therefore gains at most
> 21 images, 9.3 %**; on train the equivalent bound is 11.7 % (930 → 1064 of 1146). Seven validation
> images produce no candidate at all and are unreachable by any selection rule. These numbers were
> computed from `candidates_{val,train}.csv` before the re-ranker existed, and any reported gain is read
> against them: a re-ranker recovering half the available images is a 4.7-point gain, not a 40 %
> improvement.
>
> **What decides anything is downstream, not the re-ranker's own accuracy.** The re-ranker is scored
> exactly like any other localiser candidate: (i) the paired patient-cluster bootstrap 95 % CI of
> per-lesion box IoU against the promoted ensemble, misses scored 0, on the 243 validation lesions,
> must exclude zero; **and** (ii) a rung-3 retrain on its boxes must not lower exact validation AUC.
> **Candidate-level AUROC is diagnostic only** and is never quoted as the result. A re-ranker that
> ranks candidates well but does not move box IoU has not earned promotion.
>
> **Train-only selection.** The decision threshold τ (below which the peak rule is retained as
> fallback) and any feature scaling are fitted **on the training split only**. Validation is used for
> early stopping on candidate AUROC and for reporting, and for nothing else. The candidate set, its
> features and its labels are fixed a priori: positive iff box IoU ≥ 0.30 with **any** ground-truth
> lesion on the image, as work plan §6 specifies.
>
> **Gate B, unchanged:** the box-IoU gate passes **and** detection at IoU 0.5 reaches ≥ 0.70. For
> reference the promoted ensemble currently sits at 0.5391 on that measure at run-4 level and the
> ladder's promoted boxes at their own value; Gate B is a substantial ask, and failing it is a
> reportable result rather than a reason to re-specify.

### §15a — The re-ranker's feature set is reduced, after the first result (order of events disclosed)

**What was specified, and what it did.** §15 fixed the candidate features a priori, the set work plan §6
names plus the per-member agreement added on 9 September: peak / mean / summed probability, area,
distance to the breast edge, member support, member count, **candidate rank and the rule's own pick
flag**. The features-only model on that set was trained on the laptop and reached candidate AUROC
**0.9162 train / 0.9045 validation** — and gained **exactly nothing** downstream:

| | value |
| :--- | ---: |
| mean per-lesion box IoU, validation | **0.4700** |
| promoted three-run ensemble | **0.4700** |
| paired gate | **+0.0000 [+0.0000, +0.0000]**, 0 improved, 0 worse, **FAIL** |

**It selected candidate rank 1 on 218 of 218 validation images.** The model learned to reproduce the
peak rule rather than to correct it: `is_rule_pick` carries a standardised coefficient of +0.742, and
`area_px` (+0.932) and `mean_prob` (+0.860) both correlate with being rank 1, so the argmax never moves
off the rule's pick. A candidate AUROC of 0.90 and a downstream gain of 0.0000 is the clearest
demonstration this project has of why §15 declared the AUROC diagnostic.

**The change.** `cand_rank` and `is_rule_pick` are dropped from both models. They encode the selection
rule itself, so a model carrying them can score well on the candidate task by copying the rule, which
is precisely the behaviour observed. The reduced set is seven features, all image- or map-derived, and
is the default from here (`--feature-set norule`; `full` reproduces the specified set).

**Order of events, disclosed.** This is a specification change made **after** seeing the first result,
not a prior hypothesis. The original set remains available and **its result is reported beside the
new one**, whatever the new one shows: a re-ranker that only works after its feature set was revised in
response to a failure is a weaker claim than one that worked as specified, and the report says so.
Nothing else moves — the candidate set, the label rule, the 9.3 % ceiling, tau on train, and the
downstream box-IoU gate plus rung-3 retrain as the only deciding metrics are all unchanged.

#### 15a result — the reduced set changes nothing downstream

| | full (as specified) | norule (reduced) |
| :--- | ---: | ---: |
| candidate AUROC, train | 0.9162 | 0.9066 |
| candidate AUROC, validation | 0.9045 | 0.8955 |
| tau, selected on train | 0.70 | **0.95** |
| re-ranker picks / rule fallbacks, validation | 189 / 43 | **14 / 218** |
| mean per-lesion box IoU, validation | **0.4700** | **0.4700** |
| paired gate vs the promoted ensemble | +0.0000, FAIL | **+0.0000, FAIL** |

**Both fail identically, by different routes.** On the specified set the model copied the rule: it chose
rank 1 on 218 of 218 images. On the reduced set it could no longer see the rank, so **tau selection on
train raised tau to 0.95 and made the model defer** — 218 of 232 images fall back to the peak rule,
because deferring is what maximises box IoU. Declining is what tau exists for, and the mechanism worked
as designed. The gate reports `improved 0, worse 0`: every lesion has an identical box IoU, so even the
fourteen re-ranker picks landed on rank 1.

**What this establishes.** Dropping the two rule-encoding features cost almost nothing in ranking
quality (0.9045 -> 0.8955), so the seven map-derived features rank candidates well on their own. But
their ranking agrees with the peak rule wherever a different choice would have mattered: **the 9.3 % of
images that hold a rescuable candidate are not separable by map-derived features.** The tabular route to
§6 is exhausted, under both the specified and the revised specification.

What remains untested is whether **image evidence** separates them — the same crop the classifier sees,
which is what the CNN model uses and what S5 job (g) runs. If that also returns 0.0000, §6 is closed and
the honest report is that the localiser's wrong-component failures are not recoverable by re-ranking at
all, only by better maps.

## 20. §4 — The rung 3 vs rung 5 mechanism on test (P1; descriptive)

*Why here: Names `ladder_error_analysis`, whose validation run is quoted in §11's evidence.*

Append to "## Also run on test, once, descriptively":

> the per-lesion analysis of `notebooks/scripts/ladder_error_analysis.py` (AUC of rungs 2, 3 and 5 inside
> bins of the localiser's box IoU and of rung 5's replayed jitter IoU; the oracle's AUC on each box source's
> victims versus the rest; bad-box rates by subtlety, density, shape, margins, size and multi-lesion images;
> the paired 3−5, 2−3 and 2−5 AUC differences with patient-cluster CIs), specified and run on validation
> before the freeze and executed once on test unchanged.

Runbook (after `src.ladder --stage test`):

```bash
PYTHONPATH=. python notebooks/scripts/ladder_error_analysis.py --boxes-dir $BOXES --suffix _bundle --split test --allow-test
```

> **Run by the pass from 18 September; see §37.** Until then the pass did not run this analysis.

## 21. §5 — Tags table rows to add

*Why here: Last: it is a summary of every tag the blocks above introduce, including the device column §11 adds.*

| tag | role |
|---|---|
| vgg16_crop_rung3_unet_bundle_s1, _s2 | seed replicates of the primary (descriptive) |
| vgg16_crop_rung3_unet_bundle_ens3 | mean-probability seed ensemble (descriptive) |
| vgg16_crop_rung5_jitter_bundle_s1, _s2 | seed replicates of rung 5 (descriptive) |
| resnet50/densenet121/efficientnet_crop_rung3_unet_bundle, rung3_arch_ens4 | architecture family at the rung-3 condition and its ensemble (descriptive) |
| vgg16_fusion_view_unet_bundle | declared view fusion at the rung-3 condition (descriptive, beside rung 3) |

## 22. §16 — Resolution and background patches both fail, and L6b separates them (drafted 10 Sep)

*Why here: Follows §15 because it cites the re-ranker's bound as one of the six nulls. Carries the cache-v3 decomposition and the corrected map check.*

L6 and L6b were pre-specified in work plan §7 before either ran: L6 as cache v3 (2048x1152) with
`--background-frac 0.40`, L6b as the same canvas with `0.0`, **conditional on L6 failing its gates**.
L6 failed, so L6b ran, as specified.

**Both fail all gates.** Paired patient-cluster bootstrap, 1,000 draws, 243 validation lesions, 129
patients, misses scored 0. Computed on the pod overnight and **re-verified on the laptop on
10 September, reproducing every figure to four decimal places** (runbook §11 puts the reported
bootstrap CI on one machine).

| gate | candidate | reference | Δ [95 % CI] | improved / worse | verdict |
| :--- | ---: | ---: | :--- | ---: | :--- |
| (a) L6 vs run 4 | 0.2190 | 0.4190 | **−0.2000** [−0.2515, −0.1499] | 41 / 130 | **FAIL** |
| (b) L6 vs the three-run ensemble | 0.2190 | 0.4700 | −0.2510 [−0.3034, −0.2029] | 37 / 147 | **FAIL** |
| (c) four-member (run1+run3+run4+L6) vs three-run | 0.4742 | 0.4700 | +0.0042 [−0.0155, +0.0242] | 95 / 71 | **FAIL** |
| (a) L6b vs run 4 | 0.2875 | 0.4190 | **−0.1315** [−0.1786, −0.0810] | 54 / 116 | **FAIL** |
| (b) L6b vs the three-run ensemble | 0.2875 | 0.4700 | −0.1825 [−0.2260, −0.1385] | 52 / 128 | **FAIL** |
| (c) four-member (run1+run3+run4+L6b) vs three-run | 0.4788 | 0.4700 | +0.0087 [−0.0066, +0.0251] | 85 / 52 | **FAIL** |

Gate A's alternative criterion also fails for both: detection at IoU 0.5 needed ≥ 0.5891 against run 4's
0.5391, and reached **0.2798** (L6) and **0.3580** (L6b).

**Gate (c) is strikingly stable across four unrelated candidates:** +0.0046 (4c), +0.0037 (L5),
+0.0042 (L6), +0.0087 (L6b). Every one is positive, small, and has an interval spanning zero. A fourth
member of this quality adds the same nothing to the ensemble whatever it changed, which says the
three-run ensemble is not short of *members* but short of a better one. L6b's +0.0087 is the largest
of the four and still fails, which bounds how much a marginal member is worth here.

**The decomposition, which is why L6b was worth its money.**

- **Resolution alone costs −0.1315** (run 4 → L6b; neither has background patches, and the only other
  difference is the v1→v2 clip ceiling, which L5 bounded at −0.0095 [−0.0469, +0.0287]).
- **Background patches cost a further −0.0685** (L6b 0.2875 → L6 0.2190, same canvas).

Both intervals exclude zero. Doubling the canvas hurt badly on its own, and adding background patches
hurt again on top. L5's null is retrospectively **vindicated** rather than undermined: its background
fraction was pinned at 0.156 by geometry, and that accident limited the damage.

**What the diagnostics rule out.** Not underfitting (L6's train Dice 0.5094 exceeds L5's 0.4679 at the
same epoch); not a calibration shift a threshold sweep would recover (peak probability is 0.79-0.83
where the model fires); not a whole-image-versus-tiled artefact (tiled is marginally *worse*). What
remains is that the model fires confidently in the wrong place, or not at all. The reading is that a
512-px patch covering half the tissue loses more surrounding parenchyma than the extra resolution
buys: the lesion is 114 px inside a 512-px window instead of 57.

**Where it goes silent, measured on the selected weights** (`l6_silent_images.py`, all 243 validation
lesions, `LOC_MASK_THRESHOLD` = 0.10): the peak is below threshold on **37 of 243 (15.2 %)**. Those
images hold markedly **smaller** lesions — median side **55.6 px against 84.4 px**, with **51.4 % of
them in the smallest size quartile** against 20.4 % of the rest — and are **less often malignant**
(0.297 against 0.490). Breast density does not separate the two groups. The failure is therefore
concentrated on exactly the cases the resolution increase was justified by: cache v3 was adopted
because the median lesion would grow from 57 px to 115 px, and the resulting model goes silent on the
small lesions that argument was about.

**A correction of provenance, disclosed.** An earlier draft of this block, and the session state it
was written from, quoted **38 %** for this quantity. That came from an interim diagnostic on the
**epoch-28 checkpoint** taken while L6 was still training, not from the selected weights. The figure
on the selected weights is **15.2 %**. The 38 % number is not a property of L6 and is retained only as
what the mid-training diagnostic showed; every downstream statement uses 15.2 %.

**Consequence for S6, recorded before S6 had a result.** §16 of the runbook was amended on 10 September
to run L7 and L8 on **cache v2 at 1024x576** rather than v3. Testing an architecture inside a regime
measured to cost −0.1315 would have made a null uninterpretable. The change is disclosed as a
specification change made after seeing L6/L6b, and the residual difference against run 4 (the v1→v2
clip ceiling) is stated wherever the gate is reported.

**A measurement defect found and fixed in the process, disclosed because it touched a reported
number.** The cache-time correctness check writes `single_box_mean_iou_8bit` into each member's map
manifest. It took the ground-truth box from the store the member *predicted* from, not the frame the
maps were *stored* in, so for any member cached with `--map-canvas` it compared two boxes in two
different frames and returned ≈ 0. It reported **0.0005 for both L6 and L6b** against L5's 0.4095, and
was therefore inoperative for exactly the two runs it needed to check. Corrected to select the store
whose canvas matches the stored maps, and recomputed from the maps already on disk:

| member | split | reported | corrected | the run's own box table |
| :--- | :--- | ---: | ---: | ---: |
| L6 | val | 0.0005 | **0.2189** | 0.2190 |
| L6 | train | 0.0006 | 0.2173 | — |
| L6b | val | 0.0005 | **0.2880** | 0.2875 |
| L6b | train | 0.0009 | 0.3040 | — |

No result changes: the maps were correct throughout, and the gates were computed from the box tables,
not from this check. Two things follow that are worth reporting. First, the corrected figures
reproduce each run's full-resolution box table to within 8-bit quantisation, so **area-downsampling
the maps from 2048x1152 to the ensemble's 1024x576 costs essentially nothing** — gate (c) therefore
tests the ensemble contribution rather than being confounded by the frame change, which is sharper
than the hedge work plan §7 wrote in advance. Second, a check that cannot be run now records why and
writes no number, rather than writing one that means nothing.

## 23. §17 — The S6 runs deviated from their specification and were re-run (drafted 10 Sep)

*Why here: Follows §16, whose cache-v2 decision it references, and must precede §7a because §7a's branch resolves on the re-run results this block defines.*

**Disclosed because it is a re-run, and a re-run is the thing a pre-registration is most entitled to
be suspicious of.** Work plan §0 forbids a second attempt at a failed step "without a new, named
change". This is not that: the runs did not do what their specification said, and the correction is
to make them do it.

**The specification.** Runbook §16, written 9 September: L7 and L8 are the run-4 recipe with the
encoder swapped — "same 512-px patches, same sampler, same 0.5 Dice + 0.5 BCE, **same 100 epochs**,
same merged masks". Run 4's schedule is a 5-epoch warm-up then cosine to 1e-6 across 100 epochs, and
`OneCycleLR` is constructed for exactly that many steps.

**What the first attempt did.** Launched with `--patience 25`, carried from the prepared command in
the session handover. L7 stopped at **epoch 35 of 100** (best epoch 9); L8 was killed at epoch 18 with
its best still at epoch 2 and a stop due near 27. Both were therefore measured with the learning-rate
schedule un-annealed, roughly a third of the way through the recipe they were being compared against.

**Why patience fired at all, and why nobody caught it earlier.** L7's training loss was still falling
monotonically at the stop (0.7871 → 0.1349) and its validation metric was oscillating in a 0.33–0.40
band with no downward trend, so patience fired on **validation noise, not overfitting**. It went
unnoticed because L5, L6 and L6b each ran the full 100 epochs: patience was configured for them too
and never bound, so its effect had never been observed.

**The consequence for the gate.** It made L7 vs run 4 a **two-variable** comparison — encoder plus
truncated schedule — which is precisely the confound that moved L7/L8 to cache v2 in the first place
(§16). A gate computed on it measures the training budget as much as the architecture.

**The correction.** Both members re-run at the written specification: 100 epochs, early stopping
disabled (`--patience` set equal to `--epochs`, and the launch script now refuses to start if patience
is lower), best-epoch selection on `val_hard_iou` unchanged, everything else identical. **Branch §7a
resolves on the 100-epoch results only.**

**The truncated results are kept and reported beside them**, labelled *truncated, inconclusive*, with
their gates:

| gate | candidate | reference | Δ [95 % CI] | verdict |
| :--- | ---: | ---: | :--- | :--- |
| (a) L7 (35 ep, truncated) vs run 4 | 0.4281 | 0.4190 | +0.0091 [−0.0331, +0.0514] | FAIL |
| (a') detect-at-0.5 | 0.5185 | 0.5391 | −0.0206 | FAIL |
| (b) L7 (35 ep, truncated) vs three-run | 0.4281 | 0.4700 | −0.0419 [−0.0942, +0.0054] | FAIL |

They are kept rather than discarded because they are the only evidence for what the truncation cost,
and because a re-run whose superseded attempt has vanished is not auditable. Read as a bound rather
than a result: **even truncated at a third of its schedule, L7 is the only candidate since run 4 to
score above it** (+0.0091 against L5's −0.0095, L6's −0.2000 and L6b's −0.1315), while sitting inside
the ±0.02 single-seed noise floor work plan §0 sets for a localiser gate.

**Order of events, disclosed.** The deviation was found on 10 September while L8 was still training,
by checking whether the in-training metric was the same statistic as the gate's. L8 was killed at
epoch 18 and both members re-launched at specification. No S6 gate on a 100-epoch run existed at the
time of this decision, so it cannot have been made to obtain a particular outcome. The freeze date in
§7a was set before any of this and does not move.

**One further point, carried for the report.** `val_hard_iou` swings ±0.035 epoch to epoch on 243
validation images. Best-epoch selection takes a max over those evaluations, so the selected value is
an upward-biased estimate — mild across 100 epochs, worse across a truncated run. It is a selection
metric and never a reported one; every reported number is the gate, computed from the box tables.

## 24. §7a — The freeze date is set: 14 September, test pass 15 September (decided 10 Sep)

*Why here: It is the stopping rule, and it reads the count of nulls §16 completes. Its L9 and L10 arms are **superseded by §7b**, which must be read immediately after it.*

§7 above deliberately left the freeze date unset and tied it to Gate C. **That tie is now cut.** The
freeze is **14 September 2026** and the single test pass is **15 September 2026**, *regardless* of
whether Gate C is reached and regardless of the outcome of L7, L8 or L10.

> **Freeze: 14 September 2026.** No candidate, configuration, seed or analysis enters after that date.
> **Test pass: 15 September 2026**, once, via `notebooks/scripts/run_test_pass.sh`, under
> `docs/test_pass_preregistration.md` unchanged.

**Why the date is now calendar-set rather than gate-set.** §7 made the freeze conditional on a
performance milestone. That is a defensible rule only while the milestone is plausibly reachable, and
after L6 and L6b it is not: six separate attacks on the run-4 recipe have returned nulls (4b capacity,
4c regularisation, L5 input representation, L6 resolution + background patches, L6b resolution alone,
and the re-ranker on two feature sets), the promoted ensemble has not moved from box IoU 0.4700 since
run 4, and Gate C asks for 0.55. Leaving the freeze tied to a milestone the evidence says will not be
met converts a stopping rule into an open-ended search, which is the failure mode a pre-registration
exists to prevent. Setting the date makes the stopping rule binding again.

**What still runs, and under what condition** — fixed here, before L7 and L8 have any result:

| run | condition |
| :--- | :--- |
| **L7, L8** | running; gated as single runs and as ensemble members exactly as L5/L6/L6b were |
| **L10** (Faster R-CNN detector, work plan §9) | runs **once, 11 September only**, and **only if L7 and L8 both fail gates (a) and (b)** |
| **L9** (`tu-convnext_tiny`, a third U-Net member) | runs **only if L7 or L8 passes** gates (a) and (b) |
| **re-ranker CNN** | **closed; does not run again** |

The two conditions are deliberately mutually exclusive: a failure of both members sends the last
attempt to a different architecture *family* (a detector), while a success sends it to a third member
of the family that worked. Neither branch can be chosen after seeing which would look better, because
both are determined by gates (a) and (b), which are computed before either decision is taken.

**The re-ranker CNN is closed, and reported as attempted rather than as a result.** It was trained
twice on a pod (candidate AUROC 0.8898 and 0.8927) and its box export failed on both occasions,
because `select_tau` reaches the v1 per-lesion store, which cannot exist on a training machine while
the test partition is frozen. Both figures are **below** the features model's 0.9045. A third attempt
would be a third run of a component whose best-case downstream effect is bounded at 9.3 % of
validation images and whose tabular form delivered exactly +0.0000 by two independent routes (§15a).
**The §6 result of record is the features model: +0.0000, gate FAIL, on both the specified and the
reduced feature set.** The CNN is reported as attempted, never gated, with the reason it could not
complete.

**Order of events, disclosed.** This freeze date was set on 10 September, after L6 and L6b had failed
all gates and after their confidence intervals were re-verified on the laptop, but **before L7 and L8
produced any result** — they were launched the same morning and no S6 gate had been computed when
this was written. The date therefore cannot have been chosen to include or exclude any S6 outcome.
The test partition remains unread.

## 25. §7b, §7c — The promotion cascade, the final localiser set, and the run calendar (drafted 10 Sep)

*Why here: **Last of the pending blocks.** It adds the gate (c) arm §7a has no wording for, withdraws L9, makes L10 unconditional and fixes the cascade. §7c corrects the reading §7b's prose gave L10b's gate (c) — the figures in §7b's table were always right — and shares this position because it is read as part of the same ruling.*

### §7b — The promotion cascade, the final localiser set, and the run calendar (drafted 10 Sep)

**Order of events, first, because everything below depends on it.** This block was written on
**10 September 2026, between 07:45 and 08:15 UTC**, while the 100-epoch L7 was still training (it was
at roughly epoch 55 of 100 when drafting began). Exactly these S6 results had been seen at the time of
writing, and no others:

| seen | value |
| :--- | :--- |
| truncated L7 (35 of 100 epochs, spec deviation §17) gate (a) vs run 4 | +0.0091 [−0.0331, +0.0514] **FAIL** |
| truncated L7 gate (a') detect-at-0.5 | −0.0206 **FAIL** |
| truncated L7 gate (b) vs the three-run ensemble | −0.0419 [−0.0942, +0.0054] **FAIL** |
| truncated L7 gate (c) four-member vs three-run | **+0.0246 [+0.0057, +0.0432] PASS** |
| 100-epoch L7 | **nothing — no gate computed, run in flight** |
| L8 | **nothing — not yet started** |

**No 100-epoch gate of any kind existed when this was written.** The truncated L7's gate (c) pass is
the reason the (c) arm is being written at all — §7a's branch was worded on gates (a) and (b) only,
and a member that fails those while passing (c) falls between its two arms. The arm is written now,
before any 100-epoch gate exists, so that it cannot be shaped by the result it will judge. Times are
UTC throughout; the pod ran on UTC and every log timestamp in this project is UTC.

### The gate (c) arm, and the cascade

**The promotion cascade triggers on any of gates (a), (b) or (c).** What is promoted differs:

- **(a) and/or (b) pass** → the candidate's **single-run box table** is the promotion candidate, but
  **only if (b) passes**. (a) is measured against run 4, whose box IoU is 0.4190, and the promoted
  ensemble stands at 0.4700: a run may beat run 4 and still be worse than what is deployed. Promoting
  on (a) alone could therefore *lower* the promoted box set, so **(a) alone never promotes**. It
  remains reportable as evidence about the recipe.
- **(c) passes** → the **four-member ensemble's box table** is the promotion candidate: runs
  1 + 3 + 4 + the candidate, member set fixed a priori, rule selected on train only.
- **Both apply** → the promotion candidate is whichever has the **higher validation box IoU**, chosen
  before any rung-3 retrain is run.

**In every case promotion is final only if the rung-3 retrain on the promoted boxes does not fall
below the CUDA rung-3 reference of 0.8145 by more than the single-seed noise floor (±0.02).** A gate
is necessary and never sufficient: box IoU on victims is not AUC, which is the ladder's own finding.

> **Floor restated by §35:** one-sided, against the three-seed mean 0.814022 (floor 0.794022), not seed 28's 0.8145.

**The 100-epoch L7 is the L7 candidate.** The truncated run's gate (c) pass is recorded beside it,
labelled *truncated run, spec deviation (§17)*, and **does not compete with the rerun** — it is
evidence about what truncation cost, not a candidate for promotion. If the rerun's gate (c) fails
where the truncated one passed, both are reported, and the honest reading is that the pass did not
survive the correction.

### L7v — the pipeline-validation run, declared before it ran

The 10 September audits found seven unintended deviations between L7/L8 as run and run 4 (§17 and
the session state). The port was corrected, but a corrected port is a claim about code, and the
gate is a claim about a model. **L7v tests the port itself before any encoder claim rests on it.**

L7v is the corrected pipeline with encoder **`vgg16`** — the closest replicate of run 4 the port can
make — trained on **cache v1**, run 4's own cache, from a test-free export of the v1 per-lesion
store. It therefore **declares no intended variable**: its encoder is run 4's. Every remaining
difference is inherent to porting a Keras VGG16-encoder U-Net to `segmentation_models_pytorch`
(framework, decoder, pretrained-weight source and its preprocessing convention, encoder BatchNorm),
and the decoder is the one known difference §7b names.

**It is a pipeline check and it is also gated as a candidate**, like every other member.

**Acceptance, fixed before it landed:** single-run validation box IoU within **±0.03 of run 4's
0.4190** — run 4's own single-seed noise band — with a training-loss trajectory of the same shape,
**and a best epoch in the later half of training**. Run 4's best epoch was **62 of 100**; a best
epoch below 20 is a flag even if the box IoU is inside the band, because it would indicate the
schedule is not being traversed as run 4 traversed it. Inside the band and later-half → the pipeline
is validated and L7a follows. Outside → **the difference is a pipeline defect until proven
otherwise, and no L7a or L8a runs until it is found.**

L7a (resnet34) and L8a (efficientnet-b4) are the candidates; each declares `encoder` as its single
intended variable. **L7 and L8 as originally run are not candidates** and are recorded as
*no-aug / fp16 / unnormalised — unattributable*; the truncated L7 is recorded separately as
*truncated run, spec deviation* (§17). Neither competes with the corrected runs.

### L9 is reinstated as a round-one candidate (declared 10 Sep, before L8a has a gate)

**Order of events.** Written on 10 September while L8a was at epoch 48 of 100 with **no gate of any
kind**, and before the combined multi-member gate. L7v and L7a have each passed gate (c)
individually; nothing about L8a, L9 or the combination is known.

**§7b withdrew L9, and this reverses that.** The withdrawal was explicit about its reason — L9 was
listed under future work as *"withdrawn here only for calendar reasons"*, against §7a's 14 September
freeze. **That calendar no longer binds**: §7b itself replaced it with sequencing by result, and the
localiser set is now final when the last candidate in flight has a laptop-computed gate. A candidate
withdrawn for a deadline that has been superseded should be reinstated, or the withdrawal quietly
becomes a scientific judgement it never was.

**The second reason is the result that has since arrived.** Six interventions aimed at a better
localiser failed — resolution, background patches, regularisation, whole-image fine-tuning,
re-ranking, box fusion. The **encoder** is the first lever that moved it: L7a (resnet34) beat run 4 by
**+0.0744 [+0.0330, +0.1193]** on gate (a), the only interval excluding zero any candidate has
produced against run 4. Having found the one lever that works, stopping after two settings of it is
not a defensible stopping rule.

**What L9 separates that L7a and L8a cannot.** Encoder is confounded with capacity in the candidates
run so far:

| member | encoder | parameters | family |
| :--- | :--- | ---: | :--- |
| run 4 / L7v | VGG16 | **31.4 M** | plain conv stack |
| L7a | resnet34 | 24.4 M | residual |
| L8a | efficientnet-b4 | 20.2 M | inverted-residual, compound-scaled |
| **L9** | **`tu-convnext_tiny`** | **31.9 M** | ConvNeXt |

Both candidates so far are **smaller** than the VGG16 baseline, so "the encoder helped" and "a smaller
encoder helped" are not distinguishable from them. `convnext_tiny` at 31.9 M is the closest match to
run 4's 31.4 M in the whole registry, so it varies **architecture family at almost constant capacity**
— which is what makes the encoder claim attributable rather than a size effect.

**Specification.** Identical to L7a and L8a in every setting except `--encoder`: the corrected port,
cache v1, 512-px patches, batch 8, 100 epochs, patience non-binding, `--background-frac 0.0`, fp32,
run 4's augmentation. The config diff must return **zero deviations** with `--intended encoder` and
the same inherent list the other torch members declare (framework, decoder, pretrained weights, input
scaling, encoder BN).

**Gated exactly as every other candidate** — (a), (a′), (b), (c) — and it enters the multi-passer
rule on the same terms: if it passes (c) individually it becomes a member of the single combined
ensemble, in the declaration order above.

**Weights: no token needed, and this was verified rather than assumed.** On the S6 pod, in an empty
Hugging Face cache with no token in the environment and no token file present,
`smp.Unet("tu-convnext_tiny", encoder_weights="imagenet")` downloaded 110 MB and built: **31.9 M
parameters, 512-px forward pass OK**. The earlier check on 10 September was run against a warm cache
and therefore proved less than it appeared to.

### L10b — L10's recipe correction, in L10's slot (declared 10 Sep, before any L10b result)

**Order of events.** Declared **after** seeing L10's per-epoch validation curve and **before** L10b
has been run, launched or scored. L10's own gate set is recorded beside it, computed from the run as
it happened.

**This is the same move L7a was to L7, and it takes L10's slot rather than opening a round two.**
§17 recorded that L7 deviated from run 4 in seven unintended ways and was re-run as L7a, with L7 kept
as *"no-aug / fp16 / unnormalised — unattributable"*. L10 deviates from the members it is gated
against in two ways that were never declared as intended variables, and correcting them is the same
correction. **L10b occupies L10's position in the declaration order** — after L8a and L9, before the
seed replicates — and L10 is recorded beside it as **"the constant-LR run"**.

**The two deviations, and the evidence for each.**

**1. No learning-rate schedule.** Every U-Net member uses run 4's shape: linear warm-up over 5 epochs
then cosine decay to `lr_min = 1e-6`, and each peaks late — run 4 at epoch 62 of 100, L7a at 55, L8a
at 73, L9 at 73. L10 trains at a **constant 1e-4 with no warm-up** (`"schedule": "constant",
"warmup_epochs": 0` in its own manifest). Its curve peaks at **epoch 2 of 30** and declines
monotonically while training loss keeps falling — 0.1743 to 0.0467:

| epoch | 0 | 1 | **2** | 5 | 10 | 15 | 20 | 28 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val box IoU | 0.4529 | 0.4704 | **0.5071** | 0.4600 | 0.4618 | 0.4131 | 0.3667 | 0.3798 |

Fast fit then memorisation, which is what a constant LR on a COCO-pretrained detector over 1,146
images produces. **"30 epochs is too many" is a statement about the constant-LR recipe, not about the
detector**, and it is why extending the run would have been the wrong correction. L10b gets the
U-Net members' schedule shape over its full length.

**2. Selection on the gate's own metric.** L10 checkpoints `best.pt` on **`val_box_iou`** — the
quantity gate (a) and gate (b) score. Choosing the epoch that maximises the metric of record, on the
same 243 validation lesions the gate uses, is selection on the test of record in miniature. **L10b
selects on `val_loss`**, the Faster R-CNN multi-task loss evaluated on the validation split with no
gradient. That is not the gate's metric, and it is the same criterion the U-Net members already carry
as their secondary checkpoint (`best_valloss.pt`). `val_box_iou` is still recorded every epoch, so
the curve remains visible; it simply no longer chooses.

**Everything else identical**, and the config diff must show only the declared changes with
`--family detector --intended encoder --inherent framework batch_size`.

**L10 is not discarded.** Its gate set is reported with both the `best.pt` and `last.pt` box tables
and the selection caveat stated: *selected on the gate's own metric at epoch 2 of 30, on a curve that
declines monotonically thereafter.* If L10b also fails, **the detector family is a null on this data
under a recipe matched to the U-Net members**, which is a finding. As L10 stands, neither that
reading nor its opposite is available.

**The set closes when L10b and the seed replicates have gated**, and the combined gate follows.

### L10 is withdrawn: it was trained by a stale detector, and is recorded, not gated

**What was found, and when.** L10's gate set was computed and reported first — (a) +0.0880
[+0.0504, +0.1281] PASS, (a′) +0.1399 PASS, **(b) +0.0370 [+0.0049, +0.0675] PASS**, (c) via box
fusion −0.0150 FAIL. Reading its manifest to fix seventeen fields the config diff could not find,
the manifest turned out to hold **13 keys, one of them `amp`** — a flag removed from `detector.py`
under the fp32 rule. Pod A's copy was **260 lines against the laptop's 417**, still carrying `--amp`,
`GradScaler` and `autocast`.

**So L10 is §17 repeating.** L7 was re-run as L7a because it was trained by code the laptop had
already corrected; L10 is in the same position. Its gates are **withdrawn as results** and it is
**recorded, not gated**, exactly as L7 is. What the run actually contained cannot be established: the
13-field manifest does not record augmentation, precision, input handling or the box rule, so whether
the Phase 2 corrections were present is unknown rather than merely undocumented.

**Why it was not caught.** `s6_l10_detector.sh`'s pre-flight checked that `detector.py` **exists**.
Existence is not currency. That gap has now cost two runs, and is closed by
`src_manifest.sh` / `assert_src_current.sh`, wired into eleven launchers.

**L10 and L10b differ in TWO respects, and no later reading may treat L10b as isolating either.**

| | L10 | L10b |
| :--- | :--- | :--- |
| code | stale port: `--amp`, GradScaler, autocast, 13-field manifest | current port, 49-field manifest |
| schedule | constant LR, no warm-up | warm-up then cosine, the U-Net members' shape |
| selection | `val_box_iou` — the gate's own metric — at epoch 2 of 30 | `val_loss`, not the gate's metric |

**No L10a, and the reason is recorded.** An intermediate run at L10's recipe on current code would
isolate code-currency from recipe. It is not run, because **L10 is withdrawn and therefore not a
valid comparator**: a confound only matters against a baseline one intends to attribute a difference
to, and there is none. **L10b is gated against run 4 and the promoted ensemble, both of clean
provenance.**

### L10b's config diff against run 4: what is declared and what is inherent (11 Sep)

**Order of events.** Written **after** L10b's gate (a) was refused by `config_diff_vs_run4.py` and
**before** any L10b gate number existed. The refusal is what prompted the ruling; the ruling changes
which rows are *declared*, not what any gate computes.

The diff reported two rows differing from run 4 that were neither declared nor inherent:

| row | run 4 | L10b | ruling |
| :--- | ---: | ---: | :--- |
| `steps_per_epoch` | 287 | 573 | **inherent to the detector family** |
| `epochs` | 100 | 30 | **a declared L10b variable** |

**`steps_per_epoch` is derived, not chosen.** The detector trains on whole images, so 1146 / 2 = 573;
run 4's patch sampler gives 1146 × 2 / 8 ≈ 287. `batch_size` is already declared inherent for the
family, and a value that is pure arithmetic on an inherent field cannot be an independent deviation.
Added to `DETECTOR_INHERENT`.

**`epochs` is a choice the family does not force, so it is declared rather than exempted.** 30 stands.
§7b's correction was to the schedule **shape** — warm-up plus cosine to a low floor, replacing
constant LR — **not to its length**. And 30 binds at neither end: L10 peaked at epoch 2 and declined
monotonically; L10b peaked at epoch 10 with the cosine completing. Extending to 100 would be a new
choice made **after** seeing the curve, and would traverse a decline already measured twice. Gate (a)
carries the reading: a detector at 30 epochs against a U-Net at 100, with the evidence on record that
neither is length-limited.

With both declared, the diff authorises L10b at **0 deviations**, intended variables
`['encoder', 'epochs']`.

### L10b RESULT — gate (a) passes, (c) misses, and the inversion does not reproduce

| gate | candidate | reference | diff | 95% CI | verdict |
| :--- | ---: | ---: | ---: | :--- | :--- |
| (a) vs run 4 | 0.4925 | 0.4190 | **+0.0735** | [+0.0326, +0.1173] | **PASS** |
| (a′) detect@0.5 vs run 4 | 0.5556 | 0.5391 | +0.0165 | — | FAIL (needs ≥ +0.05) |
| (b) vs promoted | 0.4925 | 0.4700 | +0.0224 | [−0.0108, +0.0541] | FAIL |
| (c) box fusion of all four | 0.4993 | 0.4700 | +0.0293 | [−0.0019, +0.0584] | FAIL |

**L10b passes gate (a):** a Faster R-CNN at 30 epochs beats run 4's U-Net at 100 by +0.0735 on mean
box IoU, interval clear of zero. It does not pass (b) or (c), so it is **not** a member of the
combined set and configurations B and C are not formed — but **"the detector family is a null on
this data" is now false**, and §7b's fallback reading of L10's failure cannot be carried over.

**It is not, however, the only (a) pass, nor the strongest single localiser.** Every (a) result on
record, ordered:

| candidate | single-member box IoU | vs run 4 | 95% CI | (a) |
| :--- | ---: | ---: | :--- | :--- |
| L10 (withdrawn) | 0.5071 | +0.0880 | [+0.0504, +0.1281] | PASS |
| **L7a** | **0.4934** | **+0.0744** | [+0.0330, +0.1193] | **PASS** |
| **L10b** | **0.4925** | **+0.0735** | [+0.0326, +0.1173] | **PASS** |
| L8a | 0.4808 | +0.0617 | [+0.0199, +0.1036] | PASS |
| L9 | 0.4802 | +0.0611 | [+0.0245, +0.0979] | PASS |
| L7v s1 | 0.4360 | +0.0169 | [−0.0186, +0.0509] | fail |
| L7v | 0.4349 | +0.0158 | [−0.0217, +0.0559] | fail |
| run 4 s2 | 0.4184 | −0.0007 | [−0.0371, +0.0300] | fail |
| L7v s2 | 0.4139 | −0.0052 | [−0.0384, +0.0308] | fail |
| run 4 s1 | 0.4055 | −0.0135 | [−0.0464, +0.0174] | fail |

**Four current candidates pass (a)** — L7a, L10b, L8a, L9 — and the nominal best of them is **L7a at
0.4934**, not L10b. The gap between them is **0.0009**, well inside the **±0.008** single-seed scale
of §20a.2, so the two are **not distinguishable** and neither should be called the stronger.

**The finding is sharper than "the best single localiser is not a member".** L7a passes (a) **and**
(c). L10b is statistically **tied with L7a** as a single localiser and **fails (c) on a lower bound of
−0.0019**. So what the gates separate is not single-member strength at all: two localisers
indistinguishable on their own contribute differently to the ensemble, and the (c) gate is what sees
it. **The rule governs and L10b is out; the closeness of the margin is reported, not softened** — at
a larger n that interval would very likely clear zero, and the rule would then admit it. What is
recorded is the decision the declared rule produces on the data actually collected.

**(c) misses by −0.0019 on the lower bound and is reported as a fail**, on the same rule applied to
the L7v seeds: an interval that nearly excludes zero has not excluded zero.

**The box-fusion inversion does not reproduce.** §18/§19's measurement on L10 is restated against
L10b as that section said it would be:

| | L10 (withdrawn) | **L10b** |
| :--- | ---: | ---: |
| detector alone | 0.5071 | 0.4925 |
| promoted ensemble alone | 0.4700 | 0.4700 |
| **box fusion of all four** | **0.4551** | **0.4993** |
| fusion vs the better input | −0.0520 (inverted) | **+0.0068** (does not invert) |

With L10 the fused result was **worse than either input**; with L10b it is **better than both**. The
ordering that §18 called an inversion was therefore a property of **that** detector's boxes, not of
the fusion route — and L10's boxes came from the stale port, selected at epoch 2 on the gate's own
metric. §7b's decision to gate the detector through box fusion is **recorded and unchanged**, as it
was before; what changes is that the evidence said to have falsified its premise did not survive the
recipe correction. **The magnitude figures in §18/§19 that rest on L10's boxes should be read as
withdrawn with L10**, and L10b's row above is the one that stands.

### The box-fusion route inverts (§18, §19 revisited)

| | val box IoU |
| :--- | ---: |
| L10 alone | **0.5071** |
| runs 1+3+4 alone (promoted) | 0.4700 |
| **box fusion of all four** | **0.4551** |

**The route loses to either input.** §18 measured a −0.0339 handicap on three comparable members;
with **one strong and three weaker** members it does not merely cost, it **inverts** — the fused
result is worse than the best input alone and worse than the ensemble alone.

**§7b's decision to gate the detector through box fusion was taken on train, when the detector was
expected to be a weak addition to a stronger ensemble. The data has falsified that assumption.** The
decision is **recorded, not changed**, because it was declared in advance and the run it governed has
been made; changing a gating route after seeing what it does to a candidate is the failure this
document exists to prevent. It is the reason §19a's rasterisation alternative and the §19b hybrid
were pre-registered, and why the combination rule declared before L10 gated provides configuration C.

**The 0.5071 figure inherits L10's provenance.** The ordering — fusion below both inputs — does not
depend on which code produced L10's boxes, because all three tables exist and are compared directly.
The *magnitude* does. L10b will restate both.

### Conditional: L7a seed replicates, declared before any seed gate exists

**Order of events.** Written while L7v seed 2 was still training and **no seed replicate of any kind
has been gated**. Declaring a conditional rule after the seed gates would let the condition be chosen
to fit them; before is the only time it is legitimate, and this is that time.

**The rule.** **If §20's L7v seed replicates pass gate (c)**, two **L7a** seed replicates follow —
seeds 1 and 2, identical to L7a in every setting but `--seed`, gated under the same arms, entering
the **same seeds slot** of the declaration order. **If the L7v seeds fail (c), no further seeds run.**

**Why L7a and why cheap.** If seed diversity is what helps, then seeds of the **best single member**
are the cheapest remaining members available: L7a is the strongest localiser the project has (gate
(a) +0.0744 [+0.0330, +0.1193], the only interval excluding zero against run 4), a replicate costs
~70 minutes on the 5090, and it needs no new payload, no new store and no new code. Every other route
to a new member — a new encoder, a new resolution, external data — costs more and has failed more
often.

**The seeds slot order becomes:** L7v s1, L7v s2, **L7a s1, L7a s2**, run4 s1, run4 s2. The L7a pair
is inserted after the L7v pair because it is conditional on it, and before the Keras pair because
§20a's 2×2 reading is about frameworks and does not depend on how many torch seeds exist.

**What it does not change.** §20a's four-cell reading stands exactly as written: it asks whether
torch seeds and Keras seeds help, not how many of each there are. The L7a pair is a **member**
question — does another independent fit of the best recipe add anything — not a diversity question.
And the multi-passer rule is unchanged: each replicate is gated for (c) individually and joins the
combined ensemble only on its own gate.

**If the condition does not fire, that is recorded as a result**: L7v's seeds failing (c) would say
that an independent fit of a member already in the ensemble adds nothing, which makes further seeds
of any member pointless and is worth stating plainly rather than leaving as an unexplained absence.

### The multi-passer rule (written 10 Sep, before any L7a/L8a gate exists)

**Order of events.** Written after L7v's gate (c) passed and **before any other candidate has been
gated**: L7a was at epoch 0 and L8a had not started. The rule below therefore cannot have been
shaped by which candidates turn out to pass.

**L7v's gate (c) result, the one that made this necessary.** Four-member ensemble
(runs 1+3+4+l7v) **0.4932** against the promoted 0.4700, **+0.0231 [+0.0062, +0.0391]**,
improved 104 / worse 69, **PASS**. The four prior four-member sweeps returned +0.0046, +0.0037,
+0.0042 and +0.0087, none excluding zero.

**Why a rule is needed now.** §7b's (c) arm was written for one passing candidate. If two or more
pass, "promote the four-member ensemble" is ambiguous — there would be several, and choosing among
them after seeing their validation numbers is selection on the test of record.

1. **Each candidate is gated for (c) individually**, as a fourth member against the promoted
   three-run ensemble, exactly as now. This is unchanged and is what every candidate faces.
2. **The final promotion candidate is ONE ensemble**: runs 1 + 3 + 4 **plus every candidate that
   individually passed (c)**. The member set is fixed by that rule rather than searched; the sweep
   rule is selected on **train** only; and it is gated **once** against 0.4700.
3. **If the combined ensemble fails where a subset passed**, the candidate is the **largest passing
   subset in declaration order** — L7v, L7a, L8a, **L9**, **L10b** (L10's slot; L10 itself is
   recorded beside it as the constant-LR run), then the seed replicates, ordered
   among themselves as **L7v s1, L7v s2, [L7a s1, L7a s2 if the L7v pair passes (c)], run4 s1,
   run4 s2** (§20a and the L7a conditional). **No post-hoc
   subset search.** Declaration order is fixed here, before the results exist, precisely so that
   "which subset" cannot become a free parameter tuned on validation.
4. **The seed replicates are eligible members under the same rule**, declared now — §20's L7v seeds
   **and** the run-4 Keras seed replicates of §20a. If implementation diversity helps, seed diversity
   may too, and it is measured the same way rather than assumed either way. §20a fixes in advance
   what each combination of the two seed families will be taken to mean.
5. **Promotion is final only if** the rung-3 retrain on the promoted boxes does not fall below the
   CUDA rung-3 reference **0.8145** by more than the single-seed floor (±0.02). A box-IoU gate is
   necessary and never sufficient.

> **Floor restated by §35:** one-sided, against the three-seed mean 0.814022 (floor 0.794022), not seed 28's 0.8145.

**The interpretation, recorded as it stands.** L7v declares **no intended variable** — it is a
vgg16 re-implementation of run 4, and gate (a) found it individually indistinguishable from run 4
(+0.0158, interval spanning zero). Yet adding it to the ensemble helps, detectably. So what passed
is not a *better* member but **ensemble diversity from an equivalent one**: the same architecture and
recipe reached through a different framework, decoder implementation and preprocessing convention
makes errors decorrelated enough to be worth averaging.

That reframes the six nulls. **Every previous attempt sought a better member — more capacity,
regularisation, a cleaner input, more resolution, a different patch mix, better candidate selection —
and each returned nothing. The thing that worked was a different member of equal quality.** It is a
statement about the ensemble rather than about any localiser, and it is the most useful thing the
localiser thread has produced.

### L9 is withdrawn from the pre-freeze plan

`tu-convnext_tiny` would need a run, a laptop-computed gate and a place in the cascade, on top of L8,
L10 and L6c, and the localiser set is closed when those are gated. It is **withdrawn and reported as future work**. This
is a scheduling decision, not a judgement on the encoder; it was verified to build unauthenticated on
the S6 pod (31.9 M params, 512-px forward pass), so nothing blocks it later. §7a's L9 arm — "runs only
if L7 or L8 passes (a) and (b)" — is superseded by this withdrawal.

### L10 runs unconditionally, after L8

§7a made L10 conditional on L7 **and** L8 both failing (a) and (b). **That condition is removed: L10
runs once, after L8 lands and is gated, regardless of the L7/L8 outcome**, as a distinct localiser family rather
than a fallback. Recorded rationale:

- it has the strongest precedent on DDSM specifically (Ribli et al., Faster R-CNN on DDSM);
- it optimises a **box** objective directly, which is the objective the gate measures — every
  candidate so far optimises a per-pixel mask and derives a box by thresholding and taking a
  connected component, so all six share one failure mode;
- it handles **multi-lesion images natively**, where the U-Net members merge masks per mammogram;
- it is the most diverse member available, which is what an ensemble gate rewards.

**Detector-inherent rows, declared.** L10 is audited by the same
`config_diff_vs_run4.py` as every U-Net member, under `--family detector --intended encoder`, which
exempts only what the architecture forces: the multi-task RPN+RoI loss (not pixel Dice+BCE), the
constant schedule and single parameter group (there is no encoder/decoder split to mirror), the
patch-sampler fields (it trains on whole images), the box rule (highest-scoring detection, not
largest component), and **input normalisation**. That last one is worth stating precisely: the
**channel replication is shared** with the U-Net members (`to_encoder_input(..., normalise=False)`),
but the ImageNet mean/std is applied by torchvision's own `GeneralizedRCNNTransform`, which is part
of the detection model. Applying it externally as well would normalise twice. The code **asserts
the transform still carries the ImageNet constants** rather than assuming it, and prints them at
build. Everything else — augmentation, precision, optimiser family, weight decay, cache, masks,
seed, thresholds — must still match run 4, and the diff fails L10 if any does not.

**How L10's gates are read — recorded 10 September, after §18/§19 measured the route.** The two
halves answer different questions and are **reported separately, never combined into one verdict**:

| gate | what it tests | how to read it |
| :--- | :--- | :--- |
| **(a) vs run 4, (b) vs the promoted ensemble** | **the detector itself** — its own single-run box table against the same references every other member faced | like-for-like; no allowance |
| **(c) via weighted box fusion with the promoted ensemble** | **the integration**, not the detector | carries the **measured −0.034 box-IoU handicap** (§18): box fusion cost that much against map averaging on runs 1+3+4, where both routes were available |

The route was chosen on train before any L10 result existed (§19a: box fusion 0.5646 against
rasterised-map averaging 0.4420). **A (c) result near zero therefore does not mean the detector adds
nothing** — it means the detector plus a route known to cost 0.034 nets out near zero, which is a
different statement and is how the report must put it.

**Gated as a single run on (a) and (b), and via box fusion with the promoted ensemble** (weights
selected on train, `notebooks/scripts/box_fusion.py`), **not via map averaging.** A detector produces
boxes, not probability maps, so the map-averaging route the U-Net members share does not apply to it;
box fusion in mammogram coordinates is the route work plan §7 already fixed in advance for exactly
this case.

### §7b — the gate set so far, and which arm each candidate fires

Recorded as each gate landed, on the laptop, from the box tables.

| candidate | (a) vs run 4 0.4190 | (a′) detect@0.5 | (b) vs promoted 0.4700 | (c) four-member vs 0.4700 | arm |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L7v** vgg16 | +0.0158 [−0.0217, +0.0559] FAIL | +0.0041 FAIL | −0.0352 FAIL | **0.4932, +0.0231 [+0.0062, +0.0391] PASS** | **(c)** |
| **L7a** resnet34 | **0.4934, +0.0744 [+0.0330, +0.1193] PASS** | **+0.0823 PASS** | +0.0234 [−0.0156, +0.0602] FAIL | **0.4988, +0.0288 [+0.0109, +0.0494] PASS** | **(c)** |
| L8a efficientnet-b4 | — | — | — | — | in flight |

**L7a fires the (c) arm alone, not the (a)/(b) arm.** (a) and (a′) pass, but §7b's (a)/(b) arm
promotes the single-run box table **only if (b) passes**, and (b) fails. (a) is measured against run 4
at 0.4190 while the deployed set stands at 0.4700, so promoting on (a) alone could *lower* the
promoted boxes; **(a) alone never promotes**. The "both apply, take the higher validation box IoU"
clause therefore does not fire either, because the (a)/(b) arm yields no candidate.

What (a) does establish, and it is the first time in the project: **resnet34 is a genuinely better
single localiser than run 4** — seven interventions aimed at a better localiser, and the encoder swap
is the one that moved it. It is still not distinguishable from the three-run ensemble it would have
to replace, which is why it does not promote on its own.

**Two candidates have now passed (c) individually, so the multi-passer rule is live.** The promotion
candidate is one **five-member** ensemble, runs 1+3+4+l7v+l7a, member set fixed by the rule rather
than searched, rule selected on train, gated **once** against 0.4700 when the set is final.
ens4(l7v) at 0.4932 and ens4(l7a) at 0.4988 are **not competing candidates**; they are two members of
that one ensemble. §23.2b's overlap result predicts the combined gate improves on +0.0288 and not by
much.

### How a box-only (c)-passer combines with the map ensemble (declared before L10 has any gate)

**Order of events.** Written while **L9 was training and L10 had not started**, so no detector gate of
any kind exists. L7v, L7a and L8a have each passed (c) individually; nothing about L10 is known.

**The gap this closes.** §7b's multi-passer rule says the promotion candidate is one ensemble
containing every individual (c)-passer. That assumed every passer is a **map** member. L10 is
deliberately not one: §19a measured rasterising its boxes into a map at **−0.0696** against the
promoted ensemble, twice box fusion's −0.0339, so L10 is gated by box fusion and is kept out of
`MEMBER_TENSORS` so that no code path can average it as a map. The rule below says how a box-only
passer reaches the promotion candidate, and it is written before there is a result to shape it.

**Membership and combination are separate questions.** L10's own gate (c) — box fusion of runs 1+3+4
plus L10 against the promoted 0.4700 — decides whether it is a passer at all, exactly as every other
candidate's (c) decides for it. What follows applies only if it passes.

**Three configurations, all declared here.**

| | configuration | route | handicap |
| :--- | :--- | :--- | :--- |
| **A** | the combined **map** ensemble — runs 1+3+4 plus every **map-member** (c)-passer, rule selected on train | map averaging, as §7b already declares | none |
| **B** | **box fusion of L10 with A's boxes**, weights selected on **train** over the declared grid (§18 A, `box_fusion.py`) | box fusion | **−0.0339 measured** (§18) |
| **C** | the **§19b hybrid**: A's box wherever A produced one, **L10's box only for the lesions where A produced none** | substitution, one-directional | **none** |

**C is added to the proposal, and the reason is the one §18 already measured.** Box fusion localises
worse but **misses less** — 11 misses for the promoted map ensemble against 7 for fusion A. Those are
complementary failures, and §23.2b found the map ensemble's remaining loss concentrated in the **68 of
78 victims that no four-member ensemble rescues**. A detector's plausible value is therefore precisely
*where the ensemble produced no box at all*, which is what the hybrid targets and what box fusion
dilutes by touching rows the ensemble already handles. The hybrid **cannot lose IoU on those rows —
they are untouched** — so unlike B it carries no handicap, and its machinery is already pre-registered
and already exercised on runs 1+3+4 (§19b, +0.0032, a null there).

**The promotion candidate is whichever declared configuration has the higher validation box IoU,
chosen before any rung-3 retrain.** This is not new machinery: it is the rule §7b already applies when
its (a)/(b) and (c) arms both yield a candidate. It costs one declared degree of freedom over a fixed
set of configurations, not a search. **If L10 also passes (b)**, its single-run box table joins the
same comparison under §7b's existing (a)/(b) arm — no further rule is needed.

**If L10 fails (c), A stands alone** and B and C are not formed. A failed candidate does not get a
second route to the promotion candidate.

**The reading, fixed in advance, because a loss here is ambiguous and would otherwise be misread.**

- **B beats A** → the detector adds enough to overcome a measured −0.0339 route penalty, which is a
  strong result and is reported as one.
- **A beats B** → this does **not** establish that the detector adds nothing. The route may have eaten
  the gain, and §18 measured that the route costs about that much. **C is what separates the two**,
  because it cannot lose on the rows A already handles: if C also fails to beat A, the detector adds
  nothing *even where the ensemble has no box*, and that is the finding.
- **C beats A** → the detector's contribution is specifically in recovering the ensemble's misses,
  which is the direction §23.2b named as the one that matters for a detect-then-classify pipeline.
- **All three within noise of each other** → the box-only member does not change the promoted boxes by
  any route, and A stands. Reported as a null about the detector family, not about L10's training.

**Promotion is final only if** the rung-3 retrain on the promoted boxes does not fall more than the
single-seed floor below the CUDA rung-3 reference — unchanged, and it applies to whichever
configuration wins.

### L6c — the resolution test L6 and L6b could not make

L6 changed two things at once against run 4: the canvas doubled **and** the context ratio halved, because
the patch stayed at 512 px while the canvas went to 2048x1152. L6b removed the background patches but
kept both of those. So neither run isolates resolution: what they measured is resolution-plus-lost-context,
and the diagnostics point at the lost context (the lesion sits in a window covering half the tissue).

**L6c holds the context ratio at run 4's value and doubles the resolution:** cache v3, **1024-px
patches** on the 2048x1152 canvas — the same fraction of tissue per patch as 512 px on 1024x576 —
same steps per epoch, **batch 2** for memory, `--background-frac 0.0`, 100 epochs, patience not
binding, everything else the run-4 recipe.

**Conditional on capacity, and the condition is recorded here rather than decided later.** It is
queued on the 5090 after L8 **only if** the pod is otherwise idle and the volume has headroom. As
measured on 10 September the volume is **30 GB with ~12 GB free**, cache v3 is **17.2 GB and is not on
this volume**, so L6c requires either freeing ~9 GB after L7/L8 land and are fetched, or a larger
volume. **If it cannot run before the deadline below, it is the first item of future work** and is
reported as a designed-but-unrun experiment, not as a null.

### Sequencing: by result, not by calendar

**There are no dated deadlines.** L7/L8, L10 and L6c are sequenced by results: each runs, is gated on
the laptop, and the next step follows from the gate. A candidate that crashes and is not relaunched,
or that is abandoned, is recorded as failed and does not enter the cascade.

**The localiser set is final when the last candidate in flight has a laptop-computed gate.** The
freeze is set at that point, and the test pass follows it. Neither is tied to a date here: §7a's
freeze date is superseded by this rule, because a calendar freeze and a result-sequenced run plan
cannot both bind, and the run plan is the one that determines what there is to freeze.

**Pod stopping is manual** throughout; no run in this plan depends on an automatic stop.

### One promotion cascade, on the final set

The cascade runs **once**, on the set as it stands when the last gate is computed. Retrained on the
promoted boxes:

- rung 3, **3 seeds plus `ens3`**;
- the **fusion pair store re-exported** on the promoted boxes, the 3-seed fusion ensemble retrained,
  and **S4 job (e) rerun**.

> **As executed, see §35:** both halves were trained by the store trainers outside the cascade, which asserted and recorded them.

Recomputed on the laptop **from prediction tables**, not retrained: the primary-endpoint preview, the
ladder contrasts and their Holm adjustment, the fusion comparisons that follow from (e), the
localiser-victim and failure/success cuts, and Grad-CAM grounding.

**Not recomputed, and labelled "measured on the runs 1+3+4 localiser" wherever they appear:** Family A,
the §12 uncertainty and deferral thread, and the re-ranker. These are descriptive results about a
specific localiser, and re-deriving them would cost runs the calendar does not have; naming the
localiser they were measured on is the honest alternative to silently letting them read as properties
of the promoted system.

**Note added 19 September 2026, before the freeze: the state of the two lists above, and the order
of events.** (i) In the list above, "Family A" means the architecture comparison at the rung-3
condition (the `{resnet50,densenet121,efficientnet}_crop_rung3_unet_bundle` tags and
`rung3_arch_ens4`; §28), which is how the promotion cascade's own step 7 names it. It does not mean
Family A as the opening section defines it, which uses oracle crops at m = 0.20 and no localiser
boxes; that family was re-measured on the one-session ledger in §32, and the label does not apply to
it. The rung-3 architecture comparison has not been re-measured on A24's boxes and keeps the label.
(ii) The §12 uncertainty and deferral thread was re-measured on A24 in §30. The label applies to the
E3 figures, which are kept; §30's figures are measured on A24. (iii) The re-ranker has not been
re-measured and keeps the label.

(iv) The paragraph above lists the localiser-victim and failure/success cuts and Grad-CAM grounding
as recomputed at promotion. The promotion cascade of 12 September did not recompute them: on that
day the ladder error analysis was labelled as measured on the runs 1+3+4 localiser because rung 5 on
the promoted boxes did not yet exist, and Grad-CAM was held pending a weights-path question. That
ruling was written into `promotion_cascade.sh` step 7 and not into this document. Both analyses were
computed on A24 on 18 September. The ladder error analysis is `ladder_error_analysis_val_cuda0917`,
the source of §32's mechanism figures, produced by `notebooks/scripts_v2/analyse_ladder.py`; the
script the pass runs, `notebooks/scripts/ladder_error_analysis.py`, reproduced it in the `--split
val` dry run `dryrun_val_20260918f`, with every numeric field and the per-lesion table equal.
Grad-CAM for `vgg16_crop_rung3_unet_promoted_cuda0917` was produced by the 18 September `--split
val` dry runs, the boxes folder taken from the promotion pointer. The paragraph above is therefore
right about the outcome and wrong about the route.

The omission was found on 19 September while the handover was being written, before the freeze token
or any test table existed. This note changes no stage, input, comparator or threshold of the test
pass, which runs the §4 error analysis and Grad-CAM on the test partition from the promotion
pointer's boxes and tags as declared (§37). The protocol text above the marker and its recorded hash
are unaffected.

### The test pass

**If `run_test_pass.sh` fails before any test output is written, it is fixed and re-run, with the
failure logged.** Once **any** test output exists, the pass is final — no second pass, whatever the
failure. A **full validation dry run of the script happens before the freeze**, so a failure that can be
found without touching test rows is found then. The script has no `--split val` mode today (it is
test-only), so that mode is written as part of the dry run.

> **Superseded by §36**, which resolves this paragraph's conflict with "Not permitted after the pass".

### Future work, stated so the report does not end on a null

Recorded as written in `docs/plan_forward_2026-09-10.md`, which is the forward plan from 10 September.
These are **designed and not pursued**, and the reasons are given so that "future work" is a judgement
on the record rather than a list of things nobody got to.

**VinDr-Mammo negatives — background supervision.** Design: about **18,000 lesion-free images** used as
negative background supervision for the localiser, put through the **same canvas and the same intensity
path** as CBIS-DDSM so the only new thing is the tissue, and **gated as a member** under the same arms
as every other candidate.

*Not pursued, and the reason is not time.* The diagnosis it treats — that the localiser fires
confidently in the wrong place because it has never been taught what ordinary tissue looks like — is
**plausible but never demonstrated**. The two experiments aimed at it did not establish it: **L5** added
background patches and was a null, and **L6** confounded background with resolution so neither run
isolates it (§16). Committing a large external dataset to an undemonstrated diagnosis is the same
mistake the localiser thread has already made six times. And the **digital/film domain gap falls exactly
on what the negatives would teach**: VinDr is full-field digital mammography, CBIS-DDSM is digitised
film, and "what normal tissue looks like" is precisely the property that does not transfer between them.
The design is recorded so it is a concrete item, not a gesture.

**VinDr mass pretraining.** Design: **1,113 boxed mass images**, median lesion **0.089 of image width**,
which matches CBIS-DDSM's scale statistics closely enough to pretrain on. *Not pursued:* it is
**comparable in size to the existing training set** (1,146 images), so it is not a step change — it
would roughly double the data, against a localiser whose ceiling six interventions have failed to move.

**L6c** — cache v3 at **1024-px patches with the context ratio matched** to run 4's: the resolution test
L6 and L6b could not make, because both changed resolution and context together. *Blocked on volume
headroom*, not on design; the specification is in §7b above.

**Also out of scope:** external pretraining generally, multi-scale members beyond L6c's single
alternative scale, and a third localiser round.

### §7c — The L10b gate (c) reading is withdrawn and replaced (erratum with disclosure, 12 Sep)

**Order of events, stated because the reading changed after the result.** §7b recorded L10b's gates on
11 September with the **correct figures in its table** and a **wrong reading in its prose**. The wrong
reading propagated the same evening into the advisor handover, and from there into the 12 September
handover, blueprint v5.5 (items 36 and 39, open question 17) and the Chapter 5 draft. It was found on
12 September by reading `gate_boxfusion4l10b_vs_ens3.json` back against the prose — before the appendix
was applied and before the freeze. **Nothing measured changes, no run is repeated, and no gate verdict
moves.** L10b was not a member before this block and is not a member after it.

**The erratum.** L10b's gate (c) is **+0.029270**, 95 % CI **[−0.001886, +0.058384]**, 243 lesions /
129 patients, 1 000 iterations, 113 lesions improved against 82 worse. The figure **−0.0019** quoted in
§7b's prose, and everywhere downstream, is the interval's **lower bound** read as though it were the
point estimate. §7b's table row "(c) box fusion of all four" carries the correct value and stands.

**Withdrawn: "single-member strength does not predict ensemble contribution."** The claim was carried by
L7a passing (c) while L10b missed, the two being tied as single localisers (0.493443 against 0.492494, a
gap of 0.000949, inside §20a.2's ±0.008 seed scale). On the corrected figures **they are tied in (c) as
well**: L7a **+0.028758**, L10b **+0.029270**, a difference of **0.000512**. The claim asserts the
opposite of what the table shows and is withdrawn entire, with the sentence in §7b that states it
("two localisers indistinguishable on their own contribute differently to the ensemble") withdrawn with it.

**What the data supports in its place — gate (c)'s verdict is set by interval width, not effect size.**
Every (c) measurement on record, same 243 lesions, 129 patients, 1 000 iterations, reference 0.470047:

| candidate | route | (c) effect | 95 % CI | width | verdict |
| :--- | :--- | ---: | :--- | ---: | :--- |
| L9 | map averaging | +0.030267 | [+0.011092, +0.049220] | 0.038127 | PASS |
| **L10b** | **box fusion** | **+0.029270** | **[−0.001886, +0.058384]** | **0.060270** | **FAIL** |
| L7a | map averaging | +0.028758 | [+0.010881, +0.049448] | 0.038567 | PASS |
| L8a | map averaging | +0.027616 | [+0.007855, +0.049039] | 0.041184 | PASS |
| L7v | map averaging | +0.023127 | [+0.006166, +0.039058] | 0.032892 | PASS |
| L7v s1 | map averaging | +0.014686 | [−0.001333, +0.032615] | 0.033948 | FAIL |
| L7v s2 | map averaging | +0.012240 | [−0.002575, +0.027654] | 0.030230 | FAIL |
| L6b | map averaging | +0.008731 | [−0.006636, +0.025146] | 0.031782 | FAIL |
| run 4c | map averaging | +0.004640 | [−0.008637, +0.015845] | 0.024482 | FAIL |
| L6 | map averaging | +0.004159 | [−0.015519, +0.024165] | 0.039684 | FAIL |
| L5 | map averaging | +0.003665 | [−0.009639, +0.014854] | 0.024493 | FAIL |
| run 4 s1 | map averaging | +0.002594 | [−0.010312, +0.015261] | 0.025574 | FAIL |
| run 4 s2 | map averaging | −0.000796 | [−0.020619, +0.016804] | 0.037424 | FAIL |

**Two candidates whose point estimates differ by 0.000512 receive opposite verdicts.** L10b's interval is
**0.060270** against the four passers' **0.032892 to 0.041184**. Width, not effect, decides.

**The width is a property of the route, not of the candidate.** L10b's (c) is the only one on this list
measured through **box fusion**; every other row is map averaging. Box fusion does two things map
averaging does not. It **depresses the level** — §18 measured the penalty directly at **−0.0339
[−0.0690, −0.0015]** on runs 1+3+4, the three comparable members — and it **widens the interval**, being
a discretising operation: it selects among whole boxes rather than averaging a continuous field, so the
per-lesion outcome moves in steps and the bootstrap spread grows. **Only the second sets the verdict.**
The gate charged L10b for both effects and charged no other candidate for either.

**So L10b posts the second-highest (c) effect in the set while carrying a handicap the passers did not.
Its failure is a property of the measurement route, not of the candidate.** What it would have scored on
a comparable route is **not estimated here and must not be quoted**: §18's −0.0339 was measured on a
different member set and is not transferable to this one.

**What stands.** **L10b remains not a member.** The gate was applied as declared and its verdict stands
as procedurally made — the same treatment §13's declaration receives, for the same reason: a rule that
only binds when its result is liked is not a rule. **Only the recorded reason changes**, from a finding
about single-member strength to a limitation of the comparison.

**This strengthens the withdrawal of "the detector family is a null on this data"**, already recorded in
§7b as the consequent of a conditional whose antecedent never fired. On the corrected reading the
detector's one gated integration route not only failed to establish a null — it returned the second
largest effect measured, through the only handicapped route in the comparison.

**Future work, and why it is a limitation rather than an oversight.** A detector should be measured on a
route comparable to map averaging before the family is judged. The asymmetry is **inherent**: a detector
emits boxes and a U-Net emits maps, so there is no route on which the two are compared without one of
them being converted first. §19a's box-to-map rasterisation is the nearest candidate and carries its own
untested conversion loss. This is a **limitation of the comparison that the design could not avoid**, not
a step that was skipped.

**The two-band reading is demoted, not repaired.** The reading — that same-pipeline candidates give (c)
effects of +0.004 to +0.009 and all fail, differently-implemented candidates +0.023 to +0.030 and all
pass, the two bands not overlapping — describes **eight chosen candidates**. Of the **thirteen** measured
above, **five do not fit**: L10b at +0.029270 sits inside the upper band and fails; L7v s1 (+0.014686)
and L7v s2 (+0.012240) fall in the gap between the bands; run 4 s1 (+0.002594) and run 4 s2 (−0.000796)
fall below the lower band's floor. **The predictive framing is what dies** — band membership does not
determine the verdict. The **magnitudes stand as §23 already uses them**: the range of the passers, not a
band that decides anything. A table drawn over all thirteen would not have supported the claim, and that
is the point the full table above makes.

## 26. §18 — Box-level fusion of the existing members (pre-registered 10 Sep, before running)

*Why here: Self-contained; pre-registered before running. Follows §7b, whose L10 box-fusion route it exercises.*

**Written before the fusion is run and before any of its numbers exist.** The inputs are all on
disk and their single-run validation box IoUs are already known (run 1 0.3853, run 3 0.3874,
run 4 0.4190, L7-no-aug 0.4281); nothing below is chosen with knowledge of a fused result.

**Why now.** Work plan §7 fixed box-level fusion in advance as the contingency for a member that
wins alone but fails the four-member map-averaging sweep, and §7b makes L10 gate "via box fusion
with the promoted ensemble, not map averaging", because a detector emits boxes and has no
probability map to average. That integration path has never been exercised on real members. Running
it on the existing ones tests the path itself while it can still be fixed, and does so on CPU at no
GPU cost.

**Two configurations, both declared here:**

| # | member set | status |
| ---: | :--- | :--- |
| **A** | runs 1 + 3 + 4 — the **declared set**, the same three the promoted map ensemble uses | the pre-registered configuration |
| **B** | runs 1 + 3 + 4 + **L7 (no-aug)** | **separately labelled**, exploratory |

Configuration B is reported apart from A and never merged with it. Its fourth member is the
`no-aug / fp16 / unnormalised` run that §17 records as **unattributable**: it is not a candidate and
cannot become one. It is included because it is the only non-VGG16 box table in existence and the
question here is whether the *fusion path* works on a heterogeneous set, not whether that member is
good. **No promotion can follow from B alone.**

**Method, fixed here.** Weighted box fusion in mammogram coordinates (`notebooks/scripts/box_fusion.py`),
clustering member boxes at IoU ≥ 0.3. **Weights are selected on TRAIN by mean box IoU** over the
declared weight grid, and applied unchanged to validation — the same train-only selection discipline
the map-ensemble sweep uses, and for the same reason: a weight chosen on validation would make the
gate a fit to its own test.

**Gate, unchanged.** Validation per-lesion box IoU, misses scored 0, paired patient-cluster bootstrap
against **the promoted three-run ensemble at 0.4700**, CI excluding zero, on the 243 validation
lesions.

**What each outcome means, decided in advance:**

- **A passes** → it is a promotion candidate under the §7b (c)-type rule: the promotion is of the
  fused box table, and is final only if a rung-3 retrain on those boxes does not fall below the CUDA
  rung-3 reference by more than the single-seed floor. A gate alone is not a promotion.
- **A fails** → the result stands as a null of the same kind as the four map-averaging sweeps, and it
  **validates the L10 integration path on real members**, which is what §7b needs before L10 is
  gated that way. A failure here is therefore informative rather than wasted.
- **B** is descriptive in either direction.

**Prediction, recorded so it can be wrong.** The four map-averaging sweeps returned +0.0046, +0.0037,
+0.0042 and +0.0087, none excluding zero. Box fusion combines the same members' *boxes* rather than
their maps, so it is a different operation, but it draws on the same underlying members and the
expectation is another small positive that does not clear the interval. Recording the expectation
means a pass would be a genuine surprise rather than a rationalised one.

### §18 result — both fail, and A fails on the wrong side

| configuration | val mean box IoU | Δ vs promoted 0.4700 | 95 % CI | improved / worse | verdict |
| :--- | ---: | ---: | :--- | ---: | :--- |
| **A** runs 1+3+4 | 0.4361 | **−0.0339** | **[−0.0690, −0.0015]** | 103 / 85 | **FAIL** |
| **B** + L7 no-aug (exploratory) | 0.4509 | −0.0192 | [−0.0670, +0.0238] | 107 / 89 | **FAIL** |

Weights selected on train, as declared: A `{run1 0.5, run3 0.5, run4 2.0}` at train IoU 0.5646;
B `{run1 0.5, run3 0.5, run4 1.0, l7noaug 2.0}` at 0.5913.

**Configuration A is not merely a null: its interval excludes zero on the NEGATIVE side.** Box
fusion of runs 1, 3 and 4 is **detectably worse** than averaging the same three members' probability
maps. The recorded prediction — "another small positive that does not clear the interval" — was
wrong in direction, and is left standing above as written.

**Why, and it is not a defect in the fusion code.** The full metric set shows the two operations
trading different things:

| configuration | mean IoU | misses | detect@0.5 | detect@0.3 |
| :--- | ---: | ---: | ---: | ---: |
| promoted map ensemble | **0.4700** | 11 | **0.6214** | **0.6790** |
| box fusion A | 0.4361 | 7 | 0.5638 | 0.6132 |
| box fusion B | 0.4509 | **3** | 0.5597 | 0.6420 |

**Box fusion misses less and localises worse.** It recovers images where the map ensemble produced no
box at all — 11 misses down to 7, and to 3 once a fourth member is added — but the boxes it produces
are looser, so detection at both IoU thresholds falls. Averaging maps *before* thresholding lets
members reinforce each other pixel by pixel and sharpens the result; fusing boxes *after* thresholding
can only average corners that have already been committed to, and a confident wrong box drags the
fused box towards it. That is a mechanism, and it is consistent with rung 6's finding that a
confident wrong box is worse than no box.

**What follows, per the outcomes declared before running:**

1. **No promotion.** A fails, so no rung-3 retrain is triggered and nothing changes.
2. **The L10 integration path is validated on real members** — which is what this run was for. Box
   fusion loads heterogeneous member tables, clusters at IoU ≥ 0.3, selects weights on train,
   applies them unchanged to validation, and produces a gateable table in the promoted schema. The
   path works; on these members its output is worse than map averaging.
3. **A caveat L10's gate must now carry.** §7b gates L10 "via box fusion with the promoted ensemble,
   not map averaging", because a detector has no probability map. This result shows that route is
   **not neutral**: it costs about 0.034 of box IoU on members where both routes are available. A
   detector fused this way is therefore handicapped relative to a U-Net member gated by map
   averaging, and **L10's fusion gate must be read against that handicap** rather than as a
   like-for-like comparison. Recorded here so the allowance is made before L10 runs, not after.
4. **Configuration B changes nothing.** Its fourth member is unattributable and it is reported only
   as evidence the path handles a heterogeneous set.

---

## 27. §19, §19a, §19b — The detector's integration route, and a hybrid rule (pre-registered 10 Sep, before running)

*Why here: Follows §18, whose measured box-fusion handicap is its premise. Both pre-registered before running.*

### §19 — The detector's integration route, and a hybrid rule (pre-registered 10 Sep, before running)

**Written before either is run.** §18 established that box fusion is not a neutral substitute for
map averaging — it cost 0.0339 of box IoU on runs 1+3+4, with the interval excluding zero. §7b gates
L10 by box fusion because a detector emits boxes and has no probability map. That handicap is now
measured, so it should be removed rather than allowed for.

### §19a — Box-to-map rasterisation, so a detector can join map averaging

**The idea.** A detector's boxes can be turned back into a probability-like map: each predicted box
is painted onto the 1024x576 canvas as a score-weighted region. The rasterised map then enters the
ordinary map-averaging ensemble alongside the U-Net members, and the detector is gated by the same
route as everything else.

**Two variants, both declared now:**

- **box** — constant intensity equal to the box's score inside the box, zero outside.
- **gauss** — a Gaussian centred on the box centre with standard deviations `sigma_frac x (h/2, w/2)`,
  which softens the corners a hard box commits to. **`sigma_frac` is selected on TRAIN only**, over
  the grid {0.25, 0.35, 0.50, 0.70, 1.00}, by mean train box IoU.

**Score.** Each member's box carries `cand1_sum`, the summed probability of the component it was
derived from. Scores are normalised per member to its own maximum, so a member with systematically
larger blobs does not dominate by scale alone. A table without that column uses a constant 1.0, and
which applied is recorded.

**Geometry.** Boxes are stored in mammogram coordinates; they are forward-transformed to the canvas
with each lesion's own `Letterbox` record (`geometry.Letterbox.forward_box`), the exact inverse of
the transform that produced them. The averaged map is then read by the bundle's own
`REFERENCE_RULE` — largest surviving component at `LOC_MASK_THRESHOLD`, `LOC_MIN_COMPONENT_PX` — so
a rasterised member and a U-Net member are scored by one rule.

**L10's route is chosen by train IoU, before validation is looked at.** Both routes are declared here:

| route | how L10 joins |
| :--- | :--- |
| **R1 map averaging via rasterised boxes** | L10's boxes are rasterised and averaged with the promoted members' maps |
| **R2 weighted box fusion** | §7b's original route, weights on train |

**The rule: whichever route has the higher TRAIN mean box IoU is the one L10 is gated by.** Fixed
before L10 exists, computed on train only, and the validation gate is run once on the winner.
Reporting the loser's validation number as well would be selecting on validation by another name, so
it is not computed.

**Validated now, on members that have both.** Runs 1, 3 and 4 have boxes *and* cached maps, so
rasterised-box averaging can be compared directly with the promoted map ensemble on the same three
members. **This is a diagnostic of what the rasterisation itself costs**, not a promotion candidate:
it answers "how much of a member's map is recoverable from its box alone?" — an upper bound on how
well any box-only member can integrate. It is reported whatever it shows.

### §19b — A hybrid rule: the ensemble's box, with fusion only where it has none

**The idea comes from §18's own numbers.** Box fusion localises worse but **misses less**: 11 misses
for the promoted map ensemble against 7 for fusion A. Those are complementary failures. The hybrid
takes the promoted ensemble's box wherever it produced one, and box-fusion A's box **only for the 11
lesions where the ensemble produced none**.

**It cannot lose IoU on the rows the ensemble already handles** — they are untouched — so it is a
one-directional test of whether fusion's recovered misses are worth having. The 11 substituted rows
score 0 under the ensemble by definition (a miss is scored 0), so any box fusion supplies there is a
gain unless it is also a miss.

**Gate, unchanged:** validation per-lesion box IoU, misses scored 0, paired patient-cluster bootstrap
against the promoted ensemble at **0.4700**, CI excluding zero, 243 lesions.

**A pass is a promotion candidate under the §7b (c)-type rule** — the promoted box table becomes the
hybrid, and promotion is final only if a rung-3 retrain on it does not fall below the CUDA rung-3
reference by more than the single-seed floor. **A fail is recorded** as a null.

**Stated in advance, because it bounds the result:** 11 of 243 lesions is 4.5 % of the set. Even
perfect boxes on all 11 would add at most about 0.045 x (mean IoU of a good box) to the mean — of
order +0.02 — so this is a small-effect test by construction and the interval is unlikely to exclude
zero. It is run because the effect is one-directional and cheap, not because it is expected to pass.

### §19 results

**19a — rasterisation costs more than box fusion, so L10 takes the box-fusion route.**

Variant and `sigma_frac` selected on train only, as declared:

| variant | sigma_frac | train mean box IoU |
| :--- | ---: | ---: |
| **box** | — | **0.4420** (selected) |
| gauss | 0.25 | 0.1546 |
| gauss | 0.35 | 0.2827 |
| gauss | 0.50 | 0.4306 |
| gauss | 0.70 | 0.3584 |
| gauss | 1.00 | 0.2123 |

The hard box wins; softening the corners helps only as `sigma_frac` approaches the box itself
(0.50 is the best Gaussian and still below it), and wide Gaussians collapse, because a broad blob
thresholds into a component far larger than the lesion.

**Diagnostic, runs 1+3+4, the same three members both routes can use:**

| route | val mean box IoU | vs promoted 0.4700 |
| :--- | ---: | ---: |
| promoted map ensemble | **0.4700** | — |
| weighted box fusion (§18 A) | 0.4361 | −0.0339 |
| **rasterised-box averaging** | **0.4004** | **−0.0696** |

These are point estimates; only the box-fusion row carries a bootstrap CI (§18). **Rasterisation
loses roughly twice what box fusion loses.** The reason is visible in the construction: a rasterised
map is a box that has forgotten it was ever a distribution, so averaging three of them recovers only
the *agreement between corners*, while averaging three real maps recovers agreement pixel by pixel.
The answer to "how much of a member's map is recoverable from its box alone?" is: on these members,
about 85 % of the ensemble's box IoU, and less than box fusion manages from the same boxes.

**L10's route, decided on train and before any L10 result exists**, per the rule declared above:
box fusion's train mean box IoU is **0.5646** against rasterised averaging's **0.4420**, so
**L10 is gated by route R2, weighted box fusion.** The rasteriser is kept and reported as the
diagnostic it was declared to be; its validation number is not used to choose anything.

**The §18 caveat therefore stands unchanged**: L10's box-fusion gate carries a measured handicap of
about 0.034 box IoU relative to a map-averaged U-Net member, and must be read against it.

**19b — the hybrid gains what the arithmetic allowed, and no more.**

| | val mean box IoU | Δ vs 0.4700 | 95 % CI | improved / worse | verdict |
| :--- | ---: | ---: | :--- | ---: | :--- |
| hybrid (ensemble box; fusion only on its misses) | 0.4733 | **+0.0032** | **[+0.0000, +0.0090]** | 1 / 0 | **FAIL** |

Of the promoted ensemble's **11 misses**, box fusion A supplied a non-miss box for only **4**, and
only **1** of those four overlapped the lesion at all. The gate reports `improved 1, worse 0` — the
construction guaranteed nothing could get worse, and almost nothing got better.

**This sharpens §18's reading rather than repeating it.** Box fusion misses less *in aggregate*
(7 against 11), but its recoveries are largely **not the same lesions** the ensemble missed, and
where they do coincide the fused box usually still misses the lesion. The two methods' failures are
complementary in count and not in identity — which is why combining them recovers almost nothing,
and is a more useful statement than "the hybrid failed".

The lower bound is exactly +0.0000: with one lesion improved and none worse, the bootstrap cannot
separate the result from zero. Recorded as a null, nothing promoted.

---

## 28. §20, §20a — The localiser seed-noise floor (pre-declared 10 Sep, before running)

*Why here: Descriptive; must be read against every single-run gate above it, so it comes last of the substantive blocks. §20a declares the run-4 Keras replicates and fixes, in advance, what each combination of the two seed families means; it follows §7b, whose multi-passer rule makes them eligible members.*

### §20 — The localiser seed-noise floor (pre-declared 10 Sep, before running)

**Every single-run localiser gate in this project is single-seed.** Work plan §0 sets the classifier
noise floor at ±0.02 for a single-seed comparison and asserts that "localiser recipe noise ≈ 0.00",
on the evidence that run 4c reproduced run 4 to 0.000 on the gate metric. That is one replicate of
one recipe, and it has been carrying a lot of weight: **every gate verdict** — L5's −0.0095, L7's
+0.0091, the four gate-(c) values between +0.0037 and +0.0087, §18's −0.0339, §19b's +0.0032 — is
read against an assumed floor that was never measured for the localiser.

**L7v seeds 1 and 2** are run at the identical configuration, changing only `--seed`. Their spread
against L7v (seed 28) is the localiser's single-seed noise floor, measured rather than assumed.

- **Descriptive, not a gate.** No promotion, no branch and no selection depends on it.
- **Read against every single-run gate**, retrospectively. If the spread turns out comparable to the
  gate deltas above, then several of those verdicts are within noise and the report must say so —
  including the ones that suited the argument.
- **Launched after L8a**, so it never delays a candidate.
- Reported as the three-run spread (min, max, SD) of validation box IoU, and quoted beside the
  ±0.02 figure the plan has been using.

Declared before the runs exist because a noise floor measured after seeing which verdicts it would
overturn is not a noise floor.

### §20a — Run-4 seed replicates, and what the pair of seed families will mean (declared 10 Sep)

**Declared before either seed family has been run**, and before L7a or L8a has a gate.

**The runs.** Run 4 repeated at **seeds 1 and 2** in Keras/TensorFlow, identical recipe, on an Ampere
pod in a later session. They are seed replicates *within the framework the promoted members already
live in*, where §20's L7v seeds are seed replicates *within the torch port*.

**They are eligible members** under the multi-passer rule (§7b), in the same **seeds slot** of the
declaration order. Within that slot the order is fixed here: **L7v s1, L7v s2, run4 s1, run4 s2.**

**Why the pair is worth having.** L7v passed gate (c) while being individually indistinguishable from
run 4, which says the ensemble gained from a *decorrelated equivalent* member. Two explanations fit
that result equally well, and the evidence in hand cannot separate them:

- **framework diversity** — L7v helps because it is the first non-Keras member, and its errors differ
  in kind from runs 1/3/4;
- **seed diversity** — L7v helps because it is simply another independent fit, and any independent fit
  of the same recipe would do as well.

Running seed replicates in **both** frameworks separates the two, and the reading is fixed now:

| L7v seeds | run-4 seeds | reading |
| :--- | :--- | :--- |
| help | do **not** help | **framework diversity** — what the ensemble lacked was a member from a different implementation, not merely another fit |
| help | help | **seed diversity** — any independent fit of the recipe adds decorrelated error, and the framework is incidental |
| do **not** help | help | the benefit is Keras-internal and L7v's pass is not about diversity at all; the port would then need re-examining before anything rests on it |
| neither helps | | L7v's gate (c) pass was specific to that member and does **not** generalise; it is reported as a single result rather than as a principle, and the multi-passer ensemble is whatever passed on its own |

All four cells are written out deliberately. Declaring only the two expected outcomes would leave the
other two free to be interpreted after the fact, which is the failure this document exists to prevent.

**Both families are gated exactly as every other candidate**: individually for (c) against the
promoted three-run ensemble, then as members of the single combined ensemble the multi-passer rule
defines. They also remain descriptive for the noise-floor purpose §20 states, and that reading does
not depend on any of them passing a gate.

**How they run.** `notebooks/scripts/s7_run4_seeds.sh`, on a Keras pod (runbook §2's TensorFlow 2.17 /
CUDA 12.3 stack, Ampere or Ada), against the same cache v1 the torch members use. It declares `--seed`
as the intended variable to the config diff and, uniquely among the candidates, declares **no inherent
deviations** — same framework, same trainer, same cache, same recipe as run 4, so there are none to
declare. Its pre-flight asserts TensorFlow sees a CUDA GPU **and that ReLU is applied in the compiled
graph**, which is the laptop defect that put every training run on a pod in the first place. Run 4 took
17,382 s on a Kaggle P100; on an Ada/Ampere card these should be roughly an hour each.

**Cost and sequencing.** Two Keras runs of about an hour, and they follow the torch seeds, so they never
delay a candidate. Sequenced by result like everything else in §7b: if the pod time is not there, they are
reported as designed-but-unrun and the three unresolved cells of the table stay open in the write-up
rather than being quietly filled in. The running total stands at about $16 against a $10-20 cap, so this is a real constraint and not a formality.

### 20a.1 The run-4 seed (c) gates are the null every other (c) pass has never had

**Order of events, stated exactly, because this one is not clean.** Written **11 Sep 05:59 UTC**. At
that moment:

| gate | state when these readings were written |
| :--- | :--- |
| `gate_ens4run4_s2_vs_ens3.json` | **existed on disk** (written 05:55 UTC) and was **not opened**. Listed by name only. |
| `gate_ens4run4_s1_vs_ens3.json` | did not exist — its sweep had been cut short and was re-queued |
| `gate_ens4l7v_s1_vs_ens3.json` | its landing was in progress, no JSON written |
| `gate_ens4l7v_s2_vs_ens3.json`, l10b | not started |

So for three of the four seed members these readings are genuinely prior. For **run4 s2 the file
existed unread**, and that is a weaker claim than "declared before the number existed". It is named
here rather than rounded up to the stronger one. Anyone reading this may discount run4 s2's
contribution accordingly; the other three carry the design.

**What the gate means.** `gate_ens4run4_sN_vs_ens3` adds to the promoted three-member ensemble a
member carrying **no new information** — same recipe, same cache, same framework, same trainer, only
a different draw. It is therefore the counterfactual the four declared (c) passes (L7v, L7a, L8a, L9,
effects +0.0231 to +0.0303) have never been measured against: *what does a fourth member buy when the
fourth member is nothing new?*

**The three readings, fixed before the numbers are read:**

1. **Run-4 seed (c) fails, the four passes stand** → the gain **requires a differently-implemented
   member, not merely a fourth one**. The current conclusion holds as written.
2. **Run-4 seed (c) passes at comparable magnitude** → the four passes are **explained by ensemble
   size**, and implementation diversity is **not established**. §23.1's correlation evidence becomes
   *suggestive rather than demonstrative*, since a same-recipe replicate also decorrelates somewhat.
3. **Passes but clearly smaller** → **both effects are real and separable**, and the difference
   between them **is** the implementation-diversity component. That difference is then the effect
   size to report, not the raw (c) effect.

**The boundary between readings 2 and 3, declared 11 Sep before any of the four gate JSONs was
opened.**

**Primary test — the paired difference of differences.** For each declared pass *m* ∈ {L7v, L7a, L8a,
L9} and each run-4 seed *s* ∈ {s1, s2}, both (c) effects are measured on **the same 243 lesions**
against **the same reference ensemble**, so the quantity is paired per lesion and the reference
cancels exactly:

> ( IoU<sub>i</sub>[ens3 + run4_s] − IoU<sub>i</sub>[ens3] ) − ( IoU<sub>i</sub>[ens3 + m] − IoU<sub>i</sub>[ens3] )
> = IoU<sub>i</sub>[ens3 + run4_s] − IoU<sub>i</sub>[ens3 + m]

Bootstrapped **patient-clustered on those 243 lesions, 1000 draws, misses scored 0** — the protocol
every other gate in this document uses. Against the four passes' **mean**, the paired per-lesion
quantity is IoU<sub>i</sub>[ens3 + run4_s] − mean<sub>m</sub> IoU<sub>i</sub>[ens3 + m]; the point
estimate equals the mean of the four individual differences, and the interval is computed on that
per-lesion quantity rather than assembled from four intervals.

| primary CI | reading |
| :--- | :--- |
| **excludes zero below** | **reading 3** — clearly smaller. Both effects are real and separable, and **this difference is the implementation-diversity effect size** to report. |
| **contains zero** | **reading 2** — not distinguishable. Reported as **a failure to distinguish, not a demonstration of equivalence**; the test may simply lack the power, and the wording must not convert an absent difference into an established one. |
| **excludes zero above** | **reading 2 a fortiori** — the same-recipe replicate helps at least as much as the differently-implemented member, so ensemble size explains the passes. |

**Reported against all four declared passes individually and against their mean.**

**Why not a containment rule.** The criterion first offered here — *does the seed effect's CI contain
the four passes' mean effect of +0.0275?* — was **rejected before any number was read**. With
intervals of roughly ±0.02, it would contain +0.0275 for almost any seed effect between +0.01 and
+0.045, so its most likely output is "comparable" **whatever the truth**, and "comparable" is the
reading that overturns the localiser conclusion. A boundary whose answer is driven by interval width
rather than by the effect is not a test. It is **kept as a secondary descriptive reading**, reported
beside the primary one and labelled as descriptive, because it is still informative about where the
seed effect sits relative to the passes — but nothing turns on it.

**If the two run-4 seeds disagree** — one clearly smaller, one not distinguishable — the family
reading is **unresolved** and is reported as unresolved. It is not settled by choosing the seed that
agrees with either conclusion. *(This sentence is the author's, not a ruling: it is declared here
because leaving it open would mean deciding it after seeing the two numbers, which is worse.)*

### 20a.1 RESULT — the null buys nothing, and that is what demonstrates the claim

**Both seed replicates gated, and they agree.** Neither adds anything to the promoted three-run
ensemble:

| seed | own gate (c) | vs 0.4700 | 95% CI | (c) |
| :--- | ---: | ---: | :--- | :--- |
| **run4 s1** (primary) | 0.4726 | +0.0026 | [−0.0103, +0.0153] | FAIL |
| run4 s2 (confirmatory) | 0.4693 | −0.0008 | [−0.0206, +0.0168] | FAIL |

**run4 s1 is quoted as the primary evidence and run4 s2 as confirmatory**, under the rule declared
in §20a.1's order-of-events table: s1's gate JSON **did not exist** when the readings were fixed, so
its reading is fully prior, while s2's existed unopened. **The design's strength rests on s1.** The
two agree, so the weaker provenance on s2 costs nothing here — but the record says which one carries
the design, rather than presenting a four-number table as uniformly prior.

**Primary test — paired difference of differences,** as declared above. **run4 s1 first, as the
primary evidence:**

| comparison | ens4 candidate / reference | difference | 95% CI |
| :--- | ---: | ---: | :--- |
| run4 s1 vs L7v | 0.4726 / 0.4932 | −0.0205 | [−0.0345, −0.0062] |
| run4 s1 vs L7a | 0.4726 / 0.4988 | −0.0262 | [−0.0472, −0.0087] |
| run4 s1 vs L8a | 0.4726 / 0.4977 | −0.0250 | [−0.0457, −0.0063] |
| run4 s1 vs L9 | 0.4726 / 0.5003 | −0.0277 | [−0.0471, −0.0105] |
| **run4 s1 vs their mean** | 0.4726 / 0.4975 | **−0.0248** | **[−0.0414, −0.0104]** |

**run4 s2, confirmatory:**

| comparison | difference | 95% CI |
| :--- | ---: | :--- |
| run4 s2 vs L7v | −0.0239 | [−0.0438, −0.0055] |
| run4 s2 vs L7a | −0.0296 | [−0.0588, −0.0052] |
| run4 s2 vs L8a | −0.0284 | [−0.0553, −0.0052] |
| run4 s2 vs L9 | −0.0311 | [−0.0591, −0.0041] |
| **run4 s2 vs their mean** | **−0.0282** | **[−0.0523, −0.0060]** |

**Every interval excludes zero below — all four individually and against the mean, for both seeds,
ten comparisons out of ten.**

**Reading 1, not reading 3.** Readings 2 and 3 were both written for the case where the seed's own
(c) *passes*; s2's failed. The governing statement is therefore **the gain requires a
differently-implemented member, not merely a fourth one**, and the difference of differences is not
choosing between 2 and 3 — it is the **effect size that quantifies reading 1**:

> **+0.0248, 95% CI [+0.0104, +0.0414]** (run4 s1, primary), confirmed at **+0.0282
> [+0.0060, +0.0523]** (run4 s2), in favour of a differently-implemented member over a same-recipe
> replicate, on 243 validation lesions in 129 patients.

**§23.1 is demoted from the argument to corroboration of it.** The correlation evidence — that L7v's
per-lesion errors decorrelate from runs 1/3/4 more than those runs do from each other — was the
mechanism story offered for the (c) passes. It cannot carry the claim on its own, because a
same-recipe replicate also decorrelates somewhat, and §23.1 has no null to say how much of the
decorrelation is worth anything. **The null is what demonstrates the claim**; §23.1 now corroborates
it by naming the mechanism, and is written that way rather than as the evidence.

**The bound on the claim, stated here rather than left for a reader to find.** run4 s2 replicates a
recipe **already represented in the ensemble** — run 4 is one of the three promoted members. So what
the null establishes is:

> a fourth map from an **already-represented recipe** adds nothing, while a fourth map from an
> **unrepresented recipe** adds +0.028.

That is **not** the same as isolating "implementation diversity". Separating *different
implementation* from *a recipe the ensemble has not seen* would require a seed replicate of an
**absent** member — a second draw of L7v, L7a, L8a or L9 compared against a first draw of the same
recipe — which does not exist. §20's L7v seed replicates are the nearest thing and they test a
different question (whether a second draw **within** the torch port helps, not whether a second draw
of an unrepresented recipe helps once its first draw is already in). The experiment that would
separate the two is **not worth making** at this stage of the project, so the limit is declared
rather than closed.

**Provenance, and which seed carries the design.** run4 s2's gate JSON existed on disk, unopened,
when the readings were declared (§20a.1 order-of-events table). run4 s1's did not exist at all, so
its reading is **fully prior**.

- **If s1 agrees with s2**, s1 is quoted as the **primary evidence** and s2 as **confirmatory**, and
  the write-up says plainly that the design's strength rests on s1 because of the unread-file
  provenance on s2.
- **If s1 disagrees with s2**, the family reading is **unresolved** and is reported as unresolved. It
  is not settled by picking the seed that agrees with either conclusion.

Computed by `notebooks/scripts/seed_null_dod.py`, which imports the gate's own `_gate_tables` rather
than restating a paired IoU difference; result in
`dod_run4_s1_vs_passes.json` and `dod_run4_s2_vs_passes.json` under `outputs/results/localiser_bundle/`.

### 20a.3 RESULT — the 2×2 is complete: under the declared rule, neither family helps

**The declared rule.** "Helps" means **passes gate (c)**: the four-member ensemble beats the promoted
three-run ensemble with a patient-clustered 95% CI excluding zero. Nothing else.

| seed | family | four-member ensemble | vs promoted | 95% CI | (c) |
| :--- | :--- | ---: | ---: | :--- | :--- |
| l7v s1 | torch port | 0.4847 | +0.0147 | [−0.0013, +0.0326] | **FAIL** |
| l7v s2 | torch port | 0.4823 | +0.0122 | [−0.0026, +0.0277] | **FAIL** |
| run4 s1 | Keras | 0.4726 | +0.0026 | [−0.0103, +0.0153] | **FAIL** |
| run4 s2 | Keras | 0.4693 | −0.0008 | [−0.0206, +0.0168] | **FAIL** |

**Both L7v bounds are fails, and are reported as fails.** −0.0013 and −0.0026 are below zero; a rule
that admits an interval because it *nearly* excludes zero is not a rule. They are not near-misses,
and the word is not used of them.

**The shapes of the two families are not identical, and that is worth one line.** The L7v seeds sit
at **+0.0147 and +0.0122 with lower bounds just below zero**; run4 s2 sits at **−0.0008 with a
symmetric interval**. Descriptively, a second draw *within the torch port* is a small positive that
the data cannot separate from zero, while a second draw of run 4 is centred on nothing at all. The
declared rule governs the conclusion; this paragraph records the difference rather than smoothing it
into "neither helps".

**The declared cell, verbatim, is the fourth row of §20a's table:**

> *neither helps* — L7v's gate (c) pass was specific to that member and does **not** generalise; it is
> reported as a single result rather than as a principle, and the multi-passer ensemble is whatever
> passed on its own.

**Ruling of 11 Sep, and how it differs from that cell.** The ruling is that *L7v's own (c) pass came
from the framework rather than from being a second draw*. **These are not the same claim**, and the
difference is recorded rather than resolved by wording:

- the **declared cell** says the pass does not generalise — it attributes nothing, and is silent on
  mechanism;
- the **ruling** attributes the pass to the framework, which is a *positive* claim and one the seed
  families alone cannot support, because neither family tests "framework" — they test "second draw".

What the seed families establish is **only the negative half**, and it should be quoted that way:
**being a second draw is not the mechanism, in either framework.** No reader should take the
framework claim from the seed families alone, and this paragraph exists so that none can. What supports the positive half is **separate evidence, not seed
evidence** — L7a, L8a and L9, three further torch-port members with different encoders, each passing
(c) at +0.0288, +0.0276 and +0.0303. A property shared by four torch members and absent from a
run-4 replicate is what makes "framework" the surviving explanation. **The ruling is adopted on that
combined basis and the basis is named**, so that a reader can see the positive half does not come
from the 2×2.

**Read beside §20a.1's +0.0282, the two say the same thing from opposite directions.** §20a.1: a
second draw of run 4 adds nothing while a differently-implemented member adds +0.0282. §20a.3: a
second draw of *L7v* adds nothing either. Together: **a second draw of a represented recipe adds
nothing, whichever recipe it replicates** — the decorrelation that matters is not the decorrelation
between two fits of one recipe.

**Consequence for the member set.** The seeds are **not members**: none passed (c).

> The map set is **runs 1 + 3 + 4 + L7v + L7a + L8a + L9 = 7**.

**The L7a conditional does not fire,** and it closes on substance rather than on a technicality. Its
declared condition was *if §20's L7v seed replicates pass (c), two L7a seed replicates follow*; they
did not pass, so the branch is closed by its own terms. The evidence agrees with the condition
independently: §20a.1 measured that seeds of a represented recipe add nothing **regardless of which
member they replicate**, so two L7a seed replicates would have been expected to add nothing, and the
condition and the measurement point the same way. Nothing further is launched.

### 20a.2 The single-seed scale: ±0.008, and what it applies to

Run 4's own family, three seeds of one recipe, on the 243 validation lesions:

| run | seed | val mean IoU |
| :--- | ---: | ---: |
| run 4 | 28 | 0.4190 |
| run4 s1 | 1 | 0.4055 |
| run4 s2 | 2 | 0.4184 |

**Sample SD 0.0076, range 0.0135.** Both replicates land **below** their parent and **neither (a) CI
excludes zero** (s1 −0.0135 [−0.0464, +0.0174]; s2 −0.0007 [−0.0371, +0.0300]). `detect@0.5` is
0.5185 for s1 and **identical to run 4** at 0.5391 for s2.

**This is the scale every single-seed localiser comparison in the project must now be read against,
including the four (c) passes.** It does not invalidate them — the four (c) effects sit at 3–4× this
SD — but it is the right unit, and two consequences follow:

- a single-seed member's *own* val mean IoU is worth ±0.008 before anything else is said about it, so
  differences between members of that size are not differences;
- run 4's 0.4190, the reference in every gate (a), is **one draw from a distribution with this
  spread**. Gate (a) compares against that draw, not against the recipe's expectation. That is how
  (a) was declared and it does not change, but a reading of "L-member fails (a)" now carries the
  caveat that (a)'s reference carries ±0.008 of its own.

---

## 29. §21 — Scale test-time augmentation on the promoted members (pre-registered 10 Sep, before running)

*Why here: Self-contained; pre-registered before running. Follows §20 because its result is read against the seed floor.*

**Written before it is run.** Nothing below is chosen with knowledge of a result.

**The idea.** The promoted ensemble averages three members' maps at one scale. Lesions span a wide
size range — the median lesion side is 57 px on this canvas, with quartiles at 44 and 75 — and a
network trained on 512-px patches has one receptive field. Averaging predictions over **scales**
0.9, 1.0 and 1.1 as well as over the horizontal flip already used gives each member several looks at
the same lesion, and costs nothing but inference.

**Method, fixed here.** For each promoted member and each validation image: resize the canvas by
each scale factor, predict, resize the map back to 1024x576, and average across {0.9, 1.0, 1.1} x
{identity, horizontal flip} — six passes per member. Members are then averaged as now, and the
bundle's own `REFERENCE_RULE` reads the result, so a TTA ensemble and the promoted ensemble are
scored by one rule.

**Nothing is selected.** The scale set is declared here and not tuned; there is no train-time
selection to make because TTA has no parameters beyond the set itself. Declaring it in advance is
what keeps it from becoming a search over scale sets with the gate as the objective.

**Gate, unchanged:** validation per-lesion box IoU, misses scored 0, paired patient-cluster
bootstrap against the promoted ensemble at **0.4700**, CI excluding zero, 243 lesions. CPU only.

**Outcomes, decided in advance.** A pass is a promotion candidate under the §7b (c)-type rule and
needs a rung-3 retrain that does not fall. A fail is recorded as a null. **Either way the cost is
reported**: six passes per member is a 6x inference cost at deployment, which for a screening system
is a real consideration and not a free improvement — a result that passes on IoU but sextuples
inference time is reported with both numbers.

**Prediction, recorded so it can be wrong.** The bundle's rule sweep already includes flip TTA as a
selectable option. Scale may differ from flip, because it addresses lesion size rather than
orientation. Expectation: a small positive, of the same order as the gate-(c) values (+0.004 to
+0.009), not clearing the interval.

> **Correction, 10 September, after this block was written.** The sentence originally here claimed
> the selected configuration uses `tta=False`, i.e. that flip TTA had not helped on train. That is
> **wrong**: the promoted configuration is cfg 2814, `{ensemble: run1+run3+run4, tta: True,
> breast_filter: False, rule: peak, threshold: 0.15, min_px: 256}` — train selection **did** choose
> flip TTA. The error is corrected here rather than deleted, and the paragraph it contaminated in
> the result below is corrected too.

### §21 result — TTA hurts, monotonically in the amount of scale

| configuration | val mean box IoU | Δ vs the same-rule baseline | Δ vs promoted 0.4700 |
| :--- | ---: | ---: | ---: |
| baseline: plain maps, no TTA, `REFERENCE_RULE` | 0.4110 | — | — |
| flip only (scales = {1.0}) | 0.3986 | −0.0123 | — |
| flip x scales {0.95, 1.0, 1.05} | 0.3554 | −0.0556 | — |
| **flip x scales {0.9, 1.0, 1.1} — the declared set** | **0.3064** | **−0.1046** | **−0.1637** |

**Pre-registered gate:** 0.3064 against the promoted ensemble's 0.4700, **−0.1637
[−0.2032, −0.1302]**, improved 31 / worse 152, **FAIL** — and the interval excludes zero on the
negative side, so this is not a null but a detectable harm. The prediction recorded above (+0.004 to
+0.009, not clearing the interval) was **wrong in direction and in magnitude**, and is left standing.

**Two comparisons are quoted because they answer different questions.** Against the promoted
ensemble (−0.1637) is the pre-registered gate. Against the same-rule no-TTA baseline (−0.1046) is
the like-for-like: the promoted ensemble uses its own train-selected rule (peak, threshold 0.15,
min 256 px) while this experiment holds `REFERENCE_RULE` fixed on both sides, so the −0.1046 is what
TTA itself costs and the difference between the two figures is the selected rule's contribution.

**The gradient is the finding.** Harm rises monotonically with the amount of scale — −0.0123 for
flip alone, −0.0556 at ±5 %, −0.1046 at ±10 % — which rules out a coding artefact in the rescaling
and points at the mechanism: averaging a map with zoomed copies of itself **dilates the blob**, and
under a fixed threshold followed by largest-component selection a dilated blob yields a larger box
and a lower IoU. It is the same mechanism §19a found when wide Gaussians collapsed, arrived at from
a different direction.

**Flip-only does NOT corroborate the sweep — it contradicts it, and the contradiction is the
interesting part.** Train selection chose **`tta=True`** (cfg 2814), so flip TTA helped on train.
Measured here on validation it costs **−0.0123**. The two are not strictly comparable — the sweep
selected flip *under its own rule* (peak, threshold 0.15, min 256 px) while this experiment holds
`REFERENCE_RULE` fixed (largest, 0.10, 64 px) — so the honest statement is that **flip TTA's value
depends on the rule it is paired with**, and it does not survive being moved to a different one.
That is a caution about the selected configuration rather than about flip: a rule and a TTA setting
chosen together on train are a package, and neither transfers on its own.

**Nothing promoted.** The deployment cost — six forward passes per member, eighteen for the
three-member ensemble — would have been reported beside any gain; there is no gain to report it
against.

---

## 30. §22 — A relative-threshold box rule (pre-registered 10 Sep, before running)

*Why here: Self-contained; pre-registered before running. Follows §21, whose dilation mechanism is its premise.*

**Written before it is run.**

**The idea, and why it follows from what is already measured.** Every box in this project is derived
by thresholding a probability map at a **fixed** value (`LOC_MASK_THRESHOLD` = 0.10) and taking the
largest surviving component. A fixed threshold treats a confident map and a diffuse one the same
way. §19a found that widening a blob inflates the box and costs IoU; §21 found the same thing again
when TTA dilated the map. Both point at the box being too large when the map is diffuse — which is
exactly what a fixed threshold produces.

A **relative** threshold adapts: `t = k * peak`, where `peak` is the maximum probability on that
image. A confident map thresholds high and yields a tight box; a diffuse one thresholds low and
still yields a box rather than a miss.

**Method, fixed here.**

- Rule: threshold at `t = k * peak(image)`, keep components of at least `LOC_MIN_COMPONENT_PX`,
  take the largest, exactly as the fixed rule does after thresholding.
- **`k` is swept on TRAIN only**, over {0.20, 0.30, 0.40, 0.50, 0.60, 0.70}, by mean box IoU.
- **Optional one-step binary erosion** before component selection, as a second declared axis
  {off, on}: erosion shrinks a dilated blob and is the direct counter to the mechanism above.
- Member set is **fixed a priori** to the promoted three, runs 1 + 3 + 4. Only the rule is selected,
  and only on train.

**It is implemented as a separate sweep, and this matters.** The bundle's frozen configuration grid
is indexed by `cfg_id`, and the promoted selection is identified as **cfg 2814** and asserted in
three places after every sweep. Adding a rule to `config.LOC_BUNDLE_RULES` would change the grid's
size and ordering, so cfg 2814 would silently point at a different configuration and every
assertion that currently guards the promoted selection would be comparing the wrong thing. The
relative-threshold rule is therefore swept by its own script, writing its own candidate box table,
leaving the frozen grid untouched.

**Gate, unchanged:** validation per-lesion box IoU, misses scored 0, paired patient-cluster
bootstrap against the promoted ensemble at **0.4700**, CI excluding zero, 243 lesions. CPU only.

**Outcomes, decided in advance.** A pass is a promotion candidate under the §7b (c)-type rule and
needs a rung-3 retrain that does not fall. A fail is recorded.

**Prediction, recorded so it can be wrong.** The two prior findings both say the box is too large
when the map is diffuse, and a relative threshold is the natural fix, so this is the first
intervention in the series with a mechanism already measured behind it. Expectation: a small
positive, plausibly the first to clear the interval — but the four gate-(c) results between +0.0037
and +0.0087 are the base rate, and nothing in this project has yet cleared it.

### §22 result — the relative threshold is worse, and overfits its own selection

| | train mean box IoU | val mean box IoU |
| :--- | ---: | ---: |
| promoted ensemble (fixed threshold, cfg 2814) | 0.5288 | **0.4700** |
| relative threshold, selected on train | **0.5207** | 0.4309 |

Selected on train: `k = 0.20`, erosion **off**, flip TTA **on**. Grid top rows: `k=0.2` 0.5207,
`k=0.3` 0.5181, `k=0.2 + erosion` 0.5153.

**Gate: −0.0392 [−0.0743, −0.0074], improved 88 / worse 75, FAIL** — the interval excludes zero on
the negative side, so this is detectable harm rather than a null.

**Two things worth carrying.**

1. **It loses on train too.** 0.5207 against the promoted configuration's 0.5288, so the relative
   rule was never ahead: it was selected as the best *of its own family*, and that family is worse
   than the fixed threshold the project already uses. The mechanism the prediction rested on — the
   box is too large when the map is diffuse — is real (§19a, §21) but a peak-relative threshold is
   not the fix, because the peak is itself unstable on a diffuse map: dividing by a noisy maximum
   adds variance rather than removing it.
2. **It generalises worse.** Train → validation falls 0.0898 (0.5207 → 0.4309) against the promoted
   configuration's 0.0588 (0.5288 → 0.4700). A rule with a continuous parameter swept on train has
   more room to fit the training split than a rule chosen from a small discrete grid, and it did.

**Prediction, wrong again.** The recorded expectation was "plausibly the first to clear the
interval". That is now the **third consecutive optimistic prediction to be wrong** in this series
(§18 predicted a small positive and got detectable harm; §21 the same; §22 the same). The pattern is
itself worth reporting: every intervention that reasons from a measured mechanism to a proposed fix
has made the localiser worse, which is consistent with the thread's overall finding that the
constraint is not one the box rule can reach. Erosion, the one axis specifically designed to counter
dilation, was **not selected** even within its own family.

**Nothing promoted.** The frozen grid was not touched; cfg 2814 still names the promoted
configuration and its assertions are unaffected.

---

---

## 31. §23 — Three diagnostics of the gate (c) pass (pre-registered 10 Sep, before running)

*Why here: Descriptive, and it reads §7b's gate (c) arm, §20a's two explanations and §22's result, so it follows all of them. Its 23.3 hand-off check is infrastructure, reported with the same flag-not-fix rule.*

**All three are descriptive.** No gate, no branch, no promotion and no selection depends on anything
below. They exist because L7v's gate (c) pass is the project's only positive localiser result and an
unexplained positive result is worth less than an explained one. Written before any of them is run,
and before L7a has a gate.

### 23.1 Per-lesion member error correlation

**Status after §20a.1's result (added 11 Sep): corroboration, not the argument.** This section was
written as the mechanism story for the gate (c) passes, and it was careful even then to call itself
"evidence about mechanism, not a test of it". §20a.1 now supplies the test, and the demotion is
explicit: **a same-recipe replicate also decorrelates somewhat**, and nothing here can say how much
of the decorrelation is worth anything, because §23.1 has no null. The run-4 seed null is what
demonstrates the claim — a fourth map from an already-represented recipe buys −0.0008 [−0.0206,
+0.0168] while a fourth map from an unrepresented one buys +0.023 to +0.030. §23.1's correlations
**name the mechanism** behind that difference and are reported in that role.

Per-lesion **validation box IoU** for `run1`, `run3`, `run4`, `l7v`, extended to `l7a` and `l8a` as
they land, and for the promoted three-run ensemble **E3**. Misses score **0**, the convention every
gate in this project uses.

Reported: **Pearson and Spearman** correlation of per-lesion box IoU, pairwise between members and
between each member and E3; and agreement on the binary miss indicator at **IoU < 0.5** (the
detect@0.5 convention) as **Jaccard** and the **phi** coefficient. 243 lesions, 129 patients.

**No confidence intervals.** A patient-cluster bootstrap over the same 243 lesions would dress a
descriptive number as an inferential one, and nothing here is being tested.

**The reading, fixed now.** If L7v's correlation with each of runs 1/3/4 is **lower** than the
run1/run3/run4 pairwise correlations, that is *consistent with* the framework-diversity arm of §20a
and is evidence about mechanism, not a test of it — §20a's seed families are what separates the two
explanations. If L7v's correlations are **comparable** to the Keras pairwise ones, then the gate (c)
pass is not obviously explained by decorrelation and the honest report says the mechanism is
unexplained. Both outcomes are reported.

### 23.2 Where the gate (c) gain lands

The gate counted **104 lesions improved, 69 worse** (the remaining 70 unchanged) for
ens4(l7v) = runs 1+3+4+l7v, cfg 6175, against E3 at cfg 2814. Improved means ens4 box IoU strictly
greater than E3's on that lesion.

Stratified three ways, each defined here:

1. **E3 victim class** — the project's existing definition, `ladder_error_analysis.victims_contrast`:
   **no box, or box IoU < 0.3**, versus the rest.
2. **Lesion size quartile** — `box_area_frac` from `artifacts/lesion_geometry.csv`, quartiles cut on
   the **validation** lesions only.
3. **Prior-miss status** — E3 box IoU **< 0.5**, the detect@0.5 convention gate (a′) uses.

Reported per stratum: counts improved / worse / unchanged, net mean ΔIoU, and the share of the
overall **+0.0231** that the stratum contributes.

**The reading, fixed now.** If the gain concentrates in E3's victims and prior misses, L7v is
recovering lesions the promoted set loses, which is the direction that matters for a
detect-then-classify pipeline. If it concentrates in lesions E3 already localises well, the gain is
IoU refinement on boxes that already contain the lesion, which the ladder's own finding says is worth
less — box IoU on victims is not AUC. **Neither pattern changes the gate (c) verdict**, which stands
on the paired CI; this says what the verdict is made of.

### 23.3 The cascade's hand-off to the test pass, dry-run on a copy

`promotion_cascade.sh` produces the promoted boxes and the retrained tags; `run_test_pass.sh` consumes
them. The two have never been run against each other. This checks, **on a copy of the repo state and
without reading a single test row**, that what the cascade writes is what the test pass expects:
output paths, weight and history tags, and the tag table in `docs/test_pass_preregistration.md`.

**Any mismatch is flagged, not fixed.** A fix applied in the same pass that finds the mismatch cannot
be reviewed against what was there before, and the point of a pre-registered protocol is that changes
to it are visible. Mismatches are listed in the session state and fixed as separate, named changes.

Nothing here creates or reads `docs/.freeze_token`; the loader guard means the test split cannot be
read without it in any case.

> **Corrected by §38:** until 18 September the guard did not cover every route to the test split; the contacts this allowed are listed in §33.


### 23.2b Do two passing members rescue the same lesions? (pre-declared 10 Sep, before running)

**Declared after L7a's gate set and before the overlap is computed.** L7v and L7a have each passed
gate (c) individually, so §7b's multi-passer rule makes the promotion candidate a single
**five-member** ensemble, runs 1+3+4+l7v+l7a. Whether that ensemble can add anything over either
four-member one depends on whether the two members fix the *same* lesions or *different* ones, and
that is measurable now, before the combined gate is run.

**Definitions, fixed here.**

- **E3 victim**: no box, or box IoU < 0.3 — §23.2's definition, unchanged.
- **Rescued**: an E3 victim whose four-member ensemble box IoU reaches **≥ 0.5**, the detect@0.5
  convention gate (a′) uses. A victim that merely improves without crossing 0.5 is counted
  separately as *improved*, because a box that is still wrong is not a rescue.
- **Overlap**: victims rescued by **both** ens4(l7v) and ens4(l7a); **union**: rescued by either.

**The reading, fixed before the numbers exist.**

| overlap | reading | what it predicts for the five-member gate |
| :--- | :--- | :--- |
| **low** (well under half the union) | the two members fail on different lesions and fix different ones | the combined ensemble can approach the **union**, and should exceed both four-member ensembles |
| **high** (most of the union) | both are fixing the same lesions the Keras trio loses — a shared weakness, not two independent ones | the combined ensemble adds little over the better of the two, and may land **between** them |

**This is a prediction, not a gate.** The five-member ensemble is gated once, against 0.4700, with its
rule selected on train, exactly as the multi-passer rule states, and that gate stands whatever this
table says. Recorded now so that the prediction is on the record before the result, rather than
becoming an explanation afterwards.


#### §23.2c result — the marginals halve, and neither declared cell fits cleanly

Computed 10 September from cached maps, no training, rule selected on **train** for every member
set. `notebooks/scripts/marginal_member_curve.sh` → `outputs/results/localiser_bundle/marginal_member_curve.json`.

| n | members | val box IoU | vs 0.4700 | marginal |
| ---: | :--- | ---: | ---: | ---: |
| 3 | runs 1+3+4 (promoted, cfg 2814) | 0.4700 | — | — |
| 4 | + l7v | 0.4932 | +0.0231 | +0.0231 |
| 4 | + l7a | 0.4988 | +0.0288 | +0.0288 |
| 4 | + l8a | 0.4977 | +0.0276 | +0.0276 |
| 5 | + l7a + l8a (cfg 12895) | 0.5099 | +0.0398 | **+0.0111** |
| **6** | + l7v + l7a + l8a (cfg 26340) | **0.5149** | **+0.0449** | **+0.0051** |

**The marginals halve at every step — +0.0288, +0.0111, +0.0051.** That geometric decay is
dilution's signature, and §23.2b predicted it from a rescue-overlap count before it was measured.
Extrapolating the ratio, a seventh member adds roughly **+0.002**.

**Neither declared cell fits cleanly, and the honest statement is recorded rather than forced into
one.** §23.2c's first reading required *"marginal gains shrink **and** the six-member value sits near
the four-member ones"*. The marginals do shrink, but the six-member value is **+0.0161 above the best
four-member ensemble**, which is not "near". The second reading required a roughly linear rise, and
the rise is anything but linear. So: **dilution is real and the ceiling is close, but it has not been
reached at six members** — the headroom in adding members is small and shrinking, not gone.

Two things follow, and both are stated so the number is not over-read:

- **0.5149 is the highest validation box IoU in this project's history**, and it is **descriptive**.
  The promotion candidate is decided by the combined gate, run once with a paired patient-cluster CI
  through `combined_gate.sh`, on the member set §7b's multi-passer rule fixes. This curve cannot
  change that set: membership follows from each candidate's own gate (c).
- The six-member rule differs from every four-member one — cfg 26340 selects **threshold 0.30**
  where the four-member selections took 0.15–0.20 — so the combined ensemble is not any of these
  configurations enlarged; its rule is selected on train in its own right.


#### §23.2b result — low overlap, but the ensemble dilutes both members

Run 10 September, after §23.2b was declared and while L8a was still training.
`notebooks/scripts/victim_rescue_overlap.py` → `outputs/results/gate_c_strata/victim_rescue_overlap.json`.

78 of the 243 validation lesions are E3 victims. Rescued = victim reaching box IoU ≥ 0.5.

| pair | a rescues | b rescues | both | only a | only b | union | overlap/union |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ens4(l7v) vs ens4(l7a) | 7 | 7 | 4 | 3 | 3 | **10** | **0.40** |
| l7v alone vs l7a alone | 15 | 19 | 6 | 9 | 13 | **28** | **0.21** |

**The overlap is low on both cuts**, so §23.2b's first row applies: the two members fail on different
lesions and fix different ones, and the five-member ensemble should exceed both four-member ones.
The correlation table says the same thing from the other side — **`l7v / l7a` is the least correlated
pair in the whole set** (Pearson 0.479, miss Jaccard 0.471), below every torch-vs-Keras pair and far
below the Keras trio's 0.646 mean, even though the two share framework, port, cache and recipe and
differ only in encoder.

**But the headroom is 3 victims, not 13.** The members alone rescue 15 and 19 victims; inside a
four-member ensemble that falls to 7 each. Averaging with the three Keras members **dilutes** the
minority member's rescues, and 68 of the 78 victims are rescued by neither four-member ensemble.
So the prediction is: the five-member gate should improve on +0.0288, and not by much — the ensemble
route's ceiling on victims is set by the dilution, not by the members.

**That is a prediction on the record, and the gate is what decides.** The five-member ensemble is
gated once against 0.4700 with its rule selected on train, whatever this says.

### 23.2c The marginal-member curve (declared 10 Sep, before running)

**Order of events.** Declared after L8a's gate set and **before the curve is computed**. At the time
of writing, three candidates have each passed gate (c) individually — ens4(l7v) +0.0231, ens4(l7a)
+0.0288, ens4(l8a) +0.0276 — and **no five- or six-member ensemble has been scored**. L9 is at epoch
17 of 100 and L10 and the seeds have not started.

**Explicitly descriptive, and the reason it is safe to look now.** §23.2b *inferred* the ensemble
route's ceiling from a rescue-overlap count; this **measures** it. It is computed entirely from
**cached maps — no training, no new member** — and it changes nothing:

- **The promotion candidate remains the rule-fixed set of every individual (c)-passer**, gated
  **once** against 0.4700 when the set closes after L9, L10 and the seeds. §7b's multi-passer rule
  fixes that set by a rule written before any of these results existed.
- **Seeing the six-member value early cannot change which members are in it.** Membership is
  determined by each candidate's own gate (c), not by the combined number. That is exactly why the
  multi-passer rule was written before any candidate had passed: so that a combined value seen early
  is information, not a selection opportunity.
- No gate verdict, no branch and no promotion depends on anything below.

**What is computed.** Validation box IoU for each member set, **rule selected on train** in every
case, exactly as every gate (c) sweep does:

| members | set |
| ---: | :--- |
| 3 | runs 1+3+4 — the promoted ensemble, 0.4700, cfg 2814 |
| 4 | runs 1+3+4 + **l7v** — already gated, 0.4932 |
| 4 | runs 1+3+4 + **l7a** — already gated, 0.4988 |
| 4 | runs 1+3+4 + **l8a** — already gated, 0.4977 |
| 5 | runs 1+3+4 + **l7a + l8a** |
| 6 | runs 1+3+4 + **l7v + l7a + l8a** |

Reported as the level and the **marginal gain per added member**.

**The reading, fixed in advance.** §23.2b predicted that the ensemble route's ceiling is set by
**dilution** — a minority member's rescues are averaged down as members are added — and therefore that
gains shrink with each addition. So:

- **Marginal gains shrink and the six-member value sits near the four-member ones** → dilution
  confirmed as measured, not inferred, and the ensemble route is at its ceiling. The report says the
  remaining headroom is not in adding members.
- **The curve keeps rising roughly linearly** → dilution was the wrong model, the members are
  contributing independently, and §23.2b's overlap-based inference was misleading about the mechanism
  even though its overlap counts were correct.
- **The curve turns down** → members are actively interfering, and the combined gate may come in
  below the best four-member ensemble. §7b already handles that case: the candidate becomes the
  largest passing subset in declaration order, with no post-hoc subset search.

**Recorded alongside, from the three gated four-member results.** ens4(l7v) 0.4932, ens4(l7a) 0.4988
and ens4(l8a) 0.4977 span **0.0056** — three different encoders (vgg16, resnet34, efficientnet-b4),
three different single-run box IoUs (0.4348, 0.4934, 0.4808), and the same four-member outcome.
**The four-member gain is insensitive to which torch member is added.** Taken with §23.1's finding
that every torch-vs-Keras pair is less correlated than every Keras-vs-Keras pair, the lever is
**implementation diversity, not encoder identity** — which is the same conclusion L7v's gate (a)
failure pointed at from the other direction, and it is what §20a's two seed families are designed to
test.

### §23 results — the gain is entirely in the lesions the promoted set loses

Run 10 September, after the block above was written and before L7a had a gate.

#### 23.1 Members do decorrelate across frameworks

Per-lesion validation box IoU, 243 lesions, misses scored 0.
`notebooks/scripts/member_error_correlation.py` → `outputs/results/member_correlation/`.

| pair | Pearson | Spearman | miss Jaccard | miss phi |
| :--- | ---: | ---: | ---: | ---: |
| run1 / run3 | 0.709 | 0.702 | 0.699 | 0.646 |
| run3 / run4 | 0.620 | 0.635 | 0.603 | 0.525 |
| run1 / run4 | 0.610 | 0.618 | 0.618 | 0.549 |
| **Keras mean** | **0.646** | 0.652 | 0.640 | 0.573 |
| run4 / l7v | 0.579 | 0.585 | 0.582 | 0.511 |
| run3 / l7v | 0.537 | 0.550 | 0.596 | 0.517 |
| run1 / l7v | 0.507 | 0.503 | 0.589 | 0.508 |
| **torch-vs-Keras mean** | **0.541** | 0.546 | 0.589 | 0.512 |

**Every torch-vs-Keras pair is below every Keras-vs-Keras pair**, on all four measures, with no
overlap between the two groups. The highest of the torch pairs is `run4 / l7v` (0.579), which is what
one would expect: L7v is a vgg16 re-implementation of run 4 specifically.

Against E3 itself: `E3 / l7v` 0.571, below `run3 / E3` 0.669, `run4 / E3` 0.668 and `run1 / E3`
0.616 — E3 contains the other three, so their correlation with it is partly self-correlation, and
L7v's is not.

**As pre-registered, this is evidence about mechanism and not a test of it.** It is consistent with
the framework-diversity arm of §20a and cannot distinguish it from the seed arm, because no seed
replicate of either framework exists yet. §20/§20a are what separates them.

#### 23.2 All of the gain is in E3's victims, and none of it elsewhere

ens4(l7v) at cfg 6175 against E3 at cfg 2814, 243 val lesions, 104 improved / 69 worse / 70
unchanged, overall **+0.0231**. `notebooks/scripts/gate_c_strata.py` →
`outputs/results/gate_c_strata/`.

| stratum | n | impr | worse | same | mean ΔIoU | share of the +0.0231 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **E3 victim** (no box or IoU < 0.3) | 78 | 16 | 3 | 59 | **+0.0749** | **103.9 %** |
| E3 handles it | 165 | 88 | 66 | 11 | −0.0013 | −3.9 % |
| **E3 miss** (IoU < 0.5, detect@0.5) | 92 | 22 | 10 | 60 | **+0.0651** | **106.6 %** |
| E3 hit | 151 | 82 | 59 | 10 | −0.0025 | −6.6 % |
| Q1 smallest | 61 | 24 | 7 | 30 | +0.0427 | 46.4 % |
| Q2 | 61 | 19 | 24 | 18 | −0.0148 | −16.1 % |
| Q3 | 60 | 28 | 23 | 9 | +0.0257 | 27.5 % |
| Q4 largest | 61 | 33 | 15 | 13 | +0.0389 | 42.3 % |

**This is the direction §23.2 named as the one that matters.** On the 165 lesions E3 already
localises, adding L7v is a wash — 88 improved against 66 worse, net −0.0013. The entire +0.0231 comes
from the 78 lesions E3 fails: 16 improved, 3 worse, and a mean gain of +0.0749 within that stratum.
The prior-miss cut at detect@0.5 says the same thing on a different threshold.

**Read carefully: 59 of the 78 victims are unchanged.** The gain is not a broad recovery; it is a
small number of large ones — 16 lesions where the ensemble moves from a wrong box to a right one.
That is a real gain for a detect-then-classify pipeline, and it is also a thin base, which is why the
gate's paired CI ([+0.0062, +0.0391]) is the verdict and this table is only its composition.

By size the pattern is **non-monotonic**: the smallest and largest quartiles carry 46 % and 42 % of
the gain, Q3 27 %, and Q2 is **negative** (−16 %). Nothing was pre-registered about the shape, and
with 61 lesions a quartile no reading is offered beyond "not monotonic in size".

#### 23.3 The cascade does not currently hand off to the test pass

Dry-run on a copy, no test row read. `notebooks/scripts/check_cascade_handoff.py` →
`outputs/results/cascade_handoff/`. **Flagged, not fixed**, per the rule above.

| # | severity | finding |
| ---: | :--- | :--- |
| 1 | **BLOCKER** | `promotion_cascade.sh` step 2 passes `--seed "$s"` to `src.ladder`, which has no `--seed` argument. The three-seed rung-3 retrain — the step that decides whether a promotion is final — dies at argparse. The `--dry-run` exits non-zero at that line. |
| 2 | **MISMATCH** | The cascade writes tags with suffix **`_promoted`** (`vgg16_crop_rung3_unet_promoted`, `…_ens3`, `vgg16_fusion_view_promoted`, `…_ens3`) and stores `crop_store_rung3_promoted`, `fusion_store_promoted`. `run_test_pass.sh` reads **`_bundle`** (`--rung3-suffix _bundle`, `--tag-suffix _rung3_unet_bundle`). Run as they stand, the cascade would retrain on the promoted boxes and the test pass would then evaluate the **pre-promotion** models, with no error at any point. |
| 3 | **MISMATCH** | None of the six `_promoted` tags, nor the suffix `_promoted`, appears in the pre-registration tag table (§*Tags*, and §5 of this document). The tag table is the authority on what the test pass evaluates, so a promotion cannot reach the test pass without an amendment naming its tags. |
| 4 | **MISMATCH** | `run_test_pass.sh` hard-codes `BOXES=outputs/results/localiser_bundle/boxes`. That agrees with the cascade's default **today**, but `localiser_bundle apply` refuses to overwrite an existing boxes folder, so a promotion writes a **new** one and nothing propagates the name. The test pass would crop on the superseded boxes, silently. |
| 5 | INFO | Six `_promoted` weight files do not exist yet, which is expected before the cascade has run; listed so the set that must exist is visible. |

**Why none of this was caught earlier.** `check_launcher_flags.py` reports
`promotion_cascade.sh: src.ladder` as *skipped*, because `src.ladder` has no `--parse-only`. Finding
1 is exactly the class that flag was added for, and it is invisible until the module supports it.
Extending `--parse-only` to `src.ladder`, `src.compare`, `src.explain`, `src.uncertainty`,
`src.fusion`, `src.ensemble` and `src.analysis` would close it — **recorded as the fix, not applied
here**, so that the finding and its remedy stay separately reviewable.

## 32. §24, §24a — Weighted map averaging over the final member set (pre-registered 10 Sep, before the combined gate)

*Why here: Pre-registered before the combined gate. Follows §23, whose dilution measurement is its rationale, and §7b, whose multi-passer rule fixes the member set it weights. §24a shares this position with the block it amends, as §19a/§19b and §20a do with theirs: it was written before the set closed and before any weighted number existed, and it narrows §24's grid to per-group weights because the per-member grid is infeasible at seven members.*

### §24 — Weighted map averaging over the final member set (pre-registered 10 Sep, before the combined gate)

**Order of events.** Written on 10 September after L7v and L7a had each passed gate (c)
individually, and **before** the combined multi-member gate has been computed. L8a was still
training and had no gate of any kind. Nothing below is chosen with knowledge of a combined result.

**Both a candidate and descriptive**: it is gated like any other candidate, and whichever way it goes
it says something about how the ensemble works.

### Why

The ensemble averages member maps with **equal weight**. On validation the members are not equal:
run 4 is 0.4190 single-run, L7a is 0.4934, and they get the same vote. §23.2b measured what that
costs — alone, L7v and L7a rescue **15 and 19** of E3's 78 victims; inside a four-member ensemble
that falls to **7 each**, and 68 of the 78 are rescued by neither. Averaging dilutes the minority
member. Weighting is the obvious lever on that and the project has never tried it.

### What is swept, and where

**Weights are selected on train only**, exactly as the post-processing rule (threshold, `min_px`,
rule, flip TTA) already is. The grid is declared here, and it is small:

1. **equal** — current behaviour, and the baseline the others must beat **on train**;
2. **proportional to single-run train box IoU**, normalised to sum to 1 — no free parameters, and
   the quantity it uses is measured on train;
3. **a coarse sweep over weights from {0.5, 1.0, 2.0}**, all combinations, normalised — **per
   member while the ensemble has six members or fewer, and per declared group beyond that**
   (amended 10 Sep, see below).

### §24a — The coarse family is per-group beyond six members (amended 10 Sep, before the set closed)

**Order of events.** Written **before the localiser set closed** — L9 was training, L10 had not
started, no seed replicate had been gated — and **before any weighted number of any kind existed**.
Nothing below was chosen with knowledge of a weighted result, because there were none.

**Why it had to change.** The coarse family as first written is `3^k` where **k is every map member
of the combined ensemble, run1+run3+run4 included**. Measured against the joint-pass implementation
(weights and rule selected in one pass over the cached maps, which is what makes this a grid
multiplier rather than a re-sweep per vector):

| map members | vectors | estimated sweep |
| ---: | ---: | ---: |
| 6 — the three current (c)-passers | 729 | **12.8 h** |
| 7 — plus L9 | 2,187 | 37 h |
| 8 | 6,561 | 4.6 days |
| **11 — plus the four seed replicates** | **177,147** | **120 days** |

§7b's declaration order explicitly admits eleven: L7v, L7a, L8a, L9, L10, then four seed replicates,
and §7b point 4 makes the seed replicates eligible members. **At k = 11 the per-member grid is
177,147 vectors and is not merely expensive but infeasible.** That is a defect in §24 as declared,
not a cost that turned out badly, and it is visible before the set closes.

**Why grouping preserves the declaration rather than reducing it.** §24's stated argument was always
about **groups**: *"run 4 is 0.4190 single-run, L7a is 0.4934, and they get the same vote."* That is a
claim about a quality difference between kinds of member, never about individual member identities.
Weighting by declared group tests exactly that claim.

**The groups, fixed here:**

| group | members |
| :--- | :--- |
| **Keras originals** | run1, run3, run4 |
| **torch encoders** | l7v, l7a, l8a, l9 |
| **seed replicates** | l7v s1, l7v s2, run4 s1, run4 s2 |

**3^3 = 27 vectors, regardless of k.** Equal and proportional-to-train-box-IoU are unchanged, one
vector each.

**The per-member grid is kept as written while k ≤ 6**, because there it is affordable (12.8 h) and it
is the declared form. Beyond six, group weighting applies. Which form ran is stated in the report
beside the result, along with k.

**Everything else in §24 is unchanged**: weights selected on **train only**, jointly with the rule in
one pass; the single train winner enters the gate; gated once against 0.4700 as part of the final
combined gate; and the three readings fixed in advance stand exactly as written.

The three families are searched together on train and the single best weighting enters the gate.
**No weighting is selected on validation, and none is chosen after seeing a gate.**

### How it is gated

Once, as a candidate, against the promoted **0.4700**, with the paired patient-cluster bootstrap CI,
on the same 243 lesions / 129 patients as every other gate. It is **part of the final combined gate,
not a separate run**, and it does **not** run before the member set is closed — selecting a weighting
on a set that then changes is selection on a moving target.

### The reading, fixed in advance

| result | reading |
| :--- | :--- |
| weighting improves the train selection **and** the validation gate | **dilution was the binding constraint** — the ensemble was held back by giving a 0.419 member the same vote as a 0.493 one, and a weighting selected on train fixes it |
| weighting improves **train but not validation** | the weighting overfits its own selection, the same failure §22's relative threshold showed. Equal weights stand, and the report says the lever exists but does not generalise |
| weighting improves **neither** | **member agreement carries the ensemble, not member quality.** What the average is doing is finding where members concur; a better member cannot buy more agreement by voting louder. The dilution §23.2b measured is then structural, and the ceiling on the ensemble route is not something weighting can lift |

**If the winning weighting is "equal", that is a result and is reported as one** — not as an
experiment that did not happen. It would say the equal-weight average was already the right
aggregation for these members, which is worth stating explicitly given how much of this project's
localiser story rests on it.


### 24a result — the first end-to-end measurement of §24a, and the export defect that delayed it

**Order of events.** The combined gate ran on 11 September and printed a promotion candidate. The
number behind it did not describe the configuration it named, the defect was found before any
promotion was taken, and **nothing was promoted on the withdrawn number**. No pod was deployed; the
cascade and the fusion re-run had not started and were held.

#### The defect

`select` and `sweep` built their `cfg` dict from six columns — ensemble, tta, breast_filter, rule,
threshold, min_px — and silently dropped the seventh, **`weights`**. `_bundle_boxes` took no weight
argument at all, so the winning weight vector **chose** the configuration and was then **discarded**:
the exported table was an equal-weight rebuild wearing the selected configuration's name.

`_ensemble_map`'s docstring had said "One definition shared by the sweep and the box rebuild so the
two cannot drift apart" — and the two did not drift in the arithmetic. They drifted in the **caller**,
which never passed the argument. A shared function does not protect a value that is never handed to
it.

It surfaced as a contradiction inside one program's own output:

```
[select] cfg 54485 {... threshold: 0.4 ...}  train 0.8051  val 0.5300
         combinedA24_vs_ens3                                    0.5148
```

For configuration A the same two numbers agree exactly (0.5216 = 0.5216), because A's sweep used no
weights, so nothing was dropped. The defect bites **only** where weights are in play.

#### Withdrawn

> **`gate_combinedA24_vs_ens3.json` — 0.5148, +0.0448 [+0.0203, +0.0686] — is withdrawn.** It is the
> gate of an **equal-weight** seven-member ensemble at threshold 0.4, not of §24a's group-weighted
> winner. Its value is reproduced exactly by the equivalence check below (0.5148262246), which
> confirms what it measured. **No promotion, selection or reading rests on it.**

#### The fix, and the equivalence it had to pass first

`_bundle_boxes` now takes the weight vector and passes it to the same `_ensemble_map` call the sweep
uses; `select`, `sweep` and `metrics` carry the seventh column and record it in `selected`;
`_weights_for_label` reconstructs a vector from its label by **searching `_weight_vectors`** rather
than re-deriving the arithmetic, so a label has one meaning.

Checked by §24's own discipline, on the artefact that actually leaves the program
(`notebooks/scripts/weights_boxes_equivalence.py`):

| split | rows × columns | mean box IoU, equal | mean box IoU, all-ones | element-wise |
| :--- | :--- | ---: | ---: | :--- |
| val | 243 × 67 | 0.5148262246 | 0.5148262246 | **IDENTICAL** |
| train | 1210 × 67 | 0.7468131529 | 0.7468131529 | **IDENTICAL** |

**Every column, every row, |diff| 0.000e+00.** All-ones weights reproduce the unweighted exported
table bit-identically through the fixed path — the same check that caught the `np.average`
promotion, applied one layer further out.

#### The measurement

Reconstructed from the surviving score arrays and the deterministic grid
(`notebooks/scripts/reexport_weighted_selection.py`), then gated:

| | value |
| :--- | :--- |
| grid rebuilt | **56,700** = 7 members × 9 group-weight vectors, matching the array width |
| selected | cfg **54485**, train argmax over the 3,780 full-ensemble rows — **matches** the recorded selection on all six post-processing fields |
| weights label | **`g:keras=0.5,torch=2`** |
| weight vector | (0.368421, 0.368421, 0.368421, 1.473684, 1.473684, 1.473684, 1.473684) |
| train mean box IoU | 0.8050511884 (score array 0.8050512075, \|diff\| 1.9e-08) |
| val mean box IoU | **0.5299518531** (score array 0.5299518108, \|diff\| 4.2e-08) |

**Gate: 0.5300 vs 0.4700, +0.0599, 95% CI [+0.0328, +0.0869] — PASS**, 131 lesions improved, 65
worse (`gate_combinedA24weighted_vs_ens3.json`).

**This is the first end-to-end measurement of §24a.** The train search had run before; nothing had
ever scored its winner's boxes.

**What the winning weighting is.** §24a's group search **halves the Keras members and doubles the
torch members** — run 1, run 3 and run 4 at 0.368, L7v, L7a, L8a and L9 at 1.474 after
normalisation. That is the same direction as every other result in this thread: the second
implementation carries the information the ensemble lacked. It is a **train-selected** weighting and
is reported as such, not as evidence for the direction on its own.

#### §24's declared question, answered

§24 asked whether weighting the members beats averaging them equally. **It does:** +0.0599
[+0.0328, +0.0869] against +0.0515 [+0.0266, +0.0769], a difference of +0.0084 in validation box
IoU. The search was declared before the members were known, amended to per-group before any weighted
number existed, and proved equivalent to the unweighted path at all-ones before its result was read.
Its answer is positive and it is recorded as such.

**The winning vector points the same way as three earlier results, by a fourth route.** Halving the
Keras members and doubling the torch members says the torch members carry information the ensemble
lacked — the same conclusion as the seed null (a Keras replicate adds nothing: +0.0026, −0.0008),
the difference of differences (+0.0248 [+0.0104, +0.0414] for a differently-implemented member over
a same-recipe one) and §23.1's correlation tables.

**It is consistency, not independent evidence, and is labelled so wherever it is quoted.** The
weighting was selected on **train**, over a grid that contains the answer by construction: a search
free to up-weight any group will up-weight whichever group helps, so finding that it up-weights the
torch members restates what the training data already showed. It cannot corroborate the seed null,
which is an out-of-sample test against a declared null. Four routes agreeing is worth one sentence
in the write-up; it is not four pieces of evidence.

#### The promotion candidate changes

| configuration | val box IoU | vs promoted 0.4700 | 95% CI | verdict |
| :--- | ---: | ---: | :--- | :--- |
| A — equal weights | 0.5216 | +0.0515 | [+0.0266, +0.0769] | PASS |
| **A24 — §24a group-weighted** | **0.5300** | **+0.0599** | **[+0.0328, +0.0869]** | **PASS** |
| ~~A24 as exported 11 Sep~~ | ~~0.5148~~ | ~~+0.0448~~ | — | **withdrawn** |

Under the declared rule — **highest validation box IoU among the declared configurations** — the
promotion candidate is **A24 at 0.5300**. On the withdrawn number it would have been A at 0.5216.
**The defect had reversed the candidate**, which is why no promotion was taken until it was fixed.

The §7b floor is unchanged: promotion still requires a rung-3 retrain on these boxes within 0.02 of
0.8145.

> **Floor restated by §35:** one-sided, against the three-seed mean 0.814022 (floor 0.794022), not seed 28's 0.8145.

#### The restore no longer destroys the run it protects

The exit trap restores the promoted three-run bundle files, and that overwrote the run's own
`sweep_configs.csv` — the only record of which weight vector each of 56,700 configurations used. The
selection survived in `combined_A24/`; the grid explaining it did not. It was recoverable here only
because `_grid` is a pure function and the score arrays are not among the restored files.

`combined_gate.sh` now archives the run's bundle files to `$OUTDIR/run_artefacts/` **before** the
restore, and carries `sweep_configs.csv` into the configuration folders. Same class as a fetch that
could terminate a pod on a partial transfer: **the safety action must not consume the product.**

#### §13c's counts, restated for the candidate that actually enters the cascade

§13c was written when A appeared to be the candidate. A24 is. Both are recorded; the prediction is
unchanged and now attaches to the right table.

| against the promoted ensemble | A (equal) | **A24 (weighted)** |
| :--- | ---: | ---: |
| val mean box IoU | 0.5216 | **0.5300** |
| failures, box IoU = 0 | 68 → **55** (22.6 %) | 68 → **54** (22.2 %) |
| rescued (0 → > 0) | 20 | **23** |
| lost (> 0 → 0) | 7 | **9** |
| net change | −13 | **−14** |

§13c's prediction 2 therefore stands on a net **−14** failures, not −13.

## 33. §26, §26a — The classifier ceiling at rung 2 (pre-registered 10 Sep; amended 10 Sep, before any run)

*Why here: §26a declares, before the session runs, what a moved number does to §14a's recovery figure, §26's bar and the ladder contrasts. Pre-registered before any of it runs. Follows §24 because it is the classifier half of the same forward plan, and it must precede the cascade so rung 3 is retrained once.*

### §26 — The classifier ceiling at rung 2 (pre-registered 10 Sep; amended 10 Sep, before any run)

**Order of events, and it matters here.** §26 was declared on 10 September while L8a was at epoch 79
of 100 with no gate. It was then **amended, before a single §26 run had been made**, because two facts
about the incumbent crop margin were established in between:

1. **m\* = 0.20 was selected on validation**, single seed, on the **Metal** ledger — not chosen a
   priori. `config.py` says so (`m* from the C1 validation sweep (was 0.40, unvalidated)`) and the
   pre-registration heads its model list *"m\* = 0.20, selected on val AUC, single seed"*.
2. **It was selected on a different crop route from the one it is used on**, and that transfer is
   documented nowhere. This is the substantive finding, and it stands whatever the sweep below says.

Everything in this block was written before any of it ran.

### The route transfer — a finding that needs no runs

The C1 sweep trained `vgg16_crop_m{000..100}` with **`crop_sizing="mask_margin"`**, `box_source=None`:
the crop comes from the **lesion mask's** extent, expanded by m. The ladder's rung 2, rung 3, the
promotion cascade and the fusion rerun all use **`crop_sizing="box_provider"`** — the crop comes from a
**box**, oracle (`artifacts/lesion_bbox_lcc.csv`, the largest connected component) at rungs 1–2 and the
U-Net's predicted box at rung 3.

**So m\* = 0.20 was selected on `mask_margin`, single seed, on Metal, and then applied to
`box_provider` — the route every deployable number in this project is computed on — without ever being
re-selected there.** No document records the transfer. It goes in **Limitations** whatever §26 finds.

**What the data already on disk says about it, with no new runs.** Exact validation AUC (§8), CUDA:

| route | three seeds | mean | SD |
| :--- | :--- | ---: | ---: |
| `mask_margin` m020, patience 25 (`vgg16_crop_m020_p25_cuda`, `_s1`, `_s2`) | 0.8753 / 0.8744 / 0.8738 | **0.8745** | **0.0008** |
| `box_provider` oracle rung 2 (`vgg16_crop_rung2_oracle_cuda`, `_s1`, `_s2`) | 0.8861 / 0.8669 / 0.8663 | **0.8731** | **0.0113** |

**The two routes agree at the shared margin to within each other's noise** (0.0014 apart, against SDs
of 0.0008 and 0.0113), so the untested transfer **looks benign at m = 0.20**. **This does not license
the transfer at any other margin**: the two routes crop from different geometry, and there is no reason
their margin responses must have the same shape. It licenses exactly one point.

**Separately, and as its own observation: the seed spread differs fourteen-fold between the routes** —
SD 0.0008 on `mask_margin` against 0.0113 on `box_provider`. Three runs of the same recipe on the
oracle-box route disagree by 0.02 while three on the mask route agree to 0.001. That is not explained
here, and it is the reason for the reordering below.

**And seed 28 is not the cause.** At **+1.15 SD** from the three-seed mean it is not an outlier by any
usual standard, so the spread is a property of the condition rather than of one bad run. Three
conditions, all **vgg16, patience 25, CUDA, exact validation AUC**, differing only in how the crop is
obtained:

| condition | three seeds | mean | **SD** |
| :--- | :--- | ---: | ---: |
| rung 2, `box_provider(oracle)` | 0.8861 / 0.8669 / 0.8663 | 0.8731 | **0.0113** |
| rung 3, `box_provider(unet)` | 0.8145 / 0.8178 / 0.8098 | 0.8140 | **0.0040** |
| `mask_margin` m020 | 0.8753 / 0.8744 / 0.8738 | 0.8745 | **0.0008** |

**The ordering is the wrong way round.** Rung 2 is the *most* controlled condition in the ladder —
perfect boxes, no localiser error — and it has the **widest** spread, nearly three times rung 3's,
which adds a U-Net's localisation error on top of the same crop machinery. A condition that removes a
source of variance should not be noisier than the condition that keeps it. This is recorded as an open
observation, not explained: nothing in the record accounts for it, and inventing a mechanism after the
fact is exactly what this document exists to prevent.

**How the ceiling is cited, fixed here.** The headline figure is the **multi-seed mean** — three seeds
now, five once §26(b) lands. **Seed 28 is retained as the pre-registered single-seed test** and is
quoted beside it, because it is the run every earlier ceiling claim was made on. **Both are stated
wherever the ceiling is cited**, so a reader can see that the single-seed value (0.8861) sits 0.013
above the three-seed mean (0.8731) and know which one any downstream claim rests on.

**What §26(b) will settle, and it bears on the bar directly.** If five seeds narrow the SD toward
**0.004** — rung 3's figure — then the three-seed 0.0113 was a **small-sample artefact**, not a
property of the oracle-box route, and the bar computed from it (0.9069) was inflated by the estimate
rather than by the condition. If the SD stays near 0.0113 it is a route property and the wide bar is
real. **The bar is computed from this number**, so the distinction decides whether §26 can conclude
anything at all, and it is stated in the report either way.

### Why rung 2, and why now

**Rung 2 is the ceiling that decides whether any localiser gain can reach the target.** Rung 3 — the
deployable system — is rung 2 minus the localiser's cost, currently 0.059–0.072. If the ceiling sits
where it does now, no realisable localiser improvement gets the system to 0.90, and the localiser
thread has been optimising against a bound nobody has checked.

**And every knob in rung 2 was fixed on the defective Metal ledger.** tensorflow-metal 1.2.0 does not
apply ReLU in the compiled graph (§9), so the margin, the patience and the architecture choice were all
selected against numbers the CUDA ledger later replaced.

All of it runs at **rung 2, oracle boxes, `box_provider` route, on CUDA, from exported crop stores**,
about 100 s per run.

### (b) FIRST: seeds 3 and 4, and the bar recomputed from five seeds

**Two runs, and they come before anything else in §26.** The decision rule's threshold is
`max(0.01, 3 × pooled seed SD)`, and on three seeds that SD is **0.0113**, driven by a single run:
seed 28 at 0.8861 against 0.8669 and 0.8663. Three seeds cannot say whether 28 is an outlier or the
other two are, and **fixing a threshold on a three-seed SD that one run inflates is the same error
this document exists to prevent, in miniature**.

So seeds **3 and 4** run first at `box_provider` oracle, m = 0.20, and **the bar is recomputed from all
five seeds before the rule is applied to anything**. Reported with it: whether seed 28 is an outlier —
its distance from the five-seed mean in units of the five-seed SD, stated whichever way it falls.

`ens5` is built from the five prediction tables with `src.ensemble` (`--method mean --splits val`), as
`ens3` was. `ens3` gained **+0.006** over its members at fusion; this measures what the ensemble is
worth at the oracle condition, where there is no localiser error for it to average over.

### (a) The margin sweep on the operational route — NOT a re-examination of C1

**It runs on `box_provider(oracle)`, because that is the route the cascade's rung 3 consumes.** It is
therefore **not** a point-for-point re-examination of the C1 sweep, and no comparison of that kind is
claimed: C1 was a different crop route. **This is the first margin sweep ever run on the operational
route.**

**C1, recorded for what it was** — validation-selected, single seed, Metal, `mask_margin`, six margins,
`outputs/results/margin_sweep.csv`, best validation AUC per margin:

| m | tag | val AUC (best epoch) |
| ---: | :--- | ---: |
| 0.00 | `vgg16_crop_m000` | 0.8446 |
| 0.10 | `vgg16_crop_m010` | 0.8523 |
| **0.20** | `vgg16_crop_m020` | **0.8837** |
| 0.40 | `vgg16_crop_m040` | 0.8711 |
| 0.80 | `vgg16_crop_m080` | 0.8568 |
| 1.00 | `vgg16_crop_m100` | 0.8278 |

m = 0.20 won by **0.013** over m = 0.40, single seed, on a scoring path since found defective.

**Design: m = 0.10 / 0.20 / 0.30 / 0.40, three seeds each, input fixed at 224.** Three seeds because
the route's spread is what made the three-seed bar unusable; a single-seed margin curve on this route
would be uninterpretable for the same reason.

**0.80 and 1.00 are dropped, and the reason is measured, not aesthetic:** at those margins **20.9 %**
and **25.8 %** of crops are clamped at the image border and never achieve the intended geometry (median
crop long side 581 and 646 px against 388 at m = 0.20; `artifacts/lesion_geometry.csv`). Those two
points measure clamping, not margin.

**No resolution-matched arm, and the rejection is recorded rather than left implicit.** At a fixed
224 input, **margin and lesion-pixels-on-target are the same knob**: the crop is `box × (1+m)` original
pixels resized to 224, so the lesion occupies `224/(1+m)` pixels — 224 at m = 0, 187 at m = 0.20, 160
at m = 0.40. **A margin winner is therefore not attributable to context versus resolution**, and that
limitation is stated in the report. The obvious fix — vary input size to hold lesion pixels constant —
was considered and **rejected**, because changing the input size moves the ImageNet-pretrained backbone
off its operating point: `vgg16_crop_m020_448_cuda` scores **0.8155** against the same geometry at 224
scoring **0.8745**, a loss of **0.059** from resolution alone. That swaps one confound for a larger one
rather than attributing anything.

**The reading, fixed in advance.** A margin beating m = 0.20 by more than the recomputed bar becomes
the classifier for the cascade's rung 3 and the fusion rerun, and the report then states plainly that
**the operational route's margin had never been selected on that route**. Nothing separating means
**m = 0.20 stands as the pre-registered value, and the flatness is itself the result**: a
hyperparameter selected inside its own noise floor, on a different route, on a defective ledger, that
turns out not to matter on the route that counts. **Any margin winner is also run at rung 3**, because
a margin that helps with a perfect box need not help with a predicted one.

### (c) Patience 50, and (d) the four-backbone oracle ensemble — unchanged

**(c)** Patience 50, seed 28, against the patience-25 result. **Single-seed and descriptive**, as
declared. Patience 25 gave **+0.011** over 10 (S2), and run 4's recipe consolidates late, so the
schedule may still be truncating.

**(d)** The four-backbone ensemble at the rung-2 oracle condition. The rung-3 version was **0.835**;
the oracle version has never been measured. **Single-seed and descriptive**, as declared. One store
serves all four backbones: crops are held in [0,1] and normalisation is applied per model at batch
time, so this adds no payload.

### The decision rule

**A configuration becomes the classifier for the cascade's rung 3 and the fusion rerun only if it
beats the m = 0.20 five-seed mean by more than `max(0.01, 3 × pooled five-seed SD)`. Otherwise the
current configuration stands.** Unchanged in form; the only change is that the SD comes from **five**
seeds, computed before the rule is applied to anything.

On the three seeds available when §26 was declared the bar was **0.8731 + 0.0338 = 0.9069**, above the
project's own 0.90 target — recorded there as a consequence, and the reason (b) now runs first. The
five-seed bar replaces it and is stated before any configuration is compared against it.

**Reported either way**: every configuration's exact AUC and three-seed mean, the five-seed spread,
`ens5`, the four-backbone oracle ensemble, and the outlier check on seed 28 — each against the
five-seed mean and the recomputed bar, whether or not anything clears it.

### Where it runs, and what it must not delay

Session **S8**. §26 needs only oracle stores and is independent of the promoted boxes, so it does not
wait on the localiser set — but it must not hold up **Part 4, the fusion rerun on the promoted boxes**,
which is the remaining item with real upside. **If the combined localiser gate and the cascade are
ready before §26 has run, §26 goes second.** The decision is computed on the **laptop** from the
prediction tables (`decide_ceiling.py`), and only then does the cascade's rung 3 run.


### §26 results — nothing clears the bar, and the incumbent stands

Fifteen runs on pod B (RTX 4090), scored on the laptop through the **eager** path
(`score_store.py`, `model(x, training=False)`) into `outputs/_cuda/s8_results/` — never into
`outputs/results/`, whose filenames these share. Decision by `decide_ceiling.py`.

#### The bar, and what five seeds settled

| seed | 28 | 1 | 2 | **3** | **4** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| exact val AUC | 0.8861 | 0.8669 | 0.8663 | **0.8751** | **0.8679** |

mean **0.8725**, SD **0.0084** → bar = mean + max(0.01, 3×SD) = **0.8977**.

**1. The rung-2 ceiling is 0.8725, the five-seed mean — never seed 28 alone.** Seed 28's 0.8861 is
**+0.0136 above** the ceiling and is the value every earlier ceiling claim rested on. It is **not an
outlier** (+1.62 SD from the five-seed mean), which is precisely why it must not be quoted alone: it
is an ordinary draw from a wide distribution, and quoting the highest of five ordinary draws is
selection. Wherever the ceiling appears, **0.8725 (five seeds) is the figure**, with seed 28 shown
beside it as the pre-registered single-seed test.

**2. The noise inversion is now a five-seed result, and remains unexplained.** All vgg16,
patience 25, CUDA, exact validation AUC:

| condition | seeds | mean | **SD** |
| :--- | ---: | ---: | ---: |
| rung 2, `box_provider(oracle)` | 5 | 0.8725 | **0.0084** |
| rung 3, `box_provider(unet)` | 3 | 0.8140 | **0.0040** |

Five seeds narrowed the rung-2 SD from 0.0113 but left it at **twice rung 3's**. So it is a property
of the condition, not a small-sample artefact — and the ordering is still the wrong way round: the
*most* controlled condition in the ladder, with perfect boxes and no localiser error, is twice as
noisy as the condition that adds a U-Net's localisation error on top of the same crop machinery.
Nothing in the record accounts for it. **The practical consequence is recorded as a rule: every
rung-2 claim needs multi-seed backing.** A single-seed rung-2 number carries a ±0.025 three-sigma
band and cannot support a comparison of the size this project makes.

**3. Architectures that classify well and localise well are not the same set.** At the rung-2 oracle
condition, single seed:

| backbone | exact val AUC |
| :--- | ---: |
| vgg16 | 0.8731 (3-seed) / 0.8861 (seed 28) |
| efficientnet | 0.8694 |
| densenet121 | 0.8447 |
| **resnet50** | **0.7829** |

**resnet50 collapses at rung 2** — 0.09 below vgg16 — while **resnet34 is the best localiser encoder
the project has found** (L7a, gate (a) +0.0744, the only interval excluding zero against run 4).
Residual architectures are the strongest at proposing the box and among the weakest at classifying
the crop. Nothing in the two-stage design requires one network to do both, and this is the evidence
that it should not be assumed.

#### The margin sweep, and the route transfer resolved

| margin | 3-seed mean |
| :--- | ---: |
| 0.10 | 0.8575 |
| **0.20 (incumbent)** | **0.8731** |
| 0.30 | 0.8621 |
| 0.40 | 0.8663 |

**m = 0.20 wins on the operational route**, so the transfer §26 flagged — selected on `mask_margin`,
single seed, on the defective Metal ledger, then applied to `box_provider` without re-selection —
was **benign**. That resolves the finding rather than leaving it open. All four margins lie within
0.016 and far inside the bar, so the pre-registered reading applies: a hyperparameter selected inside
its own noise floor that turns out not to matter on the route that counts.

**§26(c)** patience 50: **0.8665**, below patience 25's 0.8731 — the schedule was not truncating.
**§26(d)** ensembles: four-backbone **0.8879**, `ens5` **0.8815**. Both descriptive, neither clears
0.8977.

**Decision: nothing clears the bar. The classifier for the cascade's rung 3 and the fusion rerun
stays vgg16, margin 0.20, patience 25** — now measured on CUDA rather than assumed from Metal, which
is what Part 2 of the forward plan was for.

**One implementation note.** `src.ensemble` resolves its inputs from `outputs/results/` regardless of
where the tables are, so it could only have built these by first copying tables into the reference
ledger's own folder — which is exactly what must not happen. The ensembles are computed in place, and
that arithmetic is checked against `src.ensemble` on the existing rung-3 `ens3`: **max |diff|
1.11e-16**.

### §26a — How the one-session re-measurement will be read (declared 17 Sep, before any of it runs)

**Why this exists.** The 17 Sep audit found that every classifier row except rung 3 and the promoted
fusion was scored from 7–8 Sep laptop weights trained on tensorflow-metal, and that the CUDA rows came
from three different sessions. All 59 rows are re-measured in one CUDA session (`_cuda0917`: one card,
TF 2.17, patience 25, seeds 28/1/2, plus 3/4 for rung 2). This block fixes, **before any number
exists**, what a moved number does to each conclusion that rests on it. Nothing in the session selects
anything: every figure it produces is a re-measurement.

**The shift scale, declared.** **δ = 0.03 per row.** Two measured effects set it: the compiled-path
AUC error on tensorflow-metal reached **0.031** (session state 9 Sep c), and patience 10 → 25 moved rung
2 by **+0.0107** (§10). So a contrast between two re-measured rows can move by up to **2δ = 0.06**. A
contrast is **robust** if its current |dAUC| is **≥ 0.10**; everything smaller is **restated from the
session**, with nothing carried over.

**§14a — the recovery figure (80.4 %, residual 1.36 × the ceiling's seed SD).**
- Recomputed **only** from session numbers: rung 3 on A24 ×3, rung 3 on the old boxes ×3, rung 2 ×5.
  The session figure **replaces** 80.4 % whatever it is.
- "Most of the localisation cost is recovered" is kept **iff** the fraction is **≥ 50 %**.
- The residual-to-SD ratio is stated plainly. No "within seed noise" wording unless it is **≤ 1.0**.
- If the cost (ceiling − old-box rung 3) is **< 2 ×** the larger of the two seed SDs, the fraction is
  declared **uninterpretable** and not quoted.

**§26 — the ceiling, the bar and the margin conclusion.** The session re-measures the bar's inputs
(rung 2, five seeds) but **not** the challengers it was compared against: margins 0.10/0.30/0.40,
patience 50 and the backbone variants were measured on 10 Sep (S8). The decision rule therefore
**cannot be re-applied like-for-like and is not re-applied**: **m = 0.20 stands as decided**. The
re-measured mean, SD and bar are reported beside the 10 Sep values (0.8725, 0.0084, 0.8977).
**§32 carries them: 0.873473, 0.003726, 0.884652.** The best
challenger today is 0.8694, a gap of 0.0283. A 10 Sep challenger that would clear the re-measured bar
is reported as a **cross-session observation, not a decision**.

**The ladder (validation, descriptive; the inferential test is on test).** Current |dAUC| from the
12 Sep dry run (Metal neighbours) and the 12 Sep cascade (9 Sep CUDA neighbours):
- **Robust (≥ 0.10), expected to keep sign:** rung 2 vs rungs 4/6/7 (+0.151 to +0.336); rung 3 vs
  rungs 4/6/7 (+0.141 to +0.342); rung 4 vs 6 (+0.166 to +0.185); rung 6 vs 7 (−0.146 to −0.158).
  **A sign change in any of these is reported as a finding that invalidates δ**, not smoothed over.
- **Restated from the session:** rung 2 vs rung 3 (−0.0056 / +0.0197, which already differ in sign
  across the two ledgers) and rung 4 vs rung 7 (+0.020 / +0.027).

**Family A (a pre-registered Holm family).** vgg16 vs resnet50 (+0.110) is robust.
**§32 re-measures it at +0.0810, below this block's own 0.10 bar, so it is restated rather than robust.** The other five
contrasts (|dAUC| 0.007–0.065 on validation) are restated, and the Metal-era ranking "EfficientNet ≈
DenseNet" is **not carried over**: §26's CUDA backbone table already orders them differently.

**The floor gate is unchanged at 0.794022.** If the session's rung-3 three-seed mean falls below it,
the pointer is **not** rewritten and the result comes back. The promotion's box-IoU decision stands,
but an unmet §7b condition is reported, **not overridden**.

**What does not move.** The pre-registered test-pass analyses, the primary endpoint's definition, the
Holm families and their membership, and every threshold above are fixed by this block. The shifted
figures in §7b, §13c and §14a are corrected in an erratum block **after** the session, citing this
block for how each is read.

## 34. §27 — The test-pass fusion half and A5, specified (rulings of 11 Sep)

*Why here: Rulings of 11 Sep, taken **after** the validation fusion results were known and disclosed as such. Last of the substantive blocks because it specifies what the pass runs, and it follows §13/§13a/§13b/§13c (whose bar and mechanism it cites) and §24a (whose `_promoted` tags it names).*

**Order of events, and it is the whole disclosure.** `run_test_pass.sh`'s fusion section was found to
be a **commented-out placeholder**, and three further gaps against the frozen text were found with
it, **after the validation fusion results on A24's boxes were known** (§13c RESULT: learned fusion
0.9261 against naive averaging 0.9342, −0.0080). Everything below is therefore decided in knowledge
of a result it affects. It is written as that, not as continuity.

**What the frozen text already specifies**, quoted so the gap is visible:

> Two paired DeLong tests, two-sided, α = 0.05, each uncorrected because each is a single
> pre-declared contrast on its own row set: context fusion vs `vgg16_fusion_control_all` on the 243
> test lesions, and **view fusion vs `vgg16_fusion_control` on the 101 test pairs (label of crop A)**.
> Neither is added to a family. A declared fusion mode (per the val rules above) is **also evaluated
> at image level against the BI-RADS anchor descriptively** (with its bootstrap CI); the primary
> endpoint remains rung 3 and is not re-assigned after the pass.

The script ran **none** of it.

### 27.1 Fusion comparators: all three

**`vgg16_fusion_control` runs because the frozen text names it.** That it is a single-crop control on
**oracle** crops, and therefore not like-for-like with the U-Net re-run, **is the report's problem to
explain, not the pass's to fix.** A declared contrast is run as declared.

**`rung3_on_pairs` and naive averaging are added**, and the addition is disclosed: it was **made after
the validation fusion results were known**. §13a is cited because it is what makes this legitimate —
§13a fixed averaging as the bar (*"The bar for the declared fusion system is therefore 0.8739, not
0.8406"*) and required that *"Every fusion comparison in the report uses a control computed on the
same pairs"*, **before `vgg16_fusion_view_unet` was trained**. So this **closes a gap between two
declared documents rather than inventing a comparator to suit a result.** §13a carried no instruction
to amend the test-pass section; that omission is what is corrected here.

**Tags** are the `_promoted` ones, and **`FUSION=view` is resolved explicitly** per §13, which
declared view and left `FUSION=context # or view` unresolved in the script.

### 27.2 Image-level fusion descriptive: yes

As the frozen text specifies: the declared fusion mode is evaluated at image level against the
BI-RADS anchor, **descriptively, with its bootstrap CI**, and **the primary endpoint is not
re-assigned after the pass** — it remains rung 3 image-level on the comparison set.

### 27.3 Ensemble: in

A5's condition fired — **three seeds of the declared systems exist** — and the validation record
reports ens3 alongside the three-seed means throughout.

**A correction to the reasoning that first proposed this, recorded because it changes what the
decision is.** The primary endpoint quotes the **single declared model, not ens3**, so this is *not* a
mismatch between the record and the pass being repaired. It is **a declared conditional being
honoured at no cost**: ens3 is an averaging of prediction tables and requires **no training at test**.

**The seed replicates are scored on test, and that expands test contact beyond the tags the frozen
text lists. Stated here rather than left as an implementation detail.**

A5 requires two things of the extra seeds: *"the seed spread is reported as the variance estimate the
ladder figure lacks"* and *"the mean-probability ensemble (`src.ensemble`) … evaluated once on test"*.
**Neither is computable without scoring the replicates on test.** A seed spread is the spread of the
seeds' own test AUCs; an ensemble of prediction tables cannot be averaged from tables that were never
written. So scoring `_s1` and `_s2` **follows from the declaration** rather than being added to it.

But the frozen text's "## Models entering the pass" section lists the single declared models, not
their replicates, and A5 nowhere says the word "evaluate" about the seeds themselves. **The
inference is therefore made explicit here**: four additional models touch the test partition — two
rung-3 seed replicates and two view-fusion seed replicates — and `run_test_pass.sh` gains the
`src.evaluate` and `src.fusion --seed` calls that score them.

> **Restated by §37:** the pass scores eight seed replicates on test.

**What this does and does not license.** It licenses scoring exactly the replicates of the systems A5
names, once, to produce the spread and the ensemble A5 requires. It licenses nothing else touching
test: no further seeds, no re-selection among them, and the ensembles **replace no primary model and
enter no Holm family**, as A5 already states. The primary endpoint remains the single declared
model.

### 27.4 TTA: formally omitted

**Decided now and recorded now**, which is what A5 requires: *"Any ensemble/TTA lines are appended to
the runbook before the freeze, or omitted; none are run ad hoc afterwards."*

**No TTA variant has been trained or evaluated on any ledger in this project**, and the irreversible
pass is the wrong place for an untested condition. Omission is permitted provided it is decided
before the freeze; it is, here.

> **Corrected by §38:** TTA was evaluated once, on validation, on 8 September, and was negative; the omission stands.

### 27.5 The audit that produced this

Every stage the pre-registration says enters the pass, against what the script ran before this
amendment:

| declared | ran? |
| :--- | :--- |
| Primary endpoint, Family A, Family B, ladder figure | yes |
| Calibration, PPV/NPV, decision curve, subgroups | yes |
| Grad-CAM grounding, MC-Dropout deferral | yes |
| **Fusion — view vs `vgg16_fusion_control`** | **no — commented out** |
| **Fusion — image-level vs anchor, descriptive** | **no — commented out** |
| **Ensemble (`src.ensemble`) on test** | **no — zero occurrences** |
| **TTA variant (`evaluate --tta`)** | **no — zero occurrences** |

Three of the four gaps are now closed by running the work; the fourth is closed by deciding not to.
The script is amended to match, and **the `--split val` dry run is re-run end to end afterwards**,
because the pass has changed shape and the previous dry run validated a different script.

## 35. §28 — The tag table as the promotion leaves it (executes §6; decides nothing)

*Why here: Executes §6's substitution rule and writes the resulting rows, so the tag table — §5's authority on what the pass evaluates — names the tags the pass scores. After §27, whose `_promoted` tags it writes down, and after §26a, whose ledger suffix every row carries. Found by validating the model-list guard against a dry-run rendering **before** `--apply`, which is the only point at which the table could still be fixed.*

**Written 18 September, before the freeze, after a machine check found the gap.** §6 fixed the
substitution rule in prose — *"Every `_bundle` above becomes the new suffix, `$BOXES` the new folder"*
— and §27.1 states *"**Tags** are the `_promoted` ones"*. Neither writes the resulting rows down. The
tag table is what §5 calls the authority on what the test pass evaluates, so the authority still named
the pre-promotion system while the pass would score the promoted one.

**How it was found, and why that order matters.** `notebooks/scripts/assert_model_list.py` (binding-rules
audit §4.5 #4) matches the pass's own model list against this document's tag table in both directions. It
was validated against a **dry-run rendering** of the applied pre-registration — the text
`apply_amendments.py` produces without `--apply` — precisely so a gap could still be closed: after
`--apply` the protocol is fixed and cannot be corrected. The check failed there, naming six tags
(`vgg16_crop_rung3_unet_promoted_cuda0917` and its two seeds, `vgg16_fusion_view_unet_promoted_cuda0917`
and its two seeds) that no row covers, and one row, `vgg16_crop_rung3_unet_bundle`, that nothing scores.

**This block invents nothing.** It applies §6's declared substitution to §5's rows and writes the result,
so that a checker can verify the claim instead of a reader inferring it. The promotion itself was decided
by the combined gate on validation box IoU (§7b, §24a) and recorded in `refs/promoted_pointer.json`
(cfg 54485, `run1+run3+run4+l7v+l7a+l8a+l9`); the measurement ledger is `_cuda0917` (§26a). Nothing here
re-opens either.

**The substitution, row by row.** Every tag the pass scores carries `config.LEDGER_SUFFIX = _cuda0917`,
except the context-fusion pair, which was re-measured in its own session and carries `_cuda0918`
(§37, added 18 Sep); rungs 3 and 5, and the U-Net-box fusion, additionally carry the promotion suffix
`_promoted`, which is what `refs/promoted_pointer.json` records as `rung3_suffix`.

| pre-registered tag | tag as measured |
|---|---|
| vgg16_crop_rung3_unet_bundle | vgg16_crop_rung3_unet_promoted_cuda0917 |
| vgg16_crop_rung3_unet_bundle_s1 | vgg16_crop_rung3_unet_promoted_cuda0917_s1 |
| vgg16_crop_rung3_unet_bundle_s2 | vgg16_crop_rung3_unet_promoted_cuda0917_s2 |
| vgg16_crop_rung3_unet_bundle_ens3 | vgg16_crop_rung3_unet_promoted_cuda0917_ens3 |
| vgg16_crop_rung1_oracle | vgg16_crop_rung1_oracle_cuda0917 |
| vgg16_crop_rung2_oracle | vgg16_crop_rung2_oracle_cuda0917 |
| vgg16_crop_rung4_cam | vgg16_crop_rung4_cam_cuda0917 |
| vgg16_crop_rung5_jitter | vgg16_crop_rung5_jitter_promoted_cuda0917 |
| vgg16_crop_rung5_jitter_bundle_s1 | vgg16_crop_rung5_jitter_promoted_cuda0917_s1 |
| vgg16_crop_rung5_jitter_bundle_s2 | vgg16_crop_rung5_jitter_promoted_cuda0917_s2 |
| vgg16_crop_rung6_shuffled | vgg16_crop_rung6_shuffled_cuda0917 |
| vgg16_crop_rung7_whole | vgg16_crop_rung7_whole_cuda0917 |
| vgg16_crop_m020 | vgg16_crop_m020_cuda0917 |
| resnet50_crop_m020 | resnet50_crop_m020_cuda0917 |
| densenet121_crop_m020 | densenet121_crop_m020_cuda0917 |
| efficientnet_crop_m020 | efficientnet_crop_m020_cuda0917 |
| vgg16_full | vgg16_full_cuda0917 |
| baseline_crop_m020 | baseline_crop_m020_cuda0917 |
| scaled_crop_m020 | scaled_crop_m020_cuda0917 |
| regularised_crop_m020 | regularised_crop_m020_cuda0917 |
| vgg16_fusion_view_unet_bundle | vgg16_fusion_view_unet_promoted_cuda0917 |
| vgg16_fusion_view | vgg16_fusion_view_cuda0917 |
| vgg16_fusion_control | vgg16_fusion_control_cuda0917 |
| vgg16_fusion_context | vgg16_fusion_context_cuda0918 |
| vgg16_fusion_control_all | vgg16_fusion_control_all_cuda0918 |
| resnet50_crop_rung3_unet_bundle | not scored |
| densenet121_crop_rung3_unet_bundle | not scored |
| efficientnet_crop_rung3_unet_bundle | not scored |
| rung3_arch_ens4 | not scored |
| resnet50/densenet121/efficientnet_crop_rung3_unet_bundle | not scored |

Seed replicates and the mean-probability ensemble of a measured tag carry `_s1`, `_s2` and `_ens3` on
the measured name, as §5 already declares for the pre-promotion ones.

`not scored` in the right-hand column is a statement about this pass, not a deletion: the tag stays in
the record with the numbers it has, and the pass does not evaluate it. The four such rows are the
architecture family at the rung-3 condition, for the reason below.

**Two rows of §5 do not carry over, and say why here (the reason is §26a's, not this block's).** `resnet50/densenet121/efficientnet_crop_rung3_unet_bundle`
and `rung3_arch_ens4` were the architecture family **at the rung-3 condition**. §26a chose to
re-measure Family A at the oracle `m020` condition **only**, so those four tags have no `_cuda0917`
counterpart to score — the decision is §26a's and this block merely records its consequence for the
tag table, so a reader need not join two blocks to see why a declared row is not evaluated. They stay in the record as measured on the
superseded ledger.

## 36. §29 — The compiled-path defect does not reach the localiser (measured 18 Sep, retires an open limitation)

*Why here: Retires an open limitation of §9 and narrows §26a's δ to the classifier rows, so it precedes the erratum that uses both.*

**What was open.** §9 records that tensorflow-metal 1.2.0 drops ReLU in the compiled graph, and the
project's answer was to score everything through `model(x, training=False)`. `localiser_bundle cache`
was still calling `model.predict` until **17 September**, so the stored validation maps of run1, run3
and run4 — written 8 September, and the maps the promoted ensemble was selected from — came from the
path that was replaced. The record answered this for the two localisers in use on 9 September by
direct measurement and never for the rest, which left the promotion resting on maps nobody had checked
against the current code.

**Measured, not argued.** The three Keras members were re-cached through the current eager path **on
the same device** (this laptop's Metal GPU), so the code path is the only variable, and compared at
three levels:

| member | max abs map difference | boxes moved | mean box IoU, stored -> eager |
| :--- | ---: | ---: | :--- |
| run1 | **0 / 255** | **0 / 243** | 0.393297 -> 0.393297 |
| run3 | **0 / 255** | **0 / 243** | 0.399172 -> 0.399172 |
| run4 | **0 / 255** | **0 / 243** | 0.441676 -> 0.441676 |

And at the level the promotion was decided on — the A24 group-weighted ensemble with all three
re-cached members substituted in: **box IoU 0.529952 -> 0.529952, 0 of 243 boxes move, maximum corner
shift 0 px.**

**The limitation is retired.** Not "within tolerance": bit-identical in the stored 8-bit units, for
every member, in both TTA orientations. The stored maps stand, no re-derivation follows, the
promotion's provenance is unaffected, and the report carries no caveat on this point. The reading was
declared before the number arrived (three outcomes: retire, disclose as a limitation, or stop and
decide), and the first applies.

**What this does to §26a's shift scale, and it is a strengthening.** §26a set **δ = 0.03 per row**
from two measured effects, one of which was the classifier's compiled-path AUC error of 0.031. That
error is a property of the **classification head**, and the measurement above shows the **localiser
path's error is exactly zero**. So δ governs the **classifier rows only**; the localiser's maps and
boxes carry no compiled-path uncertainty at all. Every §26a reading that used δ on a classifier
contrast is unchanged; nothing in the localiser record needs the allowance it was given.

`notebooks/scripts/eager_vs_compiled_maps.py` -> `outputs/results/eager_vs_compiled_maps.json`.

## 37. §30 — §12, §12a and §12b re-measured on the promoted localiser (18 Sep)

*Why here: Follows §12c, which it revives with a qualification, and follows §24a, whose ensemble it is measured on.*

**Why re-run, and when that was decided.** §12's uncertainty analysis, §12a's deferral curve and
§12b's mechanism reading were all measured on **E3**, the superseded three-run ensemble, with equal
member weights. The promoted localiser is **A24**: seven members, group-weighted, and a different
failure profile — 54 miss-inclusive failures of 243 against E3's 68. There is no basis for assuming a
disagreement signal measured on three equal members transfers to seven weighted ones, so the re-run
was ordered **because the promoted localiser changed** — before any of the new numbers existed, and
in particular before anyone knew §12b would flip.

| measure | recorded (E3) | **A24** | verdict |
| :--- | :--- | :--- | :--- |
| localisation failures | 68 / 243 | **54 / 243** | fewer, as A24's lower fallback rate predicts |
| box disagreement AUROC | 0.692 [0.5652, 0.7913] | **0.7363 [0.6391, 0.8155]** | **holds and strengthens** |
| map disagreement AUROC | 0.5703 [0.4903, 0.6691] | **0.6070 [0.5014, 0.7052]** | **a null that no longer cleanly excludes an effect** — the bound clears 0.5 by 0.0014 at n = 243, which is not a positive result |
| deferral: failure rate | outside the random band from 0.10 | **from 0.05** | holds |
| deferral: mean box IoU | from 0.05 | **from 0.05** | holds |
| deferral: classifier AUC | **NONE** | outside at **one rate of ten** (0.25) | **consistent with the recorded null**; one rate is noise, not a partial change |
| §12b composition | NOT SUPPORTED | **SUPPORTED** — deferred AUC 0.7476 vs retained 0.8908, score gap 0.2448 vs 0.4834 | **revived on the promoted system**, with the qualification below |

**§12b is revived, not quietly restored.** §12c corrected §12b as "too strong" on E3, and that
correction stays in the record and stays visible: it was right about E3. On A24 the deferred set does
separate worse, which is what §12b predicted, and §12c's own decomposition on the new ledger
independently agrees — **failures n = 54, AUC 0.7536 [0.5904, 0.8803]; successes n = 189, AUC 0.8788
[0.8277, 0.9272]**.

**The flip is partly compositional, and the cut that shows it.** At the declared 25 % rate both
systems defer 61 lesions, but **39 % of them differ**: overlap 37, **Jaccard 0.44**, 24 in and 24 out.
What does *not* differ is the label mix — deferred prevalence **0.328 on both sides**, retained
**0.505 on both**, failure-capture rate 33/68 (49 %) against 26/54 (48 %), and even the swapped 24
lesions match each other at prevalence 0.375 with 9 of 24 failures each. So prevalence explains none
of it: it is **a substantially different set of lesions with an identical label mix, which separates
worse under the promoted system**. That cannot be attributed to the mechanism alone. One further
caveat, stated because it limits the cut: the rung-3 classifier changed too (`_cuda0917`), so this
isolates composition rather than the classifier.

Products: `uncertainty_val_A24.{csv,json}`, `deferral_curve_val_A24.{csv,json}`, beside the E3 files,
which are kept.

## 38. §31 — §23.1's correlation table over the promoted set (18 Sep)

*Why here: Extends §23.1 from five members to seven; §20a.1 remains the test it corroborates.*

**What the recorded figure covers.** §23.1's table was computed on **five of the seven promoted
members** — run1, run3, run4, l7v, l7a, plus E3 as the reference — because l8a was still training and
l9 had not landed. Its torch-vs-Keras separation therefore rested on a **single torch pair**, which is
thin support for a claim about a seven-member ensemble. Run 2 replicates were never part of §23.1:
they are §20a.1's subject.

**The promoted set, all seven.** Declared as a measurement of the promoted ensemble and reported
**beside** the recorded five-member snapshot rather than replacing it.

| | recorded (5 members) | **promoted set (7 members)** |
| :--- | ---: | ---: |
| Keras pairwise mean Pearson | 0.646 | **0.646** |
| torch vs Keras mean Pearson | 0.525 | **0.538** |
| torch pairwise mean Pearson | — | **0.608** (6 pairs) |

Per-member mean box IoU: run1 0.3853, run3 0.3874, run4 0.4190, l7v 0.4349, l7a 0.4934, l8a 0.4808,
l9 0.4802.

**The separation holds across all four torch members and widens slightly**, so §23.1's mechanism
evidence is **strengthened**: it now rests on four torch members against three Keras ones rather than
on one pair. Wherever §23.1's correlations are quoted, that is the figure to quote and that is the
sentence to attach. §20a.1 remains the test; §23.1 remains corroboration of it.

`outputs/results/member_correlation_promoted7/`, whose `summary.json` records the member set, each
table's digest and its row count, so the snapshot says what it was computed over.

## 39. §32 — Erratum: the figures §26a scheduled, measured on the one-session ledger (18 Sep)

*Why here: It corrects figures in §7b, §13c and §14a using §26a's declared readings, so every block it touches precedes it.*

§26a required that the shifted figures in §7b, §13c and §14a be corrected **after** the session,
citing it for how each is read. This is that block. Every number below comes from `_cuda0917`, scored
through the test pass's own route, and replaces the figure beside it.

**§14a and §13c's Prediction 1 — the recovery figure.** The record carried **two** different values
for the same quantity: §13c said 79.5 % with residual 1.07 x the ceiling's seed SD, §14a said 80.4 %
with 1.36 x, because they used three and five ceiling seeds respectively. **Both are replaced.**

| quantity | §13c | §14a | **session** |
| :--- | ---: | ---: | ---: |
| rung 3 on A24, three-seed mean | 0.861005 | — | **0.865390** |
| ceiling (rung 2) mean | 0.8731 (3 seeds) | 0.872478 (5) | **0.873473** (5) |
| ceiling seed SD | 0.0113 | 0.008405 | **0.003726** |
| localisation cost | 0.059092 | 0.058456 | **0.059019** |
| recovered | 79.5 % | 80.4 % | **86.3 %** |
| residual / ceiling SD | 1.07 x | 1.36 x | **2.17 x** |

"Most of the localisation cost is recovered" **holds** (the §26a threshold was 50 %). "Within seed
noise" is **not** permitted: §26a allowed it only at a ratio of 1.0 or below, and the ratio rose to
2.17 because the denominator fell faster than the numerator — the residual itself shrank from 0.011473
to 0.008083. **The rung-2 seed-spread anomaly is answered by the same fact**: the five-seed SD fell
from 0.008405 to 0.003726 with route, store, margin and seed set unchanged, so the earlier spread was
a **cross-session artefact** — three sessions, different cards and dates — and not a property of the
box_provider(oracle) route.

**§13c's Prediction 2 — the fusion table.** Re-measured on the same 100 validation pairs:

| | §13c (old ledger) | **session** |
| :--- | ---: | ---: |
| one view, rung 3 | 0.9073 | **0.9077** |
| naive two-view averaging | 0.9342 | **0.9265** |
| learned view fusion (ens3) | 0.9261 | **0.9177** |
| fusion − averaging | −0.0080 [−0.0354, +0.0169] | **−0.0088 [−0.0455, +0.0177]**, DeLong p 0.563 |
| fusion − one view | +0.0189 [−0.0282, +0.0666] | **+0.0100 [−0.0391, +0.0570]**, DeLong p 0.670 |
| oracle-box − U-Net-box fusion | — | **−0.0044 [−0.0588, +0.0647]** |

Prediction 2's reading is unchanged: on well-localised pairs there is nothing for a combiner to do.

**The ladder.** All eight of §26a's robust contrasts kept their sign, so δ is not invalidated: rung 2
− rungs 4/6/7 **+0.1489 / +0.3095 / +0.1518**; rung 3 − rungs 4/6/7 **+0.1408 / +0.3014 / +0.1437**;
rung 4 − 6 **+0.1606**; rung 6 − 7 **−0.1577**. The two restated contrasts: rung 4 − rung 7 **+0.0029**,
and **rung 2 − rung 3 +0.0097, 95 % CI [−0.0268, +0.0500], DeLong p 0.599** (+0.0081 on three-seed
means).

**How rung 2 − rung 3 is to be stated, because it is the one most easily overstated.** The point
estimate fell from +0.059 to +0.010 and the contrast no longer excludes zero, **consistent with the
localiser having closed most of the gap, with the data unable to rule out that some remains**: at
n = 243 the interval still admits a real gap of 0.05, which is most of the original 0.059. This is a
**failure to distinguish, not a demonstration of equivalence**. "Indistinguishable from ground-truth
boxes" is accurate; "as good as ground truth" is not.

**The mechanism section's subject moves with it.** The victims cut and §12c were written when the
rung 2 − rung 3 gap was ~0.07. On the session's numbers the gap the mechanism explains is **rung 3 −
rung 5, +0.1035 [+0.0462, +0.1636]**, and the cut is unambiguous: the oracle scores **0.8348 on A24's
victims (n = 62) against 0.8766 on the rest**, while on jitter's random victims it scores **0.9203
against 0.8655**. Selective error falls on intrinsically harder lesions; random error does not.

**Family A.** §26a declared vgg16 vs resnet50 robust at +0.110. Re-measured it is **+0.0810**, below
the block's own 0.10 bar, so it is **no longer a robust contrast** and joins the five restated ones.
The ordering is unchanged: vgg16 0.8777, efficientnet 0.8552, densenet121 0.8465, resnet50 0.7967.
This is §26a firing against a conclusion rather than for one.

**§26's bar.** Re-measured mean **0.873473**, SD **0.003726**, bar **0.884652**, beside 10 September's
0.8725 / 0.0084 / 0.8977. Per §26a the rule is **not re-applied** — the challengers were not
re-measured — so **m = 0.20 stands as decided**; the best 10 September challenger, 0.8694, does not
clear the re-measured bar either.

**§7b's floor gate, and the one cross-ledger comparison in it.** The gate held: three-seed mean
**0.865390** against a floor of **0.794022**, margin **+0.071368**. Stated plainly, because the two
sides come from different sessions: the reference **0.814022 is a three-seed mean from the old CUDA
ledger**, the candidate a three-seed mean from `_cuda0917`. Nothing turns on it at this margin, and
the one row present in both ledgers agrees to **4e-4** — old-box rung 3 re-measures at 0.814454
against the gate's 0.814022.

**The primary endpoint did not move.** Image-level rung 3 against the BI-RADS anchor on the 209-image
comparison set: **0.8574 vs 0.8627, −0.0052 [−0.0624, +0.0582], p 0.8529**, against the previously
recorded −0.0038 (p 0.89). This is validation; the pre-registered inference is on test.

## 40. §35 — The promotion cascade as it ran: the floor gate, and the two halves it did not retrain (rulings of 11–12 Sep; recorded 18 Sep)

*Why here: After §32, whose floor-gate figures it gives the rule for, and after §7b, §13c and §24a, whose four statements of the floor carry pointers here.*

**The floor gate, as ruled on 11 September at 14:02 UTC.** That was about two hours before the
rung-3 retrain on the promoted boxes produced a number (15:57–16:03 UTC). The gate is one-sided and
like-for-like: the rung-3 three-seed mean on the promoted boxes must not fall more than 0.02 below
the CUDA three-seed mean on the three-run boxes, **0.814022**, so the floor is **0.794022**. A higher
score passes. A three-seed mean is not compared with seed 28's 0.8145; that is the like-for-like
error §26 caught.

This replaces the floor as stated in four places: §13c's Prediction 1, §7b's gate (c) arm, point 5
of §7b's multi-passer rule, and §24a's result. Each reads "0.8145 … ±0.02" and carries a pointer
here. The cascade has enforced this definition since 11 September (`FLOOR_REF=0.814022`,
`FLOOR_DROP=0.02`), but it was not written into this document until now.

Results: 0.861005 on 12 September (margin +0.066983), and 0.865390 on the one-session ledger
(margin +0.071368, §32).

**The two halves the cascade did not retrain.** §7b's "One promotion cascade" retrains rung 3
(three seeds and `ens3`) and the view-fusion ensemble on the promoted boxes. Both were instead
trained on pods by the store trainers, `train_crop_store` and `train_fusion_store`, from stores
exported on the promoted boxes. The cascade ran with `SKIP_RUNG3_HALF=1` and `SKIP_FUSION_HALF=1`:
it asserted the stores' boxes and the runs' tables before continuing, and recorded both in the
pointer's `precomputed_halves` field, with each store, its boxes folder and each prediction table's
md5. The reasons, in the order they bind:

- **Rung 3.** (1) The cascade's step 2 calls `model.fit`, and training on the laptop GPU has been
  banned since 9 September because of the defect §9 documents. (2) It would overwrite the runs the floor gate reads. (3) It would
  bring a third trainer (`src.ladder` / `src.train`, the DICOM path) into the floor gate's own
  comparison, where the 0.814022 reference and the current runs both came from `train_crop_store`.
- **Fusion.** The cascade's step 5 trains with `src.fusion --stage train`, while the CUDA fusion
  ledger and these runs came from `train_fusion_store`. Running it would bring a third trainer into
  a like-for-like comparison.

Cost is not among the reasons. Both halves were later re-measured in the one-session ledger
(§26a), and the pointer records those runs.

## 41. §37 — The pass's test scoring, restated before the freeze (18 Sep)

*Why here: After §35. Restates §27.3's count of test-scored replicates, records the rung-6 donor change made before the freeze, names the three stages the pass lacked, and records the context-fusion re-measurement, and lists the defects fixed before the freeze.*

**Rung 6's donors for test lesions.** The shuffled rung gives each lesion the box of another lesion
of the same laterality and view (`src/boxes.py`). As first written, the donor permutation was drawn
over every lesion loaded. The pass loads train, val and test, so it would have re-drawn the donors
of about 220 of the 243 validation lesions and re-scored rung 6 on validation crops unlike those its
reference table and operating threshold came from. From 18 September, train/val and test lesions
are deranged separately, each with its own random stream. Every train/val donor is unchanged — all
1,453, verified byte-identical (sha256 `d227d267…`) — so rung 6's validation results stand. Test
lesions take donors only from test lesions; no test donor had been drawn before the change.

**Seed replicates scored on test.** §27.3 lists four additional models. The pass scores eight seed
replicates on test: the rung-3 and U-Net-box view-fusion replicates §27.3 names; the rung-5
replicates §2 declares; and the oracle-box view-fusion replicates, with their base run, so that
fusion (b)'s oracle comparator comes from the pass's own split (ruled 17 September, when fusion (b)
was found to read a validation table on the test path). Each is scored once, feeds the ensembles and
seed spreads A5 requires, replaces no primary model and enters no Holm family.

**§4's ladder error analysis.** §4 pre-registers `ladder_error_analysis.py` as run once on test,
unchanged. The pass did not run it until 18 September, when a pre-freeze review found it missing;
§27.5's audit had not listed it. It now runs after the ladder, with the tags §28 names and the
arguments of its validation run on the one-session ledger, which is where §32's mechanism figures
come from. Running it is compliance with §4, not an addition to it.

**The context-fusion test.** "Fusion (outside both families)" declares two paired DeLong tests on
test. The pass ran only the view one: §27 quoted both, §27.1 settled only the view half, and the
context test was dropped without a ruling; §27.5's audit did not list it. It was found on 18
September, in a read of the protocol against the appendix. The pass now runs it —
`vgg16_fusion_context` against `vgg16_fusion_control_all` on the 243 test lesions, two-sided,
α = 0.05, uncorrected, in no family — as compliance with the declaration, not an addition to it.

Neither model is in the one-session ledger, so both are re-measured in a separate session,
`_cuda0918`: one RTX 4090, the session's card; seed 28 and patience 25; the stores exported on 9
September for these two runs, in the same batch as the fusion stores the one-session ledger trained
from; and the session's code, whose md5s equal the 17 September session's once the edits logged since
are reverted (a docstring in each trainer, the guard in `data_loader.py`, the rung-6 derangement in
`boxes.py`, none of them on store training's path). The 9 September runs (`_cuda`) are not used: they
recorded no seed and no code md5, so their code cannot be verified. The tags carry `_cuda0918`, not
`_cuda0917`: the runs are not part of the 59-row session, and cross-session variation is what §32
found in rung 2's seed spread.

**Written before either run exists: the re-measured validation AUCs do not reopen the declaration.**
Context was not declared on 9 September, and whatever its re-measured validation gap, the rule is not
re-applied. The re-measurement exists only so that a pre-registered test runs on verified code.
**The pod's parity check gates both runs:** the pod scores `vgg16_crop_m020` on its store (runbook
§3), and if the result falls outside 0.8843 ± 0.001, neither run trains, the pod is terminated, and
the result is reported; the session driver refuses to train without a passing parity result. A
contrast measured on a pod that cannot reproduce the session's arithmetic would be worse than scoring
the 9 September pair.

**Measured after the pre-declaration above (18 Sep).** The pod's parity check scored 0.884269, equal to
the laptop's CPU reference, so both runs trained (`outputs/_cuda/s0918/`, every file verified by md5).
Scored on the laptop through the eager path, exact validation AUC on the 243 lesions is 0.876022 for
`vgg16_fusion_context_cuda0918` and 0.861164 for `vgg16_fusion_control_all_cuda0918`, a gap of
+0.0149; as declared, the rule is not re-applied. The pod's own scores differ from these by two and
three ranked pairs. The context AUC equals the 9 September run's to six decimals, but the two are
different models (different weights; probabilities differ by up to 0.31) that happen to order the
same number of pairs correctly.

**The two declared fusion contrasts are not comparable with each other.** View fusion and its control
come from the one-session ledger; context fusion and its control come from `_cuda0918`, and they are
measured on different rows (100 pairs against 243 lesions). Each is like-for-like with its own
control, and the two gaps are not compared.

The two tags enter the tag table, which named the context control only in prose; §28 gives their
measured names.

| tag | role |
|---|---|
| vgg16_fusion_context and vgg16_fusion_control_all | the declared context contrast and its control, on every lesion (fusion (d)) |

**The data-flow figure.** The pass also draws the partition diagram (`data_flow_figure.py`), which
counts every partition, test labels included, under the pass's token. It produces only partition
counts, which "Partition" already states.

**The order of the pass.** The test boxes are written and every test lesion's box coverage is
checked; the pass then writes its marker and scores every model; the primary endpoint and both Holm
families follow directly; the ensembles, the fusion comparisons and the descriptive stages come
last, and the ledger check closes the pass (§36).

**Defects fixed before the freeze, found by reading the pass's real-mode paths and confirmed by the
rehearsal (18 Sep).**

- The freeze token could never match the value the pass derives from it; the guard now compares
  both in a whitespace-free form.
- `src.ensemble` was called on test without `--allow-test` and without validation.
- `rung3_on_pairs.py` wrote its test table under a validation name.
- `verify_test_pass.py` checked the pre-promotion tags.
- `freeze.py` read order positions as block numbers; it now requires the appendix to be exactly
  what the applier writes and the protocol text to match its recorded hash, and the pass checks
  both before it reads test data.
- Grad-CAM labelled its cases at a threshold fitted on the test predictions; it now carries rung 3's
  validation threshold.
- Plots refused a point estimate lying outside its percentile interval; error bars are clipped at
  zero.
- The map writers created each array at its final name before filling it; they now write under a
  temporary name and rename on completion.

## 42. §6 — If the localiser is promoted (written for run 4b, branch B or C; applies to A24)

*Why here: Applies. A24 was promoted on 12 September (§24a; floor gate §35). §28 carries out this block's substitution for the tag table, and §34 for the "Selected configuration" and rung-5 paragraphs.*

Every `_bundle` above becomes the new suffix, `$BOXES` the new folder, and the "Selected configuration"
paragraph is rewritten from the new `bundle_selection.json` (or run 4b's `localise_eval` metrics); the
three-run selection stays in the document as the superseded candidate with its numbers.

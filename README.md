# Deep Learning Breast Cancer Detection

Teodor Tataru. University of London, Graduate Diploma in Computer Science (Machine Learning and
Artificial Intelligence), final project (CM3070).
Template 3.2, Project Idea 2: Deep Learning Breast Cancer Detection.

**In brief.** The system reads a whole mammogram, finds the mass itself and classifies it as benign
or malignant, with no radiologist's outline to start from. On the held-out test partition (119
patients), scored once under a protocol frozen beforehand, it reached an image-level AUC of 0.7874
[0.7195, 0.8536], statistically indistinguishable from the radiologists' own BI-RADS assessments of
the same 203 images (0.8090; difference −0.0216, p 0.550). The negatives are benign masses that
radiologists marked, not normal breasts, so the task is to tell malignant from benign masses.
Descriptively, where the localiser finds the lesion the classifier does about as well as on crops
from the radiologists' own outlines, so most of the gap to ground-truth cropping comes from the
lesions the localiser misses. The system did not reach the targets set at the start, such as an AUC
of 0.90.

## What this project is

A detect-then-classify study on the mass cases of CBIS-DDSM [1], [2]. A radiologist's lesion
outline is not available at screening, so a usable system must find the lesion itself. The
pipeline has two stages:

1. **Localiser.** An ensemble of seven U-Net-family segmentation networks (three Keras, four
   PyTorch) proposes one box per mammogram.
2. **Classifier.** A VGG16, fine-tuned from ImageNet, classifies the crop inside that box as benign
   or malignant; every lesion on the mammogram is scored from that crop.

Around the pipeline the project measures:

- **Development progression:** a from-scratch baseline, scaled, regularised, then transfer
  learning, following Chollet's workflow [3].
- **Architecture comparison:** VGG16, ResNet50, DenseNet121 and EfficientNetB0 on ground-truth
  crops.
- **The localisation-degradation ladder:** seven training conditions ("rungs") that differ only in
  where the crop comes from (1 ground-truth box without margin, 2 ground-truth box with margin,
  3 localiser box, 4 CAM box, 5 jittered box, 6 shuffled box, 7 whole image), so the cost of
  imperfect localisation can be measured against controls.
- **Two-view fusion:** learned fusion of the CC and MLO views against simple averaging.
- **Uncertainty and explanation:** MC-dropout deferral, calibration, and Grad-CAM grounding against
  the lesion masks.
- **The specialist baseline:** the radiologists' BI-RADS assessment recorded in the dataset, scored
  on the same images.

The evaluation protocol (`docs/test_pass_preregistration.md`) was frozen before the test partition
was scored, and the final system was evaluated on it once.

## Headline results

Test partition, single pass under the frozen protocol. Intervals are 95 % percentile bootstrap
intervals resampled by patient (1,000 draws); p-values are paired DeLong. The primary endpoint is
image level, as are the specialist baseline and the difference; every other row is lesion level
(243 lesions per partition). Validation figures come from the stored validation tables.

| quantity | validation | test |
| :--- | :--- | :--- |
| Full pipeline, image level (209 / 203 images) | 0.8574 | **0.7874 [0.7195, 0.8536]** |
| Specialist baseline, same images | 0.8627 | 0.8090 [0.747, 0.872] |
| Pipeline minus baseline | −0.0052, p 0.853 | **−0.0216 [−0.0988, +0.0525], p 0.550** |
| Sensitivity / specificity at the threshold fitted on validation for 0.90 sensitivity (243 lesions) | 0.9018 / 0.6031 | 0.8230 / 0.4538 |
| Development progression, AUC: baseline, scaled, regularised, VGG16 | 0.6604 (0.6647), 0.6920 (0.6859), 0.8040 (0.8275), 0.8777 (0.8768) | 0.6568, 0.6897, 0.7946, 0.8304 |
| Four architectures on ground-truth crops, AUC | 0.7967 to 0.8777 (0.8043 to 0.8768) | 0.8304 to 0.8511; no pair differs after Holm correction |
| Ladder, AUC: ground-truth box with margin, localiser box, whole image | 0.8735 (0.8749), 0.8654 (0.8652), 0.7217 (0.7211) | 0.8229, 0.7800 (0.7794), 0.7074 |
| Ladder controls, AUC: CAM box, jittered box, shuffled box | 0.7246 (0.7248), 0.7652 (0.7617), 0.5640 (0.5453) | 0.6488, 0.7339 (0.7189), 0.4871 |
| Lesions the localiser finds (168 on test): pipeline against ground-truth crops | — | 0.8395 against 0.8210 |
| Localisation failures (no box, or box IoU below 0.3: 75 on test, 65 of them complete misses): pipeline against ground-truth crops | — | 0.6235 against 0.8086 |
| MC-dropout deferral at the cut-off fitted on validation | 30.0 % deferred; retained AUC 0.8875 | 35.8 % deferred; retained AUC 0.8392 |

The pipeline, the specialist comparison, the operating point and the rows below the ladder
controls are seed 28, the pre-registered seed. In the development, architecture and ladder rows,
where more than one seed was trained, the first figure is the mean over seeds and seed 28 follows in
brackets; these means are descriptive. The validation means are over three seeds (five for the
ground-truth box with margin). On test, only the localiser box and the jittered box were scored with
three seeds; the other test figures in those rows are seed 28.

The pipeline is statistically indistinguishable from the specialist baseline; it is not better, and
the interval does not establish equivalence. The targets set at the start (AUC 0.90, sensitivity
0.85, specificity 0.80, beat the baseline) were not met. On the 168 test lesions the localiser
finds, the pipeline's AUC is close to that of ground-truth crops of the same lesions; on the 75
where it fails, the pipeline's AUC falls well below theirs. This split is descriptive. On the
subset of lesions seen in both views (101 test pairs), learned two-view fusion did not beat simple
averaging of the two views' scores (−0.0031, p 0.887); both sides are three-seed ensembles.

## Data

CBIS-DDSM is distributed by The Cancer Imaging Archive [1], [4]; its curation is described in
[2]. The images are not in this repository: `data/README.md` gives the download steps and the
layout the code expects.

The official train and test folders are pooled and re-split at patient level (`src/data_loader.py`,
`StratifiedGroupKFold`, seed 28, about 70/15/15). The loader halts if any patient appears in two
partitions.

| partition | patients | mammograms | lesions (malignant) | images in the specialist comparison |
| :--- | ---: | ---: | ---: | ---: |
| train | 644 | 1,146 | 1,210 (559) | 1,018 |
| validation | 129 | 225 | 243 (112) | 209 |
| test | 119 | 221 | 243 (113) | 203 |

The negatives are benign masses that a radiologist marked, not normal breasts; some were judged
benign from the screening films alone and never recalled or biopsied ("benign without callback").
No specificity here is a screening specificity. The specialist comparison uses mammograms whose
recorded BI-RADS assessment is not 0 (0 means incomplete).

## Test partition

The final system was evaluated on the test partition once, on 19 September 2026, under the frozen
protocol. The committed pass log records the protocol's hash, whose prefix `d84ff4bcd6e36a36`
matches `shasum -a 256 docs/test_pass_preregistration.md`.

Unless the test pass has set its token (`src/test_pass_guard.py`), `split_frames` and
`build_datasets` refuse the test partition and `fusion.build_pairs` leaves it out. Every tool
refuses a test split outside the pass except the verifier (`integrity.py verify-test-pass`), which
checks the pass's stored outputs. Some validation modes read committed tables that also hold test
rows, and use only their validation rows: `src.compare` and `src.analysis` read the specialist
comparison set, `analyse_members.py gate-c-strata` the lesion geometry, and `src.ladder` the
localiser and CAM boxes. The shuffled ladder control loads the test boxes but shuffles the test
lesions on a separate stream, so no test box reaches a validation crop.

There was earlier contact with the test partition. §33 of the protocol lists every later contact
up to the pass and records that no test result informed any selection, gate, threshold, promotion
or decision. Among them is a validation dry run of the pass script on 10 September in
which the stages without a split setting defaulted to test and scored eight laptop models of 7 and 8
September; its outputs were deleted or moved unopened, and the loader guard was added in response.
§33 does not list an earlier contact. In June 2026, before the protocol existed, prototype
classifiers were scored on the partition that later became the test set. Six of their result
files are kept in `outputs/results/_superseded/`; five of them (`baseline`, `regularised`,
`densenet121`, `resnet50` and `vgg16`) record the 243 lesions of what became the test set. No
model or weight from that work is in the final system, whose models were all trained between 7 and
18 September, but that exposure cannot be excluded as an influence on the choice of backbone.

## Repository map

```
src/                         the library: data, models, training, localisation, evaluation
  localise_torch/            PyTorch localiser members and the Faster R-CNN alternative
notebooks/                   01_eda.ipynb, 02_lesion_geometry.ipynb
notebooks/scripts/           command-line tools, the test-pass scripts, the tests
  provenance/                Kaggle notebooks as run, and run manifests, including those of the
                             final system's models
docs/                        the evaluation protocol, the record of training runs, the seeds table,
                             the run-4 localiser recipe
refs/                        reference records used by the test pass and the promotion tools
                             (see refs/README.md)
artifacts/                   frozen tables the pipeline reads (see artifacts/README.md)
outputs/results/             stored results: prediction tables, metrics, comparisons, localiser gates
outputs/results/validation/  validation-mode outputs of the test-pass script
outputs/ablation_exact/      exact validation scores of the regularisation ablations
outputs/logs/                the test-pass log and records
data/                        download instructions only
```

Figures (`outputs/figures/`), weights, tensor caches and the pod records under `outputs/_cuda/` are
not in the repository.

**Library (`src/`)**

| area | modules |
| :--- | :--- |
| configuration and data | `config`, `data_loader`, `geometry` |
| classifiers | `models`, `train`, `train_crop_store`, `fusion`, `train_fusion_store`, `ensemble` |
| localiser | `localise`, `train_localiser_l4`, `precompute_localiser`, `localise_eval`, `rerank`, `cam_localise`, `localise_torch/` |
| ladder | `boxes`, `ladder` |
| evaluation | `evaluate`, `compare`, `analysis`, `birads_baseline`, `uncertainty`, `explain` |
| test-partition guard | `test_pass_guard` |

**Tools (`notebooks/scripts/`)**: each Python tool prints its usage with `--help`. The shell
scripts take no `--help`: read them rather than running them.

| tool | purpose |
| :--- | :--- |
| `stores.py` | build, copy and re-check the stores the models train from |
| `score.py` | exact validation scores of trained checkpoints, through the eager path |
| `localiser_bundle.py` | cache probability maps, sweep ensemble configurations, gate a candidate |
| `box_rules.py` | alternative rules for turning localiser output into one box |
| `recipe.py` | the reference localiser recipe, and what a candidate changes against it |
| `session.py` | declare, check and run a one-session set of training runs |
| `analyse_ladder.py` | where the ladder's error falls, and on which lesions |
| `analyse_fusion.py` | whether a second view helps, and whether a learned combiner earns it |
| `analyse_members.py` | what each localiser member contributes to the ensemble |
| `analyse_uncertainty.py` | uncertainty scores and deferral curves |
| `decide_ceiling.py` | the classifier-ceiling decision rule over validation tables |
| `figures.py` | figures and summary tables |
| `integrity.py` | ledger, weights, preprocessing and launcher checks; the test-pass verifier |
| `freeze.py`, `assert_model_list.py` | freeze the protocol; match the scored tags against it |
| `run_test_pass.sh` | the single test pass |
| `promotion_cascade.sh`, `combined_gate.sh`, `launcher_selfcopy.sh` | localiser promotion and launcher support |

## Environments

Two virtual environments, both Python 3.11:

```
python3.11 -m venv .ddsm-env  && source .ddsm-env/bin/activate  && pip install -r requirements.txt
python3.11 -m venv .torch-env && source .torch-env/bin/activate && pip install -r requirements-torch.txt
```

`.ddsm-env` (TensorFlow 2.17, Keras 3) runs everything except the PyTorch localiser members, which
use `.torch-env` (versions pinned in `requirements-torch.txt`). `tensorflow-metal` carries a macOS
marker in `requirements.txt`, so pip skips it on other systems. On Linux, `requirements.txt`
installs TensorFlow without its CUDA libraries, so TensorFlow sees no GPU; to use an NVIDIA GPU,
also run `pip install "tensorflow[and-cuda]==2.17.0"` inside `.ddsm-env`. This was tested on an
RTX 4090 (see section 2b on TF32).

On the Apple GPU, TensorFlow's compiled graph silently dropped ReLU activations in this project's
models. Every reported figure is therefore computed from a stored prediction table scored through
the eager path, and every AUC is exact. The models behind the reported results were trained on
NVIDIA GPUs (Kaggle T4 and P100; RunPod RTX 4090, RTX 5090 and A100); statistics ran on a laptop.
Laptop training appears in the record only as the superseded runs of 7–9 September; earlier laptop
training is not recorded there.

## Reproducing

Run every command from the repository root with `PYTHONPATH=.` set, in a fresh clone: several tools
rewrite their stored outputs in place, and `git status` then shows whether each rebuild matched.

### 1. From the repository alone

These read only committed files. Apart from the verifier, each rewrites its stored output. On
macOS every rewrite matches the committed file byte for byte, with two exceptions noted below the
commands; on Linux there is a third, also noted there.

```
python notebooks/scripts/integrity.py verify-test-pass --split test   # 71 passed, 0 failed
python notebooks/scripts/figures.py stages
python notebooks/scripts/figures.py ablations
python notebooks/scripts/decide_ceiling.py read-session
```

<details>
<summary>The other 21 section 1 commands</summary>

```
python -m src.compare --primary vgg16_crop_rung3_unet_promoted_cuda0917 \
    --models vgg16_crop_rung2_oracle_cuda0917 vgg16_crop_rung3_unet_promoted_cuda0917 \
    vgg16_crop_rung4_cam_cuda0917 vgg16_crop_rung6_shuffled_cuda0917 \
    vgg16_crop_rung7_whole_cuda0917 --out comparison_report_val.json --split val
python -m src.compare --models vgg16_crop_m020_cuda0917 resnet50_crop_m020_cuda0917 \
    densenet121_crop_m020_cuda0917 efficientnet_crop_m020_cuda0917 \
    --out comparison_architectures_val.json --split val
python -m src.compare --models vgg16_fusion_view_unet_promoted_cuda0917 \
    vgg16_fusion_control_cuda0917 --out comparison_fusion_declared_val.json --split val
python -m src.compare --primary vgg16_fusion_view_unet_promoted_cuda0917 \
    --out comparison_fusion_imagelevel_val.json --split val
python -m src.compare --models vgg16_fusion_context_cuda0918 vgg16_fusion_control_all_cuda0918 \
    --out comparison_fusion_context_val.json --split val
python -m src.analysis --tag vgg16_crop_rung3_unet_promoted_cuda0917 \
    vgg16_crop_rung2_oracle_cuda0917 vgg16_full_cuda0917 --split val

python -m src.ensemble --tags vgg16_crop_rung3_unet_promoted_cuda0917 \
    vgg16_crop_rung3_unet_promoted_cuda0917_s1 vgg16_crop_rung3_unet_promoted_cuda0917_s2 \
    --out-tag vgg16_crop_rung3_unet_promoted_cuda0917_ens3 --method mean --splits val
python -m src.ensemble --tags vgg16_fusion_view_unet_promoted_cuda0917 \
    vgg16_fusion_view_unet_promoted_cuda0917_s1 vgg16_fusion_view_unet_promoted_cuda0917_s2 \
    --out-tag vgg16_fusion_view_unet_promoted_cuda0917_ens3 --method mean --splits val
python -m src.ensemble --tags vgg16_fusion_view_cuda0917 vgg16_fusion_view_cuda0917_s1 \
    vgg16_fusion_view_cuda0917_s2 --out-tag vgg16_fusion_view_cuda0917_ens3 \
    --method mean --splits val

python notebooks/scripts/analyse_fusion.py controls --pairs artifacts/fusion_pairs_view_val.csv \
    --rung3 outputs/results/vgg16_crop_rung3_unet_promoted_cuda0917_ens3_predictions_val.csv \
    --tag rung3_on_pairs_promoted_cuda0917_val --split val
python notebooks/scripts/analyse_fusion.py compare --split val \
    --fusion outputs/results/vgg16_fusion_view_unet_promoted_cuda0917_ens3_predictions_val.csv \
    --controls outputs/results/rung3_on_pairs_promoted_cuda0917_val_predictions_val.csv \
    --oracle-fusion outputs/results/vgg16_fusion_view_cuda0917_ens3_predictions_val.csv \
    --out outputs/results/fusion_pair_comparisons_val.json
python -m src.ladder --arch vgg16 --margin 0.20 --stage val \
    --boxes-dir outputs/results/localiser_bundle/boxes_A24 \
    --rung3-suffix _promoted_cuda0917 --ledger-suffix _cuda0917

python notebooks/scripts/analyse_members.py gate-c-strata \
    --candidate outputs/results/localiser_bundle/selected_runs134l7v/bundle_boxes_val.csv \
    --label ens4l7v
python notebooks/scripts/analyse_members.py gate-c-strata \
    --candidate outputs/results/localiser_bundle/selected_runs134l7a/bundle_boxes_val.csv \
    --label ens4l7a
python notebooks/scripts/analyse_members.py seed-null --seed-member run4_s1
python notebooks/scripts/analyse_members.py seed-null --seed-member run4_s2
python notebooks/scripts/analyse_uncertainty.py deferral \
    --uncertainty outputs/results/localiser_bundle/uncertainty_val.csv \
    --predictions outputs/results/vgg16_crop_rung3_unet_bundle_cuda_predictions_val.csv \
    --out outputs/results/localiser_bundle/deferral_curve_val.csv
python notebooks/scripts/analyse_uncertainty.py deferral \
    --uncertainty outputs/results/localiser_bundle/uncertainty_val_A24.csv \
    --predictions outputs/results/vgg16_crop_rung3_unet_promoted_cuda0917_predictions_val.csv \
    --out outputs/results/localiser_bundle/deferral_curve_val_A24.csv

mkdir -p outputs/weights/localiser_run4_vgg16enc_patch
cp refs/run4_manifest.json outputs/weights/localiser_run4_vgg16enc_patch/manifest.json
python notebooks/scripts/recipe.py build
```

</details>

The two exceptions: `src.ensemble` rebuilds each validation table byte for byte but rewrites the
matching `*_ens3_ensemble.json` without its test entry, and `recipe.py build` rewrites
`docs/run4_recipe.json` with a new generation stamp and nothing else changed. On Linux (tested on
Ubuntu 24.04), three validation comparisons (`comparison_report_val.json`,
`comparison_architectures_val.json` and `comparison_fusion_context_val.json`) also differ, in
their DeLong and McNemar statistics, by at most 2.7e-15. `git checkout -- outputs/results docs`
restores every one. `src.analysis` and `src.ladder` also draw figures, which git ignores.

The tests:

```
.ddsm-env/bin/python -B notebooks/scripts/test_member_frames.py
.ddsm-env/bin/python -B notebooks/scripts/test_normalisation_scope.py --skip-decode
.ddsm-env/bin/python -B notebooks/scripts/test_test_pass_guard.py
CUDA_VISIBLE_DEVICES="" .torch-env/bin/python -B notebooks/scripts/test_localise_torch_port.py
```

The first two need nothing else. The guard test needs the two case-description CSVs under
`data/raw/`. `test_normalisation_scope.py` without `--skip-decode` also needs the dataset and the
crop store.

### 2. With the dataset and the released weights

The trained weights of the final system are published as a release:
<https://github.com/teo-tat/ddsm-dl-screening/releases/tag/v1.0>. It holds the seven localiser
members, the three seeds of the VGG16 classifier trained on localiser-box crops (rung 3), the four
PyTorch members' manifests, `weights_layout.txt` (where each file goes under `outputs/weights/`)
and `SHA256SUMS.txt`, which covers every other file. Each PyTorch manifest must sit beside its
checkpoint.

The committed boxes come from probability maps made on two machines: the three Keras members'
maps on the author's Apple M3 Pro GPU (8 September 2026), and the four PyTorch members' maps on an
RTX 5090. Section 2a scores the classifier from those committed boxes and reproduces the stored
prediction tables exactly. Section 2b rebuilds the boxes themselves, which depends on the hardware.

#### 2a. Scoring from the committed boxes

This needs the dataset and the three classifier seeds. It crops the validation lesions from the
committed boxes (`outputs/results/localiser_bundle/boxes_A24/`, the localiser output behind the
reported results) and scores each seed:

```
T=$(mktemp -d)
python notebooks/scripts/stores.py crop --box-source unet \
    --boxes-dir outputs/results/localiser_bundle/boxes_A24 --margin 0.20 \
    --tag rung3_unet_A24 --split val
for s in "" _s1 _s2; do
    python notebooks/scripts/score.py store --model vgg16 \
        --store outputs/crop_store_rung3_unet_A24 \
        --weights outputs/weights/vgg16_crop_rung3_unet_promoted_cuda0917${s}_best.weights.h5 \
        --out "$T"/score${s}.json \
        --out-predictions "$T"/vgg16_crop_rung3_unet_promoted_cuda0917${s}_predictions_val.csv
    cmp "$T"/vgg16_crop_rung3_unet_promoted_cuda0917${s}_predictions_val.csv \
        outputs/results/vgg16_crop_rung3_unet_promoted_cuda0917${s}_predictions_val.csv
done
```

Scored on an Apple M3 Pro GPU through the eager path, all three tables match the stored ones byte
for byte; other devices were not tested. `src.ensemble` and `src.compare` read tables only from
`outputs/results/`, so to carry the check through to the comparison, copy the rebuilt tables over
the stored ones in the clone and run the section 1 commands again: no stored validation value
changes.

#### 2b. Rebuilding the boxes

The box step (`apply`) compares the mean box IoU of the boxes it exports with the value recorded
when the ensemble was selected, and refuses if they differ by more than 1e-6. The maps therefore
have to come out almost exactly as they did on the machines that made them. On the Apple GPU a
re-cache of the Keras maps matched the stored maps byte for byte (§29 of the protocol,
`outputs/results/eager_vs_compiled_maps.json`), and a regeneration of one PyTorch member's maps
on an RTX 5090 matched exactly (`refs/test_maps_manifest.json`).

On an RTX 4090 under Linux, with TensorFlow's default arithmetic, `apply` refuses: Ampere and Ada
cards use TF32 for TensorFlow's matrix operations, which shifts the Keras maps. With TF32 turned
off for the three Keras lines (`NVIDIA_TF32_OVERRIDE=0`, which matters only on NVIDIA GPUs) and
the PyTorch members on the CPU, `apply` passes. Every box IoU then matches the stored value, but
two of the 243 boxes, both of which miss their lesion, have one edge moved by 5 pixels, and 237
rows differ in columns derived from the maps (the candidate sums in all 237; candidate areas, mask
IoU, Dice and foreground fraction in at most 63). Scored from these boxes, those two predictions
change in each seed, and each validation AUC changes by 0.7e-4 to 2.0e-4. Running the PyTorch
members with `--device cuda` on the 4090 makes `apply` refuse again. The full rebuild was not run
to completion on the Apple GPU.

The commands, in the combination that passed on the RTX 4090; in that run the final `cmp` reports
a difference, for the reasons just given. The tensor caches must be in their default folders,
because the box step reads them from there; `B` and `X` are new folders for the rebuilt maps and
boxes.

<details>
<summary>The rebuild commands</summary>

```
python -m src.precompute_localiser --splits val
python notebooks/scripts/stores.py notest-copy --split val

B=outputs/results/localiser_bundle_rebuild; X=outputs/results/boxes_rebuild
mkdir -p "$B" && cp outputs/results/localiser_bundle/bundle_selection.json "$B"/

NVIDIA_TF32_OVERRIDE=0 python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" cache \
    --tag run1 --weights outputs/weights/localiser_run1_base32/best --base-filters 32 \
    --depth 4 --split val
NVIDIA_TF32_OVERRIDE=0 python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" cache \
    --tag run3 --weights outputs/weights/localiser_run3_base32_merged/best.weights.h5 \
    --base-filters 32 --depth 4 --split val
NVIDIA_TF32_OVERRIDE=0 python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" cache \
    --tag run4 --weights outputs/weights/localiser_run4_vgg16enc_patch/best.weights.h5 \
    --base-filters 32 --depth 4 --encoder vgg16 --split val
.torch-env/bin/python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" write-torch-maps \
    --weights outputs/weights/localiser_l7v_vgg16_cachev1/best.pt --tag l7v --split val \
    --tensors-dir outputs/localiser_tensors_v1_notest --canvas 1024 576 --device cpu
.torch-env/bin/python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" write-torch-maps \
    --weights outputs/weights/localiser_l7a_resnet34_cachev1/best.pt --tag l7a --split val \
    --tensors-dir outputs/localiser_tensors_v1_notest --canvas 1024 576 --device cpu
.torch-env/bin/python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" write-torch-maps \
    --weights outputs/weights/localiser_l8a_efficientnet-b4_cachev1/best.pt --tag l8a --split val \
    --tensors-dir outputs/localiser_tensors_v1_notest --canvas 1024 576 --device cpu
.torch-env/bin/python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" write-torch-maps \
    --weights outputs/weights/localiser_l9_convnext_tiny_cachev1/best.pt --tag l9 --split val \
    --tensors-dir outputs/localiser_tensors_v1_notest --canvas 1024 576 --device cpu

python notebooks/scripts/localiser_bundle.py --bundle-dir "$B" apply --split val --boxes-dir "$X"
cmp "$X"/localiser_boxes_val.csv outputs/results/localiser_bundle/boxes_A24/localiser_boxes_val.csv
```

</details>

`write-torch-maps` records only the lesion count in its manifest. `localiser_bundle.py --bundle-dir
"$B" recheck --tag <member> --split val` prints a PyTorch member's single-box IoU and records it
in the member's manifest in `$B`, for comparison with `single_box_mean_iou_8bit` in the committed
manifest under `outputs/results/localiser_bundle/maps/`. On the RTX 4090 the PyTorch members took
12 to 25 minutes each on the CPU; the Keras members took under a minute each on the GPU. To score
from the rebuilt boxes, copy `outputs/results/localiser_bundle/boxes_A24/localiser_boxes_train.csv`
into `$X` (the crop step reads the training boxes too) and run section 2a with `--boxes-dir "$X"`
and a new `--tag` and `--store`.

Several tools have no flag to move their output (`src.compare`, `src.analysis`, `src.ensemble`,
`src.ladder`, `src.evaluate`, `src.fusion`, `recipe.py build`); run anything beyond section 1 in a
clone or a copy of the repository. `src.localise_eval` refuses to write into `artifacts/`,
`src.birads_baseline` and `src.cam_localise` refuse while their frozen tables exist, and
`src.explain` and `src.uncertainty` rewrite files under `outputs/`.

### 3. Training

Training is recorded, not rerun here. `docs/remote_runs.md` records 172 completed trainings: every
run off the laptop, and the laptop runs from 7 September, each with its command (verbatim or
reconstructed), parameters, hardware and outcome. `docs/seeds_table.md` records the seeds. The
results were produced in this order: build the stores (`stores.py`), train the localiser members and
classifiers with the recorded commands, cache and gate the localiser ensemble
(`localiser_bundle.py`, `combined_gate.sh`), rebuild and retrain what depends on the promoted boxes
(`promotion_cascade.sh`), retrain the final models in one session, freeze the protocol
(`freeze.py`), then run the test pass (`run_test_pass.sh`). `freeze.py` cannot run again after the
freeze. In a clone the real pass stops at once, because the freeze token is not committed; its
`--split val` dry run writes tracked files, so run it only in a copy.

### What does not run from a clone

- `integrity.py ledger-check` needs the ledger snapshot, which is not committed.
- `analyse_ladder.py errors` and `figures.py crops` need the dataset. `figures.py data-flow`
  counts every partition, test included, and is refused outside the test pass.
- `analyse_ladder.py silent-images` needs the validation tensor cache and the dataset's case files;
  the committed peaks table, given through `--peaks`, removes only the need for a member's weights.
- `analyse_uncertainty.py scores` needs the tensor caches and member maps.
- `analyse_members.py victim-rescue` and `decide_ceiling.py ceiling` need the local pod records.
- `figures.py matrix` needs the weights and the pod records; in a clone it marks every current row
  "no weights" and rewrites `outputs/results/experiment_matrix.csv`, in a different row order.
- Every test-split mode refuses outside the test pass, except the verifier.

Some stored files are historical records: `margin_sweep.csv` has no current writer, and
`fusion_stratified_by_box_outcome_val.json`, `uncertainty_val.{csv,json}` and the two
`seed_spread_*_bundle.csv` files are in a layout the current tools no longer write. They are kept
because the record cites them.

## How the stored results were produced

The reported results were produced by an earlier, unconsolidated set of run scripts. The tools in
`notebooks/scripts/` are their consolidation; section 1 above lists the stored outputs they rebuild
from the repository alone. The test pass, too, ran on the earlier code: the code manifest recorded
at the pass (`outputs/logs/test_pass_code_manifest/manifest_20260919T170641Z.sha256`, local
material that is not committed) lists the scripts as they ran on 19 September 2026, so its code
entries no longer match this repository's code, which was consolidated and edited afterwards. Its
entries for the protocol, `refs/test_maps_manifest.json` and the 28 weights files, three of which
are the released rung-3 seeds, still match.
The earlier scripts and the pod launchers are kept outside the
repository; where `docs/remote_runs.md` cites a path under `outputs/_cuda/` or
`outputs/_superseded/`, or a launcher kept outside the repository, it refers to that local
material.

Result files are named by model tag. The final system's tags end in `_cuda0917` (one training
session on one RTX 4090, 17 September 2026) or `_cuda0918` (the context-fusion model and its
control, 18 September). The primary comparison is `outputs/results/comparison_report_test.json`,
the ladder `outputs/results/ladder_vgg16_test.csv`, and the architecture comparison
`outputs/results/comparison_architectures_test.json`.

## References

1. R. Sawyer-Lee, F. Gimenez, A. Hoogi and D. Rubin, "Curated Breast Imaging Subset of Digital
   Database for Screening Mammography (CBIS-DDSM) [Data set]," The Cancer Imaging Archive, 2016.
   doi: 10.7937/K9/TCIA.2016.7O02S9CY
2. R. S. Lee, F. Gimenez, A. Hoogi, K. K. Miyake, M. Gorovoy and D. L. Rubin, "A curated
   mammography data set for use in computer-aided detection and diagnosis research," *Scientific
   Data*, vol. 4, 170177, 2017. doi: 10.1038/sdata.2017.177
3. F. Chollet, *Deep Learning with Python*. Shelter Island, NY: Manning, 2018.
4. K. Clark, B. Vendt, K. Smith, J. Freymann, J. Kirby, P. Koppel, S. Moore, S. Phillips,
   D. Maffitt, M. Pringle, L. Tarbox and F. Prior, "The Cancer Imaging Archive (TCIA): Maintaining
   and Operating a Public Information Repository," *Journal of Digital Imaging*, vol. 26, no. 6,
   pp. 1045–1057, 2013. doi: 10.1007/s10278-013-9622-7

## Licence

MIT (see `LICENSE`). The dataset has its own licence and data usage policy; see its TCIA page.

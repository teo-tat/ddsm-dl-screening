# Remote runs

Every training run in this project that executes somewhere other than the laptop, in the
order the campaigns run, followed by the laptop runs that have no record anywhere else.
Times are UTC. Laptop training before 7 September is not recorded here: 185 files survive from
it, 183 of them from 27 to 29 June under `outputs/_superseded/pre_m020_histories/`,
`outputs/results/_superseded/` and `outputs/weights/_superseded/`, and two from a localiser
smoke test on 6 September.

**172 completed trainings are recorded here**: 6 Kaggle localiser runs (section 1), 7 pod localiser
members and alternatives (section 2), 2 localiser seed replicates (section 3), 7 classifier ablations (section 4),
15 ceiling and margin runs (section 5), the 59-run session (section 6), 2 context-fusion runs (section 7), 3
superseded localiser runs (section 8), 35 completed runs in the cancelled or superseded pod work
(section 9) and 36 laptop runs (section 10). Section 9 also lists the launches that trained nothing,
which are not in the total.

Cards in use: Kaggle supplies a **Tesla T4** (two of them per notebook) or a **Tesla
P100-PCIE-16GB**; RunPod supplies an **RTX 5090**, an **RTX 4090** or an **A100-SXM4-80GB**;
the laptop's own **Apple M3 Pro** GPU (tensorflow-metal) ran the section 10 work before training
moved off the machine. Each entry states the card its own manifest or notebook output
records, or says that the card is not recorded.

Each entry gives the tag, what the run tests, where and when it ran, the entry point and the
command, the key parameters, the stores it reads, the files it writes, where its result is
reported, and the status of its record:

- **verbatim** — the launch command survives exactly as it was issued, in a notebook, a cells
  file, a launcher script or a generated commands file.
- **reconstructed** — the command is rebuilt from the run's own manifest, its tag and the
  code, and is marked as a reconstruction wherever it appears.

Outcome words used below: **member** (feeds the final system), **null** (tested, changed
nothing), **withdrawn** (kept out of the comparison on its settings), **superseded**
(replaced by a later run of the same thing), **diverged** (training failed), **reference**
(used to set a baseline or a noise floor).

---

## Glossary

| term | plain meaning |
| :--- | :--- |
| localiser | the network that finds the lesion in a whole mammogram and proposes a box around it |
| classifier | the network that reads one crop and outputs a malignancy probability |
| box | a rectangle in image coordinates; a crop is the image inside the box plus a margin |
| oracle box | the box drawn from the dataset's own lesion mask — perfect localisation, used as a ceiling, never available in deployment |
| localiser box | the box the trained localiser proposes — what a deployed system would actually have |
| member | one trained localiser whose boxes are combined with the others' into the ensemble the pipeline uses |
| promoted member | a member that passed its gates and is part of the final seven-member ensemble |
| gate | a pre-declared numerical test a candidate must pass before it is allowed into the ensemble: its box overlap must beat a named reference by more than a stated amount, with a bootstrap interval that excludes zero |
| the ladder | seven training conditions that differ only in where the crop comes from, so the cost of imperfect localisation can be measured |
| — oracle tight | condition 1: the mask's own box, no margin |
| — oracle at margin | condition 2: the mask's own box widened by the fixed margin — the ceiling of the deployable crop |
| — localiser box | condition 3: the U-Net ensemble's box — the realisable system |
| — CAM box | condition 4: a box from a weakly-supervised activation map, using no mask annotation |
| — jittered box | condition 5: the oracle box displaced by the localiser's own measured error |
| — shuffled box | condition 6: a plausible box taken from a different lesion — wrong location, right statistics |
| — whole image | condition 7: no localisation at all |
| margin | how far a crop is grown beyond its box, as a fraction of box size; the project's fixed value is 0.20 |
| ledger | the set of validation numbers used for every comparison, all produced by one code path on one machine; the current one is the 17 September session's |
| noise floor | the spread across seeds of the same run, which fixes how large a difference has to be before it means anything |
| the bar | the value a candidate must beat to change a decision: the incumbent's mean plus the larger of 0.01 and three times the pooled seed spread |
| cache v1 | the localiser's image cache at 1024×576, film percentiles clipped at p1/p99 — the cache every promoted member uses |
| cache v2 | the same canvas with the clip ceiling raised to p99.8 |
| cache v3 | a double canvas, 2048×1152 — a deliberately unfavourable regime for 512-pixel patches |
| per-lesion store | one row per lesion, so a mammogram with two lesions appears twice with one mask each |
| merged store | one row per mammogram, with the masks of all its lesions combined |

---

## 1. Localiser, Kaggle

The segmentation network that proposes a lesion box. Six runs: three promoted, one diverged,
two controls. The notebooks are in `notebooks/scripts/provenance/kaggle/`; the weights
manifests of runs 4, 4b and 4c are copied into
`notebooks/scripts/provenance/manifests/`.

### run 1 — `localiser_run1_base32`

The first U-Net that works: 32 base filters, depth 4, trained on the per-lesion store at
1024×576. It becomes the reference later localiser candidates are gated against, and one of
the seven promoted members.

| | |
| :--- | :--- |
| platform | Kaggle notebook, TensorFlow, Tesla T4 |
| date | 7 September, from 08:37 |
| entry point | `notebooks/scripts/provenance/kaggle/run_1.ipynb`, 7 cells |
| parameters | base filters 32, depth 4, input 1024×576 whole images, batch 4, Adam at learning rate 1e-3, loss 0.5 × binary cross-entropy + 0.5 × soft Dice, mixed float16 compute with a float32 output head, epoch cap 100 and 88 epochs run, early stopping patience 20 from epoch 20, learning-rate reduction ×0.5 on patience 8 with floor 1e-6, seed 28 |
| stores read | the per-lesion tensor store, attached to the notebook |
| writes | `best.weights.h5`, a history and a run record; landed as `outputs/weights/localiser_run1_base32/best/` |
| reported | the localiser table, as a gate reference and a promoted member |
| outcome | **member** — one of the seven promoted |
| record | **verbatim** (notebook, with its own outputs) |

### run 2 — `localiser_run2_base48`

Asks whether more capacity helps: 48 base filters under mixed precision. It diverges to NaN
partway through training and is abandoned. Kept because the capacity question it asks stays
unanswered, and because the divergence is why later runs train in float32.

| | |
| :--- | :--- |
| platform | Kaggle notebook, TensorFlow, mixed precision, Tesla T4 |
| date | 7 September, from 11:01 |
| entry point | `notebooks/scripts/provenance/kaggle/run2_base48.ipynb`, 7 cells |
| parameters | base filters 48, depth 4, input 1024×576, batch 4, Adam at 1e-3, loss 0.5 × BCE + 0.5 × soft Dice, float16 compute with a float32 output head, epoch cap 120 with 48 epochs run and the best at 30, selection on validation loss, seed 28 |
| stores read | the per-lesion tensor store |
| writes | a partial history; no usable weights |
| reported | the localiser table, as a failed run |
| outcome | **diverged** |
| record | **verbatim** (notebook) |

### run 3 — `localiser_run3_base32_merged`

Repeats run 1 on a merged store — one row per mammogram, with the masks of all its lesions
combined — in float32, to test whether the contradictory supervision on multi-lesion images
is what limits run 1. It is promoted as a member.

| | |
| :--- | :--- |
| platform | Kaggle notebook, TensorFlow, Tesla T4 |
| date | 7 September, 13:40 to about 18:37 — the notebook starts at 13:40, the first epoch at 13:42, and the validation pass at the end runs at 18:37 |
| entry point | `notebooks/scripts/provenance/kaggle/kaggle_3.ipynb`, 7 cells |
| parameters | base filters 32, depth 4, 7,768,353 parameters, input 1024×576 whole images, batch 4, Adam at learning rate 1e-3, loss 0.5 × binary cross-entropy + 0.5 × soft Dice, float32, 287 steps per epoch, 120 epochs requested and 80 run with epoch 60 selected, selection on validation loss, early stopping patience 20 from epoch 20, learning-rate reduction ×0.5 on patience 8, output bias initialised to the training foreground prior, seed 28; the notebook asserts the merged store has 1,146 training and 225 validation rows before training |
| stores read | the merged tensor store for training (1,146 images), the per-lesion store for validation (243 lesions) |
| writes | `best.weights.h5` and a run record; landed as `outputs/weights/localiser_run3_base32_merged/` |
| reported | the localiser table |
| outcome | **member** — one of the seven promoted |
| record | **verbatim** (notebook, with its own outputs). It is the training that produced the landed weights: the run record names `run3_base32_merged`, the store assertion matches, the weights file holds exactly the 7,768,353 parameters the notebook asserts, and the laptop's own evaluation in `outputs/results/localiser_run3_eval/localiser_metrics.json` reproduces the notebook's validation figures to three decimal places. |

### run 4 — `localiser_run4_vgg16enc_patch`

A VGG16 encoder with patch training: 512-pixel lesion-oversampled patches at batch 8. Its
recipe becomes the baseline every later candidate is diffed against before it is allowed to
launch.

| | |
| :--- | :--- |
| platform | Kaggle, Tesla P100-PCIE-16GB |
| date | 8 September, 08:42 to about 13:35 — the notebook's first epoch starts at 08:45 and the manifest records 17,382 s of training. The run log records run 4 as landing on 7 September, so this saved execution may be a re-run of the same recipe. |
| entry point | `notebooks/scripts/provenance/kaggle/localiser_run4_vgg16enc_patch.ipynb`, 5 cells |
| parameters | encoder vgg16 (ImageNet), patch 512 from a 1024×576 canvas, 2 patches per image with half lesion-centred and 0.25 centre jitter, batch 8, two Adam instances at encoder 1e-4 and decoder 1e-3, 5 warm-up epochs then cosine to a floor of 1e-6, loss 0.5 × BCE + 0.5 × soft Dice, float32, 100 epochs requested and 100 run with the best at 62, selection on validation hard IoU, seed 28; input replicated to 3 channels and scaled in-graph to the Caffe VGG16 convention |
| stores read | merged tensor store for training, per-lesion store for validation |
| writes | `best.weights.h5`, `manifest.json`, `history.csv`; landed as `outputs/weights/localiser_run4_vgg16enc_patch/` |
| reported | the localiser table; its recipe is published as `refs/run4_recipe.json` and `refs/run4_manifest.json` |
| outcome | **member** — one of the seven promoted, and the gate reference |
| record | **verbatim** (notebook and cells file) |

### run 4b — `localiser_run4b_vgg16enc_whole`

A control for run 4: the same encoder fine-tuned on whole 1024×576 images, warm started from
run 4's weights, with lower encoder and decoder learning rates.

| | |
| :--- | :--- |
| platform | Kaggle, Tesla P100-PCIE-16GB |
| date | 8 September (the notebook copy holds no outputs, so no time survives) |
| entry point | `notebooks/scripts/provenance/kaggle/localiser_run4b_vgg16enc_whole.ipynb`, 6 cells |
| parameters | encoder vgg16, whole-image training at 1024×576, batch 4, warm start from run 4's `best.weights.h5`, Adam at encoder 3e-5 and decoder 3e-4, 3 warm-up epochs then cosine to 1e-6, loss 0.5 × BCE + 0.5 × soft Dice, float32, 40 epochs requested, seed 28 |
| stores read | merged tensor store, per-lesion store; run 4's weights as the initialisation |
| writes | `best.weights.h5`, `manifest.json`, `history.csv`; landed as `outputs/weights/localiser_run4b_vgg16enc_whole/` |
| reported | the localiser table; its gate results are `outputs/results/localiser_bundle/gate_run4b_vs_run1.json` and `gate_run4b_vs_ens3.json` |
| outcome | **null** |
| record | **verbatim** (notebook and cells file) |

### run 4c — `localiser_run4c_vgg16enc_patch_reg`

A second control for run 4: the same patch regime with regularisation added.

| | |
| :--- | :--- |
| platform | Kaggle, Tesla P100-PCIE-16GB |
| date | 8 September 17:36 to 9 September, about 01:34 — the first epoch starts at 17:40 and the manifest records 28,400 s of training |
| entry point | `notebooks/scripts/provenance/kaggle/localiser_run4c_vgg16enc_patch_reg.ipynb`, 6 cells |
| parameters | run 4's recipe — encoder vgg16, patch 512, batch 8, Adam at 1e-4 / 1e-3, 5 warm-up epochs then cosine to 1e-6, loss 0.5 × BCE + 0.5 × soft Dice, float32, seed 28 — with decoder dropout 0.3, decoder L2 1e-4, lesion-centred fraction 0.30 drawn per patch instead of exactly 0.5, and elastic displacement of 3–6 px smoothed at 32–64 px; 120 epochs requested and 120 run, best at 59 |
| stores read | merged tensor store, per-lesion store |
| writes | `best.weights.h5`, `manifest.json`, `history.csv`; landed as `outputs/weights/localiser_run4c_vgg16enc_patch_reg/` |
| reported | the localiser table; its gate results are `outputs/results/localiser_bundle/gate_run4c_vs_run1.json` and `gate_run4c_vs_ens3.json` |
| outcome | **null** |
| record | **verbatim** (notebook and cells file) |

---

## 2. Localiser, RunPod RTX 5090

Four PyTorch members trained with one variable between them — the encoder — plus two seed
replicates and a detector that is withdrawn. Seven runs. All the encoder runs share a cache,
a patch size, a batch size, a schedule and a seed, so their differences are attributable to
the encoder alone.

The common command, from the pod launch script:

```
python -m src.localise_torch.train \
    --encoder <ENCODER> --cache <merged cache v1> --val-cache <per-lesion cache v1> \
    --out-dir <run folder> --epochs 100 --batch-size 8 --patch 512 \
    --patience 100 --background-frac 0.0 \
    --write-maps --write-boxes --run-name <NAME>
```

Common parameters, from each run's manifest: input 1024×576 cache, 512-pixel patches,
1-channel images replicated to 3 and normalised with ImageNet statistics; two Adam instances
at encoder 1e-4 and decoder 1e-3, no weight decay, 5 warm-up epochs then cosine to a floor of
1e-6; loss 0.5 × BCE + 0.5 × soft Dice; float32; 100 epochs requested and 100 run; patience
100, which cannot bind at that cap; seed 28 unless stated.

### `localiser_l7v_vgg16_cachev1`

The VGG16 encoder in PyTorch, matching run 4's encoder in a different framework; it is the
run that validates the port before the other encoders are compared through it.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 5090, the torch environment |
| date | 10 September, 08:40 to 10:35 |
| entry point | the common command above, with `--encoder vgg16 --run-name l7v` |
| parameters | as the common set; best epoch 53 |
| stores read | merged cache v1 for training, per-lesion cache v1 for validation |
| writes | `manifest.json`, checkpoint, probability maps and box tables under its run folder; manifest copied to `notebooks/scripts/provenance/manifests/localiser_l7v_vgg16_cachev1_manifest.json` |
| reported | the localiser table and the member-ensemble analyses |
| outcome | **member** — one of the seven promoted |
| record | **reconstructed**. No command line survives for this run alone. The launch script for the next two members, transcribed from the pod launch script and kept outside the repository, opens by stating that it is this run's exact command with only `--encoder`, `--out-dir` and `--run-name` changed; the run's manifest in `notebooks/scripts/provenance/manifests/` corroborates every setting above. |

### `localiser_l7a_resnet34_cachev1`

ResNet34 in the same harness: the encoder-capacity comparison against the VGG16 member.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 5090, the torch environment |
| date | 10 September, 10:46 to 12:14 |
| entry point | the common command with `--encoder resnet34 --run-name l7a` |
| parameters | as the common set; best epoch 55 |
| stores read | merged cache v1, per-lesion cache v1 |
| writes | `manifest.json`, checkpoint, maps, box tables; manifest copied to `notebooks/scripts/provenance/manifests/` |
| reported | the localiser table |
| outcome | **member** — one of the seven promoted |
| record | **verbatim**. The command is transcribed from the pod launch script into a copy kept outside the repository, and the launcher copies itself beside the run; the run's manifest corroborates the settings. |

### `localiser_l8a_efficientnet-b4_cachev1`

EfficientNet-B4 in the same harness, launched by the same loop as the ResNet34 member.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 5090, the torch environment |
| date | 10 September, 12:14 to 14:10 |
| entry point | the common command with `--encoder efficientnet-b4 --run-name l8a` |
| parameters | as the common set; best epoch 73 |
| stores read | merged cache v1, per-lesion cache v1 |
| writes | `manifest.json`, checkpoint, maps, box tables; manifest copied to `notebooks/scripts/provenance/manifests/` |
| reported | the localiser table |
| outcome | **member** — one of the seven promoted |
| record | **verbatim**. The command is transcribed from the pod launch script into a copy kept outside the repository, with the launcher's self-copy beside the run; the manifest corroborates the settings. |

### `localiser_l9_convnext_tiny_cachev1`

A ConvNeXt-Tiny encoder, queued to run after the ResNet34 and EfficientNet members.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 5090, the torch environment |
| date | 10 September, 14:10 to 16:10 |
| entry point | the common command with `--encoder tu-convnext_tiny --run-name l9`, started by the queue script that follows the previous member |
| parameters | as the common set; best epoch 73 |
| stores read | merged cache v1, per-lesion cache v1 |
| writes | `manifest.json`, checkpoint, maps, box tables; manifest copied to `notebooks/scripts/provenance/manifests/` |
| reported | the localiser table |
| outcome | **member** — one of the seven promoted |
| record | **verbatim**. The command is transcribed from the pod launch script into a copy kept outside the repository; the queue's self-copy survives and this run's does not, and the run's manifest corroborates the settings. |

### `localiser_l7v_vgg16_cachev1_s1`, `_s2`

Seed replicates of the VGG16 member, which fix the noise floor every localiser gate is read
against.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 5090, the torch environment |
| date | 10 September, 16:38 to 20:28 |
| entry point | the common command with `--seed 1` or `--seed 2` and `--run-name l7v_s<N>` |
| parameters | as the VGG16 member with seeds 1 and 2; best epochs 26 and 51 |
| stores read | merged cache v1, per-lesion cache v1 |
| writes | `manifest.json`, checkpoints, maps, box tables; both manifests copied to `notebooks/scripts/provenance/manifests/` |
| reported | the seed-spread figure behind every localiser gate reading |
| outcome | **reference** — sets the noise floor |
| record | **verbatim**. Transcribed from the pod launch script into a copy kept outside the repository; both manifests corroborate the settings. |

### `localiser_l10b_fasterrcnn_cachev1`

A detector rather than a segmentation network: Faster R-CNN with a COCO-pretrained
ResNet50-FPN backbone. It is withdrawn as a member and kept as a recorded alternative.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 5090, the torch environment |
| date | 10 September, after 16:38 |
| entry point | `python -m src.localise_torch.detector --cache <merged cache v1> --val-cache <per-lesion cache v1> --out-dir <run folder> --epochs 30 --batch-size 2 --patience 30 --schedule warmup_cosine --warmup-epochs 5 --lr-min 1e-6 --select-on val_loss` |
| parameters | ResNet50-FPN from COCO (`fasterrcnn_resnet50_fpn_v2`, whose builder gives the backbone standard, trainable batch-norm; the stem and first stage frozen, the other three stages trainable), input 1024×576 replicated to 3 channels with ImageNet statistics applied by the detector's own transform, batch 2, Adam at 1e-4 with no weight decay, 5 warm-up epochs then cosine to 1e-6, loss the Faster R-CNN multi-task objective (region-proposal classification and regression, head classification and regression), float32, 30 epochs requested and 30 run, selection on validation loss computed with the model in train mode, so the validation batches also update the batch-norm running statistics, seed 28 |
| stores read | merged cache v1, per-lesion cache v1 |
| writes | `manifest.json`, checkpoint, box tables; manifest copied to `notebooks/scripts/provenance/manifests/` |
| reported | the localiser table, as a withdrawn alternative |
| outcome | **withdrawn** |
| record | **verbatim**. Transcribed from the pod launch script into a copy kept outside the repository; the manifest corroborates the settings. |

---

## 3. Localiser seed replicates of run 4, RunPod RTX 4090

`localiser_run4_s1_cuda`, `localiser_run4_s2_cuda` repeat the Kaggle run 4 recipe on a pod at
two more seeds, so the Keras member has a noise floor measured the same way as the PyTorch
members. Two runs.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 4090, the Keras environment |
| date | 10 September, 16:48 to 20:20 |
| entry point | `python -m src.train_localiser_l4 --encoder vgg16 --merged-dir <merged cache v1> --val-dir <per-lesion cache v1> --out-dir <run folder> --run-name <NAME> --epochs 100 --batch-size 8 --patch-size 512 --seed 1` (and `--seed 2`) |
| parameters | run 4's recipe unchanged: encoder vgg16, patch 512, batch 8, Adam at encoder 1e-4 and decoder 1e-3, 5 warm-up epochs then cosine to 1e-6, loss 0.5 × BCE + 0.5 × soft Dice, float32, background fraction 0.0, 100 epochs requested and 100 run, selection on validation hard IoU; seeds 1 and 2, best epochs 66 and 57 |
| stores read | merged cache v1, per-lesion cache v1 |
| writes | `manifest.json` and a spec file per seed, checkpoints; both manifests and both specs copied to `notebooks/scripts/provenance/manifests/` |
| reported | the seed-spread figures for the Keras member |
| outcome | **reference** — sets the noise floor |
| record | **verbatim**. Transcribed from the pod launch script into a copy kept outside the repository, with a self-copy fetched back and a per-seed spec file recording the settings used; the manifests corroborate them. |

---

## 4. Classifier ablations, Kaggle

Seven runs that each change one thing against a replica of the regularised CNN, so the
contribution of each regularisation choice is measured rather than assumed. They run on
Kaggle because they need a GPU and are independent of the pod campaigns.

| | |
| :--- | :--- |
| tags | `regularised_crop_m020_kg`, and `_no_dropout_kg`, `_no_l2_kg`, `_no_se_kg`, `_no_aug_kg`, `_lr2e-3_kg`, `_overlay_lrmatched_kg` |
| platform | Kaggle notebook, TensorFlow, Tesla P100-PCIE-16GB |
| date | 8 September, from 10:21 |
| entry point | `notebooks/scripts/provenance/kaggle/ablations_kg.ipynb`, 5 cells |
| parameters | model regularised, input 224×224 crops, batch 32, Adam, binary cross-entropy, epoch cap 200 with early stopping on validation AUC, learning rate 5e-4, seed 28, platform suffix `_kg`; one change per run — none, dropout off, L2 off, squeeze-and-excite removed, augmentation off, learning rate 2e-3, dilated-mask overlay |
| stores read | the margin-0.20 crop store, and the overlay crop store for the overlay arm |
| writes | one best and one final weights file, a history and a manifest per run, under `outputs/weights/` and `outputs/results/` |
| reported | the ablation table, with an exact validation AUC scored afterwards on the laptop |
| outcome | **null** — they measure the recipe, they do not change it |
| record | **verbatim** (notebook and cells file) |

---

## 5. The classifier ceiling and the margin sweep, RunPod

Fifteen runs that ask whether the crop margin, a longer patience or another backbone lifts the
oracle-crop ceiling. They run in the order below, from one launcher, on one pod.

| group | runs | tags |
| :--- | ---: | :--- |
| two further seeds at the incumbent margin, completing a five-seed bar | 2 | `vgg16_crop_rung2_oracle` at seeds 3 and 4 |
| three margins at three seeds each | 9 | `vgg16_crop_rung2_oracle_m010`, `_m030`, `_m040`, each at seeds 28, 1 and 2 |
| the same condition at a longer patience | 1 | `vgg16_crop_rung2_oracle_p50` |
| three non-VGG backbones at the same condition | 3 | `resnet50_crop_rung2_oracle`, `densenet121_crop_rung2_oracle`, `efficientnet_crop_rung2_oracle` |

The incumbent margin's own three seeds are not re-run here: they already exist from the pod
sessions in section 9, as `vgg16_crop_rung2_oracle_cuda`, `_s1_cuda` and `_s2_cuda`. Margins 0.80 and
1.00 are excluded because a fifth and a quarter of their crops clamp at the image border.

| | |
| :--- | :--- |
| platform | RunPod pod, the Keras environment; **the card is not recorded** — these manifests carry the Linux platform string but no GPU name |
| date | 10 September, histories written from 15:40 to 16:40, one run after another in the launcher's order. These are the file dates of the histories and manifests under `outputs/_cuda/s8_results/`, not recorded times: the launcher writes no timestamped report and the manifests carry no time. The group runs before the 11 September promoted-box retrain in section 9, which its launcher is written to follow. |
| entry point | the launcher that ran on the pod, kept outside the repository, which issues `python -m src.train_crop_store --store <store> --model vgg16 --tag-suffix _rung2_oracle_m010 --platform-suffix "" --epochs 100 --patience 25 [--seed 1]` and the same line per group, with `--model resnet50` / `densenet121` / `efficientnet` for the backbone group and `--patience 50` for the patience run |
| parameters | input 224×224 crops, batch 32, Adam at 1e-3, binary cross-entropy, epoch cap 100, patience 25 (50 for the patience run), two-stage transfer (frozen head, then fine-tune), seeds 28, 1, 2 and 3, 4 |
| stores read | `crop_store_rung2_oracle`, and `crop_store_rung2_oracle_m010`, `_m030`, `_m040` — one store serves all four backbones, since normalisation is applied per model at batch time |
| writes | weights, histories, a run manifest and a train manifest per run, under `outputs/_cuda/s8_results/`; validation prediction tables scored afterwards on the laptop |
| reported | the crop-margin appendix and the backbone comparison |
| outcome | **null** — the decision is recorded in `outputs/results/ceiling_26/decision.json`, with the configurations it compared in `configurations.csv` beside it |
| record | **verbatim** (the launcher that ran on the pod, kept outside the repository) |

---

## 6. The one-session ledger, RunPod RTX 4090

Fifty-nine runs launched as one declared session, so that the ladder, architecture and
fusion numbers it produces share one code state, one card and one sitting. Twenty-six of the
**28 weights the test pass scores** come from this session; each row is recorded only after
its post-conditions verify.

| | |
| :--- | :--- |
| platform | RunPod, NVIDIA GeForce RTX 4090, the Keras environment; every row's manifest records the same card |
| date | 17 September, 15:15 to 19:00 |
| entry point | the session driver's `run` verb, over a generated commands file of 59 lines |
| command form | every row is one line of `python -m src.train_crop_store --store outputs/<store> --model <model> --seed 28 --tag-suffix <suffix> --platform-suffix '' --patience 25 --out-dir outputs/_cuda/s0917/weights` for the crop rows, or `python -m src.train_fusion_store --store outputs/<store> --arch vgg16 --seed 28 --tag-suffix _cuda0917 --platform-suffix '' --patience 25 --out-dir outputs/_cuda/s0917/weights` for the fusion rows, with `--seed 1` and `--seed 2` for the replicates and `--seed 3`, `--seed 4` at oracle-at-margin; the first line, verbatim, is `python -m src.train_crop_store --store outputs/crop_store_rung1_oracle --model vgg16 --seed 28 --tag-suffix _rung1_oracle_cuda0917 --platform-suffix '' --patience 25 --out-dir outputs/_cuda/s0917/weights` |
| families, summing to 59 | the seven ladder conditions at three seeds each, with five at oracle-at-margin (23); the earlier localiser boxes at three seeds (3); seven architectures on the margin-0.20 crop store at three seeds each (21); the whole-image reference at three seeds (3); three fusion stores at three seeds each (9) |
| parameters | patience 25 on every row, input 224×224, batch 32, epoch cap 100 per stage, Adam, binary cross-entropy, seeds 28, 1, 2 (plus 3 and 4 at oracle-at-margin) |
| stores read | every crop store and fusion store the ladder and the architecture family need |
| writes | weights, histories, run manifests and train manifests for all 59 rows, under `outputs/_cuda/s0917/`; the train manifests of the scored weights are copied to `notebooks/scripts/provenance/manifests/` |
| reported | the ladder, the architecture family, the fusion contrasts and the whole-image reference |
| outcome | **member** — 26 of the 28 weights the test pass scores |
| record | **verbatim** (the generated commands file, checked against an independent declaration of the same rows before the first launch) |

---

## 7. Context fusion, RunPod RTX 4090

Two runs that complete the fusion comparison: a context pair, in which the second crop is the
same lesion at a wide margin, and its control on the same row set. These are the **other two
of the 28 weights the test pass scores**.

| | |
| :--- | :--- |
| tags | `vgg16_fusion_context_cuda0918`, `vgg16_fusion_control_all_cuda0918` |
| platform | RunPod, NVIDIA GeForce RTX 4090, the Keras environment |
| date | 18 September, 19:01 to 19:15 |
| entry point | the session driver, second session |
| command | `python -m src.train_fusion_store --store outputs/fusion_store_vgg16_fusion_context --arch vgg16 --seed 28 --tag-suffix _cuda0918 --platform-suffix '' --patience 25 --out-dir outputs/_cuda/s0918`, and the same line with `--store outputs/fusion_store_vgg16_fusion_control_all` |
| parameters | seed 28, patience 25, batch 32, one seed each; the context run trained 59 frozen and 29 fine-tune epochs in 427 s, the control 41 frozen and 79 fine-tune epochs in 280 s |
| pre-flight | the pod must report the expected card, and must reproduce the laptop's score for a known checkpoint within a fixed tolerance, before either row trains |
| stores read | `fusion_store_vgg16_fusion_context`, `fusion_store_vgg16_fusion_control_all` |
| writes | weights, histories and manifests under `outputs/_cuda/s0918/`; both manifests copied to `notebooks/scripts/provenance/manifests/` |
| reported | the context-fusion contrast |
| outcome | **member** — 2 of the 28 weights the test pass scores |
| record | **verbatim** (the session's row table and its generated command lines) |
| code identity | this session's code-identity check compares 12 files against the copies staged before launch. Two of them — the two store exporters — have since been consolidated into `notebooks/scripts/stores.py` and are reported as retired; the other ten have all been edited since, in comments and formatting only, so none matches its staged digest. The staged copy is the one the two runs used, and it is what the session manifest records. |

---

## 8. Superseded localiser runs

Three runs that settle how the promoted members are trained, and are superseded once they
have. Their manifests stay under `outputs/_cuda/`; they are not copied into the provenance
folder, which holds the final system's models only. All three use run 4's recipe — encoder
vgg16, patch 512, batch 8, two Adam instances at 1e-4 and 1e-3, 5 warm-up epochs then cosine
to 1e-6, loss 0.5 × BCE + 0.5 × soft Dice, float32, 100 epochs requested and 100 run,
selection on validation hard IoU, seed 28 — and differ only where the table says.

| tag | date | card | what it tests | command | outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `localiser_l5_vgg16enc_cachev2_bg40` | 9 September, 15:47 to 17:16 | RTX 4090 | cache v2 with 40 % background patches requested (realised 0.156, limited by geometry) | `python -m src.train_localiser_l4 --merged-dir <cache v2 merged> --val-dir <cache v2 per-lesion> --out-dir <run folder> --run-name localiser_l5_vgg16enc_cachev2_bg40 --encoder vgg16 --epochs 100 --batch-size 8 --background-frac 0.4` — **reconstructed** from the manifest and the trainer's flags; no launcher for it survives | **superseded** by the background-fraction comparison below |
| `localiser_l6_vgg16enc_cachev3_bg40` | 9 September, 20:58 to 23:18 | A100-SXM4-80GB | the same at the cache v3 canvas | `python -m src.train_localiser_l4 --merged-dir /workspace/cache_v3_merged --val-dir /workspace/cache_v3_perlesion --out-dir <run folder> --run-name localiser_l6_vgg16enc_cachev3_bg40 --encoder vgg16 --epochs 100 --batch-size 8 --background-frac 0.4` — **reconstructed**; the overnight launcher records its evaluation and gate steps, not its training call | **superseded**; the canvas is not what limits the localiser |
| `localiser_l6b_vgg16enc_cachev3_bg00` | 10 September, 00:18 to 01:52 | A100-SXM4-80GB | the same with no background patches, isolating the background fraction | `python -m src.train_localiser_l4 --merged-dir /workspace/cache_v3_merged --val-dir /workspace/cache_v3_perlesion --out-dir <run folder> --run-name localiser_l6b_vgg16enc_cachev3_bg00 --encoder vgg16 --epochs 100 --batch-size 8 --background-frac 0.0` — **verbatim**, from the overnight launcher, kept outside the repository | **superseded**; it fixes the background fraction every promoted member uses |

None of these feeds the final system.

---

## 9. Pod work that is cancelled or superseded

Runs and sessions that do not reach the record above. They are listed because the members are
only interpretable against them, and because two of them are why the promoted members train
the way they do. The "trained" column counts runs that completed and produced weights; it
sums to 35, and the eight launches that trained nothing are counted as 0.

| session or run | date | card | trained | what it is | command | how it ends |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| first pod ladder and fusion sweep | 9 September, 10:56 to 12:07 | RTX 4090 | 15 | the seven ladder conditions, three architectures on localiser boxes, four fusion stores and one 448-pixel run | the session driver's own command list, one `src.train_crop_store` or `src.train_fusion_store` line per row | completes; **superseded** by the session in section 6, and none of its weights is scored |
| pod session 3 | 9 September, 15:05 to 15:41 | RTX 4090 | 6 | view and control fusion at seeds 1 and 2, and oracle-at-margin at seeds 1 and 2 | the same two trainers, one line per row | completes; the oracle-at-margin seeds are the incumbent three section 5 does not re-run, the fusion seeds are **superseded** by section 6 |
| pod session 4 | 9 September, 17:21 to 17:52 | RTX 4090 | 5 | localiser-box condition at seeds 1 and 2, and view fusion on U-Net boxes at three seeds | the same two trainers, one line per row | completes; **superseded** by section 6 |
| first encoder attempt, cache v2 | 10 September, 06:42 to 06:53 | RTX 5090 | 0 | ResNet34 with no background patches | the section 2 common command with `--encoder resnet34 --cache <cache v2 merged>` | cancelled at epoch 24; no weights land |
| second encoder attempt, patience 25 (`localiser_l7_resnet34_cachev2`) | 10 September, 06:53 to 07:23 | RTX 5090 | 1 | the same queue with early stopping active | the section 2 common command with `--encoder resnet34 --cache <cache v2 merged> --patience 25` | stopping fires at epoch 35 of 100 on validation noise while the training loss is still falling, so the cosine schedule never anneals and the encoder comparison gains a second variable; the member queued behind it is cancelled at epoch 18 |
| third encoder attempt, no augmentation, mixed precision, unnormalised input (`localiser_l7_noaug_fp16_unnorm`) | 10 September, 07:24 to 08:09 | RTX 5090 | 1 | the same encoder run to its full epoch budget | the section 2 common command with `--encoder resnet34 --cache <cache v2 merged> --no-augment --amp` and normalisation off | completes 100 epochs; **withdrawn** on its settings — seven unintended deviations from run 4's recipe — and the queued second member is cancelled before it starts |
| first detector (`localiser_l10_fasterrcnn_cachev2`) | 10 September, to 16:38 | RTX 5090 | 1 | Faster R-CNN on cache v2, checkpointed on the quantity its gate uses | `python -m src.localise_torch.detector --cache <cache v2 merged> --val-cache <cache v2 per-lesion> --out-dir <run folder> --epochs 30 --batch-size 2 --select-on val_box_iou` | **withdrawn**: it selects on the quantity it is gated by, and peaks at epoch 2 of 30; replaced by the detector in section 2 |
| overnight torch pre-flight | 9 September, 20:48 to 22:53 | A100-SXM4-80GB | 0 | two encoder runs queued behind a cache and gate sequence | the overnight launcher, kept outside the repository, whose queue reaches the environment check before the training steps | the cache and gate steps complete; the torch environment check fails and both encoder runs are skipped |
| run-4 seed 2, first launch | 10 September, 18:22 | RTX 4090 | 0 | the seed-2 replicate | the section 3 command with `--seed 2` | refused before launching: the pre-launch configuration diff does not return zero deviations. Relaunched at 18:45 and completes, as the run counted in section 3 |
| session driver, first launch | 17 September, about 15:11 | RTX 4090 | 0 | the 59-run session | the section 6 commands file | aborts on the first row with no manifest and no weights, the GPU being held by another process; relaunched at 15:15 and completes 59 of 59 |
| promoted-box retrain | 11 September, 15:36 to 15:53 | RTX 4090 | 3 | the localiser-box condition at three seeds on the promoted boxes | the launcher that ran on the pod, kept outside the repository, which issues `python -m src.train_crop_store --store <promoted-box store> --model vgg16 --tag-suffix _promoted --platform-suffix "" --epochs 100 --patience 25 [--seed 1|2]` | the three trainings complete; the two steps after them fail; **superseded** by section 6, which retrains the same condition |
| fusion on promoted boxes | 11 September, about 16:34 to 16:53 | RTX 4090 | 3 | view fusion at three seeds on the promoted boxes | `python -m src.train_fusion_store --store <promoted-box fusion store> --arch vgg16 --tag-suffix _promoted --platform-suffix "" --patience 25 [--seed 1|2]` | completes; **superseded** by section 6 |

The commands in this table are transcribed from the pod launch scripts, or taken from the launchers
they name, all kept outside the repository; the runs that produced weights
have manifests under `outputs/_cuda/` that corroborate their settings.

---

## 10. Laptop runs with no other record

These run on the laptop GPU, from 7 September onward. They are listed here because no
notebook, manifest or launcher in the repository records them. Thirty-six runs. Every one runs
on the Apple M3 Pro with tensorflow-metal in the Keras environment, with batch 32, Adam, binary
cross-entropy and an epoch cap of 100 per stage at the configuration's defaults; each group
states its input. Each date
below is the file date of the run's last history, in UTC, not a recorded time; where a run has
a console log, its timestamps, in local time, agree with it. The runs fall on 7, 8 and 9
September.

### The scratch and transfer models at the fixed margin

Six runs that establish the progression from a small CNN to a pretrained backbone on the same
crops. The three base models have no surviving command; the variants below them do.

| tag | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_crop_m020` | 23 | 7 September, 16:40 | `python -m src.train --model baseline --source crop --tag-suffix _m020` | **reconstructed** from the tag and the trainer's naming rule |
| `regularised_crop_m020` | 100 | 7 September, 17:36 | `python -m src.train --model regularised --source crop --tag-suffix _m020` | **reconstructed** from the tag and the trainer's naming rule |
| `vgg16_crop_m020` | 28 frozen + 68 fine-tune | 7 September, 13:21 | `python -m src.train --model vgg16 --source crop --tag-suffix _m020` | **reconstructed** from the tag and the trainer's naming rule |
| `scaled_crop_m020` | 100 | 8 September, 10:27 | `python -m src.train --model scaled --source crop --tag-suffix _m020` | **verbatim** (the laptop queue launcher, kept outside the repository) |
| `scaled_crop_m020_lr5e-4` | 100 | 8 September, 10:53 | `python -m src.train --model scaled --source crop --tag-suffix _m020 --lr 5e-4` | **verbatim** (the laptop queue launcher, kept outside the repository) |
| `vgg16_crop_m020_overlay` | 21 frozen + 35 fine-tune | 8 September, 11:12 | `python -m src.train --model vgg16 --source crop --tag-suffix _m020_overlay --mask-overlay` | **reconstructed**: the laptop queue launcher, kept outside the repository, holds this line, but the run's console log holds only `zsh: command not found: --mask-overlay`, written two seconds after its last history. Scored on validation, its weights reach AUC 0.7126 with the overlay applied and 0.8652 without it, a drop of 0.153 against 0.147 for `vgg16_crop_m020` (0.7370 and 0.8843), so the run behaves as one trained without the overlay |

`vgg16_crop_m020` is the margin-0.20 point of the sweep below as well as the transfer step
here; it is one run, listed once, and the sweep table cross-references it.

Input: 224×224 crops from the decoded crop pipeline at margin 0.20. Writes: weights, histories
and training logs under `outputs/`. Reported: the stage progression and the overlay comparison. These feed
the report's development record, not the final system.

### The other backbones and the whole-image reference

Four runs that complete the first architecture comparison on the same crops and add the
whole-image reference. Their console logs name each run's tag and source but hold no command
line.

| tag | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- |
| `resnet50_crop_m020` | 41 frozen + 36 fine-tune | 7 September, 14:31 | `python -m src.train --model resnet50 --source crop --tag-suffix _m020` | **reconstructed** from the tag and the trainer's naming rule |
| `densenet121_crop_m020` | 28 frozen + 49 fine-tune | 7 September, 14:19 | `python -m src.train --model densenet121 --source crop --tag-suffix _m020` | **reconstructed** from the tag and the trainer's naming rule |
| `efficientnet_crop_m020` | 27 frozen + 13 fine-tune | 7 September, 15:37 | `python -m src.train --model efficientnet --source crop --tag-suffix _m020` | **reconstructed** from the tag and the trainer's naming rule |
| `vgg16_full` | 15 frozen + 21 fine-tune | 7 September, 14:42 | `python -m src.train --model vgg16 --source full --tag-suffix _full` | **reconstructed** from the tag and the trainer's naming rule |

Input: 224×224, from the decoded crop pipeline at margin 0.20, or the whole mammogram resized
for `vgg16_full`. Writes: weights, histories, training logs and a console log per run under
`outputs/`. Reported: the compendium's development record (section C); the protocol's "Models
entering the pass" and its amendments §33 and §28, and for the three backbones its secondary
families; `outputs/results/stage_table.csv`, `experiment_matrix.csv` and
`validation/comparison_architectures.json` (`validation/analysis_vgg16_full.json` for the
whole-image run); `figures.py`'s stage table (`STAGE_ROWS`).
Outcome: **superseded** by the architecture family and the whole-image reference of the
session in section 6.

Six local regularisation variants were planned on the laptop; the family moved to Kaggle
(section 4), and no file shows they ran. They are not counted.

### The first ladder

Five rungs of the ladder at one seed, trained by the ladder's train stage; rungs 3 and 5 train a
day later, on the candidate bundle's boxes, below.

| tag | what it tests | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vgg16_crop_rung1_oracle` | the mask's own box, no margin | 26 frozen + 25 fine-tune | 7 September, 15:46 | `python -m src.ladder --arch vgg16 --margin 0.20 --stage train --rungs 1 2 4 6 7` | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung2_oracle` | the mask's box at margin 0.20 | 47 frozen + 31 fine-tune | 7 September, 15:59 | the same command | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung4_cam` | the activation-map box | 26 frozen + 14 fine-tune | 7 September, 16:07 | the same command | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung6_shuffled` | another lesion's box | 16 frozen + 24 fine-tune | 7 September, 16:14 | the same command | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung7_whole` | no localisation | 16 frozen + 30 fine-tune | 7 September, 16:23 | the same command | **reconstructed** from the tag and the trainer's flags |

Input: 224×224 crops from each rung's boxes at margin 0.20 (rung 1 without margin). Writes:
weights, histories and training logs under `outputs/`; no console log survives. Reported: the
protocol's "Models entering the pass" and amendment §28, and for rung 2 also §10, §14 and §14a;
`outputs/results/experiment_matrix.csv`, and for rungs 2, 4, 6 and 7
`validation/comparison_familyB_val_metal.json`. Outcome: **superseded** by the ladder of the
session in section 6.

### Rungs 3 and 5 on the candidate bundle's boxes

The localiser-box and jittered-box rungs on the boxes of the candidate localiser bundle, with two
further seeds each, and the three other backbones at rung 3.

| tag | what it tests | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vgg16_crop_rung3_unet_bundle` | rung 3 on the bundle's boxes | 23 frozen + 31 fine-tune | 8 September, 15:30 | `python -m src.ladder --arch vgg16 --margin 0.20 --stage train --rungs 3 5 --boxes-dir <bundle boxes> --rung3-suffix _bundle` | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung3_unet_bundle_s1`, `_s2` | its seeds 1 and 2 | 21 + 17 and 21 + 25 | 8 September, 15:45 and 15:53 | `python -m src.train --model vgg16 --source crop --crop-sizing box_provider --box-source unet --boxes-dir <bundle boxes> --crop-margin 0.20 --tag-suffix _rung3_unet_bundle --seed 1` (and `--seed 2`) | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung5_jitter_bundle` | rung 5 from the bundle's error model | 28 frozen + 20 fine-tune | 8 September, 15:38 | the rung-3 ladder command above | **reconstructed** from the tag and the trainer's flags |
| `vgg16_crop_rung5_jitter_bundle_s1`, `_s2` | its seeds 1 and 2 | 58 + 23 and 27 + 25 | 8 September, 17:08 and 17:54 | as the rung-3 seeds with `--box-source jitter --tag-suffix _rung5_jitter_bundle` | **reconstructed** from the tag and the trainer's flags |
| `resnet50_crop_rung3_unet_bundle` | ResNet50 at rung 3 | 18 frozen + 18 fine-tune | 8 September, 18:31 | `python -m src.ladder --arch resnet50 --margin 0.20 --stage train --rungs 3 --boxes-dir <bundle boxes> --rung3-suffix _bundle` | **reconstructed** from the tag and the trainer's flags |
| `densenet121_crop_rung3_unet_bundle` | DenseNet121 at rung 3 | 36 frozen + 11 fine-tune | 8 September, 18:39 | as above with `--arch densenet121` | **reconstructed** from the tag and the trainer's flags |
| `efficientnet_crop_rung3_unet_bundle` | EfficientNetB0 at rung 3 | 25 frozen + 13 fine-tune | 8 September, 18:43 | as above with `--arch efficientnet` | **reconstructed** from the tag and the trainer's flags |

Nine runs. Input: 224×224 crops from the bundle's boxes at margin 0.20. Writes: weights,
histories and training logs under `outputs/`, and a console log for each run except the two
seed-28 VGG16 rungs. Reported: the protocol's "Models entering the pass" and "Primary endpoint"
(rung 3 at seed 28), "Localiser selection" (rungs 3 and 5 at seed 28), and amendments §34, §2,
§3, §5 and §28;
`outputs/results/seed_spread_vgg16_crop_rung3_unet_bundle.csv`,
`seed_spread_vgg16_crop_rung5_jitter_bundle.csv`, `vgg16_crop_rung3_unet_bundle_ens3_ensemble.json`,
`rung3_arch_ens4_ensemble.json`, `uncertainty_vgg16_crop_rung3_unet_bundle.json`,
`explain_vgg16_crop_rung3_unet_bundle_val.json`, `ladder_error_analysis_val.json` and
`experiment_matrix.csv`. Outcome: **superseded** by the promoted-box retrain in section 9 and the
session in section 6.

### The first fusion runs

| tag | what it tests | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vgg16_fusion_view` | CC and MLO views of one breast | 28 frozen + 23 fine-tune | 7 September, 18:47 | `python -m src.fusion --arch vgg16 --pair view` | **reconstructed** from the tag and the trainer's flags |
| `vgg16_fusion_control` | the single-crop control on the view-paired rows | 22 frozen + 23 fine-tune | 7 September, 18:26 | `python -m src.fusion --arch vgg16 --pair control` | **reconstructed** from the tag and the trainer's flags |
| `vgg16_fusion_context` | one lesion at margins 0.20 and 1.00 | 42 frozen + 16 fine-tune | 7 September, 19:06 | `python -m src.fusion --arch vgg16 --pair context` | **reconstructed** from the tag and the trainer's flags |
| `vgg16_fusion_control_all` | the control on every lesion | 37 frozen + 52 fine-tune | 8 September, 07:13 | `python -m src.fusion --arch vgg16 --pair control --rows all`, rebuilt from the tag, which records the pair mode and the row set only. Its console log holds only `zsh: command not found: 0.20`, written at 06:56:43, before the run's first weights at 07:02. The line before it can only have ended in `--margin`, the trainer's one flag that takes 0.20, and a `--margin` with no value stops the parser (exit 2) rather than falling back to the default; so that launch did not train, and neither the launch that did nor its margin is recorded | **reconstructed** from the tag and the trainer's flags |
| `vgg16_fusion_view_unet_bundle` | view fusion on the bundle's boxes | 20 frozen + 37 fine-tune | 8 September, 18:21 | `python -m src.fusion --arch vgg16 --pair view --box-source unet --boxes-dir <bundle boxes> --tag-suffix _bundle` | **reconstructed** from the tag and the trainer's flags |

Five runs. Input: a pair of 224×224 crops per row. Writes: weights, histories, training logs
and a console log per run under `outputs/`. Reported: the protocol's "Fusion (outside both
families)", where `vgg16_fusion_view` against `vgg16_fusion_control` is the +0.0413
view-minus-control figure (0.8964 against 0.8551 on the 100 validation pairs, recomputed from
their stored validation tables) and `vgg16_fusion_view_unet_bundle`, at 0.8872, is its declared
re-run; "Models entering the pass" (the control), "Localiser selection" and amendments §1, §5,
§10, §13, §27, §28, §37 and §39;
`outputs/results/experiment_matrix.csv`, and for the context and control-all runs
`validation/family_dry.json` and `validation/smoke_ens_control_context_ensemble.json`, with
`validation/analysis_vgg16_fusion_control_all.json`. Outcome: **superseded** by the fusion rows
of the sessions in sections 6 and 7.

Six smoke tests, `vgg16_fusion_view_smoke`, `vgg16_fusion_control_smoke`,
`vgg16_fusion_context_smoke`, `vgg16_fusion_control_all_smoke`, `l4_smoke` and
`l4c_smoke_train`, run on 7 and 8 September and are kept under `outputs/_superseded/smoke/`; like
the 6 September smoke test, they are not counted.

### Other laptop runs

| tag | what it tests | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `scaled_crop_m020_augmented` (trained as `scaled_crop_m020`) | the scaled CNN with training augmentation | 100 | 7 September, 17:06 | `python -m src.train --model scaled --source crop --tag-suffix _m020`, under the trainer of that date; that it trained with augmentation is recorded by its folder name and by the session notes kept outside the repository, not by its console log | **reconstructed** from the tag and the trainer's flags |
| `rerank_cnn_norule` | a CNN that re-ranks the localiser's candidate boxes without the rule features | 38 frozen + 25 fine-tune | 9 September, 21:04 | `python -m src.rerank --model cnn --feature-set norule --tag rerank_cnn_norule --out-dir outputs/results/rerank_cnn_norule` | **reconstructed** from the tag and the trainer's flags |

Writes: for the scaled run, a history, weights, a training log and a console log, moved by the
laptop queue launcher to `outputs/_superseded/scaled_crop_m020_augmented/`; for the re-ranker,
weights, a result record holding its history, box tables and the console log
`outputs/logs/rerank_cnn_local.log`. Input: 224×224 crops for the scaled run; for the
re-ranker, a 224×224 candidate crop with the candidate's feature vector, labelled by whether the
candidate is the lesion, at patience 25. Reported: the scaled run in `figures.py`'s stage-6
curves, as superseded; the re-ranker nowhere but its own result record. Outcome: **superseded**.

### The first crop-margin sweep

One backbone at six margins, which is how the crop margin is first chosen. Five runs are
listed here; the sixth, margin 0.20, is the `vgg16_crop_m020` run in the first table above.
The sweep is superseded by the one on the operational crop route in section 5, and the report treats
this curve as history rather than as evidence, because the metric that selected each run's
epoch is the one the compiled graph corrupts on this machine.

| tag | margin | epochs run | history date | command | record |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `vgg16_crop_m000` | 0.00 | 21 frozen + 30 fine-tune | 7 September, 12:55 | `python -m src.train --model vgg16 --source crop --crop-sizing mask_margin --crop-margin 0.00 --tag-suffix _m000` | **reconstructed** |
| `vgg16_crop_m010` | 0.10 | 36 frozen + 13 fine-tune | 7 September, 13:04 | as above with `--crop-margin 0.10 --tag-suffix _m010` | **reconstructed** |
| `vgg16_crop_m040` | 0.40 | 31 frozen + 34 fine-tune | 7 September, 13:33 | as above with `--crop-margin 0.40 --tag-suffix _m040` | **reconstructed** |
| `vgg16_crop_m080` | 0.80 | 31 frozen + 29 fine-tune | 7 September, 13:44 | as above with `--crop-margin 0.80 --tag-suffix _m080` | **reconstructed** |
| `vgg16_crop_m100` | 1.00 | 44 in the full history | 7 September, 13:52 | as above with `--crop-margin 1.00 --tag-suffix _m100` | **reconstructed** |

Each line is rebuilt from the tag and the trainer's flags; none is a recorded command. Input:
224×224 crops at each run's margin. Writes:
weights, histories and training logs; the widest margin's three checkpoints are in
`outputs/weights/_superseded/`. Reported: the crop-margin appendix, as history.

---

## Where the records live

| kind | location |
| :--- | :--- |
| Kaggle notebooks, as run | `notebooks/scripts/provenance/kaggle/` |
| member and train manifests of the final system's models | `notebooks/scripts/provenance/manifests/` |
| pod launchers, with each launcher's self-copy | the launchers that ran on the pods, kept outside the repository |
| the 59-run session: declared rows and generated commands | `notebooks/scripts/session_cuda0917_rows.json`, `notebooks/scripts/session_cuda0917_commands.txt`; the session driver, kept outside the repository |
| the context-fusion session: declared rows and generated commands | `notebooks/scripts/session_cuda0918_rows.json`, `notebooks/scripts/session_cuda0918_commands.txt`, `notebooks/scripts/session_cuda0918_expect.json`; the session is replayed with `session.py --session cuda0918`; the session driver, kept outside the repository |
| session manifests, completed-row logs and per-row training logs | `outputs/_cuda/s0917/session/`, `outputs/_cuda/s0918/session/` |
| pod queue and landing logs for the superseded sessions | `outputs/_cuda/session2.log`, `session2_landing.log`, `session3_ab.log`, `session4_d.log`, `session4_e.log`, `outputs/_cuda/overnight/`, `outputs/_cuda/s8/` |
| manifests and histories of the superseded localiser runs | `outputs/_cuda/localiser_*/` |
| the ceiling and margin group | `outputs/_cuda/s8_results/` |
| run 4's recipe, published | `refs/run4_recipe.json`, `refs/run4_manifest.json` |

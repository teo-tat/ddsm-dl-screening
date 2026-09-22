# Seeds and sources of randomness

Every random draw in the pipeline is listed with its seed and the code that
consumes it. The master seed is `config.SEED = 28`; `config.set_seeds` applies
it to `PYTHONHASHSEED`, `random`, NumPy and TensorFlow. Metal is not
seed-deterministic, so two identical runs on the laptop differ at the noise
level measured by the rung-2 replicates (`config.VAL_AUC_NOISE_FLOOR = 0.02`
on validation AUC); numbers are single-seed unless a `_s<N>` tag says otherwise.

| Draw | Seed | Where | Notes |
|---|---|---|---|
| Patient partition (train / val / test) | 28 | `data_loader._three_way_split` via `StratifiedGroupKFold(shuffle=True, random_state=seed)` | Never varied. `patient_partition` is keyed on (seed, test fraction, val fraction). Identical for the classifier, the localiser, the fusion pairs and the BI-RADS comparison set |
| Weight initialisation | 28, or `--seed N` | `config.set_seeds` before `build_model` (`train._reseed`) | A replicate seed re-seeds after the datasets are built, so the partition stays on 28 |
| Training-set shuffle | 28, or `--seed N` | `data_loader._make_dataset`: `ds.shuffle(seed=…, reshuffle_each_iteration=True)`, applied after `.cache()` | Shuffling before the cache would freeze epoch-1 order; the crop-store trainer (`train_crop_store`) uses the same call |
| Online augmentation | 28, or `--seed N` | `data_loader._make_batch_augment_fn` (TF global seed) | Flip, rotation, zoom, contrast, gamma, brightness on the crop path; flip, rotation, zoom on the whole-image path |
| Localiser training shuffle and flips | 28 | `train_localiser_l4.whole_image_dataset` (shuffle), `localise._augment_pair` (flip and rotation, TF global seed) | Runs 1-3 whole images; run 4 patches |
| Run-4 patch plan | 28 + epoch | `train_localiser_l4.PatchSampler.epoch_plan`: `default_rng(seed + epoch)` | One lesion-centred patch with 25 % jitter and one anywhere in the breast box, per image per epoch; the realised fractions are logged |
| PyTorch localiser augmentation | 28 + 100000 + epoch | `src/localise_torch/train.py:223`: `default_rng(seed + 100000 + epoch)` | One generator per epoch, so an epoch's augmentation replays from its index; `dataset.augment_pair` draws a horizontal flip (p = 0.5) and a rotation angle per patch |
| Jitter rung (rung 5) | sha256(`lesion_id:28`) | `boxes._stable_rng` | Per-lesion generator, so a lesion gets the same perturbation every epoch and every run; independent of iteration order |
| Shuffled rung (rung 6) donors | 28 | `boxes.BoxProvider`: `default_rng(seed)` permutation within (laterality, view) buckets | A single cycle, so the mapping is a derangement; computed over the splits the provider is given |
| Patient-cluster bootstrap | 28 | `evaluate.bootstrap_ci`: `RandomState(seed)`, 1 000 draws | Used by compare.py, the localiser gate and ensemble.py; patients stratified on carrying a malignant lesion |
| Monte Carlo dropout | 28 | `config.set_seeds()` in `uncertainty.main`, through Python's `random`: each Keras 3 Dropout layer draws its seed from it when built | Not `tf.random.set_seed`: `uncertainty.collect` calls it before the passes, but the stateless dropout masks do not read it. Validation by default; test only inside the test pass |
| Crop-store export | none | `stores.py crop` | Deterministic: decode, crop, resize in frame order; shuffle and augmentation are re-applied at training time with the seeds above |
| Kaggle replicates (`_kg` tags) | 28 | `train_crop_store.main` | Same seeds as the local path; different hardware and TF version, hence the replica baseline |

Not seeded: DICOM decoding, intensity normalisation, letterboxing, Otsu breast
box, connected-component labelling, threshold sweeps. All are deterministic
functions of the input.

## PyTorch localiser members

| run | encoder | seed | environment | notes |
| :--- | :--- | ---: | :--- | :--- |
| L7 (withdrawn) | SMP U-Net `resnet34` (24.4 M) | 28 | `.torch-env`, torch 2.8.0, smp 0.4.0 | `torch.manual_seed(28)` + `np.random.seed(28)`; the patch plan is a pure function of `seed + epoch`, as in the Keras sampler. Withdrawn: no reported result uses it |
| L8 (withdrawn) | SMP U-Net `efficientnet-b4` (20.2 M) | 28 | same | Withdrawn: no reported result uses it |
| l10b | torchvision `fasterrcnn_resnet50_fpn_v2`, COCO weights, 2 classes | 28 | torch 2.8.0+cu128, RTX 5090 | The detector; its validation result is reported, but it is not a member of the promoted ensemble. `torch.manual_seed(28)` + `np.random.seed(28)`, and the flip and rotation draws use `default_rng(seed + 100000 + epoch)` inside the dataset (`src/localise_torch/detector.py:65`), so augmentation is reproducible independently of loader order |
| L9 | SMP U-Net `tu-convnext_tiny` (31.9 M) | 28 | torch 2.8.0+cu128, smp 0.4.0, RTX 5090 | Member of the promoted localiser ensemble; seeding as l7v. Its encoder weights come from the Hugging Face Hub, which needs no `HF_TOKEN` |
| l7v | SMP U-Net `vgg16` (23.7 M) | 28 | torch 2.8.0+cu128, smp 0.4.0, RTX 5090 | Member of the promoted localiser ensemble. `torch.manual_seed(28)` + `np.random.seed(28)`; the patch plan is a pure function of `seed + epoch` and the augmentation of `seed + 100000 + epoch` |
| l7a | SMP U-Net `resnet34` (24.4 M) | 28 | same as l7v | Member of the promoted localiser ensemble; seeding as l7v |
| l8a | SMP U-Net `efficientnet-b4` (20.2 M) | 28 | same as l7v | Member of the promoted localiser ensemble; seeding as l7v |

Seed 28 throughout, as everywhere else in the project. These runs are **not** bitwise reproducible
against the Keras family — different framework, different kernels — and are not meant to be: they are a
controlled *architecture* comparison on identical data, identical patches, identical loss and identical
scoring (`src/localise_torch/evaluate.py` imports `localiser_bundle`'s scorer rather than
re-implementing it). cuDNN autotuning is left at its default, so run-to-run variation on one machine is
possible at the third decimal; the localiser gate's noise floor is 0.02 and single-seed, so this is well
inside it.

# refs/ — load-bearing references. Nothing here is cleaned.

**The distinction this directory exists to make structural: disposable vs load-bearing.**

`outputs/` holds what runs produce. It is cleaned — on the laptop when superseded, on a pod when the
volume fills. `refs/` holds what the pipeline *reads* to decide things, and it is never cleaned.
Two of its files are written by tools: `run4_recipe.json` is a copy of `docs/run4_recipe.json`,
which `notebooks/scripts/recipe.py build` writes, and `promoted_pointer.json` is written by
`notebooks/scripts/promotion_cascade.sh` (`:517–518`; a dry run writes
`promoted_pointer.dryrun.json` instead).

On 10 September a volume cleanup on pod B ran `rm -rf outputs/weights/*` to free 5.7 GB of finished
checkpoints. It also removed run 4's reference manifest, which had been placed there because that is
where run 4's own weights live — and the next candidate's config diff had nothing to compare
against, so the pod sat idle. The weights were verified byte-exact before deletion; nothing asked
whether anything else in that directory was load-bearing.

**This is the same shape as "exists is not current":** a property that has to be remembered at the
moment of acting is a property that will eventually be forgotten. Put the references somewhere the
cleanup cannot reach, and the question stops being asked.

| file | read by |
| :--- | :--- |
| `run4_manifest.json` | `notebooks/scripts/recipe.py check` (`RUN4`, `:261`; the `--reference` default, `:551`; read at `:558`) — the reference every candidate's settings are diffed against |
| `run4_recipe.json` | no committed reader; kept as a record. It is a byte-identical copy of `docs/run4_recipe.json`, which `notebooks/scripts/test_localise_torch_port.py:176` reads and `notebooks/scripts/promotion_cascade.sh:194` requires — run 4's 62 declared fields. Two values are kept as written because `recipe.py check` compares candidates against the same values (`RUN4_CODE`, `notebooks/scripts/recipe.py:264` and `:278–281`): `framework` says keras-tf2.17, but run 4 ran on TensorFlow 2.20.0 (its Kaggle notebook, `notebooks/scripts/provenance/kaggle/localiser_run4_vgg16enc_patch.ipynb`, cell 2); `decoder` says base_filters 32, depth 4, but the pretrained U-Net has five upsampling stages, with filters (32, 64, 128, 256, 512) fixed in `src/localise.py:247` rather than taken from `LOC_DEPTH` or `LOC_BASE_FILTERS`. The "depth 4" text is also what the trainer writes into its manifests (`src/train_localiser_l4.py:348–351`), and the run-4 seed specs carry it (`notebooks/scripts/provenance/manifests/spec_run4_s1.json`, `spec_run4_s2.json`). |
| `promoted_pointer.json` | `notebooks/scripts/run_test_pass.sh:111–114` and `:150–159`, and `notebooks/scripts/integrity.py:255–257` (`verify-test-pass`) and `:3030–3032` (`cascade-handoff`) — which boxes and tag suffix a promotion produced |
| `test_maps_manifest.json` | `notebooks/scripts/run_test_pass.sh:320` — the sha256 of each test probability-map array and of `test_meta.csv`, which the pass checks before it applies the bundle's boxes |

Three more files are written here by the freeze, the test pass and the cascade's dry run, and are
not committed:
`prereg_head.sha256` (`notebooks/scripts/freeze.py:24`; read at `run_test_pass.sh:72–73`),
`test_pass_started.json` (`run_test_pass.sh:53`) and `promoted_pointer.dryrun.json`
(`promotion_cascade.sh:517`).

**`promoted_pointer.json` changed after the test pass.** On 21 September 2026, by the file's
modification time, one string in it was reworded in place: the first entry of
`precomputed_halves.rung3.reasons`. Putting the old wording back reproduces the sha256 recorded in
the pass's code manifest, so nothing else in the file changed. The pass does not read that string.
`notebooks/scripts/promotion_cascade.sh`, which writes the file, did not make the edit: it restamps
`written_at` on every write (`:542`), and `written_at` still reads 2026-09-18T06:15:39Z. No record
names what made the edit or why. The fields the pass read are unchanged and agree with the pass log,
which records the same `boxes_dir` and `rung3_suffix` (read at
`notebooks/scripts/run_test_pass.sh:113–114`) and shows that its checks of `boxes_md5`,
`bundle_selection` and `dry_run` (`:150–159`) passed. No copy of the file as it stood at the pass
was kept.

**Also load-bearing, and already protected elsewhere:** `artifacts/` (the frozen inputs listed in
`artifacts/README.md`, which are never modified or regenerated; `src/localise_eval.py:514–518`,
`src/birads_baseline.py:77–81` and `src/fusion.py:153–157` refuse to overwrite them).

**Rule for any cleanup script:** it may delete only under directories holding generated artefacts —
`outputs/`, `/workspace/runs`, `/workspace/stores` — and must refuse `refs/` and `artifacts/`
explicitly rather than by omission.

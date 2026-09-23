# artifacts/ — frozen inputs

Files here are written once and never modified or regenerated. Code and the notebooks
read them, except the two metrics files, which are records. A
file's producer is recorded here rather than inside the file, because editing a
frozen artefact to add provenance would itself break the freeze.

| artefact | written by | what it is |
| :--- | :--- | :--- |
| `birads_baseline.json` | `src/birads_baseline.py` | the BI-RADS anchor per partition |
| `birads_comparison_set.csv` | `src/birads_baseline.py` | the image rows the primary endpoint is scored on (read by `src/compare.py:68`, `src/analysis.py:170` and `notebooks/scripts/figures.py:869` via `config.BIRADS["comparison_set"]`) |
| `cam_boxes.csv` | `src/cam_localise.py` | weakly-supervised CAM boxes, the rung-4 box source |
| `cam_manifest.json` | `src/cam_localise.py` | the CAM run's configuration |
| `cam_metrics.json` | `src/cam_localise.py` | CAM localisation metrics |
| `localiser_boxes_train.csv`, `localiser_boxes_val.csv`, `localiser_boxes_test.csv` | `src/localise_eval.py` (run 1) | the run-1 localiser's box tables: the ladder's ground-truth box frame and the default reference of the localiser gates (`--reference`) |
| `localiser_error_model.json` | `src/localise_eval.py` (run 1) | run 1's measured validation box error, which the jitter rung resamples when no candidate folder (`--boxes-dir`) is given |
| `localiser_metrics.json` | `src/localise_eval.py` (run 1) | run 1's localisation metrics |
| `fusion_pairs_view_train.csv`, `fusion_pairs_view_val.csv`, `fusion_pairs_view_test.csv` | `src/fusion.py` (`FusionRun` pair construction) | the CC/MLO view pairs, fixed once; the test pass compares the fusion store's metadata with these tables (`notebooks/scripts/run_test_pass.sh:451–465`), and `src/fusion.py` checks their lesion ids when writing pairs (`src/fusion.py:147–151`) |
| `fusion_pairs_view_single_train.csv`, `_val.csv`, `_test.csv` | `src/fusion.py` (`--single-mass-only`) | the same pairs restricted to single-mass images |
| `lesion_geometry.csv` | `notebooks/02_lesion_geometry.ipynb` | per-lesion geometry (size, position) used for stratification |
| `lesion_bbox_lcc.csv` | `notebooks/02_lesion_geometry.ipynb` | per-lesion component statistics of each mask (component and speck counts, the largest component's share of the foreground, and the areas and overlap of the whole-foreground box and the largest component's box); it holds no coordinates |

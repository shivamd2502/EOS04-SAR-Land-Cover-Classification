# Scripts, by task

Run from the project root, in folder order. Every file starts with a **PROJECT CONTEXT** block
(why, results, how to run) followed by its original documentation.

| Folder | Scripts |
|---|---|
| `01_calibration/` | `eos04_radiometric_preprocessing.py` (ISRO eqs. 1-15, `--self-test`), `compare_calibration.py` (imports the first, keep them together) |
| `02_composites_and_patches/` | `generate_rgb_composites.py`, `composite_to_patches.py` (any size + DEM stats), `composite_to_patches_128.py`, `fix_exported_csv.py` |
| `03_dem_terrain/` | `collect_dem_tifs.py` |
| `04_label_validation/` | `validate_autolabels.py` |
| `05_classification_training/` | `eos04_train.py` (64/224/256 px), `eos04_train_128.py`, `eos04_train_dem.py` (SAR + DEM) |
| `06_scene_prediction/` | `eos04_predict_scene.py`, `eos04_predict_scene_128.py` |
| `07_model_evaluation/` | `collect_all_model_stats.py`, `dem_ablation_study.py`, `compare_patch_models.py`, `fair_model_comparison.py`, `prove_dem_improvement.py` |
| `08_segmentation_unet/` | `generate_seg_masks.py`, `train_unet_256.py`, `evaluate_unet_256.py` |

Renamed files (code unchanged):

| Old name | New name |
|---|---|
| `eos04_train_autolabelClassifer.py` | `05_classification_training/eos04_train.py`, the name its own usage text uses |
| `eos04_train_autolabelclassifierDEM.py` | `05_classification_training/eos04_train_dem.py` |
| `train_unet_eos04_aug.py` | `08_segmentation_unet/train_unet_256.py`, the counterpart of `evaluate_unet_256.py` |

Paths are set in the configuration block of each script (`BASE_DIR`, `PATCHES_ROOT`,
`MODEL_DIR`, `OUTPUT_DIR`). `eos04_train.py` was run several times with different
`PATCHES_ROOT` / `MODEL_DIR` values, one per patch size.

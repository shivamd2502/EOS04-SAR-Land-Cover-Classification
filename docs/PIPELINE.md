# Pipeline, step by step

All scripts read and write paths under `BASE_DIR\data\` (`BASE_DIR = r"D:\ISRO_14"`). Run them
from the project root. `scripts/`, `results/` and the local `data/` folder share the same
numbering (`00_input` holds the raw scenes and DEM ZIPs).

| Step | Script | Input → output | Why |
|---|---|---|---|
| 01 | `eos04_radiometric_preprocessing.py` | `data/00_input/raw/<scene>/` (DN, BAND_META.txt, LIA, mask, area) → `data/01_calibration/calibrated_L2B/03JUN2026_HH|HV/*.tif` | DN are not physical. L2B stores terrain-normalised γ⁰ referenced to β⁰, so the ISRO equations are required. |
| 01 | `compare_calibration.py` | ISRO σ⁰ vs a generic 3-step σ⁰ → difference table + GeoTIFF | Shows why the generic recipe is wrong for EOS-04 L2B. |
| 02 | `generate_rgb_composites.py` | calibrated HH/HV → `data/02_composites_and_patches/composites_L2B/03JUN2026/*.tif`, `ALL_composites.png` | A pre-trained CNN needs 3 channels. OPT1 = (HH σ⁰, HV σ⁰, HH/HV). |
| 02 | `composite_to_patches.py` (64/224/256 px, DEM stats), `composite_to_patches_128.py` | composite (+ `data/03_dem_terrain/DEM_TIFS/`) → `data/02_composites_and_patches/patches_<size>_L2B/03JUN2026/<composite>/patches/*.png` + `.shp` | Grid patches with per-patch SAR/DEM statistics for labelling in QGIS. |
| 02 | `fix_exported_csv.py` | QGIS attribute-table CSV → `<composite>_labels_clean.csv` | Training scripts need clean headers without geometry. |
| 03 | `collect_dem_tifs.py` | `data/00_input/DEM/*.zip` (Copernicus DSM) → `data/03_dem_terrain/DEM_TIFS/*.tif` | Terrain context for the hill/urban confusion caused by foreshortening. |
| 04 | `validate_autolabels.py` | labelled patches → `data/04_label_validation/label_validation/` | Checks the HH/HV-rule labels (Kruskal–Wallis, montages, optional Sentinel-2 NDVI). |
| 05 | `eos04_train.py` | 224/256/64 px patches + labels → `data/05_classification_training/models/EOS04_<size>_L2B/` | EfficientNet-B0 + SAR-feature MLP, 2-phase fine-tuning, class-weighted CE. |
| 05 | `eos04_train_128.py` | 128 px patches → `data/05_classification_training/models/EOS04_128_L2B/` | Same recipe on finer patches. |
| 05 | `eos04_train_dem.py` | 256 px patches with DEM stats → `data/05_classification_training/models/EOS04_256_L2B_DEM/` | 9-feature MLP (4 SAR + 5 DEM). |
| 06 | `eos04_predict_scene.py`, `eos04_predict_scene_128.py` | model + composite → `predictions_<size>_L2B/03JUN2026/` (class map, confidence GeoTIFF, RGB, CSV) | The deliverable is a georeferenced map. |
| 07 | `collect_all_model_stats.py` | all 4 checkpoints → `data/07_model_evaluation/ALL_STATS_L2B/` | One consistent report for every patch size. |
| 07 | `compare_patch_models.py`, `fair_model_comparison.py` | 256 vs 128 models → `data/07_model_evaluation/comparison*/` | Patch-size comparison (the "fair" one needs fixing, see RESULTS). |
| 07 | `dem_ablation_study.py` | same data, SAR-only vs SAR+DEM → `data/07_model_evaluation/ablation_study/` | Controlled DEM test with McNemar. |
| 07 | `prove_dem_improvement.py` | hard-coded metrics → `data/07_model_evaluation/dem_proof/` | Superseded by the ablation (see RESULTS). |
| 08 | `generate_seg_masks.py` | HH/HV band → `data/08_segmentation_unet/seg_masks/03JUN2026/` (full mask + image/mask patches) | Per-pixel pseudo-labels with the same thresholds. |
| 08 | `train_unet_256.py` | 256 px image/mask pairs → `data/08_segmentation_unet/unet_output_256/` | Pixel-level segmentation. |
| 08 | `evaluate_unet_256.py` | U-Net checkpoint → `data/08_segmentation_unet/unet_evaluation_256/` | Visual and metric check per patch. |

## Model (step 05)

```
PNG patch (224×224 after resize) ──► EfficientNet-B0 (ImageNet) ──► 1280-d ─┐
                                                                             ├─► concat ─► Dropout 0.3 ─► Linear 256 ─► Dropout 0.2 ─► Linear 3
SAR features (z-scored, 4 or 9) ──► Linear 32 ─► BN+ReLU ─► Dropout ─► Linear 32 ─┘
```

* Phase 1 (epochs 1–20): backbone frozen, only the MLP and head are trained.
* Phase 2 (from epoch 21): the last two EfficientNet blocks are unfrozen.
* Settings: AdamW, LR 1e-4, batch 16, 100 epochs.
* Loss: inverse-frequency class weights.
* Augmentation: horizontal/vertical flips and rotation.

## Labelling rule (steps 02, 04, 08)

HH/HV ratio in dB, applied to patch means in QGIS and per pixel in `generate_seg_masks.py`:

| HH/HV (dB) | Class |
|---|---|
| < 6.8 | Dense forest (strong volume scattering → high HV) |
| 6.8 – 8.2 | Mix |
| 8.2 – 12.3 | Urban (double bounce → high HH) |
| ≥ 12.3 | Unclear (dropped) |

Layover and shadow pixels from the ISRO mask are excluded (value 255 in the segmentation masks).

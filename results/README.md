# Results, by task

Copies of the outputs written by the scripts with the same number. Model weights and large
rasters stay on the training PC (see `docs/DATA.md`). Interpretation: `docs/RESULTS.md`.

| Folder | What is inside | Key number |
|---|---|---|
| `01_calibration/` | local incidence angle and layover/shadow mask quick-looks of the 03 Jun 2026 scene | – |
| `02_composites_and_patches/` | `ALL_composites.png`: raw-DN baseline vs σ⁰ / γ⁰ / RVI composites | OPT1 σ⁰ chosen |
| `03_dem_terrain/` | `dem_tile_inventory.csv`, the Copernicus DSM tiles used | 50 tiles |
| `04_label_validation/` | Kruskal–Wallis results, feature box plots, 10×10 montages per class | HH/HV H = 842.7 |
| `05_classification_training/<model>/` | training curves, confusion matrix, label encoder, feature stats | 256 px L2B: 94.50 % |
| `06_scene_prediction/<model>/` | colour land-cover map and per-patch CSV (class, confidence, row, col) | 1,262 patches at 256 px |
| `07_model_evaluation/all_models_stats_L2B/` | report, CSVs and plots for all 4 patch sizes | 93.9 – 94.5 % |
| `07_model_evaluation/dem_ablation_study/` | SAR-only vs SAR+DEM: curves, F1, confusion matrices, McNemar | 94.0 vs 92.0 %, p = 0.22 |
| `07_model_evaluation/fair_comparison_L2B/`, `comparison_v_pre_L2B/` | 256 vs 128 px comparisons | "fair" runs are invalid, see docs/RESULTS.md §3 |
| `08_segmentation_unet/training/` | loss / IoU / Dice curves and epoch-50 example predictions | val mIoU ≈ 0.94 |
| `08_segmentation_unet/evaluation_comparisons/` | composite / pseudo-mask / U-Net prediction for 10 patches | – |

Model folder names in `05_*`:
* `EOS04_L2B` (256 px), `EOS04_224_L2B`, `EOS04_128_L2B`, `EOS04_64_L2B`: final L2B models;
* `EOS04_256_L2B_DEM`: the 9-feature SAR + DEM model;
* `EOS04` and `EOS04_128`: earlier runs on the first (pre-L2B) composites, 93.5 % and 91.3 %.

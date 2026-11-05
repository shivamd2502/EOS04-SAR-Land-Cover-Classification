# Results and how far they go

All numbers below come from the saved outputs in `results/` (copied from `model/`, `validation/`,
`predictions_*` and `unet_*` on the training PC). Nothing was re-trained for this write-up.
Scene: EOS-04 MRS, 03 June 2026, Himachal Pradesh. Composite: OPT1 σ⁰ (HH σ⁰ dB, HV σ⁰ dB, HH/HV).

## 1. Patch classifier, four patch sizes (`results/07_model_evaluation/all_models_stats_L2B/`)

| Patch | Labelled | Train / Val | Best epoch | Val acc | Weighted F1 | Macro F1 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 11,663 | 9,330 / 2,333 | 34 | 94.17 % | 0.9415 | 0.9422 | 0.866 |
| 128 | 3,910 | 3,128 / 782 | 22 | 93.99 % | 0.9393 | 0.9268 | 0.890 |
| 224 | 1,307 | 1,045 / 262 | 72 | 93.89 % | 0.9383 | 0.9385 | 0.865 |
| 256 | 999 | 799 / 200 | 54 | 94.50 % | 0.9452 | 0.9498 | 0.847 |

Per-class F1 for the 256 px model: Dense forest 0.949, Mix 0.913, Urban 0.987.
Mix is the hardest class at every patch size. The re-run on the validation sets reproduces the
checkpoint accuracy exactly.

All four sizes lie within 0.6 points, and each uses its own validation set. The data do **not**
show that one patch size is better. In particular, the claim that 224 px "eliminates resize
artefacts" is not visible in accuracy: 224 px has the lowest value, 93.89 %.

## 2. SAR + DEM (`results/07_model_evaluation/dem_ablation_study/`)

| | SAR only (4 features) | SAR + DEM (9 features) |
|---|---:|---:|
| Val accuracy (same 200 patches) | **94.0 %** | 92.0 % |
| F1 Dense forest / Mix / Urban | 0.979 / 0.829 / 0.880 | 0.965 / 0.765 / 0.875 |
| Only this model correct | 5 | 1 |

McNemar χ² = 1.5, p = 0.22: no significant difference, and the direction favours SAR-only.
The separate DEM training run (`data/05_classification_training/models/EOS04_256_L2B_DEM`, 93.5 %, epoch 56) agrees with this.

`prove_dem_improvement.py` reports the opposite (+1.0 point, "Urban precision 1.000"), but it
compares **hand-typed metrics of two unrelated runs**:
* "with DEM" is in fact `data/05_classification_training/models/EOS04_L2B`, which has only the 4 SAR features;
* "without DEM" is an earlier pre-L2B run, with an estimated confusion matrix.

Its output should not be cited.

## 3. Same-location 256 vs 128 comparison (`fair_model_comparison.py`)

Both saved runs report 19.5 % (L2B) and 33.5 % (pre-L2B) accuracy for *both* models, with
100 % agreement and mean confidence 1.000. Every patch got the same class. The models reach
93–94 % when evaluated by `collect_all_model_stats.py`, so the problem is in how this script
prepares the inputs or features. The result is invalid until the script is debugged.

## 4. Labels (`results/04_label_validation/`)

999 patches: Dense forest 488, Mix 316, Urban 195.

| Feature | Kruskal–Wallis H | p | η² |
|---|---:|---:|---:|
| HH/HV ratio | 842.7 | 1.0e-183 | 0.844 |
| HV σ⁰ (mean_g) | 509.1 | 2.8e-111 | 0.509 |
| HH σ⁰ (mean_r) | 118.6 | 1.8e-26 | 0.117 |

The classes were *defined* by HH/HV thresholds, so the first row is expected by construction.
The HV and HH rows and the visual montages are the supporting evidence. The Sentinel-2 NDVI
cross-check (method 2) was not run.

## 5. Full-scene maps (`results/06_scene_prediction/`)

| Model | Patches predicted | Dense forest | Mix | Urban |
|---|---:|---:|---:|---:|
| 256 px | 1,262 | 716 | 321 | 225 |
| 224 px | 1,638 | 963 | 389 | 286 |
| 128 px | 4,941 | 3,008 | 1,054 | 879 |
| 64 px | 19,391 | 11,475 | 4,807 | 3,109 |

## 6. U-Net segmentation (`results/08_segmentation_unet/`)

* Data: 808 image/mask pairs at 256 px, with pixel pseudo-labels from the HH/HV thresholds.
* Training: 4 classes, ignore index 255 for layover/shadow.
* Validation mean IoU: 0.66 after epoch 1, about 0.94 at best (around epoch 41).
* Example patch r02048_c04608: pixel accuracy 0.974, mIoU 0.932, Dice 0.964.

## 7. Limits and open points

* **No independent ground truth.** Labels, segmentation masks and the HH/HV model input all
  come from the same ratio, so the accuracies measure agreement with the labelling rule.
  A comparison against an external land-cover product (e.g. ESA WorldCover, Bhuvan LULC) or
  Sentinel-2 NDVI would be the real test.
* **One scene, one random split.** Neighbouring patches can fall into train and val, so
  validation accuracy is likely optimistic. A spatially blocked split, or a second date
  (the 17 May 2026 scene is in `data/00_input/raw/`), would be a stronger test.
* **RVI is constant.** `rvi_mean` is constant for OPT1 (mean 0, std 1e-6 in every
  `feature_stats.json`), so the "4 SAR features" are effectively 3.
* **The DEM features of the 224 px model are all zero** (`data/05_classification_training/models/EOS04_224_L2B/feature_stats.json`).

## 8. Paper drafts and resume (local files, not in this repository)

| Statement | Status |
|---|---|
| APSCON draft: 94.50 % with 256 px patches, Kruskal–Wallis H = 842.7, η² = 0.844 | ✔ matches `results/` |
| SAR.pdf draft: "incorporating DEM features achieves 93.5 % … McNemar's test confirming statistical significance of the terrain feature contribution" | ✘ The ablation gives p = 0.22, and SAR-only is 2 points better. 93.5 % is the DEM run's accuracy, but it is not an improvement. |
| SAR.pdf draft: 224 px is the best configuration | ✘ not supported, see §1 |
| Resume: 997 (256 px) and 3,894 (128 px) patches; 93.5 % and 91.3 % | These are the earlier pre-L2B runs. The current L2B runs have 999 / 3,910 patches and reach 94.50 % / 93.99 %. |
| Resume: "classifying 6,320 SAR patches" | 6,320 is the approximate 128 px grid size. The saved 128 px prediction covers 4,941 valid patches. |

# Data

The SAR data, intermediate rasters, patches and model weights are **not** in this repository.
They total about 33 GB. The scripts expect them under `BASE_DIR\data\` (default
`D:\ISRO_14\data\`). They are arranged by task, with the same numbering as `scripts/` and
`results/`:

```
data/
  00_input/                     raw/  (EOS-04 L2B scenes)   DEM/  (Copernicus DSM ZIPs)
  01_calibration/               calibrated_L2B/03JUN2026_HH, _HV
  02_composites_and_patches/    composites_L2B/  patches_64_L2B/  patches_128_L2B/  patches_224_L2B/
                                patches_L2B/ (256 px)  patches_256_L2B_DEM/  qgis_projects/ (*.qgz)
  03_dem_terrain/               DEM_TIFS/  terrain_qc/
  04_label_validation/          label_validation/
  05_classification_training/   models/EOS04_L2B, EOS04_64_L2B, EOS04_128_L2B, EOS04_224_L2B,
                                       EOS04_256_L2B_DEM (+ earlier EOS04, EOS04_128)
  06_scene_prediction/          predictions_L2B/  predictions_64_L2B/  predictions_128_L2B/  predictions_224_L2B/
  07_model_evaluation/          ALL_STATS_L2B/  ablation_study/  comparison/  comparison_L2B/
  08_segmentation_unet/         seg_masks/  unet_output_256/  unet_evaluation_256/
```

## Source data

| Folder | Content | Source |
|---|---|---|
| `data/00_input/raw/E04_SAR_MRS_03JUN2026_…/` | EOS-04 MRS Level-2B ARD: HH/HV GeoTIFF, `BAND_META.txt`, `*_lia.tif`, `*_mask.tif`, `*_area.tif` (scene centre N31.5° E76.8°) | [ISRO Bhoonidhi](https://bhoonidhi.nrsc.gov.in) |
| `data/00_input/raw/E04_SAR_MRS_17MAY2026_…/` | second acquisition (not yet used for training) | Bhoonidhi |
| `data/00_input/DEM/` (50 ZIPs) → `data/03_dem_terrain/DEM_TIFS/` | Copernicus DSM tiles covering the scene | Copernicus DEM (ESA) |
| `docs/reference/EOS_04_Data_Products_Format_Document.pdf` | *EOS-04 Data Products Formats v1.2.5*, SAC/ISRO; marked "Security Classification: Un-Restricted" | SAC / ISRO |

## Sizes on the training PC

| Folder | Size |
|---|---|
| `00_input` | 12 GB (raw 1.6 GB, DEM ZIPs 10.4 GB) |
| `01_calibration` | 3.1 GB |
| `02_composites_and_patches` | 6.3 GB (composites 3.0 GB, each patch set ~0.7 GB) |
| `03_dem_terrain` | 7.7 GB (DEM tiles 2.5 GB, extracted terrain QC 5.2 GB) |
| `05_classification_training` | 0.2 GB |
| `08_segmentation_unet` | 0.9 GB |

Label files: `data/02_composites_and_patches/patches_<size>_L2B/03JUN2026/OPT1_sigma0/OPT1_sigma0_labels_clean.csv`.
They are exported from QGIS (`qgis_projects/<size>_classes.qgz`) and cleaned with `fix_exported_csv.py`.

The small reports, metrics and figures from these folders are copied into `results/`.

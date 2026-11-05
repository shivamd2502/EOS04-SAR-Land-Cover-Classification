# Data

The SAR data, intermediate rasters, patches and model weights are **not** in this repository.
They total about 33 GB. This page describes the folder layout the scripts expect under
`BASE_DIR` (default `D:\ISRO_14`).

## Source data

| Folder | Content | Source |
|---|---|---|
| `raw/E04_SAR_MRS_03JUN2026_…/` | EOS-04 MRS Level-2B ARD: HH/HV GeoTIFF, `BAND_META.txt`, `*_lia.tif`, `*_mask.tif`, `*_area.tif` (scene centre N31.5° E76.8°) | [ISRO Bhoonidhi](https://bhoonidhi.nrsc.gov.in) |
| `raw/E04_SAR_MRS_17MAY2026_…/` | second acquisition (not yet used for training) | Bhoonidhi |
| `DEM/` (50 ZIPs) → `DEM_TIFS/` | Copernicus DSM tiles covering the scene | Copernicus DEM (ESA) |
| `docs/reference/EOS_04_Data_Products_Format_Document.pdf` | *EOS-04 Data Products Formats v1.2.5*, SAC/ISRO; marked "Security Classification: Un-Restricted" | SAC / ISRO |

## Generated folders (created by the scripts)

| Folder | Step | Size on the training PC |
|---|---|---|
| `calibrated_L2B/03JUN2026_HH`, `_HV` | 01 | 3.1 GB |
| `composites_L2B/03JUN2026` | 02 | 3.0 GB |
| `patches_64_L2B`, `patches_128_L2B`, `patches_224_L2B`, `patches_L2B` (256), `patches_256_L2B_DEM` | 02 | ~0.7 GB each |
| `validation/` | 03-04 | 5.2 GB (mostly extracted DEM) |
| `model/EOS04_*_L2B/`, `model/ablation_study/`, `model/ALL_STATS_L2B/`, `model/comparison*/` | 05, 07 | 0.3 GB |
| `predictions_*_L2B/03JUN2026` | 06 | 20 MB |
| `seg_masks/03JUN2026`, `unet_output_256`, `unet_evaluation_256` | 08 | 0.9 GB |

Label files are `patches_<size>_L2B/03JUN2026/OPT1_sigma0/OPT1_sigma0_labels_clean.csv`.
They are exported from QGIS and cleaned with `fix_exported_csv.py`.

The small reports, metrics and figures from these folders are copied into `results/`.

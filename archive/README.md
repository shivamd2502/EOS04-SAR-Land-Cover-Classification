# Archive: first version (June 2026)

`v1_initial_pipeline/` holds the first iteration of the project, before the ISRO-documented L2B
calibration. These scripts were in this repository's root until the reorganisation. They are
kept for history, are not used by `scripts/`, and point to an old base path
(`C:\Users\shiva\OneDrive\Documents\ISRO_14`).

| File | Purpose |
|---|---|
| `step1_preprocess.py`, `step2_patch_extraction.py`, `step3_visualize.py` | generic DN → dB preprocessing, 256 px `.npy` patches, overview plots |
| `convert_to_png.py`, `find_best_patches.py` | patches to PNG, choosing patches for annotation |
| `train_classifier.py`, `predict.py`, `predict_and_shapefile.py`, `stitch_predictions.py`, `generate_patch_shapefile.py` | first EfficientNet-B0 classifier (classes None / Urban Dense / Hills Rural), prediction and shapefiles |
| `convert_masks.py`, `verify_masks.py`, `train_unet.py` | Label Studio brush masks → PNG, check, U-Net (ResNet34 encoder) |
| `inspect_lia_mask.py`, `composite_class_comparison.py` | LIA/mask inspection, composite comparison per class |
| `eos04_train_autolabelClasssifer.py` | earlier copy of today's `scripts/05_classification_training/eos04_train.py` |
| `README_v1_original.md`, `requirements_v1*.txt` | the original README and requirements |

## v1 results (`v1_initial_pipeline/results/`)

* **Classifier:** 300 hand-labelled patches (`labels/`), EfficientNet-B0, best val accuracy 96.7 %
  on 61 patches (`classifier_efficientnet/`). `classifier_quality_filtered/` is a retrain after
  a quality filter dropped 222 of 304 patches (`patch_quality_audit/`).
* **U-Net (ResNet34):** 50 hand-annotated masks, best val mIoU 0.595 (`unet_resnet34/`).
* **Stitched full-scene mask:** `stitched_scene/`.

The small dataset and subjective labels motivated v2: ISRO calibration, physics-based labels
and about 1,000 to 11,700 patches.

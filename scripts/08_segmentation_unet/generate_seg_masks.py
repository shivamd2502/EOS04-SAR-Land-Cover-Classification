# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 8 of 8 - pixel-level pseudo-labels (folder: scripts/08_segmentation_unet)
#
# WHY THIS STEP
#   Patch labels give one class per 4.6 km square. For a pixel-level map, the same
#   QGIS HH/HV-ratio rule is applied per pixel:
#   < 6.8 dB Dense_forest, 6.8-8.2 Mix, 8.2-12.3 Urban, >= 12.3 Unclear;
#   layover/shadow pixels = 255 (ignored). An optional speckle filter runs before
#   the thresholds and optional morphological cleaning after them.
#
# WHAT IT PRODUCED
#   seg_masks/03JUN2026/full_scene_mask.tif and paired image/mask patches:
#   808 at 256 px (seg_patches) and 3,358 at 128 px (seg_patches_128).
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
EOS-04 SAR Pixel-Level Segmentation Mask Generator

Purpose
-------
Generate pixel-level pseudo-label segmentation masks from the
EOS-04 L2B HH/HV ratio channel.

Current QGIS threshold rule:

    HH/HV < 6.8 dB        -> Dense_forest
    6.8 <= HH/HV < 8.2   -> Mix
    8.2 <= HH/HV < 12.3  -> Urban
    HH/HV >= 12.3        -> Unclear

The script:
1. Loads the calibrated EOS-04 composite.
2. Extracts the HH/HV ratio band.
3. Optionally applies a speckle filter.
4. Applies the QGIS thresholds pixel-by-pixel.
5. Applies the ISRO layover/shadow mask.
6. Optionally applies morphological cleaning.
7. Generates a full-scene uint8 segmentation mask.
8. Generates a colourised preview.
9. Extracts paired image/mask patches for U-Net training.

Class encoding
--------------
0   = Dense_forest
1   = Mix
2   = Urban
3   = Unclear
255 = Ignored

ISRO mask values
----------------
128 = Valid
16  = Layover -> Ignored
64  = Shadow  -> Ignored
0   = Outside scene -> Ignored
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("EOS04_SEGMENTATION")


# =============================================================================
# DEPENDENCIES
# =============================================================================

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio is not installed.")
    print("Run: pip install rasterio")
    sys.exit(1)


try:
    from scipy.ndimage import (
        uniform_filter,
        binary_opening,
        binary_closing,
    )
except ImportError:
    print("ERROR: scipy is not installed.")
    print("Run: pip install scipy")
    sys.exit(1)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = r"D:\ISRO_14"

COMPOSITE = os.path.join(
    BASE_DIR,
    "composites_L2B",
    "03JUN2026",
    "OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif"
)

MASK_TIF = os.path.join(
    BASE_DIR,
    "raw",
    "E04_SAR_MRS_03JUN2026_154005259612_23736_STUC00ZTD_32242_8_DH_D_R_N31497_E076829",
    "E04_SAR_MRS_03JUN2026_154005259612_23736_STUC00ZTD_32242_8_DH_D_R_N31497_E076829_mask.tif"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "seg_masks",
    "03JUN2026"
)


# =============================================================================
# QGIS THRESHOLDS
# =============================================================================

# These exactly match your QGIS CASE expression:
#
# CASE
#   WHEN hh_hv_rat < 6.8 THEN 'Dense_forest'
#   WHEN hh_hv_rat < 8.2 THEN 'Mix'
#   WHEN hh_hv_rat < 12.3 THEN 'Urban'
#   ELSE 'Unclear'
# END

THRESH_FOREST = 6.8
THRESH_MIX = 8.2
THRESH_URBAN = 12.3


# =============================================================================
# CLASS IDs
# =============================================================================

CLS_FOREST = 0
CLS_MIX = 1
CLS_URBAN = 2
CLS_UNCLEAR = 3

# Pixels that should not participate in training
CLS_IGNORE = 255


# =============================================================================
# CLASS NAMES
# =============================================================================

CLASS_NAMES = {
    CLS_FOREST: "Dense_forest",
    CLS_MIX: "Mix",
    CLS_URBAN: "Urban",
    CLS_UNCLEAR: "Unclear",
}


# =============================================================================
# PREVIEW COLOURS
# =============================================================================

CLASS_COLORS = {

    # Dense forest = green
    CLS_FOREST: (0, 128, 0),

    # Mix = yellow
    CLS_MIX: (255, 200, 0),

    # Urban = red
    CLS_URBAN: (220, 50, 50),

    # Unclear = dark grey
    CLS_UNCLEAR: (50, 50, 50),

    # Ignored = black
    CLS_IGNORE: (0, 0, 0),
}


# =============================================================================
# PATCH SETTINGS
# =============================================================================

DEFAULT_PATCH_SIZES = [256, 128]

DEFAULT_STRIDE = 256

# A patch must contain at least 60% non-ignored pixels.
MIN_VALID_PCT = 60.0


# =============================================================================
# 1. RATIO STATISTICS
# =============================================================================

def print_ratio_statistics(
    ratio: np.ndarray,
    title: str = "Ratio statistics"
):
    """
    Print useful statistics of the HH/HV ratio image.
    """

    finite = ratio[np.isfinite(ratio)]

    if finite.size == 0:
        log.warning("%s: no finite pixels found.", title)
        return

    log.info("%s:", title)

    log.info(
        "  Min  = %.2f dB",
        np.min(finite)
    )

    log.info(
        "  Max  = %.2f dB",
        np.max(finite)
    )

    log.info(
        "  Mean = %.2f dB",
        np.mean(finite)
    )

    percentiles = np.percentile(
        finite,
        [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )

    names = [
        "P01",
        "P05",
        "P10",
        "P25",
        "P50",
        "P75",
        "P90",
        "P95",
        "P99",
    ]

    for name, value in zip(names, percentiles):
        log.info(
            "  %-4s = %.2f dB",
            name,
            value
        )


# =============================================================================
# 2. ISRO MASK STATISTICS
# =============================================================================

def print_isro_mask_statistics(
    layover_mask: np.ndarray
):
    """
    Print the distribution of ISRO mask values.
    """

    print("\n## ISRO MASK VALUE DISTRIBUTION\n")

    unique_values, counts = np.unique(
        layover_mask,
        return_counts=True
    )

    total = layover_mask.size

    for value, count in zip(unique_values, counts):

        percentage = 100.0 * count / total

        log.info(
            "  value=%d : %12d pixels (%6.2f%%)",
            int(value),
            int(count),
            percentage
        )


# =============================================================================
# 3. FROST FILTER
# =============================================================================

def frost_filter(
    img: np.ndarray,
    kernel: int = 7,
    damping: float = 2.0
) -> np.ndarray:
    """
    Approximate Frost speckle filter.

    Parameters
    ----------
    img:
        Input HH/HV ratio image.

    kernel:
        Local filtering window.

    damping:
        Controls smoothing strength.

    Returns
    -------
    Filtered image.
    """

    img = img.astype(np.float64)

    mean = uniform_filter(
        img,
        size=kernel
    )

    mean_sq = uniform_filter(
        img ** 2,
        size=kernel
    )

    variance = mean_sq - mean ** 2

    variance = np.maximum(
        variance,
        1e-10
    )

    # Local coefficient of variation squared
    ci_sq = variance / (
        mean ** 2 + 1e-10
    )

    weight = np.exp(
        -damping * ci_sq
    )

    numerator = uniform_filter(
        img * weight,
        size=kernel
    )

    denominator = uniform_filter(
        weight,
        size=kernel
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore"
    ):

        filtered = np.where(
            denominator > 0,
            numerator / denominator,
            mean
        )

    return filtered.astype(np.float32)


# =============================================================================
# 4. LEE FILTER
# =============================================================================

def lee_filter(
    img: np.ndarray,
    kernel: int = 7
) -> np.ndarray:
    """
    Lee speckle filter.

    Parameters
    ----------
    img:
        Input image.

    kernel:
        Local filtering window.

    Returns
    -------
    Filtered image.
    """

    img = img.astype(np.float64)

    mean = uniform_filter(
        img,
        size=kernel
    )

    mean_sq = uniform_filter(
        img ** 2,
        size=kernel
    )

    variance = np.maximum(
        mean_sq - mean ** 2,
        0
    )

    noise_variance = np.mean(
        variance
    )

    weight = variance / (
        variance +
        noise_variance +
        1e-10
    )

    filtered = (
        mean +
        weight * (img - mean)
    )

    return filtered.astype(np.float32)


# =============================================================================
# 5. THRESHOLD CLASSIFICATION
# =============================================================================

def classify_ratio(
    ratio: np.ndarray
) -> np.ndarray:
    """
    Apply the exact QGIS threshold logic.

    QGIS:

        < 6.8       -> Dense_forest
        6.8 - 8.2   -> Mix
        8.2 - 12.3  -> Urban
        >= 12.3     -> Unclear

    Returns
    -------
    uint8 classification mask.
    """

    # Start everything as Unclear.
    mask = np.full(
        ratio.shape,
        CLS_UNCLEAR,
        dtype=np.uint8
    )

    # Dense forest
    mask[
        ratio < THRESH_FOREST
    ] = CLS_FOREST

    # Mix
    mask[
        (ratio >= THRESH_FOREST) &
        (ratio < THRESH_MIX)
    ] = CLS_MIX

    # Urban
    mask[
        (ratio >= THRESH_MIX) &
        (ratio < THRESH_URBAN)
    ] = CLS_URBAN

    # Unclear
    mask[
        ratio >= THRESH_URBAN
    ] = CLS_UNCLEAR

    # Invalid pixels
    mask[
        ~np.isfinite(ratio)
    ] = CLS_IGNORE

    return mask


# =============================================================================
# 6. THRESHOLD DISTRIBUTION
# =============================================================================

def print_threshold_distribution(
    mask: np.ndarray
):
    """
    Print class distribution before applying ISRO mask.
    """

    total = mask.size

    log.info(
        "Threshold distribution BEFORE ISRO mask:"
    )

    for cid, name in CLASS_NAMES.items():

        count = int(
            np.sum(mask == cid)
        )

        percentage = (
            100.0 * count / total
        )

        log.info(
            "  %-14s : %12d (%6.2f%%)",
            name,
            count,
            percentage
        )

    invalid = int(
        np.sum(mask == CLS_IGNORE)
    )

    percentage = (
        100.0 * invalid / total
    )

    log.info(
        "  %-14s : %12d (%6.2f%%)",
        "Invalid/NaN",
        invalid,
        percentage
    )


# =============================================================================
# 7. MORPHOLOGICAL CLEANING
# =============================================================================

def morphological_cleaning(
    mask: np.ndarray,
    iterations: int = 2
) -> np.ndarray:
    """
    Remove very small isolated regions.

    IMPORTANT:
    This is disabled by default because threshold validation
    should first be performed without morphology.
    """

    cleaned_mask = mask.copy()

    for cls_id in [
        CLS_FOREST,
        CLS_MIX,
        CLS_URBAN
    ]:

        binary = (
            mask == cls_id
        )

        opened = binary_opening(
            binary,
            iterations=iterations
        )

        opened = binary_closing(
            opened,
            iterations=iterations
        )

        removed = (
            binary &
            ~opened
        )

        cleaned_mask[
            removed
        ] = CLS_UNCLEAR

    return cleaned_mask


# =============================================================================
# 8. COMPLETE MASK GENERATION
# =============================================================================

def generate_pixel_mask(
    ratio_band: np.ndarray,
    layover_mask: np.ndarray | None = None,
    speckle_kernel: int = 7,
    filter_type: str = "none",
    use_morphology: bool = False,
    morphology_iterations: int = 2
) -> np.ndarray:

    """
    Generate final segmentation mask.
    """

    log.info(
        "Generating pixel-level segmentation mask..."
    )

    # -------------------------------------------------------------------------
    # STEP 1: SPECKLE FILTER
    # -------------------------------------------------------------------------

    if filter_type == "frost":

        log.info(
            "Applying Frost speckle filter "
            "(kernel=%d)...",
            speckle_kernel
        )

        ratio_processed = frost_filter(
            ratio_band,
            kernel=speckle_kernel
        )

    elif filter_type == "lee":

        log.info(
            "Applying Lee speckle filter "
            "(kernel=%d)...",
            speckle_kernel
        )

        ratio_processed = lee_filter(
            ratio_band,
            kernel=speckle_kernel
        )

    else:

        log.info(
            "No speckle filter applied."
        )

        ratio_processed = ratio_band.copy()


    # -------------------------------------------------------------------------
    # STEP 2: PROCESSED RATIO STATISTICS
    # -------------------------------------------------------------------------

    print_ratio_statistics(
        ratio_processed,
        title="Processed HH/HV ratio"
    )


    # -------------------------------------------------------------------------
    # STEP 3: THRESHOLDS
    # -------------------------------------------------------------------------

    log.info(
        "Applying thresholds:"
    )

    log.info(
        "  Forest < %.2f dB",
        THRESH_FOREST
    )

    log.info(
        "  Mix    %.2f - %.2f dB",
        THRESH_FOREST,
        THRESH_MIX
    )

    log.info(
        "  Urban  %.2f - %.2f dB",
        THRESH_MIX,
        THRESH_URBAN
    )

    log.info(
        "  Unclear >= %.2f dB",
        THRESH_URBAN
    )


    # -------------------------------------------------------------------------
    # STEP 4: CLASSIFY EVERY PIXEL
    # -------------------------------------------------------------------------

    mask = classify_ratio(
        ratio_processed
    )


    # -------------------------------------------------------------------------
    # STEP 5: PRINT THRESHOLD DISTRIBUTION
    # -------------------------------------------------------------------------

    print_threshold_distribution(
        mask
    )


    # -------------------------------------------------------------------------
    # STEP 6: APPLY ISRO MASK
    # -------------------------------------------------------------------------

    if layover_mask is not None:

        log.info(
            "Applying ISRO layover/shadow mask..."
        )

        # 16 = layover
        mask[
            layover_mask == 16
        ] = CLS_IGNORE

        # 64 = shadow
        mask[
            layover_mask == 64
        ] = CLS_IGNORE

        # 0 = outside scene
        mask[
            layover_mask == 0
        ] = CLS_IGNORE

    else:

        log.warning(
            "No ISRO mask supplied."
        )


    # -------------------------------------------------------------------------
    # STEP 7: OPTIONAL MORPHOLOGY
    # -------------------------------------------------------------------------

    if use_morphology:

        log.info(
            "Morphological cleaning ENABLED "
            "(iterations=%d).",
            morphology_iterations
        )

        mask = morphological_cleaning(
            mask,
            iterations=morphology_iterations
        )

    else:

        log.info(
            "Morphological cleaning DISABLED "
            "during validation."
        )


    # -------------------------------------------------------------------------
    # STEP 8: FINAL DISTRIBUTION
    # -------------------------------------------------------------------------

    total = mask.size

    log.info(
        "FINAL MASK DISTRIBUTION:"
    )

    for cid, name in CLASS_NAMES.items():

        count = int(
            np.sum(mask == cid)
        )

        percentage = (
            100.0 * count / total
        )

        log.info(
            "  %-16s : %12d (%6.2f%%)",
            name,
            count,
            percentage
        )

    ignored = int(
        np.sum(mask == CLS_IGNORE)
    )

    percentage = (
        100.0 * ignored / total
    )

    log.info(
        "  %-16s : %12d (%6.2f%%)",
        "Ignored",
        ignored,
        percentage
    )

    return mask


# =============================================================================
# 9. FINAL MASK STATISTICS
# =============================================================================

def mask_stats(
    mask: np.ndarray
) -> dict:

    total = mask.size

    stats = {}

    for cid, cname in CLASS_NAMES.items():

        count = int(
            np.sum(mask == cid)
        )

        stats[cname] = {
            "pixels": count,
            "pct": round(
                100.0 * count / total,
                2
            )
        }

    ignored = int(
        np.sum(mask == CLS_IGNORE)
    )

    stats["Ignored"] = {
        "pixels": ignored,
        "pct": round(
            100.0 * ignored / total,
            2
        )
    }

    return stats


# =============================================================================
# 10. MASK -> RGB
# =============================================================================

def mask_to_rgb(
    mask: np.ndarray
) -> np.ndarray:

    rgb = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8
    )

    for cid, colour in CLASS_COLORS.items():

        rgb[
            mask == cid
        ] = colour

    return rgb


# =============================================================================
# 11. COMPOSITE -> UINT8
# =============================================================================

def stretch_composite(
    composite: np.ndarray
) -> np.ndarray:

    """
    Convert 3-band float composite to uint8 RGB image.

    Uses 2nd and 98th percentile stretching independently
    for each channel.
    """

    height = composite.shape[1]
    width = composite.shape[2]

    hwc = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for i in range(3):

        channel = composite[i]

        valid = channel[
            np.isfinite(channel)
        ]

        if valid.size == 0:
            continue

        low, high = np.percentile(
            valid,
            [2, 98]
        )

        if high <= low:
            continue

        stretched = np.clip(
            (
                (channel - low) /
                (high - low)
            ) * 255,
            0,
            255
        )

        stretched[
            ~np.isfinite(channel)
        ] = 0

        hwc[:, :, i] = (
            stretched.astype(np.uint8)
        )

    return hwc


# =============================================================================
# 12. PATCH EXTRACTION
# =============================================================================

def extract_paired_patches(
    composite_hwc: np.ndarray,
    mask: np.ndarray,
    images_dir: str,
    masks_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    min_valid_pct: float = 60.0
) -> tuple[int, int, int]:

    """
    Extract aligned image and mask patches.

    A patch is saved if:

        valid pixels >= min_valid_pct

    and:

        at least one labelled class pixel exists.

    Returns
    -------
    saved, skipped_ignored, skipped_no_class
    """

    os.makedirs(
        images_dir,
        exist_ok=True
    )

    os.makedirs(
        masks_dir,
        exist_ok=True
    )

    H, W, _ = composite_hwc.shape

    saved = 0
    skipped_ignored = 0
    skipped_no_class = 0

    row = 0

    while row + patch_size <= H:

        col = 0

        while col + patch_size <= W:

            img_patch = composite_hwc[
                row:row + patch_size,
                col:col + patch_size
            ]

            mask_patch = mask[
                row:row + patch_size,
                col:col + patch_size
            ]


            # -------------------------------------------------------------
            # Check ignored percentage
            # -------------------------------------------------------------

            valid_pct = (
                100.0 *
                np.mean(
                    mask_patch != CLS_IGNORE
                )
            )

            if valid_pct < min_valid_pct:

                skipped_ignored += 1

                col += stride
                continue


            # -------------------------------------------------------------
            # Check for actual class pixels
            # -------------------------------------------------------------

            labelled = np.sum(
                (mask_patch == CLS_FOREST) |
                (mask_patch == CLS_MIX) |
                (mask_patch == CLS_URBAN)
            )

            if labelled == 0:

                skipped_no_class += 1

                col += stride
                continue


            # -------------------------------------------------------------
            # Save aligned pair
            # -------------------------------------------------------------

            filename = (
                f"r{row:05d}_c{col:05d}.png"
            )

            image_path = os.path.join(
                images_dir,
                filename
            )

            mask_path = os.path.join(
                masks_dir,
                filename
            )

            Image.fromarray(
                img_patch
            ).save(
                image_path
            )

            Image.fromarray(
                mask_patch
            ).save(
                mask_path
            )

            saved += 1

            col += stride

        row += stride


    return (
        saved,
        skipped_ignored,
        skipped_no_class
    )


# =============================================================================
# 13. MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate EOS-04 pixel-level "
            "segmentation masks."
        )
    )


    # -------------------------------------------------------------------------
    # Input/output arguments
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--composite",
        default=COMPOSITE
    )

    parser.add_argument(
        "--mask-tif",
        default=MASK_TIF,
        help="ISRO layover/shadow mask GeoTIFF"
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR
    )


    # -------------------------------------------------------------------------
    # Patch sizes
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--patch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_PATCH_SIZES
    )


    # -------------------------------------------------------------------------
    # Speckle filtering
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--speckle-filter",
        default="none",
        choices=[
            "none",
            "frost",
            "lee"
        ],
        help=(
            "Speckle filter. "
            "Default=none for threshold validation."
        )
    )

    parser.add_argument(
        "--speckle-kernel",
        type=int,
        default=7
    )


    # -------------------------------------------------------------------------
    # Morphology
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--morphology",
        action="store_true",
        help=(
            "Enable morphological cleaning. "
            "Disabled by default."
        )
    )

    parser.add_argument(
        "--morphology-iterations",
        type=int,
        default=2
    )


    # -------------------------------------------------------------------------
    # Patch valid percentage
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--min-valid",
        type=float,
        default=MIN_VALID_PCT,
        help=(
            "Minimum percentage of non-ignored "
            "pixels required in a patch."
        )
    )


    args = parser.parse_args()


    # =========================================================================
    # HEADER
    # =========================================================================

    print()
    print("=" * 70)
    print("EOS-04 PIXEL-LEVEL SEGMENTATION MASK GENERATION")
    print("=" * 70)

    print()
    print("Composite:")
    print(args.composite)

    print()
    print("ISRO mask:")
    print(args.mask_tif)

    print()
    print("Thresholds:")
    print(
        f"  Dense forest : HH/HV < {THRESH_FOREST:.2f} dB"
    )
    print(
        f"  Mix          : {THRESH_FOREST:.2f} - "
        f"{THRESH_MIX:.2f} dB"
    )
    print(
        f"  Urban        : {THRESH_MIX:.2f} - "
        f"{THRESH_URBAN:.2f} dB"
    )
    print(
        f"  Unclear      : >= {THRESH_URBAN:.2f} dB"
    )

    print()
    print("Speckle filter:")
    print(
        f"  {args.speckle_filter}"
        f" (kernel={args.speckle_kernel})"
    )

    print()
    print("Morphological cleaning:")
    print(
        "  ENABLED"
        if args.morphology
        else "  DISABLED"
    )

    print()
    print("Patch sizes:")
    print(
        f"  {args.patch_sizes}"
    )

    print()
    print("Output:")
    print(
        args.output_dir
    )

    print("=" * 70)
    print()


    # =========================================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    # =========================================================================
    # LOAD COMPOSITE
    # =========================================================================

    log.info(
        "Loading composite..."
    )

    if not os.path.isfile(
        args.composite
    ):

        log.error(
            "Composite file not found:"
        )

        log.error(
            args.composite
        )

        sys.exit(1)


    with rasterio.open(
        args.composite
    ) as src:

        composite = src.read(
            [1, 2, 3]
        ).astype(
            np.float32
        )

        meta = src.meta.copy()

        transform = src.transform
        crs = src.crs


    C, H, W = composite.shape

    log.info(
        "Scene dimensions: H=%d W=%d",
        H,
        W
    )


    # =========================================================================
    # EXTRACT HH/HV RATIO
    # =========================================================================

    # Band 3 = B band = HH/HV ratio
    ratio_band = composite[2]


    log.info(
        "Original HH/HV ratio:"
    )

    print_ratio_statistics(
        ratio_band,
        title=""
    )


    # =========================================================================
    # LOAD ISRO MASK
    # =========================================================================

    layover_mask = None


    if os.path.isfile(
        args.mask_tif
    ):

        log.info(
            "Loading ISRO mask..."
        )

        with rasterio.open(
            args.mask_tif
        ) as src:

            layover_mask = src.read(
                1
            ).astype(
                np.uint8
            )

            mask_H = src.height
            mask_W = src.width


        # ---------------------------------------------------------------------
        # IMPORTANT DIMENSION CHECK
        # ---------------------------------------------------------------------

        if (
            mask_H != H or
            mask_W != W
        ):

            log.error(
                "Dimension mismatch!"
            )

            log.error(
                "Composite : H=%d W=%d",
                H,
                W
            )

            log.error(
                "ISRO mask : H=%d W=%d",
                mask_H,
                mask_W
            )

            log.error(
                "The composite and ISRO mask must have "
                "the same dimensions before pixel-wise masking."
            )

            sys.exit(1)


        print_isro_mask_statistics(
            layover_mask
        )


        valid_pct = (
            100.0 *
            np.mean(
                layover_mask == 128
            )
        )

        log.info(
            "ISRO valid pixels: %.2f%%",
            valid_pct
        )


    else:

        log.warning(
            "ISRO mask not found:"
        )

        log.warning(
            args.mask_tif
        )

        log.warning(
            "Continuing without ISRO mask."
        )


    # =========================================================================
    # GENERATE SEGMENTATION MASK
    # =========================================================================

    seg_mask = generate_pixel_mask(
        ratio_band=ratio_band,
        layover_mask=layover_mask,
        speckle_kernel=args.speckle_kernel,
        filter_type=args.speckle_filter,
        use_morphology=args.morphology,
        morphology_iterations=args.morphology_iterations
    )


    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================

    stats = mask_stats(
        seg_mask
    )


    print()
    print("=" * 70)
    print("FINAL PIXEL-LEVEL CLASS DISTRIBUTION")
    print("=" * 70)

    for cname, values in stats.items():

        percentage = values["pct"]

        bar = "█" * int(
            percentage / 2
        )

        print(
            f"{cname:<18} : "
            f"{values['pixels']:>12,} "
            f"({percentage:>6.2f}%) "
            f"{bar}"
        )


    # =========================================================================
    # SAVE FULL-SCENE MASK
    # =========================================================================

    mask_path = os.path.join(
        args.output_dir,
        "full_scene_mask.tif"
    )


    output_meta = meta.copy()

    output_meta.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate"
    )


    with rasterio.open(
        mask_path,
        "w",
        **output_meta
    ) as dst:

        dst.write(
            seg_mask,
            1
        )


    log.info(
        "Saved full-scene mask:"
    )

    log.info(
        mask_path
    )


    # =========================================================================
    # SAVE COLOURISED PREVIEW
    # =========================================================================

    preview_path = os.path.join(
        args.output_dir,
        "full_scene_mask_rgb.png"
    )


    rgb_mask = mask_to_rgb(
        seg_mask
    )


    Image.fromarray(
        rgb_mask
    ).save(
        preview_path
    )


    log.info(
        "Saved colourised preview:"
    )

    log.info(
        preview_path
    )


    # =========================================================================
    # STRETCH COMPOSITE
    # =========================================================================

    log.info(
        "Stretching composite to uint8..."
    )


    composite_hwc = stretch_composite(
        composite
    )


    # =========================================================================
    # EXTRACT PATCHES
    # =========================================================================

    total_pairs = {}


    print()
    print("=" * 70)
    print("PATCH EXTRACTION")
    print("=" * 70)


    for patch_size in args.patch_sizes:

        log.info(
            "Extracting %d x %d paired patches...",
            patch_size,
            patch_size
        )


        if patch_size == 256:

            folder = "seg_patches"

        else:

            folder = (
                f"seg_patches_{patch_size}"
            )


        images_dir = os.path.join(
            args.output_dir,
            folder,
            "images"
        )


        masks_dir = os.path.join(
            args.output_dir,
            folder,
            "masks"
        )


        saved, skipped_ignored, skipped_no_class = (
            extract_paired_patches(
                composite_hwc,
                seg_mask,
                images_dir,
                masks_dir,
                patch_size=patch_size,
                stride=patch_size,
                min_valid_pct=args.min_valid
            )
        )


        total_pairs[
            patch_size
        ] = saved


        log.info(
            "Patch size %d:",
            patch_size
        )

        log.info(
            "  Saved pairs              : %d",
            saved
        )

        log.info(
            "  Skipped (> ignored)      : %d",
            skipped_ignored
        )

        log.info(
            "  Skipped (no class pixels): %d",
            skipped_no_class
        )


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 70)
    print("MASK GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("Full-scene mask:")
    print(mask_path)

    print()
    print("Class encoding:")
    print("0   = Dense_forest")
    print("1   = Mix")
    print("2   = Urban")
    print("3   = Unclear")
    print("255 = Ignored")

    print()
    print("Colourised preview:")
    print(preview_path)

    print()
    print("Paired patch folders:")

    for patch_size, number in total_pairs.items():

        if patch_size == 256:

            folder = "seg_patches"

        else:

            folder = (
                f"seg_patches_{patch_size}"
            )

        print()
        print(
            f"{folder}/"
        )

        print(
            f"  images/ <- "
            f"{number} composite patches "
            f"({patch_size}x{patch_size})"
        )

        print(
            f"  masks/  <- "
            f"{number} mask patches "
            f"(same filenames)"
        )


    print()
    print("=" * 70)

    print(
        "NEXT STEP:"
    )

    print(
        "1. Inspect full_scene_mask_rgb.png"
    )

    print(
        "2. Compare it with the QGIS classification"
    )

    print(
        "3. Verify forest/mix/urban boundaries"
    )

    print(
        "4. Verify several areas manually"
    )

    print(
        "5. Only then proceed to U-Net training"
    )

    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
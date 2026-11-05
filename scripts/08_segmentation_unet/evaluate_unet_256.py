# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 8 of 8 - U-Net evaluation (folder: scripts/08_segmentation_unet)
#
# WHY THIS STEP
#   Loads the trained U-Net and shows composite | pseudo-label mask | prediction
#   side by side with pixel accuracy, mean IoU and Dice per patch, so the
#   segmentation can be checked visually.
#
# WHAT IT PRODUCED
#   results/08_segmentation_unet/evaluation_comparisons/ (10 patches; e.g.
#   r02048_c04608: pixel acc 0.974, mIoU 0.932, Dice 0.964). The large
#   all_patch_comparisons.png stays local (20 MB).
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
EOS-04 U-Net 256x256 PATCH EVALUATION

Purpose
-------
Load a trained EOS-04 U-Net model and evaluate it on selected
256x256 SAR patches.

Comparison:
    Original Composite
            |
            v
    Ground Truth Mask
            |
            v
    U-Net Prediction

ONLY 256x256 patches are used.

Input:
    D:/ISRO_14/seg_masks/03JUN2026/seg_patches/images/
    D:/ISRO_14/seg_masks/03JUN2026/seg_patches/masks/

Model:
    D:/ISRO_14/unet_output_256/best_unet_eos04_256.pth

Output:
    D:/ISRO_14/unet_evaluation_256/

Classes:
    0 = Dense_forest
    1 = Mix
    2 = Urban
    3 = Unclear
    255 = Ignored
"""

# ============================================================
# IMPORTS
# ============================================================

import random
import numpy as np

from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\ISRO_14")

# ------------------------------------------------------------
# ONLY 256x256 PATCHES
# ------------------------------------------------------------

IMAGE_DIR = (
    BASE_DIR
    / "data" / "08_segmentation_unet" / "seg_masks"
    / "03JUN2026"
    / "seg_patches"
    / "images"
)

MASK_DIR = (
    BASE_DIR
    / "data" / "08_segmentation_unet" / "seg_masks"
    / "03JUN2026"
    / "seg_patches"
    / "masks"
)

# ------------------------------------------------------------
# TRAINED MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    BASE_DIR
    / "data" / "08_segmentation_unet" / "unet_output_256"
    / "best_unet_eos04_256.pth"
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "data" / "08_segmentation_unet" / "unet_evaluation_256"
)

PREDICTION_DIR = (
    OUTPUT_DIR
    / "predicted_masks"
)

COMPARISON_DIR = (
    OUTPUT_DIR
    / "comparisons"
)

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

PATCH_SIZE = 256

NUM_CLASSES = 4

IGNORE_INDEX = 255

CLASS_NAMES = {
    0: "Dense_forest",
    1: "Mix",
    2: "Urban",
    3: "Unclear"
}

# ------------------------------------------------------------
# NUMBER OF PATCHES TO EVALUATE
# ------------------------------------------------------------

NUM_PATCHES = 10

# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

SEED = 42

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS COLOURS
# ============================================================

CLASS_COLORS = {

    # Dense forest
    0: (0, 128, 0),

    # Mix
    1: (255, 200, 0),

    # Urban
    2: (220, 50, 50),

    # Unclear
    3: (80, 80, 80),

    # Ignored
    255: (0, 0, 0)
}


# ============================================================
# DOUBLE CONVOLUTION
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            3,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.out_conv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        p1 = self.pool(e1)

        e2 = self.enc2(p1)

        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        p3 = self.pool(e3)

        e4 = self.enc4(p3)

        p4 = self.pool(e4)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(p4)

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # ====================================================
        # OUTPUT
        # ====================================================

        output = self.out_conv(d1)

        return output


# ============================================================
# SET SEED
# ============================================================

def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD PATCH
# ============================================================

def load_patch(
    image_path,
    mask_path
):

    # --------------------------------------------------------
    # Load RGB composite
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image = np.array(
        image
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Check image size
    # --------------------------------------------------------

    if image.shape[:2] != (
        PATCH_SIZE,
        PATCH_SIZE
    ):

        raise ValueError(
            f"Invalid image size: "
            f"{image.shape}"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = image / 255.0

    # --------------------------------------------------------
    # H,W,C -> C,H,W
    # --------------------------------------------------------

    image_tensor = torch.tensor(
        np.transpose(
            image,
            (2, 0, 1)
        ),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Load ground truth mask
    # --------------------------------------------------------

    mask = Image.open(
        mask_path
    )

    mask = np.array(
        mask
    ).astype(
        np.int64
    )

    if mask.shape != (
        PATCH_SIZE,
        PATCH_SIZE
    ):

        raise ValueError(
            f"Invalid mask size: "
            f"{mask.shape}"
        )

    return (
        image_tensor,
        mask,
        image
    )


# ============================================================
# MASK TO RGB
# ============================================================

def mask_to_rgb(
    mask
):

    height, width = mask.shape

    rgb = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    for class_id, colour in CLASS_COLORS.items():

        rgb[
            mask == class_id
        ] = colour

    return rgb


# ============================================================
# PIXEL ACCURACY
# ============================================================

def calculate_pixel_accuracy(
    prediction,
    target
):

    valid = (
        target != IGNORE_INDEX
    )

    total = np.sum(
        valid
    )

    if total == 0:

        return 0.0

    correct = np.sum(
        (
            prediction == target
        )
        & valid
    )

    return (
        correct /
        total
    )


# ============================================================
# CLASS-WISE IOU
# ============================================================

def calculate_iou(
    prediction,
    target
):

    ious = {}

    valid = (
        target != IGNORE_INDEX
    )

    for class_id in range(
        NUM_CLASSES
    ):

        pred_class = (
            prediction == class_id
        ) & valid

        target_class = (
            target == class_id
        ) & valid

        intersection = np.logical_and(
            pred_class,
            target_class
        ).sum()

        union = np.logical_or(
            pred_class,
            target_class
        ).sum()

        if union == 0:

            ious[class_id] = np.nan

        else:

            ious[class_id] = (
                intersection /
                union
            )

    return ious


# ============================================================
# DICE
# ============================================================

def calculate_dice(
    prediction,
    target
):

    dice_scores = {}

    valid = (
        target != IGNORE_INDEX
    )

    for class_id in range(
        NUM_CLASSES
    ):

        pred_class = (
            prediction == class_id
        ) & valid

        target_class = (
            target == class_id
        ) & valid

        intersection = np.logical_and(
            pred_class,
            target_class
        ).sum()

        pred_sum = np.sum(
            pred_class
        )

        target_sum = np.sum(
            target_class
        )

        denominator = (
            pred_sum +
            target_sum
        )

        if denominator == 0:

            dice_scores[class_id] = np.nan

        else:

            dice_scores[class_id] = (
                2.0 *
                intersection /
                denominator
            )

    return dice_scores


# ============================================================
# CREATE COMPARISON IMAGE
# ============================================================

def save_comparison(
    image,
    ground_truth,
    prediction,
    filename,
    accuracy,
    mean_iou,
    mean_dice
):

    # --------------------------------------------------------
    # Convert masks to RGB
    # --------------------------------------------------------

    gt_rgb = mask_to_rgb(
        ground_truth
    )

    pred_rgb = mask_to_rgb(
        prediction
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # --------------------------------------------------------
    # Original composite
    # --------------------------------------------------------

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Original SAR Composite",
        fontsize=14
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(
        gt_rgb
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=14
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[2].imshow(
        pred_rgb
    )

    axes[2].set_title(
        "U-Net Prediction",
        fontsize=14
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    fig.suptitle(
        f"Pixel Accuracy: {accuracy:.4f}   |   "
        f"Mean IoU: {mean_iou:.4f}   |   "
        f"Mean Dice: {mean_dice:.4f}",
        fontsize=15
    )

    plt.tight_layout()

    output_path = (
        COMPARISON_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CREATE COMPARISON SHEET
# ============================================================

def create_comparison_sheet(
    examples
):

    if len(examples) == 0:

        return

    rows = len(
        examples
    )

    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(15, 5 * rows)
    )

    if rows == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, example in enumerate(
        examples
    ):

        image = example["image"]

        gt = example["gt"]

        pred = example["pred"]

        gt_rgb = mask_to_rgb(
            gt
        )

        pred_rgb = mask_to_rgb(
            pred
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        axes[row, 0].imshow(
            image
        )

        axes[row, 0].set_title(
            "Original Composite"
        )

        axes[row, 0].axis(
            "off"
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        axes[row, 1].imshow(
            gt_rgb
        )

        axes[row, 1].set_title(
            "Ground Truth"
        )

        axes[row, 1].axis(
            "off"
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        axes[row, 2].imshow(
            pred_rgb
        )

        axes[row, 2].set_title(
            "U-Net Prediction"
        )

        axes[row, 2].axis(
            "off"
        )

        # ----------------------------------------------------
        # Filename + metrics
        # ----------------------------------------------------

        fig.text(
            0.5,
            1.0 - (
                row + 0.5
            ) / rows,
            (
                f"{example['filename']}   |   "
                f"Accuracy={example['accuracy']:.4f}   |   "
                f"IoU={example['mean_iou']:.4f}   |   "
                f"Dice={example['mean_dice']:.4f}"
            ),
            ha="center",
            fontsize=11
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "all_patch_comparisons.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Combined comparison sheet:"
    )

    print(
        output_path
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading trained U-Net..."
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    model = UNet(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Best model checkpoint
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Checkpoint format: "
            "training checkpoint"
        )

        if "epoch" in checkpoint:

            print(
                f"Trained epoch: "
                f"{checkpoint['epoch']}"
            )

        if "mean_iou" in checkpoint:

            print(
                f"Validation Mean IoU: "
                f"{checkpoint['mean_iou']:.4f}"
            )

    else:

        model.load_state_dict(
            checkpoint
        )

        print(
            "Checkpoint format: "
            "model state dictionary"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Device: {DEVICE}"
    )

    return model


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    set_seed()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EOS-04 U-NET 256x256 PATCH EVALUATION"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Patch size : 256 x 256"
    )

    print(
        "128x128 patches : NOT USED"
    )

    print()
    print(
        "Input:"
    )

    print(
        IMAGE_DIR
    )

    print()
    print(
        "Ground Truth:"
    )

    print(
        MASK_DIR
    )

    print()
    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{IMAGE_DIR}"
        )

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            f"Mask directory not found:\n"
            f"{MASK_DIR}"
        )

    # --------------------------------------------------------
    # Find paired patches
    # --------------------------------------------------------

    image_files = sorted(
        IMAGE_DIR.glob("*.png")
    )

    pairs = []

    for image_path in image_files:

        mask_path = (
            MASK_DIR /
            image_path.name
        )

        if mask_path.exists():

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

    print()
    print(
        f"Total 256x256 paired patches: "
        f"{len(pairs)}"
    )

    if len(pairs) == 0:

        raise RuntimeError(
            "No paired 256x256 patches found."
        )

    # --------------------------------------------------------
    # Select patches
    # --------------------------------------------------------

    number_to_select = min(
        NUM_PATCHES,
        len(pairs)
    )

    selected_pairs = random.sample(
        pairs,
        number_to_select
    )

    print()
    print(
        f"Selected patches for evaluation: "
        f"{number_to_select}"
    )

    for image_path, _ in selected_pairs:

        print(
            f"  {image_path.name}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluation containers
    # --------------------------------------------------------

    examples = []

    overall_accuracy = []

    overall_ious = []

    overall_dices = []

    # --------------------------------------------------------
    # Process each patch
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "RUNNING U-NET SEGMENTATION"
    )

    print(
        "=" * 70
    )

    for patch_number, (
        image_path,
        mask_path
    ) in enumerate(
        selected_pairs,
        start=1
    ):

        print()
        print(
            f"[{patch_number}/{number_to_select}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Load patch
        # ----------------------------------------------------

        (
            image_tensor,
            ground_truth,
            image
        ) = load_patch(
            image_path,
            mask_path
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # U-Net prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )

            prediction = torch.argmax(
                output,
                dim=1
            )[0]

        prediction = (
            prediction
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = (
            calculate_pixel_accuracy(
                prediction,
                ground_truth
            )
        )

        ious = (
            calculate_iou(
                prediction,
                ground_truth
            )
        )

        dices = (
            calculate_dice(
                prediction,
                ground_truth
            )
        )

        mean_iou = np.nanmean(
            list(
                ious.values()
            )
        )

        mean_dice = np.nanmean(
            list(
                dices.values()
            )
        )

        overall_accuracy.append(
            accuracy
        )

        overall_ious.append(
            mean_iou
        )

        overall_dices.append(
            mean_dice
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"  Pixel Accuracy : "
            f"{accuracy:.4f}"
        )

        print(
            f"  Mean IoU       : "
            f"{mean_iou:.4f}"
        )

        print(
            f"  Mean Dice      : "
            f"{mean_dice:.4f}"
        )

        print(
            "  Class IoU:"
        )

        for class_id in range(
            NUM_CLASSES
        ):

            iou = ious[
                class_id
            ]

            if np.isnan(iou):

                print(
                    f"    "
                    f"{CLASS_NAMES[class_id]:15s}"
                    f": N/A"
                )

            else:

                print(
                    f"    "
                    f"{CLASS_NAMES[class_id]:15s}"
                    f": {iou:.4f}"
                )

        # ----------------------------------------------------
        # Save predicted mask
        # ----------------------------------------------------

        prediction_rgb = mask_to_rgb(
            prediction
        )

        prediction_filename = (
            image_path.stem
            + "_prediction.png"
        )

        prediction_path = (
            PREDICTION_DIR
            / prediction_filename
        )

        Image.fromarray(
            prediction_rgb
        ).save(
            prediction_path
        )

        # ----------------------------------------------------
        # Save comparison
        # ----------------------------------------------------

        comparison_filename = (
            image_path.stem
            + "_comparison.png"
        )

        save_comparison(
            image=image,
            ground_truth=ground_truth,
            prediction=prediction,
            filename=comparison_filename,
            accuracy=accuracy,
            mean_iou=mean_iou,
            mean_dice=mean_dice
        )

        # ----------------------------------------------------
        # Store example
        # ----------------------------------------------------

        examples.append(
            {
                "filename":
                    image_path.name,

                "image":
                    image,

                "gt":
                    ground_truth,

                "pred":
                    prediction,

                "accuracy":
                    accuracy,

                "mean_iou":
                    mean_iou,

                "mean_dice":
                    mean_dice
            }
        )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    overall_accuracy_mean = np.mean(
        overall_accuracy
    )

    overall_iou_mean = np.mean(
        overall_ious
    )

    overall_dice_mean = np.mean(
        overall_dices
    )

    print()
    print(
        "=" * 70
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Number of patches : "
        f"{number_to_select}"
    )

    print(
        f"Patch size        : "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print()
    print(
        f"Mean Pixel Accuracy : "
        f"{overall_accuracy_mean:.4f}"
    )

    print(
        f"Mean IoU            : "
        f"{overall_iou_mean:.4f}"
    )

    print(
        f"Mean Dice           : "
        f"{overall_dice_mean:.4f}"
    )

    # ========================================================
    # CREATE COMBINED SHEET
    # ========================================================

    create_comparison_sheet(
        examples
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Predicted masks:"
    )

    print(
        PREDICTION_DIR
    )

    print()
    print(
        "Individual comparisons:"
    )

    print(
        COMPARISON_DIR
    )

    print()
    print(
        "Combined comparison:"
    )

    print(
        OUTPUT_DIR
        / "all_patch_comparisons.png"
    )

    print()
    print(
        "Classes:"
    )

    print(
        "0   = Dense_forest"
    )

    print(
        "1   = Mix"
    )

    print(
        "2   = Urban"
    )

    print(
        "3   = Unclear"
    )

    print(
        "255 = Ignored"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
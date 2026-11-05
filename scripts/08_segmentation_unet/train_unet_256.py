# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 8 of 8 - U-Net segmentation (folder: scripts/08_segmentation_unet)
# (file was named train_unet_eos04_aug.py)
#
# WHY THIS STEP
#   Learns a pixel-wise map from the SAR composite to the pseudo-label masks of
#   generate_seg_masks.py, so that the map follows texture instead of a hard
#   per-pixel threshold. U-Net (encoder-decoder with skip connections) keeps
#   fine boundaries. Training uses 4 classes, ignore_index = 255 for
#   layover/shadow, batch 4, 50 epochs, LR 1e-4 and augmentation.
#
# WHAT IT PRODUCED (results/08_segmentation_unet/training/)
#   Validation mean IoU rises from 0.66 (epoch 1) to about 0.94 (best near epoch 41);
#   best_unet_eos04_256.pth / final_unet_eos04_256.pth in unet_output_256/.
#   CAVEAT: the targets are threshold pseudo-labels, so IoU measures how well the
#   rule is reproduced, not real land-cover accuracy.
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
EOS-04 U-Net Semantic Segmentation
===================================

Trains a U-Net using ONLY 256x256 EOS-04 SAR patches.

Input:
    seg_masks/03JUN2026/seg_patches/images/
    seg_masks/03JUN2026/seg_patches/masks/

Classes:
    0 = Dense_forest
    1 = Mix
    2 = Urban
    3 = Unclear
    255 = Ignored

The 128x128 patches are NOT used.

Features:
    - 256x256 RGB composite input
    - U-Net architecture
    - Train / validation split
    - CrossEntropyLoss
    - Ignore index 255
    - Dice score
    - Pixel accuracy
    - IoU per class
    - Mean IoU
    - Best model checkpoint
    - Training curves
    - Validation prediction examples
"""

import os
import random
import numpy as np
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\ISRO_14")

IMAGE_DIR = (
    BASE_DIR
    / "seg_masks"
    / "03JUN2026"
    / "seg_patches"
    / "images"
)

MASK_DIR = (
    BASE_DIR
    / "seg_masks"
    / "03JUN2026"
    / "seg_patches"
    / "masks"
)

OUTPUT_DIR = BASE_DIR / "unet_output_256"

# IMPORTANT:
# Only 256x256 patches are used.
# seg_patches_128 is never accessed.

PATCH_SIZE = 256

NUM_CLASSES = 4

CLASS_NAMES = {
    0: "Dense_forest",
    1: "Mix",
    2: "Urban",
    3: "Unclear",
}

IGNORE_INDEX = 255

TRAIN_RATIO = 0.80

BATCH_SIZE = 4

EPOCHS = 50

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class EOS04Dataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_files = sorted(
            self.image_dir.glob("*.png")
        )

        valid_pairs = []

        for image_path in self.image_files:

            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                valid_pairs.append(
                    (image_path, mask_path)
                )

        self.pairs = valid_pairs

        print()
        print("=" * 70)
        print("DATASET")
        print("=" * 70)

        print(
            f"Image directory : {self.image_dir}"
        )

        print(
            f"Mask directory  : {self.mask_dir}"
        )

        print(
            f"Paired patches  : {len(self.pairs)}"
        )

        print(
            f"Patch size      : {PATCH_SIZE} x {PATCH_SIZE}"
        )

        print("=" * 70)

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        # ----------------------------------------------------
        # Load RGB composite
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        image = np.array(image).astype(np.float32)

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = image / 255.0

        # H,W,C -> C,H,W

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = Image.open(mask_path)

        mask = np.array(mask).astype(np.int64)

        # Check dimensions

        if mask.shape != (PATCH_SIZE, PATCH_SIZE):

            raise ValueError(
                f"Mask size {mask.shape} does not match "
                f"expected {(PATCH_SIZE, PATCH_SIZE)} "
                f"for {mask_path}"
            )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # Encoder

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

        # Bottleneck

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # Pooling

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Decoder

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

        # Final segmentation layer

        self.out_conv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        p1 = self.pool(e1)

        e2 = self.enc2(p1)

        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        p3 = self.pool(e3)

        e4 = self.enc4(p3)

        p4 = self.pool(e4)

        # Bottleneck

        b = self.bottleneck(p4)

        # Decoder

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

        output = self.out_conv(d1)

        return output


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(
    prediction,
    target,
    num_classes,
    ignore_index=255
):

    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()

    dice_scores = []

    for class_id in range(num_classes):

        pred_class = (
            prediction == class_id
        )

        target_class = (
            target == class_id
        )

        valid = (
            target != ignore_index
        )

        pred_class = pred_class & valid
        target_class = target_class & valid

        intersection = np.logical_and(
            pred_class,
            target_class
        ).sum()

        pred_sum = pred_class.sum()

        target_sum = target_class.sum()

        denominator = (
            pred_sum + target_sum
        )

        if denominator == 0:

            continue

        dice = (
            2.0 * intersection
            / denominator
        )

        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return float(
        np.mean(dice_scores)
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    prediction,
    target,
    num_classes,
    ignore_index=255
):

    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()

    ious = []

    for class_id in range(num_classes):

        valid = (
            target != ignore_index
        )

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

            ious.append(np.nan)

        else:

            ious.append(
                intersection / union
            )

    return ious


# ============================================================
# PIXEL ACCURACY
# ============================================================

def pixel_accuracy(
    prediction,
    target,
    ignore_index=255
):

    valid = (
        target != ignore_index
    )

    correct = (
        prediction[valid]
        == target[valid]
    ).sum()

    total = valid.sum()

    if total == 0:

        return 0.0

    return (
        correct.item()
        / total.item()
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    running_loss = 0.0

    for images, masks in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        running_loss
        / len(loader.dataset)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_dice = []

    all_accuracy = []

    all_ious = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            dice = dice_score(
                predictions,
                masks,
                NUM_CLASSES,
                IGNORE_INDEX
            )

            accuracy = pixel_accuracy(
                predictions,
                masks,
                IGNORE_INDEX
            )

            ious = calculate_iou(
                predictions,
                masks,
                NUM_CLASSES,
                IGNORE_INDEX
            )

            all_dice.append(dice)

            all_accuracy.append(
                accuracy
            )

            all_ious.append(ious)

    mean_loss = (
        running_loss
        / len(loader.dataset)
    )

    mean_dice = np.mean(
        all_dice
    )

    mean_accuracy = np.mean(
        all_accuracy
    )

    mean_ious = np.nanmean(
        np.array(all_ious),
        axis=0
    )

    mean_iou = np.nanmean(
        mean_ious
    )

    return (
        mean_loss,
        mean_dice,
        mean_accuracy,
        mean_ious,
        mean_iou
    )


# ============================================================
# SAVE EXAMPLE PREDICTIONS
# ============================================================

def save_prediction_examples(
    model,
    dataset,
    epoch,
    num_examples=3
):

    model.eval()

    example_dir = (
        OUTPUT_DIR
        / "predictions"
    )

    example_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    indices = random.sample(
        range(len(dataset)),
        min(
            num_examples,
            len(dataset)
        )
    )

    with torch.no_grad():

        for i, index in enumerate(indices):

            image, mask = dataset[index]

            input_tensor = (
                image.unsqueeze(0)
                .to(DEVICE)
            )

            output = model(
                input_tensor
            )

            prediction = torch.argmax(
                output,
                dim=1
            )[0]

            image_np = (
                image
                .permute(1, 2, 0)
                .numpy()
            )

            mask_np = mask.numpy()

            pred_np = (
                prediction
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Create figure
            # ------------------------------------------------

            fig = plt.figure(
                figsize=(15, 5)
            )

            ax1 = fig.add_subplot(
                1, 3, 1
            )

            ax1.imshow(
                image_np
            )

            ax1.set_title(
                "Input Composite"
            )

            ax1.axis("off")

            ax2 = fig.add_subplot(
                1, 3, 2
            )

            ax2.imshow(
                mask_np,
                vmin=0,
                vmax=3
            )

            ax2.set_title(
                "Ground Truth Mask"
            )

            ax2.axis("off")

            ax3 = fig.add_subplot(
                1, 3, 3
            )

            ax3.imshow(
                pred_np,
                vmin=0,
                vmax=3
            )

            ax3.set_title(
                "U-Net Prediction"
            )

            ax3.axis("off")

            plt.tight_layout()

            filename = (
                example_dir
                / f"epoch_{epoch:03d}_example_{i}.png"
            )

            plt.savefig(
                filename,
                dpi=150
            )

            plt.close()


# ============================================================
# TRAINING CURVES
# ============================================================

def save_training_curves(
    train_losses,
    val_losses,
    val_dices,
    val_ious
):

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # Loss

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "U-Net Training and Validation Loss"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "loss_curve.png",
        dpi=150
    )

    plt.close()

    # Dice

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        val_dices,
        label="Validation Dice"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Dice Score"
    )

    plt.title(
        "Validation Dice Score"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "dice_curve.png",
        dpi=150
    )

    plt.close()

    # IoU

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        val_ious,
        label="Validation Mean IoU"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mean IoU"
    )

    plt.title(
        "Validation Mean IoU"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "iou_curve.png",
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("EOS-04 U-NET TRAINING")
    print("=" * 70)

    print(
        f"Device          : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Patch size      : {PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Number classes  : {NUM_CLASSES}"
    )

    print(
        "Training source : seg_patches/"
    )

    print(
        "128x128 patches : NOT USED"
    )

    print("=" * 70)

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
    # Dataset
    # --------------------------------------------------------

    dataset = EOS04Dataset(
        IMAGE_DIR,
        MASK_DIR
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "No paired 256x256 patches found."
        )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_size = int(
        TRAIN_RATIO
        * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print()

    print(
        f"Training patches   : {len(train_dataset)}"
    )

    print(
        f"Validation patches : {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    train_losses = []

    val_losses = []

    val_dices = []

    val_ious = []

    best_val_iou = -1.0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        (
            val_loss,
            val_dice,
            val_accuracy,
            class_ious,
            mean_iou
        ) = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_loss
        )

        # Save history

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        val_dices.append(
            val_dice
        )

        val_ious.append(
            mean_iou
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}]"
        )

        print(
            f"  Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"  Validation Loss  : "
            f"{val_loss:.4f}"
        )

        print(
            f"  Pixel Accuracy   : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"  Dice Score       : "
            f"{val_dice:.4f}"
        )

        print(
            f"  Mean IoU         : "
            f"{mean_iou:.4f}"
        )

        print(
            f"  Learning Rate    : "
            f"{current_lr:.6f}"
        )

        print(
            "  Class IoU:"
        )

        for class_id, iou in enumerate(
            class_ious
        ):

            print(
                f"    {CLASS_NAMES[class_id]:15s}: "
                f"{iou:.4f}"
            )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if mean_iou > best_val_iou:

            best_val_iou = mean_iou

            checkpoint_path = (
                OUTPUT_DIR
                / "best_unet_eos04_256.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss":
                        val_loss,
                    "val_dice":
                        val_dice,
                    "val_accuracy":
                        val_accuracy,
                    "mean_iou":
                        mean_iou,
                    "class_ious":
                        class_ious,
                    "num_classes":
                        NUM_CLASSES,
                    "patch_size":
                        PATCH_SIZE,
                    "class_names":
                        CLASS_NAMES,
                },
                checkpoint_path
            )

            print(
                f"  ✓ Best model saved "
                f"(Mean IoU = {mean_iou:.4f})"
            )

        # ----------------------------------------------------
        # Save prediction examples every 10 epochs
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            save_prediction_examples(
                model,
                val_dataset,
                epoch
            )

    # --------------------------------------------------------
    # Save curves
    # --------------------------------------------------------

    save_training_curves(
        train_losses,
        val_losses,
        val_dices,
        val_ious
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_path = (
        OUTPUT_DIR
        / "final_unet_eos04_256.pth"
    )

    torch.save(
        model.state_dict(),
        final_path
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Mean IoU : "
        f"{best_val_iou:.4f}"
    )

    print()

    print(
        f"Best model:"
    )

    print(
        OUTPUT_DIR
        / "best_unet_eos04_256.pth"
    )

    print()

    print(
        f"Final model:"
    )

    print(
        final_path
    )

    print()

    print(
        "Training curves:"
    )

    print(
        OUTPUT_DIR
        / "loss_curve.png"
    )

    print(
        OUTPUT_DIR
        / "dice_curve.png"
    )

    print(
        OUTPUT_DIR
        / "iou_curve.png"
    )

    print()

    print(
        "Prediction examples:"
    )

    print(
        OUTPUT_DIR
        / "predictions"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
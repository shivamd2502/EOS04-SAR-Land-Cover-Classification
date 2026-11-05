#!/usr/bin/env python3
# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 5 of 8 - SAR + DEM classifier (folder: scripts/05_classification_training)
# (file was named eos04_train_autolabelclassifierDEM.py)
#
# WHY THIS STEP
#   Steep Himalayan slopes can mimic urban backscatter. This version extends the
#   MLP branch from 4 SAR features to 9 (adding dem_mean, slp_mean, slp_std,
#   slp_max, slp_p15 from the Copernicus DSM), to test whether terrain context
#   reduces the hill / urban confusion. The architecture diagram is in the
#   docstring below.
#
# WHAT IT PRODUCED
#   data/05_classification_training/models/EOS04_256_L2B_DEM: 1,000 patches (800 / 200), val acc 93.5 % at
#   epoch 56 (training_summary.json). The raw label "Hilly" is mapped to Dense_forest here.
#   The controlled comparison is scripts/07_model_evaluation/dem_ablation_study.py:
#   on the identical split, SAR-only reached 94.0 % and SAR+DEM 92.0 %
#   (McNemar p = 0.22, not significant). DEM features did not improve accuracy.
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
===============================================================================
SAR PATCH CLASSIFIER — EOS-04 CALIBRATED COMPOSITES
SAR + DEM/TERRAIN MULTI-INPUT CLASSIFICATION
===============================================================================

Architecture
------------
RGB/SAR PNG PATCH
        |
        v
EfficientNet-B0
        |
        v
1280 CNN FEATURES
        |
        |                       SAR + DEM FEATURES
        |                              |
        |                              v
        |                       Z-SCORE NORMALISATION
        |                              |
        |                              v
        |                         Linear 9 -> 32
        |                              |
        |                         BatchNorm + ReLU
        |                              |
        |                           Dropout
        |                              |
        |                         Linear 32 -> 32
        |                              |
        +---------------+--------------+
                        |
                        v
                 CONCAT 1280 + 32
                        |
                        v
                  Linear 1312 -> 256
                        |
                      ReLU
                        |
                     Dropout
                        |
                  Linear 256 -> 3
                        |
             +----------+----------+
             |          |          |
             v          v          v
        Dense_forest   Mix       Urban

MODEL CLASSES
-------------
0 = Dense_forest
1 = Mix
2 = Urban

RAW CSV LABELS
--------------
The supplied training output showed these labels in the CSV:
    Hilly
    Mix
    Urban Dense

By default they are mapped as:
    Hilly       -> Dense_forest
    Mix         -> Mix
    Urban Dense -> Urban

IMPORTANT:
Verify that this semantic mapping is correct for your project before using
the results in a paper. If "Hilly" is not intended to mean Dense_forest,
change RAW_TO_MODEL_LABEL below.

AUXILIARY FEATURES
------------------
SAR:
    1. mean_r
    2. mean_g
    3. hh_hv_rat
    4. rvi_mean

DEM / TERRAIN:
    5. dem_mean
    6. slp_mean
    7. slp_std
    8. slp_max
    9. slp_p15
===============================================================================
"""

from __future__ import annotations

import os
import json
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = r"D:\ISRO_14"

PATCHES_ROOT = os.path.join(
    BASE_DIR,
    "data", "02_composites_and_patches", "patches_256_L2B_DEM",
    "03JUN2026",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data", "05_classification_training", "models",
    "EOS04_256_L2B_DEM",
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# COMPOSITES
# =============================================================================

COMPOSITE_NAMES = [
    "OPT1_sigma0",
    "OPT2_gamma0",
    "OPT3_rvi",
]


# =============================================================================
# MODEL CLASS NAMES
# =============================================================================

CLASS_NAMES = [
    "Dense_forest",
    "Mix",
    "Urban",
]


# =============================================================================
# RAW CSV LABEL -> MODEL LABEL
# =============================================================================
#
# IMPORTANT:
# Verify these semantic mappings with your QGIS labels.
#
# Your previous run showed:
#   Hilly
#   Mix
#   Urban Dense
#
# while the model was configured for:
#   Dense_forest
#   Mix
#   Urban
#
# Therefore the previous code discarded Hilly and Urban Dense.
#
# =============================================================================

RAW_TO_MODEL_LABEL = {
    "Hilly": "Dense_forest",
    "Mix": "Mix",
    "Urban Dense": "Urban",

    # Also support already-normalized labels.
    "Dense_forest": "Dense_forest",
    "Urban": "Urban",
}


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.20
RANDOM_SEED = 42


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# AUXILIARY FEATURES
# =============================================================================

AUX_FEATURE_COLS = [
    # SAR
    "mean_r",
    "mean_g",
    "hh_hv_rat",
    "rvi_mean",

    # DEM / terrain
    "dem_mean",
    "slp_mean",
    "slp_std",
    "slp_max",
    "slp_p15",
]


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is useful for experiments.
    # It may slightly reduce GPU performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# STEP 1 — LOAD AND MERGE LABELS
# =============================================================================

def load_labels(
    composite: str,
    patches_root: str,
) -> pd.DataFrame:
    """
    Load labels_clean_DEM.csv for one or all composites.

    Returns a DataFrame containing:
        abs_png_path
        label
        label_id
        composite
        SAR + DEM auxiliary features
    """

    if composite == "all":
        targets = COMPOSITE_NAMES
    else:
        targets = [composite]

    dfs = []

    for name in targets:

        csv_path = os.path.join(
            patches_root,
            name,
            f"{name}_labels_clean_DEM.csv",
        )

        if not os.path.isfile(csv_path):
            print(
                f"[WARN] CSV not found, skipping:\n"
                f"       {csv_path}"
            )
            continue

        df = pd.read_csv(csv_path)

        print(
            f"  Loaded {csv_path} -> {len(df)} rows"
        )

        # ---------------------------------------------------------------------
        # use_flag
        # ---------------------------------------------------------------------

        if "use_flag" in df.columns:

            before = len(df)

            df = df[
                df["use_flag"] == 1
            ].copy()

            print(
                f"    use_flag filter: "
                f"kept {len(df)} / {before} patches"
            )

        # ---------------------------------------------------------------------
        # Required label column
        # ---------------------------------------------------------------------

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column not found in:\n{csv_path}"
            )

        # ---------------------------------------------------------------------
        # Remove empty labels
        # ---------------------------------------------------------------------

        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
        )

        df = df[
            df["label"].notna()
            & (df["label"] != "")
            & (df["label"].str.lower() != "nan")
        ].copy()

        # ---------------------------------------------------------------------
        # Show RAW labels before mapping
        # ---------------------------------------------------------------------

        print("\n  Raw label distribution:")

        for label, count in df["label"].value_counts().items():
            print(
                f"    {label}: {count}"
            )

        # ---------------------------------------------------------------------
        # Map raw labels to model labels
        # ---------------------------------------------------------------------

        df["label_model"] = (
            df["label"]
            .map(RAW_TO_MODEL_LABEL)
        )

        unknown = sorted(
            df.loc[
                df["label_model"].isna(),
                "label",
            ].unique()
        )

        if unknown:
            raise ValueError(
                "\nUnknown labels found in CSV:\n"
                f"  {unknown}\n\n"
                "Update RAW_TO_MODEL_LABEL in this script "
                "before training."
            )

        # ---------------------------------------------------------------------
        # Construct absolute PNG path
        # ---------------------------------------------------------------------

        composite_dir = os.path.join(
            patches_root,
            name,
        )

        if "png_path" not in df.columns:
            raise ValueError(
                f"'png_path' column not found in:\n{csv_path}"
            )

        df["abs_png_path"] = df["png_path"].apply(
            lambda p: os.path.join(
                composite_dir,
                str(p),
            )
        )

        # ---------------------------------------------------------------------
        # Clean auxiliary features
        # ---------------------------------------------------------------------

        for col in AUX_FEATURE_COLS:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                df[col] = df[col].replace(
                    [
                        -9999,
                        -9999.0,
                        -999.0,
                    ],
                    np.nan,
                )

                df[col] = df[col].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

                # Keep the original behavior:
                # missing feature values become 0.
                df[col] = df[col].fillna(0.0)

            else:

                print(
                    f"  [WARN] Feature '{col}' "
                    f"not found in CSV. Filling with 0."
                )

                df[col] = 0.0

        # ---------------------------------------------------------------------
        # Composite identifier
        # ---------------------------------------------------------------------

        df["composite"] = name

        dfs.append(df)

    # -------------------------------------------------------------------------
    # Verify CSVs
    # -------------------------------------------------------------------------

    if not dfs:
        raise FileNotFoundError(
            f"No label CSV files found for "
            f"composite='{composite}' under:\n"
            f"{patches_root}"
        )

    merged = pd.concat(
        dfs,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Convert normalized model labels to IDs
    # -------------------------------------------------------------------------

    class_to_id = {
        name: i
        for i, name in enumerate(CLASS_NAMES)
    }

    merged["label_id"] = (
        merged["label_model"]
        .map(class_to_id)
    )

    if merged["label_id"].isna().any():
        raise ValueError(
            "Some normalized labels could not be converted "
            "to class IDs."
        )

    merged["label_id"] = (
        merged["label_id"]
        .astype(int)
    )

    # From this point onward, use MODEL labels.
    merged["label"] = merged["label_model"]

    # -------------------------------------------------------------------------
    # Print final label distribution
    # -------------------------------------------------------------------------

    print(
        f"\nLabel distribution "
        f"(model labels, composite={composite}):"
    )

    for class_name in CLASS_NAMES:

        count = int(
            (merged["label"] == class_name).sum()
        )

        print(
            f"  {class_name}: {count}"
        )

    print(
        f"  TOTAL: {len(merged)}"
    )

    # -------------------------------------------------------------------------
    # Print auxiliary features
    # -------------------------------------------------------------------------

    print(
        "\nAuxiliary features used by model:"
    )

    for i, col in enumerate(AUX_FEATURE_COLS):
        print(
            f"  {i + 1}. {col}"
        )

    return merged


# =============================================================================
# STEP 2 — FEATURE NORMALIZATION
# =============================================================================

def compute_feature_stats(
    train_df: pd.DataFrame,
) -> dict:
    """
    Compute mean/std using TRAINING data only.
    This prevents validation information from leaking into normalization.
    """

    stats = {}

    for col in AUX_FEATURE_COLS:

        mean_value = float(
            train_df[col].mean()
        )

        std_value = float(
            train_df[col].std()
        )

        stats[col] = {
            "mean": mean_value,
            "std": max(std_value, 1e-6),
        }

    return stats


def normalise_features(
    df: pd.DataFrame,
    stats: dict,
) -> np.ndarray:
    """
    Return z-score normalized auxiliary features.

    Shape:
        (N, 9)
    """

    arr = np.zeros(
        (
            len(df),
            len(AUX_FEATURE_COLS),
        ),
        dtype=np.float32,
    )

    for i, col in enumerate(AUX_FEATURE_COLS):

        values = (
            df[col].values.astype(np.float32)
        )

        arr[:, i] = (
            values - stats[col]["mean"]
        ) / stats[col]["std"]

    # Protect against any numerical issue.
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return arr


# =============================================================================
# STEP 3 — DATASET
# =============================================================================

class EOS04PatchDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        feature_arr: np.ndarray,
        transform=None,
    ):

        self.df = (
            df
            .reset_index(drop=True)
        )

        self.feature_arr = feature_arr
        self.transform = transform

        if len(self.df) != len(self.feature_arr):
            raise ValueError(
                "DataFrame and feature array lengths do not match."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        label = int(
            row["label_id"]
        )

        aux_feat = torch.tensor(
            self.feature_arr[idx],
            dtype=torch.float32,
        )

        img_path = row["abs_png_path"]

        try:
            img = Image.open(
                img_path
            ).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Could not read image:\n{img_path}\n"
                f"Error: {exc}"
            )

        if self.transform:
            img = self.transform(img)

        return (
            img,
            aux_feat,
            label,
        )


# =============================================================================
# STEP 4 — DATA AUGMENTATION
# =============================================================================

def get_transforms():

    mean = [
        0.485,
        0.456,
        0.406,
    ]

    std = [
        0.229,
        0.224,
        0.225,
    ]

    train_tf = transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=90
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    return train_tf, val_tf


# =============================================================================
# STEP 5 — MULTI-INPUT MODEL
# =============================================================================

class EOS04Classifier(nn.Module):

    def __init__(
        self,
        num_classes: int,
        num_aux_features: int = 9,
    ):

        super().__init__()

        # =====================================================================
        # CNN BRANCH — EfficientNet-B0
        # =====================================================================

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Freeze complete backbone initially.
        for param in backbone.parameters():
            param.requires_grad = False

        self.cnn_features = backbone.features
        self.pool = backbone.avgpool

        self.cnn_dim = (
            backbone.classifier[1].in_features
        )

        print(
            f"EfficientNet CNN feature dimension: "
            f"{self.cnn_dim}"
        )

        # Keep references to the last two feature blocks.
        self._phase2_params = list(
            self.cnn_features[-2:].parameters()
        )

        # =====================================================================
        # AUXILIARY SAR + DEM BRANCH
        # =====================================================================

        self.sar_branch = nn.Sequential(

            nn.Linear(
                num_aux_features,
                32,
            ),

            nn.BatchNorm1d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                32,
                32,
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.sar_dim = 32

        # =====================================================================
        # FUSION CLASSIFIER
        # =====================================================================

        fusion_dim = (
            self.cnn_dim
            + self.sar_dim
        )

        self.classifier = nn.Sequential(

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                fusion_dim,
                256,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                256,
                num_classes,
            ),
        )

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        images,
        aux_features,
    ):

        x = self.cnn_features(
            images
        )

        x = self.pool(
            x
        )

        x = x.flatten(
            1
        )

        s = self.sar_branch(
            aux_features
        )

        fused = torch.cat(
            [
                x,
                s,
            ],
            dim=1,
        )

        return self.classifier(
            fused
        )

    # =========================================================================
    # PHASE 2
    # =========================================================================

    def unfreeze_for_phase2(self):

        for param in self._phase2_params:
            param.requires_grad = True


# =============================================================================
# STEP 6 — TRAINING
# =============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, aux_feat, labels in loader:

        images = images.to(device)
        aux_feat = aux_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images,
            aux_feat,
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        correct += (
            outputs.argmax(1)
            == labels
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, aux_feat, labels in loader:

            images = images.to(device)
            aux_feat = aux_feat.to(device)
            labels = labels.to(device)

            outputs = model(
                images,
                aux_feat,
            )

            loss = criterion(
                outputs,
                labels,
            )

            preds = outputs.argmax(
                1
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            correct += (
                preds == labels
            ).sum().item()

            total += batch_size

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return (
        total_loss / total,
        correct / total,
        all_preds,
        all_labels,
    )


# =============================================================================
# STEP 7 — TRAINING CURVES
# =============================================================================

def plot_curves(
    history,
    model_dir,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    fig.suptitle(
        "EOS-04 SAR + DEM Patch Classifier — Training Results",
        fontsize=13,
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    # Loss
    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train",
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation",
    )

    axes[0].set_title(
        "Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(
        epochs,
        history["train_acc"],
        label="Train",
    )

    axes[1].plot(
        epochs,
        history["val_acc"],
        label="Validation",
    )

    axes[1].set_title(
        "Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].set_ylim(
        0,
        1,
    )

    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    out = os.path.join(
        model_dir,
        "training_results.png",
    )

    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Saved: {out}"
    )


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_cm(
    labels,
    preds,
    class_names,
    model_dir,
):

    cm = confusion_matrix(
        labels,
        preds,
        labels=list(
            range(len(class_names))
        ),
    )

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_title(
        "Confusion Matrix (Validation Set)"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    plt.tight_layout()

    out = os.path.join(
        model_dir,
        "confusion_matrix.png",
    )

    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Saved: {out}"
    )


# =============================================================================
# SAVE CLASS DISTRIBUTION
# =============================================================================

def print_class_distribution(
    df,
    title,
):

    print(
        f"\n{title}"
    )

    counts = (
        df["label_id"]
        .value_counts()
        .reindex(
            range(len(CLASS_NAMES)),
            fill_value=0,
        )
    )

    for i, class_name in enumerate(CLASS_NAMES):

        print(
            f"  {class_name}: "
            f"{int(counts.iloc[i])}"
        )

    return counts


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--composite",
        default="OPT1_sigma0",
        choices=COMPOSITE_NAMES + ["all"],
        help="Which composite(s) to train on",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
    )

    args = parser.parse_args()

    # =========================================================================
    # START
    # =========================================================================

    print(
        "=" * 70
    )

    print(
        "EOS-04 SAR + DEM PATCH CLASSIFIER — TRAINING"
    )

    print(
        f"Composite : {args.composite}"
    )

    print(
        f"Device    : {DEVICE}"
    )

    print(
        f"Auxiliary features : "
        f"{len(AUX_FEATURE_COLS)}"
    )

    print(
        "=" * 70
    )

    # =========================================================================
    # STEP 1 — LOAD LABELS
    # =========================================================================

    print(
        "\n[1] Loading labels ..."
    )

    df = load_labels(
        args.composite,
        PATCHES_ROOT,
    )

    num_classes = len(
        CLASS_NAMES
    )

    print(
        f"\nClasses ({num_classes}): "
        f"{CLASS_NAMES}"
    )

    # =========================================================================
    # VERIFY PNG FILES
    # =========================================================================

    exists_mask = df[
        "abs_png_path"
    ].apply(
        os.path.isfile
    )

    missing_count = int(
        (~exists_mask).sum()
    )

    if missing_count:

        print(
            f"[WARN] {missing_count} PNG files "
            f"missing — dropping them"
        )

        df = df[
            exists_mask
        ].copy()

    print(
        f"Final: {len(df)} usable patches"
    )

    if len(df) < 10:
        raise ValueError(
            "Too few patches to train. "
            "Check your CSV and PNG paths."
        )

    # =========================================================================
    # CRITICAL CLASS VALIDATION
    # =========================================================================

    counts_full_df = print_class_distribution(
        df,
        "Final class distribution before split:",
    )

    missing_classes = [
        CLASS_NAMES[i]
        for i, count in enumerate(counts_full_df)
        if count == 0
    ]

    if missing_classes:

        raise ValueError(
            "\nTRAINING STOPPED.\n"
            f"These classes have ZERO samples: "
            f"{missing_classes}\n\n"
            "This is not a valid 3-class training dataset. "
            "Check your CSV labels and RAW_TO_MODEL_LABEL mapping."
        )

    too_small = [
        CLASS_NAMES[i]
        for i, count in enumerate(counts_full_df)
        if count < 2
    ]

    if too_small:

        raise ValueError(
            "\nTRAINING STOPPED.\n"
            f"These classes have fewer than 2 samples: "
            f"{too_small}\n"
            "Stratified train/validation splitting is not reliable."
        )

    # =========================================================================
    # SAVE LABEL ENCODER
    # =========================================================================

    label_enc = {
        name: i
        for i, name in enumerate(
            CLASS_NAMES
        )
    }

    label_encoder_path = os.path.join(
        MODEL_DIR,
        "label_encoder.json",
    )

    with open(
        label_encoder_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            label_enc,
            f,
            indent=2,
        )

    print(
        f"\nLabel encoder saved: "
        f"{label_enc}"
    )

    # =========================================================================
    # STEP 2 — TRAIN / VALIDATION SPLIT
    # =========================================================================

    print(
        "\n[2] Splitting dataset (80/20) ..."
    )

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

    print(
        f"  Train: {len(train_df)}"
    )

    print(
        f"  Val:   {len(val_df)}"
    )

    print_class_distribution(
        train_df,
        "Training class distribution:",
    )

    print_class_distribution(
        val_df,
        "Validation class distribution:",
    )

    # =========================================================================
    # STEP 3 — FEATURE NORMALIZATION
    # =========================================================================

    print(
        "\n[3] Computing SAR + DEM "
        "feature normalization stats ..."
    )

    feat_stats = compute_feature_stats(
        train_df
    )

    feature_stats_path = os.path.join(
        MODEL_DIR,
        "feature_stats.json",
    )

    with open(
        feature_stats_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feat_stats,
            f,
            indent=2,
        )

    print(
        "\nAuxiliary feature statistics:"
    )

    for col, stats in feat_stats.items():

        print(
            f"  {col}: "
            f"mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f}"
        )

    train_feat = normalise_features(
        train_df,
        feat_stats,
    )

    val_feat = normalise_features(
        val_df,
        feat_stats,
    )

    print(
        f"\nTrain auxiliary feature shape: "
        f"{train_feat.shape}"
    )

    print(
        f"Validation auxiliary feature shape: "
        f"{val_feat.shape}"
    )

    # =========================================================================
    # STEP 4 — DATASETS
    # =========================================================================

    train_tf, val_tf = get_transforms()

    train_ds = EOS04PatchDataset(
        train_df,
        train_feat,
        train_tf,
    )

    val_ds = EOS04PatchDataset(
        val_df,
        val_feat,
        val_tf,
    )

    # =========================================================================
    # DATALOADERS
    # =========================================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    # =========================================================================
    # STEP 5 — BUILD MODEL
    # =========================================================================

    print(
        "\n[4] Building multi-input "
        "EfficientNet-B0 model ..."
    )

    model = EOS04Classifier(
        num_classes=num_classes,
        num_aux_features=len(
            AUX_FEATURE_COLS
        ),
    ).to(DEVICE)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"  Total params    : {total:,}"
    )

    print(
        f"  Trainable params: {trainable:,}"
    )

    print(
        "  CNN backbone    : Frozen initially"
    )

    print(
        f"  Auxiliary input : "
        f"{len(AUX_FEATURE_COLS)} features"
    )

    # =========================================================================
    # STEP 6 — CLASS WEIGHTS
    # =========================================================================

    counts_series = (
        train_df["label_id"]
        .value_counts()
        .sort_index()
    )

    counts_full = np.zeros(
        num_classes,
        dtype=np.float32,
    )

    for label_id, count in counts_series.items():

        counts_full[
            int(label_id)
        ] = float(count)

    if np.any(counts_full <= 0):

        raise ValueError(
            "A class has zero training samples. "
            "Cannot compute valid class weights."
        )

    # Inverse-frequency weights.
    weights_arr = (
        1.0 / counts_full
    )

    weights = torch.tensor(
        weights_arr,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Normalize average weight to approximately 1.
    weights = (
        weights
        / weights.mean()
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    print(
        "\nClass weights:"
    )

    print({
        CLASS_NAMES[i]:
        round(
            float(weights[i].item()),
            3,
        )
        for i in range(num_classes)
    })

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # =========================================================================
    # STEP 7 — TRAINING
    # =========================================================================

    print(
        f"\n[5] Training for "
        f"{args.epochs} epochs ..."
    )

    print(
        "  Phase 1 (ep 1-20): "
        "classifier head + auxiliary branch only"
    )

    print(
        "  Phase 2 (ep 21+): "
        "+ last 2 CNN blocks unfrozen"
    )

    print(
        "-" * 70
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    best_path = os.path.join(
        MODEL_DIR,
        "best_model.pth",
    )

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ---------------------------------------------------------------------
        # PHASE 2
        # ---------------------------------------------------------------------

        if epoch == 21 and args.epochs >= 21:

            print(
                "\n[Phase 2] "
                "Unfreezing last 2 CNN blocks ..."
            )

            model.unfreeze_for_phase2()

            # Rebuild optimizer so newly trainable parameters are included.
            optimizer = optim.AdamW(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters(),
                ),
                lr=args.lr * 0.1,
                weight_decay=1e-4,
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        # ---------------------------------------------------------------------
        # VALIDATE
        # ---------------------------------------------------------------------

        (
            vl_loss,
            vl_acc,
            v_preds,
            v_labels,
        ) = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        # ---------------------------------------------------------------------
        # LR scheduler
        # ---------------------------------------------------------------------

        scheduler.step(
            vl_loss
        )

        # ---------------------------------------------------------------------
        # HISTORY
        # ---------------------------------------------------------------------

        history["train_loss"].append(
            tr_loss
        )

        history["val_loss"].append(
            vl_loss
        )

        history["train_acc"].append(
            tr_acc
        )

        history["val_acc"].append(
            vl_acc
        )

        # ---------------------------------------------------------------------
        # PRINT
        # ---------------------------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"Train loss={tr_loss:.4f} "
            f"acc={tr_acc:.3f} | "
            f"Val loss={vl_loss:.4f} "
            f"acc={vl_acc:.3f} | "
            f"LR={current_lr:.2e}"
        )

        # ---------------------------------------------------------------------
        # SAVE BEST MODEL
        #
        # Primary metric = validation accuracy.
        # If accuracy ties, lower validation loss wins.
        # ---------------------------------------------------------------------

        is_better = (
            vl_acc > best_val_acc
            or (
                np.isclose(vl_acc, best_val_acc)
                and vl_loss < best_val_loss
            )
        )

        if is_better:

            best_val_acc = vl_acc
            best_val_loss = vl_loss
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": vl_acc,
                "val_loss": vl_loss,
                "class_names": CLASS_NAMES,
                "composite": args.composite,
                "aux_feature_cols": AUX_FEATURE_COLS,
                "num_aux_features": len(
                    AUX_FEATURE_COLS
                ),
                "raw_to_model_label": RAW_TO_MODEL_LABEL,
            }

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"    ✓ Best model saved "
                f"(epoch={epoch}, "
                f"val_acc={vl_acc:.3f}, "
                f"val_loss={vl_loss:.4f})"
            )

    # =========================================================================
    # STEP 8 — FINAL EVALUATION
    # =========================================================================

    print(
        "\n[6] Final evaluation ..."
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f} "
        f"({best_val_acc * 100:.1f}%)"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    # -------------------------------------------------------------------------
    # Reload exact best model
    # -------------------------------------------------------------------------

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        best_eval_loss,
        best_eval_acc,
        final_preds,
        final_labels,
    ) = validate(
        model,
        val_loader,
        criterion,
        DEVICE,
    )

    print(
        f"Reloaded best model validation accuracy: "
        f"{best_eval_acc:.4f} "
        f"({best_eval_acc * 100:.1f}%)"
    )

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            final_labels,
            final_preds,
            labels=list(
                range(num_classes)
            ),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # =========================================================================
    # STEP 9 — SAVE PLOTS
    # =========================================================================

    print(
        "\n[7] Saving plots ..."
    )

    plot_curves(
        history,
        MODEL_DIR,
    )

    plot_cm(
        final_labels,
        final_preds,
        CLASS_NAMES,
        MODEL_DIR,
    )

    # =========================================================================
    # SAVE TRAINING HISTORY
    # =========================================================================

    history_path = os.path.join(
        MODEL_DIR,
        "training_history.json",
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                key: [
                    float(v)
                    for v in values
                ]
                for key, values in history.items()
            },
            f,
            indent=2,
        )

    print(
        f"  Saved: {history_path}"
    )

    # =========================================================================
    # SAVE FINAL SUMMARY
    # =========================================================================

    summary = {
        "composite": args.composite,
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(
            best_eval_acc
        ),
        "best_validation_loss": float(
            best_eval_loss
        ),
        "num_total_patches": int(len(df)),
        "num_train_patches": int(len(train_df)),
        "num_validation_patches": int(len(val_df)),
        "class_names": CLASS_NAMES,
        "class_distribution_total": {
            CLASS_NAMES[i]: int(counts_full_df.iloc[i])
            for i in range(num_classes)
        },
        "auxiliary_features": AUX_FEATURE_COLS,
        "raw_to_model_label": RAW_TO_MODEL_LABEL,
        "best_model": best_path,
        "label_encoder": label_encoder_path,
        "feature_stats": feature_stats_path,
    }

    summary_path = os.path.join(
        MODEL_DIR,
        "training_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best model   : {best_path}"
    )

    print(
        f"Best epoch   : {best_epoch}"
    )

    print(
        f"Best val acc : "
        f"{best_eval_acc * 100:.1f}%"
    )

    print(
        f"Total patches: {len(df)}"
    )

    print(
        f"Train patches: {len(train_df)}"
    )

    print(
        f"Val patches  : {len(val_df)}"
    )

    print(
        f"Classes      : {CLASS_NAMES}"
    )

    print(
        f"Aux features : {len(AUX_FEATURE_COLS)}"
    )

    print(
        "Aux feature list:"
    )

    for i, feature in enumerate(
        AUX_FEATURE_COLS,
        start=1,
    ):
        print(
            f"  {i}. {feature}"
        )

    print(
        f"Label encoder: {label_encoder_path}"
    )

    print(
        f"Feature stats: {feature_stats_path}"
    )

    print(
        f"Summary      : {summary_path}"
    )

    print(
        "=" * 70
    )


# =============================================================================
# PROGRAM ENTRY
# =============================================================================

if __name__ == "__main__":

    set_seed(
        RANDOM_SEED
    )

    main()
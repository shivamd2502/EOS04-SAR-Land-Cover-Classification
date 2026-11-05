# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 7 of 8 - DEM ablation (folder: scripts/07_model_evaluation)
#
# WHY THIS STEP
#   The controlled test of whether terrain features help: two models with the same
#   architecture, data (1,000 patches), split (seed 42, stratified 800/200) and
#   epochs (100). The only difference is the MLP input (4 SAR vs 4 SAR + 5 DEM
#   features). McNemar's test compares their paired predictions.
#
# WHAT IT PRODUCED (results/07_model_evaluation/dem_ablation_study/)
#   SAR only 94.0 %, SAR + DEM 92.0 %. Discordant pairs: 5 only-SAR-correct vs
#   1 only-DEM-correct, chi2 = 1.5, p = 0.22 -> no significant difference.
#   On this scene, DEM features did not improve classification.
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
EOS-04 L2B SAR-only vs SAR+DEM ablation study.

Usage:
    python dem_ablation_study.py
    python dem_ablation_study.py --epochs 100
    python dem_ablation_study.py --skip-training

--skip-training loads:
    model_A_SAR_only/best_model.pth
    model_B_SAR_DEM/best_model.pth

The report is always written as UTF-8.
"""

from __future__ import annotations

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# =============================================================================
# CONFIG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ablation")

BASE_DIR = r"D:\ISRO_14"

PATCHES_ROOT = os.path.join(
    BASE_DIR, "data", "02_composites_and_patches", "patches_256_L2B_DEM", "03JUN2026"
)

COMPOSITE = "OPT1_sigma0"

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "07_model_evaluation", "ablation_study"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Dense_forest", "Mix", "Urban"]

RAW_TO_MODEL = {
    "Hilly": "Dense_forest",
    "Mix": "Mix",
    "Urban Dense": "Urban",
    "Dense_forest": "Dense_forest",
    "Urban": "Urban",
}

SAR_ONLY_FEATURES = [
    "mean_r",
    "mean_g",
    "hh_hv_rat",
    "rvi_mean",
]

SAR_DEM_FEATURES = [
    "mean_r",
    "mean_g",
    "hh_hv_rat",
    "rvi_mean",
    "dem_mean",
    "slp_mean",
    "slp_std",
    "slp_max",
    "slp_p15",
]

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
VAL_SPLIT = 0.20
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# DATASET
# =============================================================================

def load_dataset(patches_root: str, composite: str) -> pd.DataFrame:
    csv_path = os.path.join(
        patches_root,
        composite,
        f"{composite}_labels_clean_DEM.csv",
    )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info("Loading CSV: %s", csv_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "WKT" in df.columns:
        df = df.drop(columns=["WKT"])

    if "use_flag" in df.columns:
        df = df[df["use_flag"] == 1].copy()

    df["label"] = df["label"].astype(str).str.strip()
    df["label"] = df["label"].map(RAW_TO_MODEL)
    df = df[df["label"].isin(CLASS_NAMES)].copy()

    label_to_id = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }
    df["label_id"] = df["label"].map(label_to_id).astype(int)

    comp_dir = os.path.join(patches_root, composite)

    def resolve(p):
        p = str(p).strip()

        full = os.path.join(comp_dir, p)
        if os.path.isfile(full):
            return full

        bare = os.path.join(
            comp_dir, "patches", os.path.basename(p)
        )
        if os.path.isfile(bare):
            return bare

        return full

    df["abs_png"] = df["png_path"].apply(resolve)
    df = df[df["abs_png"].apply(os.path.isfile)].copy()
    df = df.reset_index(drop=True)

    for col in SAR_DEM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )
            df[col] = df[col].replace(
                [-9999, -9999.0], np.nan
            )
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    log.info(
        "Dataset: %d patches | %s",
        len(df),
        df["label"].value_counts().to_dict(),
    )

    return df


# =============================================================================
# NORMALIZATION
# =============================================================================

def compute_stats(
    train_df: pd.DataFrame,
    feat_cols: list[str],
) -> dict:
    return {
        col: {
            "mean": float(train_df[col].mean()),
            "std": max(float(train_df[col].std()), 1e-6),
        }
        for col in feat_cols
    }


def normalise(
    df: pd.DataFrame,
    stats: dict,
    feat_cols: list[str],
) -> np.ndarray:
    arr = np.zeros(
        (len(df), len(feat_cols)),
        dtype=np.float32,
    )

    for i, col in enumerate(feat_cols):
        arr[:, i] = (
            (df[col].values - stats[col]["mean"])
            / stats[col]["std"]
        ).astype(np.float32)

    return np.nan_to_num(
        arr, nan=0.0, posinf=0.0, neginf=0.0
    )


# =============================================================================
# TRANSFORMS
# =============================================================================

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class PatchDataset(Dataset):

    def __init__(self, df, feat_arr, transform):
        self.df = df.reset_index(drop=True)
        self.feat = feat_arr
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(
            row["abs_png"]
        ).convert("RGB")

        img = self.tf(img)

        feat = torch.tensor(
            self.feat[idx],
            dtype=torch.float32,
        )

        return (
            img,
            feat,
            int(row["label_id"]),
        )


# =============================================================================
# MODEL
# =============================================================================

class Classifier(nn.Module):

    def __init__(self, num_classes: int, num_features: int):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        for p in backbone.parameters():
            p.requires_grad = False

        self.cnn = backbone.features
        self.pool = backbone.avgpool
        self.cnn_dim = backbone.classifier[1].in_features

        self._phase2 = list(
            backbone.features[-2:].parameters()
        )

        self.sar = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(True),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.cnn_dim + 32, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, imgs, feats):
        x = self.pool(self.cnn(imgs)).flatten(1)
        s = self.sar(feats)
        return self.head(torch.cat([x, s], dim=1))

    def unfreeze(self):
        for p in self._phase2:
            p.requires_grad = True


# =============================================================================
# LOADERS
# =============================================================================

def make_loaders(train_df, val_df, feat_cols):
    stats = compute_stats(train_df, feat_cols)

    train_feat = normalise(
        train_df, stats, feat_cols
    )
    val_feat = normalise(
        val_df, stats, feat_cols
    )

    train_ds = PatchDataset(
        train_df, train_feat, TRAIN_TF
    )
    val_ds = PatchDataset(
        val_df, val_feat, VAL_TF
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return stats, train_ld, val_ld


def make_class_weights(train_df):
    counts = (
        train_df["label_id"]
        .value_counts()
        .sort_index()
    )

    weights = torch.tensor(
        [
            1.0 / counts.get(i, 1)
            for i in range(len(CLASS_NAMES))
        ],
        dtype=torch.float32,
    )

    weights = weights / weights.mean()
    return weights.to(DEVICE)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    preds = []
    labels_all = []

    with torch.no_grad():
        for imgs, feats, lbls in data_loader:
            imgs = imgs.to(DEVICE)
            feats = feats.to(DEVICE)
            lbls = lbls.to(DEVICE)

            out = model(imgs, feats)
            loss = criterion(out, lbls)

            total_loss += (
                loss.item() * imgs.size(0)
            )

            p = out.argmax(1)

            correct += (
                (p == lbls).sum().item()
            )

            total += imgs.size(0)

            preds.extend(p.cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(preds),
        np.array(labels_all),
    )


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(
    name,
    train_df,
    val_df,
    feat_cols,
    epochs,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    log.info(
        "Training [%s] with %d features ...",
        name,
        len(feat_cols),
    )

    stats, train_ld, val_ld = make_loaders(
        train_df, val_df, feat_cols
    )

    model = Classifier(
        len(CLASS_NAMES),
        len(feat_cols),
    ).to(DEVICE)

    class_weights = make_class_weights(train_df)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_acc = -1.0
    best_epoch = 0

    best_path = os.path.join(
        save_dir,
        "best_model.pth",
    )

    # Save normalization information.
    with open(
        os.path.join(
            save_dir,
            "normalization_stats.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "feature_columns": feat_cols,
                "stats": stats,
                "class_names": CLASS_NAMES,
                "img_size": IMG_SIZE,
            },
            f,
            indent=2,
        )

    for ep in range(1, epochs + 1):

        if ep == 21:
            model.unfreeze()

            optimizer = optim.AdamW(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters(),
                ),
                lr=LR * 0.1,
                weight_decay=1e-4,
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

            log.info(
                "[%s] Unfroze final two EfficientNet blocks",
                name,
            )

        model.train()

        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for imgs, feats, lbls in train_ld:
            imgs = imgs.to(DEVICE)
            feats = feats.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            out = model(imgs, feats)
            loss = criterion(out, lbls)

            loss.backward()
            optimizer.step()

            train_loss_total += (
                loss.item() * imgs.size(0)
            )

            train_correct += (
                (out.argmax(1) == lbls)
                .sum()
                .item()
            )

            train_total += imgs.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss_total / train_total

        val_loss, val_acc, preds, labels = evaluate_model(
            model,
            val_ld,
            criterion,
        )

        scheduler.step(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if ep % 10 == 0 or ep == 1:
            log.info(
                "[%s] ep %3d/%d | train=%.3f | val=%.3f",
                name,
                ep,
                epochs,
                train_acc,
                val_acc,
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = ep

            torch.save(
                model.state_dict(),
                best_path,
            )

            log.info(
                "[%s] best val_acc=%.4f at epoch %d",
                name,
                val_acc,
                ep,
            )

    # Reload the best checkpoint, not epoch 100.
    model.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    val_loss, best_acc, best_preds, best_labels = evaluate_model(
        model,
        val_ld,
        criterion,
    )

    report = classification_report(
        best_labels,
        best_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        best_labels,
        best_preds,
        labels=list(range(len(CLASS_NAMES))),
    )

    result = {
        "name": name,
        "feat_cols": feat_cols,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "preds": best_preds,
        "true_labels": best_labels,
        "report": report,
        "cm": cm,
        "history": history,
        "stats": stats,
        "best_path": best_path,
        "best_val_loss": val_loss,
    }

    save_history(history, save_dir)

    save_training_metadata(
        result,
        save_dir,
        epochs,
    )

    return result


# =============================================================================
# LOAD SAVED MODEL -- USED BY --skip-training
# =============================================================================

def load_saved_model(
    name,
    train_df,
    val_df,
    feat_cols,
    save_dir,
):
    """
    Load and evaluate an existing best_model.pth.

    NO TRAINING happens here.
    """

    best_path = os.path.join(
        save_dir,
        "best_model.pth",
    )

    if not os.path.isfile(best_path):
        raise FileNotFoundError(
            "\nSaved checkpoint not found:\n"
            f"{best_path}\n\n"
            "Run the script once without --skip-training."
        )

    log.info(
        "Loading saved checkpoint: %s",
        best_path,
    )

    # Recreate exactly the same split first.
    # Therefore normalization statistics are identical.
    stats, _, val_ld = make_loaders(
        train_df,
        val_df,
        feat_cols,
    )

    model = Classifier(
        len(CLASS_NAMES),
        len(feat_cols),
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    class_weights = make_class_weights(train_df)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    val_loss, val_acc, preds, labels = evaluate_model(
        model,
        val_ld,
        criterion,
    )

    report = classification_report(
        labels,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(len(CLASS_NAMES))),
    )

    metadata_path = os.path.join(
        save_dir,
        "training_metadata.json",
    )

    best_epoch = None

    if os.path.isfile(metadata_path):
        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as f:
                metadata = json.load(f)

            best_epoch = metadata.get(
                "best_epoch"
            )
        except Exception:
            pass

    history = load_saved_history(save_dir)

    if history is None:
        history = {
            "train_acc": [],
            "val_acc": [],
            "train_loss": [],
            "val_loss": [],
        }

    log.info(
        "[%s] Loaded checkpoint | val_acc=%.4f",
        name,
        val_acc,
    )

    return {
        "name": name,
        "feat_cols": feat_cols,
        "best_acc": val_acc,
        "best_epoch": best_epoch,
        "preds": preds,
        "true_labels": labels,
        "report": report,
        "cm": cm,
        "history": history,
        "stats": stats,
        "best_path": best_path,
        "best_val_loss": val_loss,
    }


# =============================================================================
# SAVING METADATA
# =============================================================================

def save_history(history, save_dir):
    with open(
        os.path.join(
            save_dir,
            "training_history.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)


def load_saved_history(save_dir):
    path = os.path.join(
        save_dir,
        "training_history.json",
    )

    if not os.path.isfile(path):
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return None


def save_training_metadata(
    result,
    save_dir,
    epochs,
):
    metadata = {
        "name": result["name"],
        "feature_columns": result["feat_cols"],
        "epochs": epochs,
        "best_epoch": result["best_epoch"],
        "best_validation_accuracy": result["best_acc"],
        "best_model_path": result["best_path"],
        "device": str(DEVICE),
        "random_seed": RANDOM_SEED,
    }

    with open(
        os.path.join(
            save_dir,
            "training_metadata.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# McNEMAR TEST
# =============================================================================

def mcnemar_test(preds_a, preds_b, true):
    correct_a = preds_a == true
    correct_b = preds_b == true

    n00 = int(
        (correct_a & correct_b).sum()
    )

    n01 = int(
        (correct_a & ~correct_b).sum()
    )

    n10 = int(
        (~correct_a & correct_b).sum()
    )

    n11 = int(
        (~correct_a & ~correct_b).sum()
    )

    disagreements = n01 + n10

    if disagreements == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "significant": False,
            "verdict": "No disagreement between models.",
        }

    chi2_stat = (
        (abs(n01 - n10) - 1) ** 2
        / disagreements
    )

    p_value = 1 - chi2.cdf(
        chi2_stat,
        df=1,
    )

    significant = p_value < 0.05

    verdict = (
        f"DEM model corrected {n10} patches "
        f"that SAR-only got wrong.\n"
        f"SAR-only corrected {n01} patches "
        f"that DEM got wrong.\n"
        f"Net improvement by DEM: {n10 - n01} patches.\n"
        + (
            "STATISTICALLY SIGNIFICANT (p<0.05)"
            if significant
            else
            "NOT statistically significant (p>=0.05)"
        )
    )

    return {
        "chi2": round(chi2_stat, 4),
        "p_value": float(p_value),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "significant": significant,
        "verdict": verdict,
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_all(res_a, res_b, mcnemar, out_dir):
    # -------------------------------------------------------------------------
    # Training curves
    # -------------------------------------------------------------------------
    if (
        len(res_a["history"]["val_acc"]) > 0
        and len(res_b["history"]["val_acc"]) > 0
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5)
        )

        fig.suptitle(
            "Ablation Study: SAR-only vs SAR+DEM - Same Dataset",
            fontsize=13,
            fontweight="bold",
        )

        axes[0].plot(
            res_a["history"]["val_acc"],
            label="Model A: SAR only",
        )

        axes[0].plot(
            res_b["history"]["val_acc"],
            label="Model B: SAR+DEM",
        )

        axes[0].set_title(
            "Validation Accuracy"
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(
            res_a["history"]["val_loss"],
            label="Model A: SAR only",
        )

        axes[1].plot(
            res_b["history"]["val_loss"],
            label="Model B: SAR+DEM",
        )

        axes[1].set_title(
            "Validation Loss"
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                "ablation_training_curves.png",
            ),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

    # -------------------------------------------------------------------------
    # F1 comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    f1_a = [
        res_a["report"][c]["f1-score"]
        for c in CLASS_NAMES
    ]

    f1_b = [
        res_b["report"][c]["f1-score"]
        for c in CLASS_NAMES
    ]

    bars_a = ax.bar(
        x - width / 2,
        f1_a,
        width,
        label=(
            f"Model A: SAR only "
            f"(acc={res_a['best_acc'] * 100:.1f}%)"
        ),
    )

    bars_b = ax.bar(
        x + width / 2,
        f1_b,
        width,
        label=(
            f"Model B: SAR+DEM "
            f"(acc={res_b['best_acc'] * 100:.1f}%)"
        ),
    )

    for bars in [bars_a, bars_b]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=9,
            )

    ax.set_title(
        "Per-Class F1: SAR-only vs SAR+DEM\n"
        "Same validation set",
        fontsize=11,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0.6, 1.1)
    ax.set_ylabel("F1 score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "ablation_f1_comparison.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Confusion matrices
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    fig.suptitle(
        "Confusion Matrices on IDENTICAL Validation Set",
        fontsize=13,
        fontweight="bold",
    )

    for ax, res in zip(
        axes, [res_a, res_b]
    ):
        cm = res["cm"]

        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )
        row_sums[row_sums == 0] = 1

        cm_pct = (
            cm.astype(float)
            / row_sums
            * 100
        )

        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            vmin=0,
            vmax=100,
            annot_kws={"size": 12},
        )

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(
                    j + 0.5,
                    i + 0.72,
                    f"n={cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

        ax.set_title(
            f"{res['name']}\n"
            f"acc={res['best_acc'] * 100:.2f}%"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "ablation_confusion_matrices.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # McNemar table
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    data = np.array([
        [mcnemar["n00"], mcnemar["n01"]],
        [mcnemar["n10"], mcnemar["n11"]],
    ])

    im = ax.imshow(
        data,
        cmap="Blues",
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Model B (DEM)\ncorrect",
        "Model B (DEM)\nwrong",
    ])

    ax.set_yticklabels([
        "Model A (SAR)\ncorrect",
        "Model A (SAR)\nwrong",
    ])

    labels_cm = [
        [
            f"Both correct\n(n={mcnemar['n00']})",
            f"Only SAR correct\n(n={mcnemar['n01']})",
        ],
        [
            f"Only DEM correct\n(n={mcnemar['n10']})",
            f"Both wrong\n(n={mcnemar['n11']})",
        ],
    ]

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels_cm[i][j],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=(
                    "white"
                    if data[i, j] > data.max() / 2
                    else "black"
                ),
            )

    significance = (
        "SIGNIFICANT"
        if mcnemar["significant"]
        else "NOT SIGNIFICANT"
    )

    ax.set_title(
        "McNemar's Test\n"
        f"chi2={mcnemar['chi2']:.3f}, "
        f"p={mcnemar['p_value']:.4f}, "
        f"{significance}"
    )

    plt.colorbar(
        im,
        ax=ax,
        label="count",
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "ablation_mcnemar.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "All plots saved to %s",
        out_dir,
    )


# =============================================================================
# REPORT
# =============================================================================

def save_report(
    res_a,
    res_b,
    mcnemar,
    n_total,
    n_train,
    n_val,
    epochs,
    out_dir,
):
    sep = "=" * 70

    lines = [
        sep,
        "ABLATION STUDY REPORT: SAR-only vs SAR+DEM",
        "EOS-04 L2B - DEM Feature Ablation",
        sep,
        "",
        "EXPERIMENTAL DESIGN:",
        "  Dataset            : IDENTICAL for both models",
        f"  Total patches      : {n_total}",
        f"  Training patches   : {n_train}",
        f"  Validation patches: {n_val}",
        f"  Train/val split    : seed={RANDOM_SEED}, stratified",
        "  Architecture       : EfficientNet-B0 + MLP",
        f"  Epochs             : {epochs} each",
        "  Only difference    : MLP input feature set",
        "",
        sep,
        "MODEL A - SAR ONLY",
        sep,
        f"  Features: {res_a['feat_cols']}",
        f"  Best epoch: {res_a['best_epoch']}",
        f"  Best validation accuracy: {res_a['best_acc'] * 100:.2f}%",
        "",
    ]

    for cls in CLASS_NAMES:
        r = res_a["report"][cls]
        lines.append(
            f"  {cls:<16} "
            f"Precision={r['precision']:.3f}  "
            f"Recall={r['recall']:.3f}  "
            f"F1={r['f1-score']:.3f}  "
            f"Support={r['support']:.0f}"
        )

    lines += [
        "",
        sep,
        "MODEL B - SAR + DEM",
        sep,
        f"  Features: {res_b['feat_cols']}",
        f"  Best epoch: {res_b['best_epoch']}",
        f"  Best validation accuracy: {res_b['best_acc'] * 100:.2f}%",
        "",
    ]

    for cls in CLASS_NAMES:
        r = res_b["report"][cls]
        diff = (
            r["f1-score"]
            - res_a["report"][cls]["f1-score"]
        )

        lines.append(
            f"  {cls:<16} "
            f"Precision={r['precision']:.3f}  "
            f"Recall={r['recall']:.3f}  "
            f"F1={r['f1-score']:.3f}  "
            f"Support={r['support']:.0f}  "
            f"F1 difference={diff:+.3f}"
        )

    accuracy_difference = (
        res_b["best_acc"]
        - res_a["best_acc"]
    ) * 100

    lines += [
        "",
        f"  Accuracy difference (DEM - SAR): "
        f"{accuracy_difference:+.2f} percentage points",
        "",
        sep,
        "McNEMAR'S TEST",
        sep,
        f"  Both correct              : {mcnemar['n00']}",
        f"  Only SAR correct          : {mcnemar['n01']}",
        f"  Only DEM correct          : {mcnemar['n10']}",
        f"  Both wrong                : {mcnemar['n11']}",
        f"  Chi2                      : {mcnemar['chi2']}",
        f"  p-value                   : {mcnemar['p_value']:.6f}",
        f"  Significant (p<0.05)      : "
        f"{'YES' if mcnemar['significant'] else 'NO'}",
        "",
        mcnemar["verdict"],
        "",
        sep,
        "OUTPUTS",
        sep,
        "  model_A_SAR_only/best_model.pth",
        "  model_B_SAR_DEM/best_model.pth",
        "  ablation_training_curves.png",
        "  ablation_f1_comparison.png",
        "  ablation_confusion_matrices.png",
        "  ablation_mcnemar.png",
        "  ablation_report.txt",
        sep,
    ]

    path = os.path.join(
        out_dir,
        "ablation_report.txt",
    )

    # CRITICAL FIX:
    # UTF-8 prevents Windows cp1252 UnicodeEncodeError.
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))

    for line in lines:
        print(line)

    log.info(
        "Report saved: %s",
        path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "EOS-04 SAR-only vs SAR+DEM ablation study"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Epochs per model (default: {EPOCHS})",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Load existing best_model.pth files "
            "instead of training."
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1"
        )

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("=" * 60)
    print("ABLATION STUDY: SAR-only vs SAR+DEM")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Dataset: {PATCHES_ROOT}")
    print(
        f"  Mode   : "
        f"{'LOAD SAVED MODELS' if args.skip_training else 'TRAIN'}"
    )
    print("=" * 60)

    # -------------------------------------------------------------------------
    # ONE dataset
    # -------------------------------------------------------------------------

    df = load_dataset(
        PATCHES_ROOT,
        COMPOSITE,
    )

    # -------------------------------------------------------------------------
    # ONE fixed train/validation split
    # -------------------------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    log.info(
        "Train=%d  Val=%d",
        len(train_df),
        len(val_df),
    )

    dir_a = os.path.join(
        OUTPUT_DIR,
        "model_A_SAR_only",
    )

    dir_b = os.path.join(
        OUTPUT_DIR,
        "model_B_SAR_DEM",
    )

    # -------------------------------------------------------------------------
    # MODEL A
    # -------------------------------------------------------------------------

    if args.skip_training:
        log.info(
            "Loading Model A - SAR only..."
        )

        res_a = load_saved_model(
            "Model A: SAR only",
            train_df,
            val_df,
            SAR_ONLY_FEATURES,
            dir_a,
        )

    else:
        log.info(
            "Training Model A - SAR only..."
        )

        res_a = train_model(
            "Model A: SAR only",
            train_df,
            val_df,
            SAR_ONLY_FEATURES,
            args.epochs,
            dir_a,
        )

    # -------------------------------------------------------------------------
    # MODEL B
    # -------------------------------------------------------------------------

    if args.skip_training:
        log.info(
            "Loading Model B - SAR+DEM..."
        )

        res_b = load_saved_model(
            "Model B: SAR+DEM",
            train_df,
            val_df,
            SAR_DEM_FEATURES,
            dir_b,
        )

    else:
        log.info(
            "Training Model B - SAR+DEM..."
        )

        res_b = train_model(
            "Model B: SAR+DEM",
            train_df,
            val_df,
            SAR_DEM_FEATURES,
            args.epochs,
            dir_b,
        )

    # -------------------------------------------------------------------------
    # McNemar
    # -------------------------------------------------------------------------

    log.info(
        "Running McNemar's test..."
    )

    mcnemar = mcnemar_test(
        res_a["preds"],
        res_b["preds"],
        res_a["true_labels"],
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    plot_all(
        res_a,
        res_b,
        mcnemar,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    save_report(
        res_a,
        res_b,
        mcnemar,
        len(df),
        len(train_df),
        len(val_df),
        args.epochs,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)

    print(
        f"Model A best accuracy: "
        f"{res_a['best_acc'] * 100:.2f}%"
    )

    print(
        f"Model B best accuracy: "
        f"{res_b['best_acc'] * 100:.2f}%"
    )

    print(
        f"DEM - SAR difference: "
        f"{(res_b['best_acc'] - res_a['best_acc']) * 100:+.2f} pp"
    )

    print(
        f"McNemar p-value: "
        f"{mcnemar['p_value']:.6f}"
    )

    print(
        f"Output folder: {OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
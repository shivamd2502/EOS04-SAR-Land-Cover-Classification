#!/usr/bin/env python3
# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 7 of 8 - one report for all patch sizes (folder: scripts/07_model_evaluation)
#
# WHY THIS STEP
#   Re-evaluates the 64 / 128 / 224 / 256 px checkpoints on their validation sets
#   with identical code, so the numbers in the paper and README come from one place.
#
# WHAT IT PRODUCED (results/07_model_evaluation/all_models_stats_L2B/)
#   val acc 64 px 94.17 %, 128 px 93.99 %, 224 px 93.89 %, 256 px 94.50 %;
#   weighted F1 0.94-0.95 for all. The re-run accuracy equals the checkpoint
#   accuracy for every model. The four accuracies lie within 0.6 points of each
#   other, with different validation sets, so patch size makes little difference.
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
collect_all_model_stats.py
==========================
Single script that evaluates ALL four L2B patch-size models
(64×64, 128×128, 224×224, 256×256) and produces a complete
stats report for every model in one place.

For each model it collects:
  • Total patches, train split, val split
  • Best val accuracy (from checkpoint) and epoch it was achieved
  • Per-class precision, recall, F1, support (re-run on val set)
  • Weighted average F1
  • Mean confidence (model certainty)
  • Ground area per patch
  • Approximate scene patch count

Outputs saved to model/ALL_STATS_L2B/
  all_models_stats.csv          ← flat table, one row per model per class
  all_models_summary.csv        ← one row per model (overall metrics)
  all_models_accuracy_bar.png   ← bar chart comparing val accuracy
  all_models_f1_heatmap.png     ← heatmap: model vs class F1
  all_models_confusion.png      ← 4 confusion matrices side by side
  all_models_report.txt         ← human-readable full report

Usage
-----
    python collect_all_model_stats.py

    # evaluate only specific patch sizes
    python collect_all_model_stats.py --patch-sizes 128 256

    # use a different composite label (default OPT1_sigma0)
    python collect_all_model_stats.py --composite OPT1_sigma0
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("all_stats")

# =============================================================================
# CONFIG — folder layout for all 4 patch-size models
# =============================================================================
BASE_DIR   = r"D:\ISRO_14"
COMPOSITE  = "OPT1_sigma0"
CLASSES    = ["Dense_forest", "Mix", "Urban"]
IMG_SIZE   = 224
VAL_SPLIT  = 0.2
RANDOM_SEED = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAR_COLS   = ["mean_r", "mean_g", "hh_hv_rat", "rvi_mean"]
PIXEL_SIZE_M = 18   # EOS-04 MRS pixel = 18 m

# Each entry: patch_size → (patches_root, model_dir)
MODEL_CONFIG = {
    64:  {
        "patches_root": os.path.join(BASE_DIR, "data", "02_composites_and_patches", "patches_64_L2B",  "03JUN2026"),
        "model_dir":    os.path.join(BASE_DIR, "data", "05_classification_training", "models", "EOS04_64_L2B"),
        "label":        "64×64",
        "ground_km":    f"{64*PIXEL_SIZE_M/1000:.2f}km × {64*PIXEL_SIZE_M/1000:.2f}km",
    },
    128: {
        "patches_root": os.path.join(BASE_DIR, "data", "02_composites_and_patches", "patches_128_L2B", "03JUN2026"),
        "model_dir":    os.path.join(BASE_DIR, "data", "05_classification_training", "models", "EOS04_128_L2B"),
        "label":        "128×128",
        "ground_km":    f"{128*PIXEL_SIZE_M/1000:.2f}km × {128*PIXEL_SIZE_M/1000:.2f}km",
    },
    224: {
        "patches_root": os.path.join(BASE_DIR, "data", "02_composites_and_patches", "patches_224_L2B", "03JUN2026"),
        "model_dir":    os.path.join(BASE_DIR, "data", "05_classification_training", "models", "EOS04_224_L2B"),
        "label":        "224×224",
        "ground_km":    f"{224*PIXEL_SIZE_M/1000:.2f}km × {224*PIXEL_SIZE_M/1000:.2f}km",
    },
    256: {
        "patches_root": os.path.join(BASE_DIR, "data", "02_composites_and_patches", "patches_L2B",     "03JUN2026"),
        "model_dir":    os.path.join(BASE_DIR, "data", "05_classification_training", "models", "EOS04_L2B"),
        "label":        "256×256",
        "ground_km":    f"{256*PIXEL_SIZE_M/1000:.2f}km × {256*PIXEL_SIZE_M/1000:.2f}km",
    },
}

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "07_model_evaluation", "ALL_STATS_L2B")


# =============================================================================
# 1. MODEL DEFINITION  (same for all patch sizes)
# =============================================================================

class EOS04Classifier(nn.Module):
    def __init__(self, num_classes: int, num_sar_features: int = 4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        for p in backbone.parameters():
            p.requires_grad = False
        self.cnn_features   = backbone.features
        self.pool           = backbone.avgpool
        self.cnn_dim        = backbone.classifier[1].in_features
        self._phase2_params = list(backbone.features[-2:].parameters())
        self.sar_branch = nn.Sequential(
            nn.Linear(num_sar_features, 32), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
        )
        fusion = self.cnn_dim + 32
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fusion, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, imgs, sar):
        x = self.pool(self.cnn_features(imgs)).flatten(1)
        s = self.sar_branch(sar)
        return self.classifier(torch.cat([x, s], dim=1))

    def unfreeze_for_phase2(self):
        for p in self._phase2_params:
            p.requires_grad = True


# =============================================================================
# 2. LOAD MODEL + CHECKPOINT METADATA
# =============================================================================

def load_model_bundle(model_dir: str) -> dict | None:
    """Loads model weights + all JSON metadata. Returns None if not found."""
    ckpt_path = os.path.join(model_dir, "best_model.pth")
    enc_path  = os.path.join(model_dir, "label_encoder.json")
    feat_path = os.path.join(model_dir, "feature_stats.json")

    for p in [ckpt_path, enc_path, feat_path]:
        if not os.path.isfile(p):
            log.warning("Missing file: %s — skipping this model.", p)
            return None

    with open(enc_path)  as f: label_enc  = json.load(f)
    with open(feat_path) as f: feat_stats = json.load(f)

    ckpt        = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    id_to_class = {v: k for k, v in label_enc.items()}
    num_classes = len(label_enc)

    model = EOS04Classifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model":        model,
        "id_to_class":  id_to_class,
        "class_to_id":  label_enc,
        "feat_stats":   feat_stats,
        "num_classes":  num_classes,
        "ckpt_val_acc": ckpt.get("val_acc", 0.0),
        "ckpt_epoch":   ckpt.get("epoch",   0),
        "class_names":  ckpt.get("class_names", list(label_enc.keys())),
    }


# =============================================================================
# 3. LOAD LABELS CSV AND RECONSTRUCT VAL SPLIT
# =============================================================================

def load_val_split(patches_root: str, composite: str) -> pd.DataFrame | None:
    """
    Finds the labels CSV, reconstructs the same 80/20 val split
    used during training (seed=42, stratify), returns val DataFrame.
    """
    comp_dir   = os.path.join(patches_root, composite)
    candidates = [
        os.path.join(comp_dir,     f"{composite}_labels_clean.csv"),
        os.path.join(comp_dir,     f"{composite}_labels.csv"),
        os.path.join(patches_root, f"{composite}_labels_clean.csv"),
    ]
    # also glob
    import glob as _g
    for folder in [comp_dir, patches_root]:
        candidates += _g.glob(os.path.join(folder, "*.csv"))
        candidates += _g.glob(os.path.join(folder, "*.xlsx"))

    label_path = None
    for c in dict.fromkeys(candidates):   # dedup, preserve order
        if os.path.isfile(c):
            label_path = c; break

    if label_path is None:
        log.warning("No label CSV found in %s", comp_dir)
        return None

    ext = Path(label_path).suffix.lower()
    df  = pd.read_excel(label_path) if ext == ".xlsx" \
          else pd.read_csv(label_path)
    df.columns = df.columns.str.strip()
    log.info("  Labels: %s  (%d rows)", label_path, len(df))

    # drop geometry
    if "WKT" in df.columns:
        df = df.drop(columns=["WKT"])

    # keep valid patches
    if "use_flag" in df.columns:
        df = df[df["use_flag"] == 1].copy()

    # label column
    if "label" not in df.columns:
        for alt in ["Label", "class", "Class"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "label"}); break

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(CLASSES)].copy()
    df["label_id"] = df["label"].map(
        {c: i for i, c in enumerate(CLASSES)}).astype(int)

    # resolve PNG path
    def resolve(p):
        p = str(p).strip()
        full = os.path.join(comp_dir, p)
        if os.path.isfile(full): return full
        bare = os.path.join(comp_dir, "patches", os.path.basename(p))
        if os.path.isfile(bare): return bare
        return full

    if "png_path" in df.columns:
        df["abs_png"] = df["png_path"].apply(resolve)
    elif "patch_id" in df.columns:
        df["abs_png"] = df["patch_id"].apply(
            lambda p: os.path.join(comp_dir, "patches", f"{p}.png"))

    df = df[df["abs_png"].apply(os.path.isfile)].copy()

    total = len(df)
    _, val_df = train_test_split(df, test_size=VAL_SPLIT,
                                  random_state=RANDOM_SEED,
                                  stratify=df["label_id"])
    val_df = val_df.reset_index(drop=True)

    log.info("  Total=%d  Train=%d  Val=%d",
             total, total - len(val_df), len(val_df))
    return val_df, total


# =============================================================================
# 4. INFERENCE ON VAL SET
# =============================================================================

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


def sar_features(df_row: pd.Series, feat_stats: dict) -> torch.Tensor:
    """Z-score normalised SAR features from CSV row."""
    raw = np.array([
        float(df_row.get("mean_r",    0) or 0),
        float(df_row.get("mean_g",    0) or 0),
        float(df_row.get("hh_hv_rat", 0) or 0),
        float(df_row.get("rvi_mean",  0) or 0),
    ], dtype=np.float32)
    # replace -9999 with 0
    raw = np.where(np.abs(raw) > 9000, 0.0, raw)
    for i, col in enumerate(SAR_COLS):
        raw[i] = (raw[i] - feat_stats[col]["mean"]) / feat_stats[col]["std"]
    return torch.tensor(raw, dtype=torch.float32)


@torch.no_grad()
def run_inference(bundle: dict, val_df: pd.DataFrame) -> tuple:
    """
    Runs the model on every row in val_df.
    Returns (true_labels, pred_labels, confidences) as lists.
    """
    model      = bundle["model"]
    feat_stats = bundle["feat_stats"]
    id_to_cls  = bundle["id_to_class"]

    trues, preds, confs = [], [], []
    batch_imgs, batch_sar, batch_true = [], [], []

    def flush():
        if not batch_imgs: return
        imgs = torch.stack(batch_imgs).to(DEVICE)
        sars = torch.stack(batch_sar).to(DEVICE)
        probs = torch.softmax(model(imgs, sars), dim=1).cpu().numpy()
        for i in range(len(batch_imgs)):
            pid  = int(probs[i].argmax())
            conf = float(probs[i].max())
            preds.append(id_to_cls.get(pid, "unknown"))
            confs.append(conf)
        trues.extend(batch_true)
        batch_imgs.clear(); batch_sar.clear(); batch_true.clear()

    for _, row in val_df.iterrows():
        try:
            img = Image.open(row["abs_png"]).convert("RGB")
        except Exception:
            continue
        batch_imgs.append(VAL_TF(img))
        batch_sar.append(sar_features(row, feat_stats))
        batch_true.append(row["label"])
        if len(batch_imgs) >= 32:
            flush()
    flush()

    return trues, preds, confs


# =============================================================================
# 5. COLLECT STATS FOR ONE MODEL
# =============================================================================

def collect_stats(patch_size: int, cfg: dict, composite: str) -> dict | None:
    """
    Full stats collection for one patch-size model.
    Returns a rich dict or None if model/data not found.
    """
    log.info("━━ Patch size %d×%d", patch_size, patch_size)

    bundle = load_model_bundle(cfg["model_dir"])
    if bundle is None:
        return None

    result = load_val_split(cfg["patches_root"], composite)
    if result is None:
        return None
    val_df, total_patches = result

    log.info("  Running inference on %d val patches ...", len(val_df))
    trues, preds, confs = run_inference(bundle, val_df)

    if len(trues) == 0:
        log.warning("  No predictions produced — skipping.")
        return None

    # per-class report
    rep = classification_report(
        trues, preds, labels=CLASSES,
        output_dict=True, zero_division=0)

    acc        = accuracy_score(trues, preds)
    mean_conf  = float(np.mean(confs))
    cm         = confusion_matrix(trues, preds, labels=CLASSES)

    # approx patches per full scene
    from math import floor
    scene_H, scene_W = 10301, 10201
    approx_scene_patches = (floor(scene_H / patch_size) *
                             floor(scene_W / patch_size))

    stats = {
        "patch_size":           patch_size,
        "label":                cfg["label"],
        "ground_area":          cfg["ground_km"],
        "total_patches":        total_patches,
        "train_patches":        total_patches - len(val_df),
        "val_patches":          len(val_df),
        "ckpt_val_acc":         bundle["ckpt_val_acc"],
        "ckpt_epoch":           bundle["ckpt_epoch"],
        "rerun_val_acc":        acc,
        "mean_confidence":      mean_conf,
        "approx_scene_patches": approx_scene_patches,
        "confusion_matrix":     cm,
        "per_class":            {
            cls: {
                "precision": rep[cls]["precision"],
                "recall":    rep[cls]["recall"],
                "f1":        rep[cls]["f1-score"],
                "support":   rep[cls]["support"],
            }
            for cls in CLASSES
        },
        "weighted_f1":  rep["weighted avg"]["f1-score"],
        "macro_f1":     rep["macro avg"]["f1-score"],
        "true_labels":  trues,
        "pred_labels":  preds,
    }

    # print per-model summary immediately
    log.info("  Checkpoint val_acc = %.3f (ep %d)",
             bundle["ckpt_val_acc"], bundle["ckpt_epoch"])
    log.info("  Re-run   val_acc  = %.3f  (on %d patches)",
             acc, len(val_df))
    for cls in CLASSES:
        pc = stats["per_class"][cls]
        log.info("    %-16s  P=%.3f  R=%.3f  F1=%.3f  n=%d",
                 cls, pc["precision"], pc["recall"],
                 pc["f1"], pc["support"])

    return stats


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_accuracy_bars(all_stats: list[dict], out_dir: str) -> None:
    labels    = [s["label"]        for s in all_stats]
    ckpt_acc  = [s["ckpt_val_acc"] * 100 for s in all_stats]
    rerun_acc = [s["rerun_val_acc"] * 100 for s in all_stats]
    wf1       = [s["weighted_f1"]  * 100 for s in all_stats]

    x, w = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w,   ckpt_acc,  w, label="Checkpoint val_acc",
                color="steelblue", alpha=0.85)
    b2 = ax.bar(x,        rerun_acc, w, label="Re-run val_acc",
                color="coral",     alpha=0.85)
    b3 = ax.bar(x + w,   wf1,       w, label="Weighted F1",
                color="seagreen",  alpha=0.85)

    for bars in [b1, b2, b3]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + 0.3,
                    f"{b.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_title("Val Accuracy & Weighted F1 across all patch sizes (L2B)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Metric (%)"); ax.set_ylim(60, 105)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_models_accuracy_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: all_models_accuracy_bar.png")


def plot_f1_heatmap(all_stats: list[dict], out_dir: str) -> None:
    rows  = [s["label"] for s in all_stats]
    data  = [[s["per_class"][c]["f1"] for c in CLASSES]
              for s in all_stats]
    df    = pd.DataFrame(data, index=rows, columns=CLASSES)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn",
                ax=ax, vmin=0.70, vmax=1.0,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("Per-Class F1 Score — all patch sizes (L2B)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("Patch size")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_models_f1_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: all_models_f1_heatmap.png")


def plot_confusion_matrices(all_stats: list[dict], out_dir: str) -> None:
    n   = len(all_stats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices — all patch sizes (L2B Gamma0)",
                 fontsize=13, fontweight="bold")

    for ax, s in zip(axes, all_stats):
        cm     = s["confusion_matrix"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, vmin=0, vmax=100, annot_kws={"size": 11})
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j+0.5, i+0.73, f"n={cm[i,j]}",
                        ha="center", va="center", fontsize=8, color="gray")
        ax.set_title(
            f"{s['label']}\n"
            f"acc={s['rerun_val_acc']*100:.1f}%  "
            f"wF1={s['weighted_f1']:.3f}",
            fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_models_confusion.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: all_models_confusion.png")


def plot_per_class_f1_lines(all_stats: list[dict], out_dir: str) -> None:
    """Line plot: how each class F1 changes as patch size grows."""
    sizes  = [s["patch_size"]  for s in all_stats]
    colors = {"Dense_forest": "seagreen", "Mix": "goldenrod", "Urban": "coral"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in CLASSES:
        f1s = [s["per_class"][cls]["f1"] for s in all_stats]
        ax.plot(sizes, f1s, marker="o", linewidth=2,
                color=colors[cls], label=cls)
        for x, y in zip(sizes, f1s):
            ax.annotate(f"{y:.3f}", (x, y),
                        textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title("Per-Class F1 vs Patch Size (L2B Gamma0)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Patch size (pixels)"); ax.set_ylabel("F1 score")
    ax.set_xticks(sizes)
    ax.set_xticklabels([s["label"] for s in all_stats])
    ax.set_ylim(0.65, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_models_f1_per_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: all_models_f1_per_class.png")


# =============================================================================
# 7. SAVE CSVs
# =============================================================================

def save_csvs(all_stats: list[dict], out_dir: str) -> None:
    # summary — one row per model
    summary_rows = []
    for s in all_stats:
        row = {
            "patch_size":           s["patch_size"],
            "label":                s["label"],
            "ground_area":          s["ground_area"],
            "total_patches":        s["total_patches"],
            "train_patches":        s["train_patches"],
            "val_patches":          s["val_patches"],
            "approx_scene_patches": s["approx_scene_patches"],
            "ckpt_val_acc_%":       round(s["ckpt_val_acc"] * 100, 2),
            "ckpt_epoch":           s["ckpt_epoch"],
            "rerun_val_acc_%":      round(s["rerun_val_acc"] * 100, 2),
            "weighted_f1":          round(s["weighted_f1"], 4),
            "macro_f1":             round(s["macro_f1"], 4),
            "mean_confidence":      round(s["mean_confidence"], 4),
        }
        for cls in CLASSES:
            pc = s["per_class"][cls]
            row[f"{cls}_precision"] = round(pc["precision"], 3)
            row[f"{cls}_recall"]    = round(pc["recall"],    3)
            row[f"{cls}_f1"]        = round(pc["f1"],        3)
            row[f"{cls}_support"]   = pc["support"]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "all_models_summary.csv"), index=False)
    log.info("Saved: all_models_summary.csv")

    # flat — one row per model per class
    flat_rows = []
    for s in all_stats:
        for cls in CLASSES:
            pc = s["per_class"][cls]
            flat_rows.append({
                "patch_size":      s["patch_size"],
                "label":           s["label"],
                "class":           cls,
                "precision":       round(pc["precision"], 3),
                "recall":          round(pc["recall"],    3),
                "f1":              round(pc["f1"],        3),
                "support":         pc["support"],
                "val_acc_%":       round(s["rerun_val_acc"] * 100, 2),
                "weighted_f1":     round(s["weighted_f1"], 4),
                "mean_confidence": round(s["mean_confidence"], 4),
            })
    pd.DataFrame(flat_rows).to_csv(
        os.path.join(out_dir, "all_models_stats.csv"), index=False)
    log.info("Saved: all_models_stats.csv")


# =============================================================================
# 8. HUMAN-READABLE TEXT REPORT
# =============================================================================

def save_text_report(all_stats: list[dict], out_dir: str) -> None:
    sep   = "=" * 72
    lines = [sep,
             "EOS-04 L2B PATCH CLASSIFIER — COMPLETE STATS REPORT",
             f"Composite: OPT1_sigma0   |   Classes: {CLASSES}",
             sep, ""]

    for s in all_stats:
        lines += [
            f"PATCH SIZE : {s['label']}",
            f"  Ground area per patch  : {s['ground_area']}",
            f"  Approx patches / scene : {s['approx_scene_patches']:,}",
            f"  Total labelled patches : {s['total_patches']:,}",
            f"  Training patches (80%) : {s['train_patches']:,}",
            f"  Validation patches(20%): {s['val_patches']:,}",
            f"  Best checkpoint epoch  : {s['ckpt_epoch']}",
            f"  Checkpoint val_acc     : {s['ckpt_val_acc']*100:.2f}%",
            f"  Re-run val_acc         : {s['rerun_val_acc']*100:.2f}%",
            f"  Weighted F1            : {s['weighted_f1']:.4f}",
            f"  Macro F1               : {s['macro_f1']:.4f}",
            f"  Mean model confidence  : {s['mean_confidence']:.4f}",
            "",
            f"  {'Class':<16} {'Precision':>10} {'Recall':>8} "
            f"{'F1':>8} {'Support':>9}",
            "  " + "─" * 54,
        ]
        for cls in CLASSES:
            pc = s["per_class"][cls]
            lines.append(
                f"  {cls:<16} {pc['precision']:>10.3f} "
                f"{pc['recall']:>8.3f} {pc['f1']:>8.3f} "
                f"{pc['support']:>9}")
        lines += ["", sep, ""]

    # best model per metric
    best_acc  = max(all_stats, key=lambda x: x["rerun_val_acc"])
    best_wf1  = max(all_stats, key=lambda x: x["weighted_f1"])
    best_conf = max(all_stats, key=lambda x: x["mean_confidence"])
    lines += [
        "SUMMARY — BEST MODEL PER METRIC",
        sep,
        f"  Highest val accuracy  : {best_acc['label']}"
        f"  ({best_acc['rerun_val_acc']*100:.2f}%)",
        f"  Highest weighted F1   : {best_wf1['label']}"
        f"  ({best_wf1['weighted_f1']:.4f})",
        f"  Highest confidence    : {best_conf['label']}"
        f"  ({best_conf['mean_confidence']:.4f})",
        "",
        "  NOTE on patch size vs feature quality:",
        "  • 64×64  → upscale 3.5× to 224: significant feature loss",
        "  • 128×128→ upscale 1.75×: some blur",
        "  • 224×224→ NO resize: CNN sees native SAR pixels (best texture)",
        "  • 256×256→ shrink to 224: slight loss (acceptable)",
        sep,
    ]

    report_path = os.path.join(out_dir, "all_models_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for line in lines:
        print(line)
    log.info("Saved: %s", report_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-sizes", nargs="+", type=int,
                        default=[64, 128, 224, 256],
                        help="Which patch sizes to evaluate")
    parser.add_argument("--composite", default=COMPOSITE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("EOS-04 L2B — COLLECTING STATS FOR ALL PATCH SIZE MODELS")
    print(f"  Patch sizes : {args.patch_sizes}")
    print(f"  Composite   : {args.composite}")
    print(f"  Device      : {DEVICE}")
    print(f"  Output      : {args.output_dir}")
    print("=" * 60)

    all_stats = []
    for ps in sorted(args.patch_sizes):
        if ps not in MODEL_CONFIG:
            log.warning("Patch size %d not in MODEL_CONFIG — skipping.", ps)
            continue
        cfg   = MODEL_CONFIG[ps]
        stats = collect_stats(ps, cfg, args.composite)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("\nNo models found. Check that model folders exist and "
              "contain best_model.pth, label_encoder.json, feature_stats.json")
        return

    print(f"\nSuccessfully evaluated {len(all_stats)} model(s). "
          f"Generating outputs ...")

    save_csvs(all_stats, args.output_dir)
    save_text_report(all_stats, args.output_dir)

    if len(all_stats) >= 2:
        plot_accuracy_bars(all_stats,         args.output_dir)
        plot_f1_heatmap(all_stats,            args.output_dir)
        plot_per_class_f1_lines(all_stats,    args.output_dir)
    plot_confusion_matrices(all_stats,        args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("  all_models_report.txt        ← full human-readable report")
    print("  all_models_summary.csv       ← one row per model")
    print("  all_models_stats.csv         ← one row per model per class")
    print("  all_models_accuracy_bar.png  ← accuracy comparison bar chart")
    print("  all_models_f1_heatmap.png    ← F1 heatmap (model × class)")
    print("  all_models_f1_per_class.png  ← F1 trend line per class")
    print("  all_models_confusion.png     ← confusion matrices side by side")


if __name__ == "__main__":
    main()
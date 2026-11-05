#!/usr/bin/env python3
# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 2 of 8 - label CSV clean-up (folder: scripts/02_composites_and_patches)
#
# WHY THIS STEP
#   QGIS exports the labelled attribute table with a MULTIPOLYGON geometry column
#   and without proper headers. The training scripts expect a clean
#   <composite>_labels_clean.csv, so this small utility renames the columns and
#   drops the geometry. Edit INPUT_PATH / OUTPUT_PATH for each patch size.
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================
"""
fix_exported_csv.py
===================
Fixes the QGIS-exported CSV by:
  1. Adding correct column headers
  2. Removing the MULTIPOLYGON geometry column (column A)
  3. Saving a clean CSV ready for training

Usage:
    python fix_exported_csv.py
    (edit the INPUT_PATH and OUTPUT_PATH below to match your file)
"""

import pandas as pd
import os

# ── EDIT THESE TWO PATHS ──────────────────────────────────────────────────────
INPUT_PATH  = r"D:\ISRO_14\data\02_composites_and_patches\patches_64_L2B\03JUN2026\OPT1_sigma0\OPT1_sigma0_labels_clean.csv"
OUTPUT_PATH = r"D:\ISRO_14\data\02_composites_and_patches\patches_64_L2B\03JUN2026\OPT1_sigma0\OPT1_sigma0_labels_clean.csv"
# ─────────────────────────────────────────────────────────────────────────────

# These are the correct column names in the order QGIS exports them
# (matches what is visible in your screenshot)
COLUMN_NAMES = [
    "geometry",        # col A  — MULTIPOLYGON (will be dropped)
    "patch_id",        # col B
    "composite",       # col C
    "r_ch",            # col D
    "g_ch",            # col E
    "b_ch",            # col F
    "row_idx",         # col G
    "col_idx",         # col H
    "px_row",          # col I
    "px_col",          # col J
    "p_size",          # col K
    "stride",          # col L
    "valid_pct",       # col M
    "mean_r",          # col N
    "mean_g",          # col O
    "mean_b",          # col P
    "std_r",           # col Q
    "hh_hv_rat",       # col R
    "rvi_mean",        # col S
    "png_path",        # col T
    "use_flag",        # col U
    "label",           # col V
]

print(f"Reading: {INPUT_PATH}")

# Read raw CSV with no header (QGIS did not write one)
df = pd.read_csv(INPUT_PATH, header=None, low_memory=False)

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Assign column names
# If QGIS exported fewer columns than expected, trim the names list to match
actual_cols = df.shape[1]
names_to_use = COLUMN_NAMES[:actual_cols]

# If more columns than names, add generic names for the extras
if actual_cols > len(COLUMN_NAMES):
    for i in range(len(COLUMN_NAMES), actual_cols):
        names_to_use.append(f"extra_{i}")

df.columns = names_to_use

print(f"Columns assigned: {list(df.columns)}")

# Drop the geometry column (MULTIPOLYGON data — not needed for training)
if "geometry" in df.columns:
    df = df.drop(columns=["geometry"])
    print("Dropped geometry column")

# Drop r_ch, g_ch, b_ch (channel meaning strings — verbose, not needed for ML)
cols_to_drop = [c for c in ["r_ch", "g_ch", "b_ch"] if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped verbose columns: {cols_to_drop}")

# Show label distribution
if "label" in df.columns:
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

# Show sample
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# Save clean CSV
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nClean CSV saved to: {OUTPUT_PATH}")
print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
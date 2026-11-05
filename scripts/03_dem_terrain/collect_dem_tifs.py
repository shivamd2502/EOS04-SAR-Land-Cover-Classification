#!/usr/bin/env python3
# ============================================================================
# PROJECT CONTEXT (EOS-04 SAR land-cover pipeline)
# Step 3 of 8 - terrain data (folder: scripts/03_dem_terrain)
#
# WHY THIS STEP
#   In the Himalayan scene, steep slopes produce foreshortening that can look like
#   urban double-bounce in SAR. To test whether terrain information helps, DEM
#   elevation and slope statistics are added per patch. The Copernicus DSM tiles
#   were downloaded as ZIP archives; this script copies only the
#   Copernicus_DSM *.tif files out of each ZIP without extracting everything.
#
# WHAT IT PRODUCED
#   DEM/ (50 ZIPs) -> DEM_TIFS/ (50 GeoTIFF tiles), used by composite_to_patches.py
#   (--dem-dir) to compute dem_mean, slp_mean, slp_std, slp_max, slp_p15.
#   Tile inventory: results/03_dem_terrain/dem_tile_inventory.csv.
#   Overview of all steps: README.md and docs/PIPELINE.md
# ============================================================================

"""
collect_dem_tifs.py

Collect only the actual Copernicus DEM GeoTIFF files from
all ZIP archives inside D:\\ISRO_14\\DEM.

The ZIP files are NOT fully extracted.

The script searches inside each ZIP archive and copies only
files that:

    1. are .tif or .tiff
    2. are located inside a DEM directory
    3. contain "Copernicus_DSM" in the filename

Output:

    D:\\ISRO_14\\DEM_TIFS
"""

from pathlib import Path
from zipfile import ZipFile, BadZipFile
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(r"D:\ISRO_14\DEM")

OUTPUT_DIR = Path(r"D:\ISRO_14\DEM_TIFS")


# ============================================================
# FIND ZIP FILES
# ============================================================

def find_zip_files():

    zip_files = sorted(SOURCE_DIR.glob("*.zip"))

    return zip_files


# ============================================================
# CHECK WHETHER FILE IS THE ACTUAL COPERNICUS DEM
# ============================================================

def is_copernicus_dem(member_name: str):

    path = Path(member_name)

    filename = path.name.lower()

    suffix = path.suffix.lower()

    # Must be GeoTIFF
    if suffix not in [".tif", ".tiff"]:
        return False

    # Must be a Copernicus DSM file
    if "copernicus_dsm" not in filename:
        return False

    # Must be inside a DEM folder
    parts = [p.lower() for p in path.parts]

    if "dem" not in parts:
        return False

    return True


# ============================================================
# COLLECT DEM FILES
# ============================================================

def collect_dem_files():

    print("=" * 90)
    print("COPERNICUS DEM TIFF COLLECTOR")
    print("=" * 90)

    print(f"\nSource directory:")
    print(SOURCE_DIR)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)

    if not SOURCE_DIR.exists():

        print("\nERROR:")
        print(f"Source directory does not exist: {SOURCE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = find_zip_files()

    print(f"\nZIP files found: {len(zip_files)}")

    if not zip_files:

        print("\nERROR: No ZIP files found.")
        print("\nExpected structure:")
        print(r"D:\ISRO_14\DEM\1.zip")
        print(r"D:\ISRO_14\DEM\2.zip")
        print(r"...")

        return

    collected = 0
    skipped = 0
    failed = 0

    collected_files = []

    # ========================================================
    # PROCESS EACH ZIP
    # ========================================================

    for index, zip_path in enumerate(zip_files, start=1):

        print("\n" + "-" * 90)

        print(f"[{index}/{len(zip_files)}]")
        print(f"ZIP: {zip_path.name}")

        try:

            with ZipFile(zip_path, "r") as z:

                members = z.namelist()

                # Find actual Copernicus DEM TIFFs
                dem_members = [
                    member
                    for member in members
                    if is_copernicus_dem(member)
                ]

                if not dem_members:

                    print("  No Copernicus DEM TIFF found.")

                    continue

                for member in dem_members:

                    member_path = Path(member)

                    original_name = member_path.name

                    output_file = OUTPUT_DIR / original_name

                    print(f"  DEM found:")
                    print(f"      {member}")

                    # ------------------------------------------------
                    # Handle duplicate filenames
                    # ------------------------------------------------

                    if output_file.exists():

                        # Check whether it is exactly the same size
                        info = z.getinfo(member)

                        existing_size = output_file.stat().st_size

                        if existing_size == info.file_size:

                            print(
                                f"      Already exists → skipping"
                            )

                            skipped += 1
                            continue

                        # Different file with same name
                        stem = output_file.stem
                        suffix = output_file.suffix

                        counter = 2

                        while True:

                            new_name = (
                                f"{stem}_{counter}{suffix}"
                            )

                            output_file = (
                                OUTPUT_DIR / new_name
                            )

                            if not output_file.exists():
                                break

                            counter += 1

                    # ------------------------------------------------
                    # Extract ONLY this TIFF
                    # ------------------------------------------------

                    print(
                        f"      Copying → {output_file.name}"
                    )

                    with z.open(member) as source:

                        with open(output_file, "wb") as destination:

                            shutil.copyfileobj(
                                source,
                                destination,
                                length=1024 * 1024
                            )

                    collected += 1
                    collected_files.append(output_file)

        except BadZipFile:

            print("  ERROR: Invalid/corrupted ZIP file.")

            failed += 1

        except Exception as e:

            print("  ERROR:")

            print(f"      {e}")

            failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 90)
    print("COLLECTION COMPLETE")
    print("=" * 90)

    print(f"\nZIP archives found : {len(zip_files)}")
    print(f"DEM TIFFs copied   : {collected}")
    print(f"Already present    : {skipped}")
    print(f"Failed archives    : {failed}")

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)

    # ========================================================
    # LIST OUTPUT FILES
    # ========================================================

    output_files = sorted(
        OUTPUT_DIR.glob("*.tif")
    ) + sorted(
        OUTPUT_DIR.glob("*.tiff")
    )

    print(f"\nTotal TIFF files in output folder: {len(output_files)}")

    print("\nCollected DEM files:")

    for i, file in enumerate(output_files, start=1):

        size_mb = file.stat().st_size / (1024 * 1024)

        print(
            f"{i:03d}. "
            f"{file.name} "
            f"({size_mb:.1f} MB)"
        )

    print("\n" + "=" * 90)

    if output_files:

        print("SUCCESS")
        print("\nThe actual Copernicus DEM TIFF files are now available at:")

        print(OUTPUT_DIR)

    else:

        print("WARNING")
        print("\nNo DEM TIFF files were collected.")

        print(
            "\nCheck whether the ZIP structure contains "
            "Copernicus_DSM files inside a DEM folder."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    collect_dem_files()
# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download the RESOLVE Ecoregions2017 shapefile and rasterize it into two
global nominal COGs: fine-grained ecoregions (ECO_ID, 847 classes) and
coarse biomes (BIOME_NUM, 15 classes).

Source: https://storage.googleapis.com/teow2016/Ecoregions2017.zip (~150MB,
EPSG:4326, CC-BY 4.0). Every polygon record carries both a unique ECO_ID
(0-846) and a BIOME_NUM (1-14) grouping ecoregions into biomes. One record,
ECO_ID=0 ("Rock and Ice" — ice sheets/bare rock, e.g. Antarctica interior,
Greenland), shares BIOME_NUM=11 with real Tundra ecoregions even though
RESOLVE gives it its own distinct display color. It's remapped to its own
biome class id (15) here so ice-sheet area doesn't get counted as Tundra.

Steps:
  1. Download the zip via aria2c (skips if already present)
  2. Extract the shapefile components (.shp/.shx/.dbf/.prj)
  3. gdal_rasterize -a ECO_ID              → layers/ecoregions.tif (UInt16)
     gdal_rasterize -sql <biome remap>     → layers/biome.tif      (Byte)
  4. Rebuild config/gis/legends/{ecoregions,biome}_legend.json from the same
     shapefile attributes (ECO_NAME/BIOME_NAME/COLOR/COLOR_BIO), so the
     legend always matches what got burned into the rasters.

Usage (inside the gdal container):
    uv run python -m scripts.gis.download_ecoregions [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

ZIP_URL      = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
ZIP_FILENAME = "Ecoregions2017.zip"
SHP_STEM     = "Ecoregions2017"
SHP_MEMBERS  = (".shp", ".shx", ".dbf", ".prj")

RAW_ZIP_DIR = Path("data/gis/ecoregions_raw_zip")
RAW_DIR     = Path("data/gis/ecoregions_raw")
LAYERS_DIR  = Path("data/gis/layers")
LEGENDS_DIR = Path("config/gis/legends")

# 30 arcsec, matching CHELSA/kg2's global grid.
RESOLUTION_DEG = 1.0 / 120.0
ROCK_AND_ICE_ECO_ID  = 0
ROCK_AND_ICE_BIOME_ID = 15

ECOREGIONS_OUT = LAYERS_DIR / "ecoregions.tif"
BIOME_OUT      = LAYERS_DIR / "biome.tif"

CITATION = (
    "Dinerstein et al. (2017). An Ecoregion-Based Approach to Protecting Half "
    "the Terrestrial Realm. BioScience, 67(6), 534-545."
)


# ── download / extract ──────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _download_zip(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--split=8",
            "--max-connection-per-server=8",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            ZIP_URL,
        ],
        check=True,
    )


def _extract_shapefile(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = out_dir / f"{SHP_STEM}.shp"
    if all((out_dir / f"{SHP_STEM}{ext}").exists() for ext in SHP_MEMBERS):
        print(f"  Shapefile already extracted: {shp_path}")
        return shp_path

    with zipfile.ZipFile(zip_path) as zf:
        for ext in SHP_MEMBERS:
            member = f"{SHP_STEM}{ext}"
            dest = out_dir / member
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(zf.read(member))
            tmp.replace(dest)
    return shp_path


# ── rasterize ────────────────────────────────────────────────────────────────
#
# gdal_rasterize writes tiled/BigTIFF output out of row order (it seeks back
# to patch the TIFF directory as tiles complete). That fails silently and
# instantly on this host's data/gis/layers/ mount (a WSL2 bind mount) even
# though the directory is otherwise perfectly writable — confirmed by the
# identical command succeeding both untiled to that same directory and fully
# tiled to /tmp. So: rasterize into a local tempdir (proven reliable), then
# land the finished file in data/gis/layers/ with a plain sequential copy,
# which is a completely different (append-only) I/O pattern that the mount
# handles fine.

def _rasterize(
    shp_path: Path,
    out_path: Path,
    *,
    field: str,
    sql: str | None = None,
    dtype: str,
    nodata: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        scratch = Path(tmp_dir) / out_path.name
        cmd = ["gdal_rasterize"]
        if sql is not None:
            # OGR's default SQL dialect has no CASE WHEN — needs SQLite's.
            cmd += ["-sql", sql, "-dialect", "SQLite"]
        cmd += [
            "-a", field,
            "-ot", dtype,
            "-a_nodata", nodata,
            "-init", nodata,
            "-te", "-180", "-90", "180", "90",
            "-tr", str(RESOLUTION_DEG), str(RESOLUTION_DEG),
            "-a_srs", "EPSG:4326",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=IF_SAFER",
            str(shp_path), str(scratch),
        ]
        _run(cmd)

        dest_tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        dest_tmp.unlink(missing_ok=True)
        with scratch.open("rb") as src, dest_tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    out_path.unlink(missing_ok=True)
    dest_tmp.replace(out_path)


def _rasterize_ecoregions(shp_path: Path, out_path: Path) -> None:
    _rasterize(shp_path, out_path, field="ECO_ID", dtype="UInt16", nodata="65535")


def _rasterize_biome(shp_path: Path, out_path: Path) -> None:
    # SELECT * (not just the computed column) — an OGR SQL query that only
    # names the computed column drops the geometry entirely, silently
    # producing an all-nodata raster with no error from gdal_rasterize.
    remap_sql = (
        f"SELECT *, (CASE WHEN ECO_ID={ROCK_AND_ICE_ECO_ID} "
        f"THEN {ROCK_AND_ICE_BIOME_ID} ELSE CAST(BIOME_NUM AS INTEGER) END) "
        f"AS BIOME_ID FROM {SHP_STEM}"
    )
    _rasterize(shp_path, out_path, field="BIOME_ID", sql=remap_sql, dtype="Byte", nodata="255")


# ── legends ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    name = name.replace("&", "and")
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_license(path: Path) -> None:
    license_path = path.with_suffix(path.suffix + ".license")
    license_path.write_text(
        "SPDX-FileCopyrightText: RESOLVE / Dinerstein et al. (2017)\n"
        "\n"
        "SPDX-License-Identifier: CC-BY-4.0\n"
    )


def _build_legends(shp_path: Path) -> None:
    import geopandas as gpd

    gdf = gpd.read_file(shp_path, ignore_geometry=True)
    records = gdf.to_dict("records")

    eco_classes = []
    for r in sorted(records, key=lambda r: int(r["ECO_ID"])):
        eco_id = int(r["ECO_ID"])
        if eco_id == ROCK_AND_ICE_ECO_ID:
            group, group_label = "rock_and_ice", "Rock and Ice"
        else:
            group, group_label = _slug(r["BIOME_NAME"]), r["BIOME_NAME"]
        eco_classes.append({
            "id": eco_id,
            "name": r["ECO_NAME"],
            "group": group,
            "group_label": group_label,
            "traits": {"color": r["COLOR"]},
        })

    biome_rows: dict[int, tuple[str, str]] = {}
    rock_ice_color = None
    for r in records:
        eco_id = int(r["ECO_ID"])
        if eco_id == ROCK_AND_ICE_ECO_ID:
            rock_ice_color = r["COLOR_BIO"]
            continue
        biome_rows[int(r["BIOME_NUM"])] = (r["BIOME_NAME"], r["COLOR_BIO"])

    # Biome is already the coarse tier (no finer sub-structure in the source
    # data to group by), so each biome is its own singleton group — same
    # group id/label a matching ecoregion would carry, keeping the two
    # legends' grouping consistent.
    biome_classes = []
    for bnum, (name, color) in sorted(biome_rows.items()):
        biome_classes.append({
            "id": bnum,
            "name": name,
            "group": _slug(name),
            "group_label": name,
            "traits": {"color": color},
        })
    biome_classes.append({
        "id": ROCK_AND_ICE_BIOME_ID,
        "name": "Rock and Ice",
        "group": "rock_and_ice",
        "group_label": "Rock and Ice",
        "traits": {"color": rock_ice_color},
    })

    eco_path = LEGENDS_DIR / "ecoregions_legend.json"
    biome_path = LEGENDS_DIR / "biome_legend.json"
    _write_json(eco_path, {"layer_id": "ecoregions", "source": CITATION, "classes": eco_classes})
    _write_json(biome_path, {"layer_id": "biome", "source": CITATION, "classes": biome_classes})
    _write_license(eco_path)
    _write_license(biome_path)
    print(f"  Wrote {eco_path} ({len(eco_classes)} classes)")
    print(f"  Wrote {biome_path} ({len(biome_classes)} classes)")


# ── main ─────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    target_ids = ("ecoregions", "biome")
    if VARS_TO_DOWNLOAD is not None and not any(v in VARS_TO_DOWNLOAD for v in target_ids):
        print("[download_ecoregions] skipped (ecoregions/biome not in VARS_TO_DOWNLOAD)")
        return

    rasters_done = ECOREGIONS_OUT.exists() and BIOME_OUT.exists()
    if rasters_done and not force:
        print(f"[skip] rasters already exist: {ECOREGIONS_OUT}, {BIOME_OUT} (--force to rebuild)")
        return

    zip_dest = RAW_ZIP_DIR / ZIP_FILENAME
    if not zip_dest.exists():
        print(f"Downloading {ZIP_URL} ...")
        _download_zip(zip_dest)
    else:
        print(f"ZIP already downloaded: {zip_dest}")

    print("Extracting shapefile...")
    shp_path = _extract_shapefile(zip_dest, RAW_DIR)

    print(f"Rasterizing ecoregions (ECO_ID) → {ECOREGIONS_OUT}")
    _rasterize_ecoregions(shp_path, ECOREGIONS_OUT)

    print(f"Rasterizing biome (BIOME_NUM, Rock & Ice → {ROCK_AND_ICE_BIOME_ID}) → {BIOME_OUT}")
    _rasterize_biome(shp_path, BIOME_OUT)

    print("Building legends from shapefile attributes...")
    _build_legends(shp_path)

    print("Cleaning up raw shapefile/zip (no longer needed once rasters+legends exist)...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(RAW_ZIP_DIR, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and rasterize RESOLVE Ecoregions2017")
    parser.add_argument("--force", action="store_true", help="Re-rasterize even if outputs already exist")
    args = parser.parse_args()
    main(force=args.force)

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download the RESOLVE Ecoregions2017 shapefile and prepare it as a native
vector source — fine-grained ecoregions (ECO_ID, 847 classes) and coarse
biomes (BIOME_NUM, 15 classes) both live in one file, rendered directly from
polygon geometry at request time (see util/tiles.py's vector dispatch)
instead of a pre-baked global raster.

A prior version of this script rasterized the shapefile globally at 1
arcsec (~30m) via gdal_rasterize — a multi-hour job producing an ~840B
pixel grid — and still showed visible mismatches against real coastlines
near the poles and around small islands. Neither problem is really about
resolution: gdal_rasterize burns onto a *fixed-degree* WGS84 grid, whose
real ground size shrinks by cos(latitude) toward the poles (cells become
tall slivers there, which a fixed lon/lat sampling grid handles poorly
regardless of how fine it is), and it doesn't split geometries that cross
the antimeridian (~180°E/W), which silently corrupts them once the raster
is later warped into a projection like Web Mercator that also has a seam
there. Serving straight from the original polygon boundaries, at whatever
resolution each tile actually needs, sidesteps both: there's no
intermediate discretized grid to introduce sampling artifacts, and the
polygons are pre-split at the seam once here instead of relying on a
raster to represent something that crosses it.

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
  3. ogr2ogr -wrapdateline           → split antimeridian-crossing polygons
  4. Compute BIOME_ID (rock & ice remap), reproject to EPSG:3857, write
     layers/ecoregions_vector.parquet (ECO_ID + BIOME_ID + geometry, one
     file backs both layers since the polygons are shared)
  5. Rebuild config/gis/legends/{ecoregions,biome}_legend.json from the same
     shapefile attributes (ECO_NAME/BIOME_NAME/COLOR/COLOR_BIO), so the
     legend always matches what's in the vector file.

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

# 1 arcsec (~30m at the equator, in line with how other "30m" products like
# SRTM/NASADEM label themselves — real width shrinks with cos(latitude) away
# from the equator). Up from 30 arcsec (~927m); no longer matches CHELSA/kg2's
# coarser grid, but nothing depends on ecoregions/biome sharing that grid —
# every layer gets independently warped to the requested tile at serve time.
# ~840B pixels globally (vs. ~933M before) — expect a much longer rasterize
# and a much larger (if still well-compressed) output file.
ROCK_AND_ICE_ECO_ID  = 0
ROCK_AND_ICE_BIOME_ID = 15

VECTOR_OUT = LAYERS_DIR / "ecoregions_vector.parquet"

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


# ── vectorize ────────────────────────────────────────────────────────────────
#
# No rasterization at all: the tile renderer (util/tiles.py) burns these
# polygons directly onto each requested tile's own grid, in its own
# projection, on demand. This step just needs to hand it clean geometry —
# split at the antimeridian and reprojected once — plus the two attribute
# columns (ECO_ID, BIOME_ID) it needs to color by.

def _wrapdateline_shapefile(shp_path: Path, tmp_dir: Path) -> Path:
    """Split polygons crossing the antimeridian (~180°E/W) so none of them
    wrap the long way around the globe once reprojected. Source ecoregions
    genuinely do cross it (e.g. Bering Sea / Chukotka), and a projection
    with a seam there (Web Mercator included) renders an unsplit crossing
    polygon as a shape spanning nearly the whole map width instead of a
    normal-looking region hugging the seam on both sides.
    """
    out_path = tmp_dir / f"{SHP_STEM}_wrapped.shp"
    _run(["ogr2ogr", "-wrapdateline", "-datelineoffset", "10", str(out_path), str(shp_path)])
    return out_path


def _build_vector_source(shp_path: Path, out_path: Path) -> None:
    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmp_dir:
        wrapped_path = _wrapdateline_shapefile(shp_path, Path(tmp_dir))
        gdf = gpd.read_file(wrapped_path)

    gdf["geometry"] = gdf["geometry"].make_valid()
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty]

    rock_and_ice = gdf["ECO_ID"].astype(int) == ROCK_AND_ICE_ECO_ID
    gdf["ECO_ID"] = gdf["ECO_ID"].astype("int32")
    gdf["BIOME_ID"] = gdf["BIOME_NUM"].astype("int32")
    gdf.loc[rock_and_ice, "BIOME_ID"] = ROCK_AND_ICE_BIOME_ID

    gdf = gdf[["ECO_ID", "BIOME_ID", "geometry"]].to_crs(epsg=3857)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_out.unlink(missing_ok=True)
    gdf.to_parquet(tmp_out, compression="zstd")
    out_path.unlink(missing_ok=True)
    tmp_out.replace(out_path)


# ── legends ──────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    name = name.replace("&", "and")
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


# Biomes that map to a single concept group (no fan-out needed) but still
# need a group/attributes override instead of the default singleton — either
# because they share a group with a sibling biome (e.g. all three conifer
# forest variants), or because a member needs a "solo_group_label" (a name —
# "Taiga" — to use only when it's the sole representative of its group in a
# given description; the moment a sibling conifer forest joins it, the
# generic "Conifer Forests" label takes over instead — see
# util/descriptions.py's _build_nominal_lines "solo_group_label" handling).
# name -> {group, group_label, attributes, solo_group_label?}.
_BIOME_GROUP_OVERRIDES: dict[str, dict] = {
    "Tropical & Subtropical Moist Broadleaf Forests": {
        "group": "broadleaf_forest",
        "group_label": "Broadleaf Forests",
        "attributes": ["moist", "tropical & subtropical"],
    },
    "Tropical & Subtropical Dry Broadleaf Forests": {
        "group": "broadleaf_forest",
        "group_label": "Broadleaf Forests",
        "attributes": ["dry", "tropical & subtropical"],
    },
    "Tropical & Subtropical Coniferous Forests": {
        "group": "conifer_forest",
        "group_label": "Conifer Forests",
        "attributes": ["tropical & subtropical"],
    },
    "Temperate Broadleaf & Mixed Forests": {
        "group": "broadleaf_and_mixed_forest",
        "group_label": "Broadleaf & Mixed Forests",
        "attributes": ["temperate"],
    },
    "Temperate Conifer Forests": {
        "group": "conifer_forest",
        "group_label": "Conifer Forests",
        "attributes": ["temperate"],
    },
    "Boreal Forests/Taiga": {
        "group": "conifer_forest",
        "group_label": "Conifer Forests",
        "solo_group_label": "Boreal Forests/Taiga",
        "attributes": ["boreal"],
    },
}

# broadleaf_forest varies along two independent dimensions at once (zone and
# moisture) rather than the single dimension every other biome group varies
# along, so it needs the axis system (each axis independently collapses to
# nothing if its members disagree) instead of the plain attribute list — see
# util/descriptions.py's _build_nominal_lines attribute_axes handling. The
# "zone" axis has only one member (all broadleaf_forest entries are
# tropical & subtropical) but still needs to be declared so it's exposed as
# a comparable modifier — otherwise it would only ever appear glued to
# "moist"/"dry" and could never match e.g. grassland's own "tropical &
# subtropical" modifier when both land in the same sentence.
_BIOME_ATTRIBUTE_AXES: dict[str, list[dict]] = {
    "broadleaf_forest": [
        {"name": "moisture", "values": ["moist", "dry"]},
        {"name": "zone", "values": ["tropical & subtropical"]},
    ],
}

# A harder case: several biomes bundle multiple conceptually *distinct*
# land-cover types into one compound name (e.g. "Grasslands, Savannas &
# Shrublands") because RESOLVE's source pixels can't tell them apart. A
# straight group-by-name-match (like the override above) can't split those
# back into separate concepts — describing "how often is this species in
# grassland" vs. "in shrubland" independently requires the SAME biome
# fraction to count fully toward more than one group at once. name -> list
# of {group, group_label, attributes} the class's full fraction fans out to
# — see util/descriptions.py's _build_nominal_lines "memberships" handling.
_BIOME_GROUP_FANOUT: dict[str, list[dict]] = {
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": [
        {"group": "grassland", "group_label": "Grasslands", "attributes": ["tropical & subtropical"]},
        {"group": "savanna", "group_label": "Savannas", "attributes": ["tropical & subtropical"]},
        {"group": "shrubland", "group_label": "Shrublands", "attributes": ["tropical & subtropical"]},
    ],
    "Temperate Grasslands, Savannas & Shrublands": [
        {"group": "grassland", "group_label": "Grasslands", "attributes": ["temperate"]},
        {"group": "savanna", "group_label": "Savannas", "attributes": ["temperate"]},
        {"group": "shrubland", "group_label": "Shrublands", "attributes": ["temperate"]},
    ],
    "Flooded Grasslands & Savannas": [
        {"group": "grassland", "group_label": "Grasslands", "attributes": ["flooded"]},
        {"group": "savanna", "group_label": "Savannas", "attributes": ["flooded"]},
    ],
    "Montane Grasslands & Shrublands": [
        {"group": "grassland", "group_label": "Grasslands", "attributes": ["montane"]},
        {"group": "shrubland", "group_label": "Shrublands", "attributes": ["montane"]},
    ],
    "Deserts & Xeric Shrublands": [
        {"group": "shrubland", "group_label": "Shrublands", "attributes": ["desert"]},
    ],
    "Mediterranean Forests, Woodlands & Scrub": [
        {"group": "forest", "group_label": "Forests", "attributes": ["mediterranean"]},
        {"group": "shrubland", "group_label": "Shrublands", "attributes": ["mediterranean"]},
    ],
}


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
    # data to group by), so most biomes are their own singleton group — same
    # group id/label a matching ecoregion would carry, keeping the two
    # legends' grouping consistent — except the handful overridden above
    # (same-noun variant merge, or fanned out into several concept groups).
    biome_classes = []
    for bnum, (name, color) in sorted(biome_rows.items()):
        override = _BIOME_GROUP_OVERRIDES.get(name)
        fanout = _BIOME_GROUP_FANOUT.get(name)
        if fanout:
            biome_classes.append({
                "id": bnum,
                "name": name,
                "memberships": fanout,
                "traits": {"color": color},
            })
        elif override:
            biome_classes.append({
                "id": bnum,
                "name": name,
                **override,
                "traits": {"color": color},
            })
        else:
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
    _write_json(biome_path, {
        "layer_id": "biome",
        "source": CITATION,
        "classes": biome_classes,
        "attribute_axes": _BIOME_ATTRIBUTE_AXES,
    })
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

    if VECTOR_OUT.exists() and not force:
        print(f"[skip] vector source already exists: {VECTOR_OUT} (--force to rebuild)")
        return

    zip_dest = RAW_ZIP_DIR / ZIP_FILENAME
    if not zip_dest.exists():
        print(f"Downloading {ZIP_URL} ...")
        _download_zip(zip_dest)
    else:
        print(f"ZIP already downloaded: {zip_dest}")

    print("Extracting shapefile...")
    shp_path = _extract_shapefile(zip_dest, RAW_DIR)

    print(f"Building vector source (ECO_ID + BIOME_ID, dateline-split, EPSG:3857) → {VECTOR_OUT}")
    _build_vector_source(shp_path, VECTOR_OUT)

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

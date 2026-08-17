# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build the self-hosted OpenMapTiles-schema vector basemap that replaces the
Stadia Maps light/dark + Toner background/lines/labels tiles.

Steps:
  1. Download the OSM planet PBF (aria2c, resumable) to data/tmp/ — never
     pushed to prod, same treatment as the DEM/landcover raw downloads.
  2. Download (and cache) the planetiler-openmaptiles jar.
  3. Run it to produce a single planet-wide PMTiles archive at
     data/gis/tiles/basemap.pmtiles. One data file serves every theme —
     "standard" vs "variable" (background+labels only) is a style-layer
     concern, not a data concern.
  4. Copy the two hand-maintained base styles from config/gis/tile_styles/
     (Positron/Dark Matter forks, committed to the repo since they're code)
     into data/gis/tiles/styles/, and derive the "variable" (clutter-hidden)
     counterpart of each so tileserver-gl's raster-rendering path — needed
     for the Leaflet + fallback views, which can't do runtime layer
     toggling like the MapLibre globe view can — has all four as
     ready-to-render style.json files.
  5. Download/cache the OpenMapTiles font glyphs used by the styles.

The finished artifacts (basemap.pmtiles, styles/, fonts/) land under
data/gis/tiles/, so they ride the *existing* rebuild.py `push` stage
(rclone sync of data/ → gambaby) — no separate deploy path needed, and
since it's part of the normal data/ tree this box's own WHEREWILD_DATA_ROOT
already points at, it's immediately usable for local dev/testing without
pushing anywhere. Not part of rebuild.py's STAGES: OSM data doesn't need
daily freshness, so this runs on its own cron (e.g. monthly), independently
of the GBIF-crawl-triggered taxonomy rebuild.

The raw planet PBF / planetiler jar / font zip are cached under
data/gis/tiles/_src/ rather than data/tmp/ — _push_stage() in rebuild.py
deletes data/tmp/ on *every* push (it's meant as pure scratch space), which
would force a re-download of the ~94GB planet file after each push if the
cache lived there. data/gis/tiles/_src/ is instead added to _push_stage()'s
rclone --exclude list (alongside elevation.tif/temporal) so it never leaves
this box — it survives both the wipe (wipe_data_dir() skips "gis" entirely)
and the push, staying warm for the next rebuild.

Requires a JRE — see Dockerfile (openjdk-21-jre-headless).

Usage (inside the gdal container):
    # full planet (hours, ~94GB download, needs a real box)
    uv run python scripts/gis/build_basemap_tiles.py [--force] [--skip-download]

    # quick end-to-end validation against a tiny named extract instead of
    # the planet — uses planetiler's own --area/--download (no aria2c planet
    # fetch), writes to data/gis/tiles/_test/basemap-<area>.pmtiles so it
    # can't collide with a real basemap.pmtiles. Minutes, not hours.
    uv run python scripts/gis/build_basemap_tiles.py --area monaco
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Pin deliberately, like the GDAL/uv base images in Dockerfile — bump by
# hand when picking up a new planetiler release, don't float. Verify against
# https://github.com/onthegomap/planetiler/releases before bumping; the jar
# is a single build with OpenMapTilesMain as its default entrypoint (no
# separate "-openmaptiles" artifact despite what you'd guess from the repo
# name — confirmed by running it: "Building OpenMapTilesProfile profile").
PLANETILER_JAR_URL = (
    "https://github.com/onthegomap/planetiler/releases/download/v0.10.2/planetiler.jar"
)
PLANETILER_JAR_SHA256 = os.environ.get(
    "WW_PLANETILER_JAR_SHA256",
    "f310bd0413e2e4512b27f4046d418664e8e1d3bf31603c2a70e23de06c167e4d",
)

# Geofabrik mirrors the official planet file with resumable HTTP, which
# plays nicer with aria2c's multi-connection split than the OSMF server.
PLANET_PBF_URL = "https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf"

OPENMAPTILES_FONTS_URL = (
    "https://github.com/openmaptiles/fonts/releases/download/v2.0/v2.0.zip"
)
# Keep this list tight — each entry is a whole font-stack directory of PBF
# glyph ranges. Add more only if a style actually references them.
FONT_STACKS = ("Noto Sans Regular", "Noto Sans Bold", "Noto Sans Italic")

TILES_DIR     = Path("data/gis/tiles")

# Raw/intermediate downloads — kept out of data/tmp/ (wiped on every push,
# see module docstring) and out of the rclone push itself (see
# rebuild.py's _push_stage exclude list, which needs "gis/tiles/_src/**"
# added alongside the existing elevation.tif/temporal excludes).
SRC_DIR       = TILES_DIR / "_src"
PLANET_PBF    = SRC_DIR / "planet-latest.osm.pbf"
JAR_PATH      = SRC_DIR / "planetiler.jar"
FONTS_ZIP     = SRC_DIR / "openmaptiles-fonts.zip"

PMTILES_OUT   = TILES_DIR / "basemap.pmtiles"
TEST_DIR      = TILES_DIR / "_test"
STYLES_OUT_DIR = TILES_DIR / "styles"
FONTS_OUT_DIR  = TILES_DIR / "fonts"

STYLES_SRC_DIR = Path("config/gis/tile_styles")
BASE_STYLES = ("standard-light", "standard-dark")

# OpenMapTiles source layers to hide for the "variable" (background+labels)
# theme — the vector-tile equivalent of Stadia's Toner "background" tile:
# shapes and place names stay, road network/buildings/POI clutter goes, so
# the heatmap overlay has something to sit on top of without competing for
# attention. Matches the SATELLITE/VARIABLE mode comment in
# speciesOccurrenceMapHelpers.ts.
VARIABLE_MODE_HIDDEN_SOURCE_LAYERS = frozenset({
    "transportation",
    "transportation_name",
    "building",
    "poi",
    "housenumber",
    "aeroway",
})


def _run(cmd: list[str], **kwargs) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def _aria2_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--split=16",
            "--max-connection-per-server=16",
            "--min-split-size=1M",
            "--file-allocation=none",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            url,
        ],
        check=True,
    )


def _verify_sha256(path: Path, expected: str) -> None:
    if not expected:
        print(f"  WARNING: no checksum configured for {path.name} — skipping verification")
        return
    _run(["sha256sum", "-c", "-"], input=f"{expected}  {path}\n")


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def _download_planet(force: bool) -> None:
    if PLANET_PBF.exists() and not force:
        print(f"  Planet PBF already downloaded: {PLANET_PBF}")
        return
    print(f"  Downloading planet PBF from {PLANET_PBF_URL} (this is ~94GB, expect 4-7 hours to build after download)...")
    _aria2_download(PLANET_PBF_URL, PLANET_PBF)


def _download_planetiler_jar() -> None:
    if JAR_PATH.exists():
        print(f"  planetiler-openmaptiles jar already cached: {JAR_PATH}")
        return
    print(f"  Downloading {PLANETILER_JAR_URL} ...")
    _aria2_download(PLANETILER_JAR_URL, JAR_PATH)
    _verify_sha256(JAR_PATH, PLANETILER_JAR_SHA256)


def _build_pmtiles(force: bool, area: str) -> Path:
    """Runs planetiler. area="planet" uses the aria2c-downloaded planet PBF
    (--osm-path); anything else is a named extract planetiler downloads
    itself (--area/--download) — see PLANET.md's list of valid area names
    (Geofabrik region/country names, or "planet").
    """
    is_test = area != "planet"
    out_path = (TEST_DIR / f"basemap-{area.replace(' ', '_')}.pmtiles") if is_test else PMTILES_OUT

    if out_path.exists() and not force:
        print(f"  PMTiles already built: {out_path} (use --force to rebuild)")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the .pmtiles extension on the tmp path — planetiler infers its
    # output archive format from the filename extension, so a ".tmp" suffix
    # (e.g. from Path.with_suffix) makes it fail with "Unsupported format".
    tmp_out = out_path.with_name(out_path.stem + ".tmp.pmtiles")
    tmp_out.unlink(missing_ok=True)

    # RAM budget per planetiler's own guidance (PLANET.md): if system RAM is
    # >=1.5x the PBF size, keep the node-location cache fully in-memory with
    # a large heap; otherwise (our case — 62GB RAM vs a ~94GB planet PBF)
    # use a much smaller heap and let the (already-default) mmap-backed node
    # cache spill to disk/OS page cache instead. A heap sized for the
    # in-memory regime on a box that doesn't have the RAM for it just
    # starves the OS/mmap page cache and slows things down.
    # Override WW_PLANETILER_HEAP_GB if this box's specs differ.
    cmd = ["java", "-jar", str(JAR_PATH)]
    if is_test:
        cmd += ["--area", area, "--download"]
    else:
        heap_gb = os.environ.get("WW_PLANETILER_HEAP_GB", "20")
        cmd = ["java", f"-Xmx{heap_gb}g", "-jar", str(JAR_PATH), "--osm-path", str(PLANET_PBF)]
    cmd += ["--output", str(tmp_out), "--force"]

    print(f"  Running planetiler ({' '.join(cmd[1:3])}...) → {tmp_out}")
    _run(cmd)

    tmp_out.replace(out_path)
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")
    return out_path


def _derive_variable_style(standard: dict) -> dict:
    """Same style, clutter layers hidden — see VARIABLE_MODE_HIDDEN_SOURCE_LAYERS."""
    variable = json.loads(json.dumps(standard))  # deep copy
    for layer in variable.get("layers", []):
        if layer.get("source-layer") in VARIABLE_MODE_HIDDEN_SOURCE_LAYERS:
            layer.setdefault("layout", {})["visibility"] = "none"
    return variable


def _build_styles() -> None:
    STYLES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in BASE_STYLES:
        src = STYLES_SRC_DIR / f"{name}.json"
        style = json.loads(src.read_text())

        (STYLES_OUT_DIR / f"{name}.json").write_text(json.dumps(style, indent=2))

        variable_name = name.replace("standard-", "variable-")
        variable_style = _derive_variable_style(style)
        (STYLES_OUT_DIR / f"{variable_name}.json").write_text(json.dumps(variable_style, indent=2))
        print(f"  Wrote {name}.json + {variable_name}.json")


def _build_fonts(force: bool) -> None:
    if FONTS_OUT_DIR.exists() and not force:
        print(f"  Fonts already extracted: {FONTS_OUT_DIR}")
        return

    if not FONTS_ZIP.exists():
        print(f"  Downloading {OPENMAPTILES_FONTS_URL} ...")
        _aria2_download(OPENMAPTILES_FONTS_URL, FONTS_ZIP)

    import zipfile
    FONTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FONTS_ZIP) as zf:
        members = [
            m for m in zf.namelist()
            if any(m.startswith(f"{stack}/") for stack in FONT_STACKS)
        ]
        print(f"  Extracting {len(members)} glyph files for {len(FONT_STACKS)} font stacks...")
        for member in members:
            zf.extract(member, FONTS_OUT_DIR)


def main(force: bool = False, skip_download: bool = False, area: str = "planet") -> None:
    is_test = area != "planet"

    if not is_test:
        print("--- Planet PBF ---")
        if not skip_download:
            _download_planet(force)
        elif not PLANET_PBF.exists():
            raise RuntimeError(f"--skip-download passed but {PLANET_PBF} doesn't exist")

    print("--- planetiler jar ---")
    _download_planetiler_jar()

    print(f"--- PMTiles build ({area}) ---")
    out_path = _build_pmtiles(force, area)

    print("--- Styles ---")
    _build_styles()

    print("--- Fonts ---")
    _build_fonts(force)

    print(f"\nDone. {out_path} + styles + fonts under {TILES_DIR}")
    if is_test:
        print("Test build — not the real basemap.pmtiles, won't be picked up by anything automatically.")
    else:
        print("Next: rebuild.py --stage push (or the next scheduled --push run) syncs this to gambaby.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build self-hosted OpenMapTiles-schema basemap tiles")
    parser.add_argument("--force", action="store_true", help="Rebuild PMTiles/fonts even if output already exists")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the planet PBF download; fail if it isn't already present",
    )
    parser.add_argument(
        "--area", default="planet",
        help=(
            "Planetiler area name: 'planet' (default, full build via the aria2c-downloaded "
            "planet PBF) or any Geofabrik region/country name (e.g. 'monaco', 'rhode island') "
            "for a fast end-to-end test build via planetiler's own --download."
        ),
    )
    args = parser.parse_args()
    main(force=args.force, skip_download=args.skip_download, area=args.area)

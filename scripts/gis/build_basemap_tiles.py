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

Progress is tracked in data/basemap_state.json and pushed to
WHEREWILD_STATUS_PUSH_URL (same mechanism as scripts/build_temporal.py) so
it shows up on the /status page — but only for real (--area planet) runs,
not --area test builds.

The finished artifacts (basemap.pmtiles, styles/, fonts/) land under
data/gis/tiles/, which is part of the normal data/ tree this box's own
WHEREWILD_DATA_ROOT already points at, so it's immediately usable for
local dev/testing without pushing anywhere. Pushing to prod is this
pipeline's *own* scoped rclone sync (_push_basemap_files, step 6 below) —
deliberately NOT rebuild.py's `push` stage, even though that stage's own
rclone sync of the whole data/ tree would technically pick these files up
too (it now excludes gis/tiles/** entirely, see its _push_stage). This
pipeline runs on its own cron (e.g. quarterly), entirely independent of the
GBIF-crawl-triggered taxonomy rebuild — going through rebuild.py's stage
runner instead would fire the taxonomy pipeline's own alert/notification
channel for a rebuild that never happened, and couple two unrelated
pipelines' failure modes together (a taxonomy-side push failure — e.g. a
stray root-owned file on the remote — has no reason to ever block or get
blamed on a basemap tile build, and vice versa).

The raw planet PBF / planetiler jar / font zip are cached under
data/gis/tiles/_src/ rather than data/tmp/ — wipe_data_dir() in rebuild.py
deletes data/tmp/ on *every* push (it's meant as pure scratch space), which
would force a re-download of the ~94GB planet file after each taxonomy
rebuild if the cache lived there. data/gis/tiles/_src/ (and _test/) are
instead excluded from _push_basemap_files()'s own rclone sync below, so
they never leave this box — staying warm for the next build.

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
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

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

# Raw/intermediate downloads — kept out of data/tmp/ (wiped on every
# taxonomy-pipeline push, see module docstring) and out of this module's own
# push (_push_basemap_files excludes "_src/**"/"_test/**" below).
SRC_DIR       = TILES_DIR / "_src"
PLANET_PBF    = SRC_DIR / "planet-latest.osm.pbf"
JAR_PATH      = SRC_DIR / "planetiler.jar"
FONTS_ZIP     = SRC_DIR / "openmaptiles-fonts.zip"

PMTILES_OUT   = TILES_DIR / "basemap.pmtiles"
TEST_DIR      = TILES_DIR / "_test"
STYLES_OUT_DIR = TILES_DIR / "styles"
FONTS_OUT_DIR  = TILES_DIR / "fonts"

STYLES_SRC_DIR = Path("config/gis/tile_styles")
# All independent hand-authored files now — variable-light.json is NOT
# derived from standard-*.json (it used to be, hiding clutter layers, but
# Toner-style "variable" is a genuinely different monochrome design: light
# land/dark water vs. standard's full-color CARTO-derived look, confirmed
# against openmaptiles/maptiler-toner-gl-style's real values — see
# variable-light.json's metadata note). No variable-dark: 'variable' mode is
# always the light/dark-water Toner look regardless of app theme — it reads
# best under the heatmap overlay in both, and matches the old Stadia
# background's behavior (never split by mode either). See
# getBackgroundTileUrl in speciesOccurrenceMapHelpers.ts.
ALL_STYLES = (
    "standard-light",
    "standard-dark",
    "standard-voyager",
    "variable-light",
    "labels",
)

# State tracking, same shape/mechanism as scripts/build_temporal.py's
# _TEMPORAL_STATE_PATH/_push_temporal_state — a local file for `/status` to
# read directly (GamBase) plus a push to WHEREWILD_STATUS_PUSH_URL for when
# this runs on gambaby and the API needs it pushed rather than read locally.
# Only written for real (--area planet) runs — a --area monaco test build
# isn't something the status page should show.
BASEMAP_STATE_PATH = Path("data/basemap_state.json")
STATUS_PUSH_URL = os.environ.get("WHEREWILD_STATUS_PUSH_URL", "")

def _run_streaming(cmd: list[str]) -> None:
    """Like _run, but for long enough commands that swallowing stdout/stderr
    until exit (as _run does) would leave a multi-hour job's own progress
    logging invisible the whole time it's running. Lets the child inherit our
    stdout/stderr directly — since run_basemap_tiles.sh redirects the whole
    script's output to logs/basemap_tiles.log, that's where it lands, live.
    """
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


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


def _push_basemap_state(state: dict) -> None:
    if not STATUS_PUSH_URL:
        return
    try:
        url = STATUS_PUSH_URL.rstrip("/") + "/internal/basemap-state"
        httpx.post(url, json=state, timeout=5)
    except Exception as exc:
        print(f"basemap status push: failed: {exc}")


def _push_basemap_files() -> None:
    """Syncs just data/gis/tiles/ (basemap.pmtiles, styles/, fonts/) to
    prod. Deliberately its own rclone invocation rather than rebuild.py's
    `push` stage — see this module's docstring for why the two pipelines'
    pushes need to stay fully separate. Reuses the same WW_SYNC_DEST/
    WW_PUSH_MIN_FREE_GB/WW_RCLONE_TRANSFERS env vars as rebuild.py's
    _push_stage for consistency, but is otherwise independent: no data/tmp
    cleanup (irrelevant here) and no /internal/reload ping (nothing on the
    API side caches basemap tiles in memory — they're proxied straight
    through to tileserver-gl per request, and the separate
    /internal/basemap-state push already tells the API about the new
    build_date for cache-busting).
    """
    dest = os.environ.get("WW_SYNC_DEST")
    if not dest:
        raise RuntimeError("WW_SYNC_DEST env var must be set (e.g. gambaby:/path/to/data)")
    tiles_dest = dest.rstrip("/") + "/gis/tiles"

    min_free_gb = float(os.environ.get("WW_PUSH_MIN_FREE_GB", "50"))
    check = subprocess.run(
        ["rclone", "about", "--json", dest.split(":")[0] + ":"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        info = json.loads(check.stdout)
        free_gb = info.get("free", 0) / 1024 ** 3
        if free_gb < min_free_gb:
            raise RuntimeError(
                f"Destination has only {free_gb:.1f} GB free (minimum {min_free_gb} GB). "
                "Free up space before pushing."
            )
        print(f"  Destination free space: {free_gb:.1f} GB")

    transfers = os.environ.get("WW_RCLONE_TRANSFERS", "16")
    flags = [
        "--exclude", "_src/**",
        "--exclude", "_test/**",
        "--transfers", transfers,
        "--stats-one-line", "--stats", "1m",
    ]
    print(f"  rclone sync {TILES_DIR} → {tiles_dest}")
    r = subprocess.run(["rclone", "sync", str(TILES_DIR), tiles_dest, *flags])
    if r.returncode != 0:
        raise RuntimeError(f"rclone sync failed with exit code {r.returncode}")


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

    print(f"  Running planetiler ({' '.join(cmd[1:3])}...) → {tmp_out}", flush=True)
    _run_streaming(cmd)

    tmp_out.replace(out_path)
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")
    return out_path


# Required credit per CARTO's CC-BY 4.0 design license (config/gis/tile_styles/
# standard-*.json's metadata) plus OpenMapTiles'/OSM's own attribution
# requirements — see LICENSE.md at github.com/CartoDB/basemap-styles. Placed
# on the vector source itself so MapLibre's AttributionControl picks it up
# automatically in the globe view with no frontend wiring; the Leaflet path's
# static MAP_TILE_ATTRIBUTION_SELF_HOSTED constant carries the same credits
# separately since Leaflet doesn't read style-source metadata.
_ATTRIBUTION_HTML = (
    '&copy; <a href="https://carto.com/" target="_blank">CARTO</a>, '
    '&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> '
    '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
)


def _build_styles(build_date: str) -> None:
    STYLES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    attribution = f"{_ATTRIBUTION_HTML} &middot; Basemap built {build_date}"
    for name in ALL_STYLES:
        src = STYLES_SRC_DIR / f"{name}.json"
        style = json.loads(src.read_text())
        style["sources"]["openmaptiles"]["attribution"] = attribution
        (STYLES_OUT_DIR / f"{name}.json").write_text(json.dumps(style, indent=2))
        print(f"  Wrote {name}.json")


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
    started_at = datetime.now(UTC)
    t_start = time.perf_counter()

    def _write_state(status: str, *, stage: str | None = None, error: str | None = None) -> None:
        if is_test:
            return
        state: dict = {
            "status": status,
            "pid": os.getpid() if status == "running" else None,
            "started_at": started_at.isoformat(),
            "stage": stage,
            "area": area,
        }
        if status in ("completed", "failed"):
            state["completed_at"] = datetime.now(UTC).isoformat()
            state["duration_s"] = round(time.perf_counter() - t_start)
        if error is not None:
            state["error"] = error
        BASEMAP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BASEMAP_STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(BASEMAP_STATE_PATH)
        _push_basemap_state(state)

    try:
        _write_state("running", stage="planet_download")
        if not is_test:
            print("--- Planet PBF ---")
            if not skip_download:
                _download_planet(force)
            elif not PLANET_PBF.exists():
                raise RuntimeError(f"--skip-download passed but {PLANET_PBF} doesn't exist")

        _write_state("running", stage="planetiler_jar")
        print("--- planetiler jar ---")
        _download_planetiler_jar()

        _write_state("running", stage="pmtiles_build")
        print(f"--- PMTiles build ({area}) ---")
        out_path = _build_pmtiles(force, area)

        _write_state("running", stage="styles")
        print("--- Styles ---")
        _build_styles(started_at.date().isoformat())

        _write_state("running", stage="fonts")
        print("--- Fonts ---")
        _build_fonts(force)

        if not is_test:
            _write_state("running", stage="push")
            print("--- Push ---")
            _push_basemap_files()
    except Exception as exc:
        _write_state("failed", error=str(exc))
        raise

    _write_state("completed")
    print(f"\nDone. {out_path} + styles + fonts under {TILES_DIR}")
    if is_test:
        print("Test build — not the real basemap.pmtiles, won't be picked up by anything automatically.")
    else:
        print(f"Pushed to prod ({os.environ.get('WW_SYNC_DEST', '<WW_SYNC_DEST>')}/gis/tiles).")


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

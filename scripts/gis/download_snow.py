# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download NSIDC-0791 (MODIS/Terra Global Annual 0.01Deg CMG Snow Cover
Climatology, Johnston et al. 2024) and build six global COGs:

  scdur.tif   SCD climatology  (avg days/yr w/ snow, ratio)
  scsl.tif    CSS climatology  (max consecutive days, ratio)
  sfsl.tif    FSS climatology  (first-to-last span, ratio)
  sper.tif    SP  climatology  (% of snow year, ratio)
  ssper.tif   SSP climatology  (% of full season, ratio)
  sreg.tif    SSC, water year 2023 (ordinal, 5 classes)

Each of the six source NetCDF files (one per parameter, ~200MB-1.8GB) holds
BOTH a per-water-year variable (2001-2023) and a "_climatology" variable
(the 23-year average) — see the NSIDC-0791 user guide. Five of our six
layers use the climatology variable directly, matching every other static
"normal" layer in the catalog (bio1-19, scd/swe/swb, ...).

sreg is the exception: its climatology variant is documented as a
per-pixel *average of the 0-4 class codes across years* ("Value between
0-4", not "Integer between 0-4" like the per-year variable) — i.e. a blended
score, not a valid class label. So sreg instead takes the single most
recent water year (2023) from the per-year SSC variable, which is a genuine
5-class integer field (0=No Snow ... 4=Perennial Snow, 255=Fill/Water).

Data access requires a free NASA Earthdata login (the files sit behind
urs.earthdata.nasa.gov, confirmed via an anonymous request redirecting to
the login page) — set EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env,
mirroring the GBIF_USER/GBIF_PASSWORD convention already used for GBIF.

Usage (inside the gdal container):
    uv run python scripts/gis/download_snow.py [--force]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import rasterio

_raw_vars = os.environ.get("VARS_TO_DOWNLOAD", "")
VARS_TO_DOWNLOAD: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD", "")

BASE_URL = (
    "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/"
    "MODIS-Related/NSIDC-0791/1/2000/03/01/"
    "NSIDC-0791_{param}_0.01Deg_WY2001-2023_V01.0.nc"
)

RAW_DIR    = Path("data/gis/nsidc_0791_raw")
LAYERS_DIR = Path("data/gis/layers")

# (catalog layer id, NSIDC file short name, use climatology variable?)
_TARGETS: list[tuple[str, str, bool]] = [
    ("scdur", "SCD", True),
    ("scsl",  "CSS", True),
    ("sfsl",  "FSS", True),
    ("sper",  "SP",  True),
    ("ssper", "SSP", True),
    ("sreg",  "SSC", False),
]

# Multiplier applied ON TOP OF whatever native packing scale the source
# NetCDF variable carries (auto-detected, see _detect_scale_offset) — used
# only for the deliberate unit conversion from NSIDC's native 0-1 fraction
# to the 0-100 percent this catalog displays SP/SSP as. Everything else is
# already in its native display unit (days), so no extra multiplier.
_DISPLAY_MULTIPLIER: dict[str, float] = {
    "sper":  100.0,
    "ssper": 100.0,
}


CATALOG_PATH = Path("config/gis/catalog.json")


def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _nsidc_layers(catalog: dict) -> dict[str, dict]:
    return {
        layer["id"]: layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("source") == "nsidc_0791"
    }


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def _netrc_path() -> Path:
    """Write a project-local .netrc for NASA Earthdata auth (not ~/.netrc,
    so this never clobbers a developer's own netrc entries for other hosts).
    curl follows the data host -> urs.earthdata.nasa.gov login -> data host
    redirect chain automatically when given netrc credentials + a cookie
    jar to carry the resulting session across hosts — the standard,
    NASA-documented access pattern for Earthdata-gated downloads.
    """
    if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
        raise OSError(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set (free account: "
            "https://urs.earthdata.nasa.gov/users/new) to download NSIDC-0791."
        )
    path = RAW_DIR / ".netrc"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"machine urs.earthdata.nasa.gov login {EARTHDATA_USERNAME} password {EARTHDATA_PASSWORD}\n"
    )
    path.chmod(0o600)
    return path


def _cleanup_credentials() -> None:
    """Remove the local .netrc/.cookies files written for NASA Earthdata
    auth once they're no longer needed for this run, rather than leaving
    them on disk indefinitely. Minimizes how long the plaintext credential
    file actually exists — chmod 0o600 already restricts who can read it,
    and it's gitignored (under data/), but curl's --netrc-file mechanism
    has no alternative that avoids a plaintext file existing at all while
    a download is in flight.
    """
    (RAW_DIR / ".netrc").unlink(missing_ok=True)
    (RAW_DIR / ".cookies").unlink(missing_ok=True)


def _download(param: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return
    netrc = _netrc_path()
    cookie_jar = RAW_DIR / ".cookies"
    url = BASE_URL.format(param=param)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  Fetching {url} ...", flush=True)
    _run([
        "curl", "-fL",
        "--netrc-file", str(netrc),
        "-c", str(cookie_jar), "-b", str(cookie_jar),
        "-o", str(tmp),
        url,
    ])
    tmp.replace(dest)


def _find_variable_pair(nc_path: Path) -> tuple[str, str]:
    """Return (base_subdataset, climatology_subdataset) for a NSIDC-0791
    file, auto-detected from whatever "X" / "X_climatology" pair actually
    exists rather than assumed from the documented short parameter names
    (SCD, SP, ...) — those turned out not to match the files' real NetCDF
    variable names (confirmed empirically: the SCD file exposes
    "snow_cover_duration" / "snow_cover_duration_climatology", not
    "SCD" / "SCD_climatology"). Every file has exactly one such pair, per
    the NSIDC-0791 user guide section 1.2.2 ("each file contains a unique
    parameter variable and corresponding average variable (climatology)").
    """
    with rasterio.open(nc_path) as ds:
        subdatasets = ds.subdatasets
    names = [s.rsplit(":", 1)[-1].strip('"') for s in subdatasets]
    climatology_names = [n for n in names if n.endswith("_climatology")]
    if len(climatology_names) != 1:
        raise ValueError(
            f"Expected exactly one *_climatology subdataset in {nc_path}, "
            f"found {climatology_names} (all variables: {names})"
        )
    clim_name = climatology_names[0]
    # Don't derive the base name from clim_name by string-stripping "_climatology"
    # — confirmed unreliable: the SSC file pairs "snow_classes" (plural) with
    # "snow_class_climatology" (singular), so it isn't even a clean prefix.
    # Each file has exactly one non-climatology data variable, so just take
    # whichever one that is.
    non_climatology = [n for n in names if n != clim_name]
    if len(non_climatology) != 1:
        raise ValueError(
            f"Expected exactly one non-climatology data variable alongside "
            f"{clim_name!r} in {nc_path}, found {non_climatology} (all variables: {names})"
        )
    base_name = non_climatology[0]
    return subdatasets[names.index(base_name)], subdatasets[names.index(clim_name)]


def _detect_nodata(path: str) -> float | None:
    with rasterio.open(path) as ds:
        return ds.nodata


def _is_packed(subdataset: str) -> bool:
    """True if GDAL parsed a non-identity CF scale_factor/add_offset off
    this NetCDF variable's own metadata — i.e. it needs unpacking.

    Caught a real bug via this: scdur/scsl/sfsl's raw values turned out to
    be packed at scale=0.01 (confirmed empirically — computed render_max
    landed at ~36522-36523, exactly 100x a plausible ~365-day ceiling), but
    the COGs were being built from the raw packed integers with no
    unpacking applied anywhere, so every pixel was 100x too large.
    """
    with rasterio.open(subdataset) as ds:
        scale = ds.scales[0] if ds.scales else 1.0
        offset = ds.offsets[0] if ds.offsets else 0.0
    return scale not in (1.0, None) or offset not in (0.0, None)


def _build_cog(subdataset: str, out_path: Path, *, band: int | None = None, nominal: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nodata = _detect_nodata(subdataset)
    packed = _is_packed(subdataset)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / out_path.name
        cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=YES",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-a_srs", "EPSG:4326",
        ]
        if band is not None:
            cmd += ["-b", str(band)]
        if packed:
            # Bake the file's own scale_factor/add_offset into the actual
            # pixel values (GDAL applies whatever it parsed off the source
            # metadata — no manual scale/offset math on our end) so the COG
            # holds true, sanely-scaled values and nothing downstream ever
            # needs to know packing happened.
            cmd += ["-unscale", "-ot", "Float32"]
        if nodata is not None:
            cmd += ["-a_nodata", str(nodata)]
        elif nominal:
            # SSC's documented fill/water value — set explicitly in case the
            # NetCDF's own _FillValue attribute isn't picked up by GDAL for
            # this particular band selection.
            cmd += ["-a_nodata", "255"]
        cmd += [subdataset, str(tmp_out)]
        _run(cmd)

        dest_tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        dest_tmp.unlink(missing_ok=True)
        # Cross-filesystem-safe landing (tmp_dir and data/gis/layers/ may be
        # different filesystems, e.g. a WSL2 bind mount) — see the matching
        # comment in download_glwd.py._build_cog for why plain os.rename
        # isn't safe here.
        with tmp_out.open("rb") as fsrc, dest_tmp.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=64 * 1024 * 1024)
    out_path.unlink(missing_ok=True)
    dest_tmp.replace(out_path)


def _most_recent_band(subdataset: str) -> int:
    """Return the last band index (1-based) of a per-year (time, lat, lon)
    subdataset — GDAL exposes the netCDF time dimension as sequential bands
    in chronological order, so the last band is water year 2023.
    """
    with rasterio.open(subdataset) as ds:
        return ds.count


def main(force: bool = False) -> None:
    target_ids = [t[0] for t in _TARGETS]
    if VARS_TO_DOWNLOAD is not None and not any(v in VARS_TO_DOWNLOAD for v in target_ids):
        print(f"[download_snow] skipped (none of {target_ids} in VARS_TO_DOWNLOAD)")
        return

    catalog = _load_catalog()
    layers = _nsidc_layers(catalog)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for layer_id, param, use_climatology in _TARGETS:
            if VARS_TO_DOWNLOAD is not None and layer_id not in VARS_TO_DOWNLOAD:
                continue
            layer = layers[layer_id]
            out_path = LAYERS_DIR / layer["filename"]
            if out_path.exists() and not force:
                print(f"[skip] {layer_id} — already at {out_path} (--force to rebuild)")
                continue

            raw_path = RAW_DIR / f"NSIDC-0791_{param}_0.01Deg_WY2001-2023_V01.0.nc"
            print(f"[download_snow] {layer_id} ({param})")
            _download(param, raw_path)

            base_sub, clim_sub = _find_variable_pair(raw_path)

            if use_climatology:
                print(f"  Building COG (climatology) -> {out_path}")
                _build_cog(clim_sub, out_path, nominal=False)
            else:
                band = _most_recent_band(base_sub)
                print(f"  Building COG (most recent year, band {band}) -> {out_path}")
                _build_cog(base_sub, out_path, band=band, nominal=True)
    finally:
        _cleanup_credentials()

    # render_min/render_max aren't computed here — scripts/gis/build_overviews.py
    # computes those for every continuous layer as the true min/max of
    # valid pixel values, and runs right after this stage in the rebuild
    # pipeline, so anything set here would be immediately overwritten.
    # sreg's fixed 0-4 ordinal bounds are set directly in catalog.json and
    # untouched by either script.
    print("[download_snow] done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download NSIDC-0791 snow cover climatology")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs already exist")
    args = parser.parse_args()
    main(force=args.force)

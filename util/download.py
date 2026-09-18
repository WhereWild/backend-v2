# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Species-data download: bundles a taxon's occurrence data and precomputed
stats into the same ZIP shape as the custom-upload archive (see
util.upload.build_archive), so it can be mounted offline via the exact same
local data-source path the custom-upload flow already uses.

Works for any rank. Leaf ranks (subspecies-equivalents) use their own
occurrence.parquet; higher ranks aggregate every descendant leaf's
observations via util.stats.collect_taxon_df — the same rollup already used
by the location-filtered stats endpoint, deduped by catalogNumber. Stats
(numerical/nominal/ordinal/circular/density/density_grid) are read straight
out of GLOBAL_STATS_DIR filtered by taxon_key, since the tree pipeline
already computes and stores per-taxon aggregates there for every rank —
non-leaf included — before deleting the original per-node files at
consolidation (see scripts/process_tree.py::run_consolidation).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from util.stats import (
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    DENSITY_GRID_FILE,
    GLOBAL_STATS_DIR,
    NOMINAL_STATS_FILE,
    NUMERICAL_STATS_FILE,
    ORDINAL_STATS_FILE,
    collect_taxon_df,
)
from util.storage import ParquetStorage
from util.taxa import TaxonRecord
from util.upload import (
    _add_metadata_to_archive,
    _add_ternary_classification_overlay,
    _build_layer_meta,
    _build_temporal_var_meta,
    _package_archive,
    build_description_profile_for_df,
)

# Duplicated (not imported) from main.py's identically-named helper to avoid
# a circular import (main.py -> util.download; the reverse would cycle).
# Keep in sync if the taxon dict's image_* field naming ever changes.


def _license_label(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"/publicdomain/zero/([^/]+)/", url)
    if m:
        return f"CC0 {m.group(1)}"
    m = re.search(r"/licenses/([^/]+)/([^/]+)/", url)
    if m:
        parts = m.group(1).split("-")
        return "CC " + "-".join(p.upper() for p in parts) + " " + m.group(2)
    return url


def _image_fields(taxon: TaxonRecord) -> dict:
    """Return unified image_* fields, preferring iNat over GBIF backup."""
    prefix = "inat_preferred" if taxon.get("inat_preferred_image") else "gbif_backup"
    license_url = taxon.get(f"{prefix}_image_license") or None
    return {
        "image_url": taxon.get(f"{prefix}_image") or None,
        "image_license": _license_label(license_url),
        "image_license_url": license_url,
        "image_creator": taxon.get(f"{prefix}_image_creator") or None,
        "image_rights_holder": taxon.get(f"{prefix}_image_attribution") or None,
    }

_STATS_FILES = (
    NUMERICAL_STATS_FILE,
    NOMINAL_STATS_FILE,
    ORDINAL_STATS_FILE,
    CIRCULAR_STATS_FILE,
    DENSITY_FILE,
    DENSITY_GRID_FILE,
)


def _copy_taxon_stats(work_dir: Path, taxon_key: str, storage: ParquetStorage) -> None:
    """Copy this taxon's rows out of each global consolidated stats file,
    matching the per-taxon files the tree pipeline itself produces before
    consolidation deletes them."""
    for filename in _STATS_FILES:
        path = GLOBAL_STATS_DIR / filename
        if not storage.exists(path):
            continue
        table = storage.read_table(path, filters=[("taxon_key", "=", taxon_key)])
        if table.num_rows == 0:
            continue
        rows = table.to_pylist()
        for row in rows:
            row.pop("taxon_key", None)
        pq.write_table(pa.Table.from_pylist(rows), work_dir / filename)


def _archive_filename(taxon: TaxonRecord) -> str:
    name = taxon.get("scientific_name") or taxon.get("common_name") or str(taxon["taxon_key"])
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "species"
    return f"{slug}-{taxon['taxon_key']}.zip"


def _add_location_gid(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a single most-specific locationGid column from the tree's
    level0Gid/level1Gid/level2Gid columns — same combination logic as
    util.upload.enrich_with_gadm, but without redoing the GADM spatial join
    since the tree's occurrence data already carries the level GIDs. The
    frontend's per-observation location filter joins on locationGid
    specifically, not the level columns, so without this every observation
    silently fails to match any location filter even though the location
    list itself (built straight from the level columns) looks fine.
    """
    if not {"level0Gid", "level1Gid", "level2Gid"} & set(df.columns):
        return df
    result = df.copy()
    level2 = result.get("level2Gid")
    level1 = result.get("level1Gid")
    level0 = result.get("level0Gid")
    location_gid = level2 if level2 is not None else pd.Series(None, index=result.index)
    if level1 is not None:
        location_gid = location_gid.where(location_gid.notna(), level1)
    if level0 is not None:
        location_gid = location_gid.where(location_gid.notna(), level0)
    result["locationGid"] = location_gid
    return result


def _add_media_license_label(df: pd.DataFrame) -> pd.DataFrame:
    """Each occurrence row stores a raw mediaLicense URL, not a display label
    (see scripts/populate_tree.py) -- split it into mediaLicenseUrl (the raw
    URL) + mediaLicense (a human-readable label, e.g. "CC BY 4.0"), matching
    SpeciesOccurrence.mediaLicense/mediaLicenseUrl's own convention (see
    main.py's identical derivation for the live /gis/... occurrence routes:
    media_license_url = the raw column, media_license = _license_label(it)).
    Without this, occurrence.parquet carried a bare URL under the name the
    frontend expects to already be a short label, so every per-occurrence
    photo's license silently failed to render.
    """
    if "mediaLicense" not in df.columns:
        return df
    result = df.copy()
    result["mediaLicenseUrl"] = result["mediaLicense"]
    # .map(..., na_action="ignore") leaves NaN rows as NaN instead of passing
    # them to _license_label, which expects str | None, not a bare float NaN.
    result["mediaLicense"] = result["mediaLicense"].map(_license_label, na_action="ignore")
    return result


def build_species_archive(
    taxon: TaxonRecord, storage: ParquetStorage,
) -> tuple[Path, str, Path] | None:
    """Bundle a taxon's data into a downloadable ZIP, same shape as the
    custom-upload archive. Returns None if the taxon has no observations
    (nothing to download). Caller is responsible for deleting the returned
    work_dir after the response has been sent.
    """
    df = collect_taxon_df(taxon, storage=storage)
    if df is None or df.empty:
        return None
    df = _add_location_gid(df)
    df = _add_media_license_label(df)

    layer_meta = _build_layer_meta()
    for row in _build_temporal_var_meta(df):
        layer_meta[row["id"]] = row

    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-download-"))
    archive_name = _archive_filename(taxon)
    try:
        _copy_taxon_stats(work_dir, str(taxon["taxon_key"]), storage)
        _add_ternary_classification_overlay(work_dir, layer_meta)
        archive_path = _package_archive(work_dir, df, layer_meta, archive_name, include_csv=False)
        # Always included (not gated behind an option, unlike the custom
        # upload's own checkbox/image field) -- a species already HAS a real
        # description and image, so there's no "opt in" question here, and a
        # re-imported species download needs to exercise the exact same
        # description/image display path a custom upload does.
        description_profile = build_description_profile_for_df(work_dir, df)
        image_fields = _image_fields(taxon)
        _add_metadata_to_archive(
            archive_path,
            description_profile=description_profile,
            image_url=image_fields["image_url"],
            image_license=image_fields["image_license"],
            image_license_url=image_fields["image_license_url"],
            image_creator=image_fields["image_creator"],
            image_rights_holder=image_fields["image_rights_holder"],
        )
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return archive_path, archive_name, work_dir

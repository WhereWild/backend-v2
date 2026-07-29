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
    _add_ternary_classification_overlay,
    _build_layer_meta,
    _build_temporal_var_meta,
    _package_archive,
)

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

    layer_meta = _build_layer_meta()
    for row in _build_temporal_var_meta(df):
        layer_meta[row["id"]] = row

    work_dir = Path(tempfile.mkdtemp(prefix="wherewild-download-"))
    archive_name = _archive_filename(taxon)
    try:
        _copy_taxon_stats(work_dir, str(taxon["taxon_key"]), storage)
        _add_ternary_classification_overlay(work_dir, layer_meta)
        archive_path = _package_archive(work_dir, df, layer_meta, archive_name, include_csv=False)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return archive_path, archive_name, work_dir

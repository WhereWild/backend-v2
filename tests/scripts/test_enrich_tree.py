# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.enrich_tree as et

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_CATALOG_JSON = {
    "categories": [
        {
            "id": "bioclimate",
            "layers": [
                {
                    "id": "bio1",
                    "filename": "bio1.tif",
                    "value_type": "interval",
                    "scale_factor": 0.1,
                    "add_offset": -273.15,
                },
                {
                    "id": "swe",
                    "filename": "swe.tif",
                    "value_type": "ratio",
                    "scale_factor": 0.1,
                    "add_offset": 0.0,
                },
                {
                    "id": "kg2",
                    "filename": "kg2.tif",
                    "value_type": "nominal",
                    "scale_factor": None,
                    "add_offset": None,
                },
            ],
        }
    ]
}

FAKE_TAXON = {
    "taxon_key": "2923970",
    "path": "Plantae_6/Opuntia_2923970",
    "scientific_name": "Opuntia_humifusa",
    "common_name": "devil's tongue",
    "rank": "SPECIES",
}

FAKE_CATALOG = {
    "6": {
        "taxon_key": "6",
        "path": "Plantae_6",
        "scientific_name": "Plantae",
        "common_name": "Plants",
        "rank": "KINGDOM",
    },
    "2923970": {**FAKE_TAXON},
    "9999": {
        "taxon_key": "9999",
        "path": "Fungi_9999",
        "scientific_name": "Fungi",
        "common_name": "Fungi",
        "rank": "SPECIES",
    },
}


def _make_occurrences_parquet(path: Path, rows: dict | None = None) -> None:
    """Write a minimal consolidated occurrences.parquet for worklist tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "decimalLatitude":  [40.0, 41.0],
        "decimalLongitude": [-105.0, -106.0],
        "catalogNumber":    ["obs1", "obs2"],
        "hilbertIdx":       pa.array([1000, 1001], type=pa.int32()),
        "taxon_key":        ["2923970", "2923970"],
    }
    if rows:
        data.update(rows)
    pq.write_table(pa.table(data), path)


def _mock_rasterio_open(values: list[float], nodata: float | None = None):
    """Return a mock rasterio dataset whose sample() yields scalar values."""
    ds = MagicMock()
    ds.__enter__ = lambda s: s
    ds.__exit__ = MagicMock(return_value=False)
    ds.nodata = nodata
    ds.sample = MagicMock(return_value=iter([[v] for v in values]))
    # Force _sample_cog_batch onto the ds.sample() path (not the in-memory path).
    ds.dtypes = ["float32"]
    ds.width = 100_000
    ds.height = 100_000
    return ds


def _make_worklist(missing_layers: list[str], hilbert_vals: list[int] | None = None) -> pa.Table:
    n = 2
    if hilbert_vals is None:
        hilbert_vals = [1000, 1001]
    return pa.table({
        "catalogNumber":    pa.array(["obs1", "obs2"], type=pa.string()),
        "hilbertIdx":       pa.array(hilbert_vals,     type=pa.int32()),
        "decimalLatitude":  pa.array([40.0, 41.0],     type=pa.float64()),
        "decimalLongitude": pa.array([-105.0, -106.0], type=pa.float64()),
        "missingLayers":    pa.array([missing_layers] * n, type=pa.list_(pa.string())),
    })


# ---------------------------------------------------------------------------
# _load_layers
# ---------------------------------------------------------------------------

def test_load_layers_returns_all(tmp_path):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    with patch.object(et, "CATALOG_PATH", cat_path):
        layers = et._load_layers()
    assert len(layers) == 3
    assert layers[0]["id"] == "bio1"
    assert layers[2]["id"] == "kg2"


# ---------------------------------------------------------------------------
# _stale_gis_columns
# ---------------------------------------------------------------------------

def test_stale_gis_columns_noop():
    existing = {"decimalLatitude", "catalogNumber", "taxon_key", "bio1"}
    assert et._stale_gis_columns(["bio1"], existing) == []


def test_stale_gis_columns_removes_unknown():
    existing = {"decimalLatitude", "catalogNumber", "taxon_key", "bio1", "old_layer"}
    assert et._stale_gis_columns(["bio1"], existing) == ["old_layer"]


def test_stale_gis_columns_keeps_temporal_prefixed():
    existing = {"decimalLatitude", "catalogNumber", "taxon_key", "temperature_2m_avg_24h"}
    with patch.object(et, "_temporal_layer_ids", return_value=frozenset({"temperature_2m"})):
        assert et._stale_gis_columns([], existing) == []


# ---------------------------------------------------------------------------
# _iter_worklist_batches / _build_worklist_chunk
# ---------------------------------------------------------------------------

def test_worklist_batches_no_occurrences_file(tmp_path):
    with patch.object(et, "OCCURRENCES_FILE", tmp_path / "occurrences.parquet"), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=100))
    assert batches == []


def test_worklist_batches_unknown_root(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path)
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "load_catalog", return_value={}):
        batches = list(et._iter_worklist_batches(["bio1"], "999", row_limit=100))
    assert batches == []


def test_worklist_batches_yields_sorted_batch(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    # hilbertIdx intentionally out of order
    _make_occurrences_parquet(occ_path, rows={"hilbertIdx": pa.array([2000, 1000], type=pa.int32())})
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=1000))
    assert len(batches) == 1
    hilbert_vals = batches[0].column("hilbertIdx").to_pylist()
    assert hilbert_vals == sorted(hilbert_vals)
    assert batches[0].column("missingLayers").to_pylist()[0] == ["bio1"]  # bio1 absent → all missing


def test_worklist_batches_splits_on_row_limit(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path)  # 2 rows
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=1))
    assert len(batches) == 2
    assert all(b.num_rows == 1 for b in batches)


def test_worklist_batches_excludes_present_nonnull_layer(tmp_path):
    # bio1 present and non-null for both rows → no rows missing it
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path, rows={"bio1": [1.0, 2.0]})
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=1000))
    assert batches == []


def test_worklist_batches_partial_nulls(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path, rows={"bio1": [1.0, None]})
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=1000))
    assert len(batches) == 1
    assert batches[0].num_rows == 1
    assert batches[0].column("catalogNumber").to_pylist() == ["obs2"]


def test_worklist_batches_scopes_by_taxon_key_subtree(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    pq.write_table(pa.table({
        "decimalLatitude":  [40.0, 50.0],
        "decimalLongitude": [-105.0, -110.0],
        "catalogNumber":    ["obs1", "obs2"],
        "hilbertIdx":       pa.array([1000, 2000], type=pa.int32()),
        "taxon_key":        ["2923970", "9999"],  # 2923970 under Plantae_6, 9999 under Fungi_9999
    }), occ_path)
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=1000))
    assert len(batches) == 1
    assert batches[0].column("catalogNumber").to_pylist() == ["obs1"]  # Fungi row excluded


def test_scope_taxon_keys_unknown_root(tmp_path):
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        assert et._scope_taxon_keys("999") is None


def test_scope_taxon_keys_includes_self_and_descendants(tmp_path):
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        keys = set(et._scope_taxon_keys("6"))
    assert keys == {"6", "2923970"}  # Plantae_6 itself + its descendant, not Fungi_9999


def test_scope_taxon_keys_unions_two_roots(tmp_path):
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        keys = set(et._scope_taxon_keys(("6", "9999")))
    assert keys == {"6", "2923970", "9999"}


def test_scope_taxon_keys_all_roots_unknown_returns_none(tmp_path):
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        assert et._scope_taxon_keys(("111", "222")) is None


def test_scope_taxon_keys_partial_unknown_root_returns_found_subset(tmp_path):
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        keys = set(et._scope_taxon_keys(("6", "999")))
    assert keys == {"6", "2923970"}


# ---------------------------------------------------------------------------
# _sample_cog
# ---------------------------------------------------------------------------

def test_sample_cog_empty():
    result = et._sample_cog(Path("x.tif"), "bio1", np.array([]), np.array([]), 1.0, 0.0)
    assert result == []


def test_sample_cog_applies_scale_offset():
    mock_ds = _mock_rasterio_open([2731.0, 2830.0], nodata=65535.0)
    lats = np.array([40.0, 41.0])
    lons = np.array([-105.0, -106.0])
    with patch("rasterio.open", return_value=mock_ds):
        result = et._sample_cog(Path("bio1.tif"), "bio1", lats, lons, 0.1, -273.15)
    assert pytest.approx(result[0], abs=0.01) == 2731.0 * 0.1 - 273.15
    assert pytest.approx(result[1], abs=0.01) == 2830.0 * 0.1 - 273.15


def test_sample_cog_nodata_becomes_none():
    mock_ds = _mock_rasterio_open([65535.0], nodata=65535.0)
    lats = np.array([40.0])
    lons = np.array([-105.0])
    with patch("rasterio.open", return_value=mock_ds):
        result = et._sample_cog(Path("bio1.tif"), "bio1", lats, lons, 0.1, -273.15)
    assert result == [None]


def test_sample_cog_swe_nodata_becomes_zero():
    mock_ds = _mock_rasterio_open([65535.0], nodata=65535.0)
    lats = np.array([40.0])
    lons = np.array([-105.0])
    with patch("rasterio.open", return_value=mock_ds):
        result = et._sample_cog(Path("swe.tif"), "swe", lats, lons, 0.1, 0.0)
    assert result == [0.0]


def test_sample_cog_no_nodata():
    mock_ds = _mock_rasterio_open([5.0], nodata=None)
    lats = np.array([40.0])
    lons = np.array([-105.0])
    with patch("rasterio.open", return_value=mock_ds):
        result = et._sample_cog(Path("kg2.tif"), "kg2", lats, lons, 1.0, 0.0)
    assert result == [5.0]


def test_sample_cog_nominal_no_transform():
    mock_ds = _mock_rasterio_open([15.0], nodata=65535.0)
    lats = np.array([40.0])
    lons = np.array([-105.0])
    with patch("rasterio.open", return_value=mock_ds):
        result = et._sample_cog(Path("kg2.tif"), "kg2", lats, lons, 1.0, 0.0)
    assert result == [15.0]


# ---------------------------------------------------------------------------
# _process_batch (now returns a staging table instead of writing files)
# ---------------------------------------------------------------------------

def test_process_batch_empty():
    worklist = pa.table({
        "catalogNumber": pa.array([], type=pa.string()),
        "hilbertIdx": pa.array([], type=pa.int32()),
        "decimalLatitude": pa.array([], type=pa.float64()),
        "decimalLongitude": pa.array([], type=pa.float64()),
        "missingLayers": pa.array([], type=pa.list_(pa.string())),
    })
    assert et._process_batch(worklist, []) is None


def test_process_batch_unknown_layer(capsys):
    worklist = _make_worklist(["ghost_layer"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": 0.0}]
    et._process_batch(worklist, layers)
    out = capsys.readouterr().out
    assert "unknown layer" in out


def test_process_batch_missing_file(tmp_path, capsys):
    worklist = _make_worklist(["bio1"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": 0.0}]
    with patch.object(et, "LAYERS_DIR", tmp_path / "nonexistent"):
        et._process_batch(worklist, layers)
    out = capsys.readouterr().out
    assert "not found" in out


def test_process_batch_full_flow(tmp_path):
    worklist = _make_worklist(["bio1"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": -273.15}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    mock_ds = _mock_rasterio_open([2731.0, 2830.0], nodata=65535.0)
    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch("rasterio.open", return_value=mock_ds):
        (layers_dir / "bio1.tif").touch()
        staging = et._process_batch(worklist, layers)
    assert staging is not None
    assert staging.num_rows == 2
    vals = dict(zip(staging.column("catalogNumber").to_pylist(), staging.column("bio1").to_pylist()))
    assert pytest.approx(vals["obs1"], abs=0.01) == 2731.0 * 0.1 - 273.15


def test_process_batch_none_scale_offset(tmp_path):
    worklist = _make_worklist(["kg2"])
    layers = [{"id": "kg2", "filename": "kg2.tif", "scale_factor": None, "add_offset": None}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "kg2.tif").touch()
    mock_ds = _mock_rasterio_open([15.0, 3.0], nodata=65535.0)
    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch("rasterio.open", return_value=mock_ds):
        staging = et._process_batch(worklist, layers)
    vals = dict(zip(staging.column("catalogNumber").to_pylist(), staging.column("kg2").to_pylist()))
    assert pytest.approx(vals["obs1"]) == 15.0


def test_process_batch_empty_missing_layers(tmp_path):
    worklist = pa.table({
        "catalogNumber":    pa.array(["obs1", "obs2"], type=pa.string()),
        "hilbertIdx":       pa.array([1000, 1001],     type=pa.int32()),
        "decimalLatitude":  pa.array([40.0, 41.0],     type=pa.float64()),
        "decimalLongitude": pa.array([-105.0, -106.0], type=pa.float64()),
        "missingLayers":    pa.array([[], []],         type=pa.list_(pa.string())),
    })
    assert et._process_batch(worklist, []) is None


# ---------------------------------------------------------------------------
# _process_batch — derived elevation (slope) paths
# ---------------------------------------------------------------------------

def test_process_batch_derived_elevation_not_found(tmp_path, capsys):
    worklist = _make_worklist(["slope"])
    layers = [{"id": "slope", "filename": None, "scale_factor": None, "add_offset": None}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch.object(et, "DERIVED_FROM_ELEVATION", frozenset({"slope"})):
        et._process_batch(worklist, layers)
    out = capsys.readouterr().out
    assert "elevation.tif not found" in out


def test_process_batch_derived_slope_success(tmp_path):
    worklist = _make_worklist(["slope"])
    layers = [{"id": "slope", "filename": None, "scale_factor": None, "add_offset": None}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "elevation.tif").touch()

    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch.object(et, "DERIVED_FROM_ELEVATION", frozenset({"slope"})), \
         patch.object(et, "sample_slope_batch", return_value=[5.5, 12.0]):
        staging = et._process_batch(worklist, layers)
    vals = dict(zip(staging.column("catalogNumber").to_pylist(), staging.column("slope").to_pylist()))
    assert pytest.approx(vals["obs1"]) == 5.5
    assert pytest.approx(vals["obs2"]) == 12.0


# ---------------------------------------------------------------------------
# _write_staging_batch / _finalize_enrichment
# ---------------------------------------------------------------------------

def test_write_staging_batch_noop_on_empty(tmp_path):
    staging_dir = tmp_path / "staging"
    with patch.object(et, "STAGING_DIR", staging_dir):
        et._write_staging_batch(1, None)
        et._write_staging_batch(2, pa.table({"catalogNumber": pa.array([], type=pa.string())}))
    assert not staging_dir.exists() or not list(staging_dir.glob("*.parquet"))


def test_write_staging_batch_writes_file(tmp_path):
    staging_dir = tmp_path / "staging"
    table = pa.table({"catalogNumber": ["obs1"], "bio1": [1.0]})
    with patch.object(et, "STAGING_DIR", staging_dir):
        et._write_staging_batch(1, table)
    files = list(staging_dir.glob("*.parquet"))
    assert len(files) == 1
    assert pq.read_table(files[0]).num_rows == 1


def test_finalize_enrichment_noop_when_nothing_to_do(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path, rows={"bio1": [1.0, 2.0]})
    before = occ_path.stat().st_mtime_ns
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"):
        et._finalize_enrichment(["bio1"])
    assert occ_path.stat().st_mtime_ns == before  # untouched, nothing staged or stale


def test_finalize_enrichment_coalesces_updates_without_clobbering(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path, rows={"bio1": [None, 5.0]})  # obs1 missing, obs2 already has a value
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    pq.write_table(
        pa.table({"catalogNumber": ["obs1"], "bio1": [9.9]}),
        staging_dir / "batch_00001.parquet",
    )
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "STAGING_DIR", staging_dir):
        et._finalize_enrichment(["bio1"])
    out = pq.read_table(occ_path).to_pandas().set_index("catalogNumber")
    assert pytest.approx(out.loc["obs1", "bio1"]) == 9.9      # filled from update
    assert pytest.approx(out.loc["obs2", "bio1"]) == 5.0      # untouched, preserved


def test_finalize_enrichment_adds_new_column(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path)  # no bio1 column yet
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    pq.write_table(
        pa.table({"catalogNumber": ["obs1", "obs2"], "bio1": [1.1, 2.2]}),
        staging_dir / "batch_00001.parquet",
    )
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "STAGING_DIR", staging_dir):
        et._finalize_enrichment(["bio1"])
    out = pq.read_table(occ_path).to_pandas().set_index("catalogNumber")
    assert pytest.approx(out.loc["obs1", "bio1"]) == 1.1
    assert pytest.approx(out.loc["obs2", "bio1"]) == 2.2


def test_finalize_enrichment_drops_stale_columns(tmp_path):
    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path, rows={"old_layer": [1.0, 2.0]})
    with patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"):
        et._finalize_enrichment(["bio1"])  # old_layer no longer in catalog
    out = pq.read_table(occ_path)
    assert "old_layer" not in out.schema.names


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_nothing_to_do(tmp_path, capsys):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "OCCURRENCES_FILE", tmp_path / "occurrences.parquet"), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"), \
         patch.object(et, "load_catalog", return_value={}):
        et.main()
    out = capsys.readouterr().out
    assert "Completed" in out


def test_main_processes_batch_and_finalizes(tmp_path, capsys):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "bio1.tif").touch()

    occ_path = tmp_path / "occurrences.parquet"
    _make_occurrences_parquet(occ_path)

    worklist = _make_worklist(["bio1"])
    mock_ds = _mock_rasterio_open([2731.0, 2830.0], nodata=65535.0)
    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "OCCURRENCES_FILE", occ_path), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"), \
         patch.object(et, "LAYERS_DIR", layers_dir), \
         patch.object(et, "_iter_worklist_batches", return_value=iter([worklist])), \
         patch("rasterio.open", return_value=mock_ds):
        et.main()
    out = capsys.readouterr().out
    assert "processing batch" in out
    assert "Completed" in out
    result = pq.read_table(occ_path).to_pandas().set_index("catalogNumber")
    assert pytest.approx(result.loc["obs1", "bio1"], abs=0.01) == 2731.0 * 0.1 - 273.15


def test_main_skips_empty_batch(tmp_path, capsys):
    empty_batch = pa.table({
        "catalogNumber":    pa.array([], type=pa.string()),
        "hilbertIdx":       pa.array([], type=pa.int32()),
        "decimalLatitude":  pa.array([], type=pa.float64()),
        "decimalLongitude": pa.array([], type=pa.float64()),
        "missingLayers":    pa.array([], type=pa.list_(pa.string())),
    })
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "OCCURRENCES_FILE", tmp_path / "occurrences.parquet"), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"), \
         patch.object(et, "_iter_worklist_batches", return_value=iter([empty_batch])):
        et.main()
    out = capsys.readouterr().out
    assert "Completed" in out


def test_main_vars_to_enrich_filters_layers(tmp_path):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    captured_layer_ids = []

    def fake_iter_batches(layer_ids, root_key, *, row_limit):
        captured_layer_ids.extend(layer_ids)
        return iter([])

    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "OCCURRENCES_FILE", tmp_path / "occurrences.parquet"), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"), \
         patch.object(et, "VARS_TO_ENRICH", ["bio1"]), \
         patch.object(et, "_iter_worklist_batches", side_effect=fake_iter_batches):
        et.main()

    assert captured_layer_ids == ["bio1"]


def test_main_vars_to_enrich_none_uses_all_layers(tmp_path):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    captured_layer_ids = []

    def fake_iter_batches(layer_ids, root_key, *, row_limit):
        captured_layer_ids.extend(layer_ids)
        return iter([])

    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "OCCURRENCES_FILE", tmp_path / "occurrences.parquet"), \
         patch.object(et, "STAGING_DIR", tmp_path / "staging"), \
         patch.object(et, "VARS_TO_ENRICH", None), \
         patch.object(et, "_iter_worklist_batches", side_effect=fake_iter_batches):
        et.main()

    assert set(captured_layer_ids) == {"bio1", "swe", "kg2"}

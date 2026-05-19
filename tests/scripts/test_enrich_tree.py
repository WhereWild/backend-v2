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
                    "id": "kg0",
                    "filename": "kg0.tif",
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


def _make_occurrence_parquet(path: Path, extra_cols: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "decimalLatitude": [40.0, 41.0],
        "decimalLongitude": [-105.0, -106.0],
        "catalogNumber": ["obs1", "obs2"],
        "hilbertIdx": [1000, 1001],
        "eventTimestamp": [None, None],
        "coordinateUncertaintyInMeters": [10.0, 20.0],
        "obscured": ["No", "No"],
        "gbifRegion": ["NORTH_AMERICA", "NORTH_AMERICA"],
        "level0Gid": ["USA", "USA"],
        "level1Gid": ["USA.5", "USA.5"],
        "level2Gid": ["USA.5.1", "USA.5.2"],
        "dp": ["", ""],
        "vitality": ["", ""],
        "rcs": ["flowers", ""],
    }
    if extra_cols:
        data.update(extra_cols)
    pq.write_table(pa.table(data), path)


def _mock_rasterio_open(values: list[float], nodata: float | None = None):
    """Return a mock rasterio dataset whose sample() yields scalar values."""
    ds = MagicMock()
    ds.__enter__ = lambda s: s
    ds.__exit__ = MagicMock(return_value=False)
    ds.nodata = nodata
    ds.sample = MagicMock(return_value=iter([[v] for v in values]))
    return ds


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
    assert layers[2]["id"] == "kg0"


# ---------------------------------------------------------------------------
# _atomic_write
# ---------------------------------------------------------------------------

def test_atomic_write_creates_file(tmp_path):
    dest = tmp_path / "out.parquet"
    table = pa.table({"x": [1, 2, 3]})
    et._atomic_write(dest, table)
    assert dest.exists()
    assert pq.read_table(dest).num_rows == 3


def test_atomic_write_replaces_existing(tmp_path):
    dest = tmp_path / "out.parquet"
    pq.write_table(pa.table({"x": [9]}), dest)
    et._atomic_write(dest, pa.table({"x": [1, 2]}))
    assert pq.read_table(dest).num_rows == 2


# ---------------------------------------------------------------------------
# _drop_stale_gis_columns
# ---------------------------------------------------------------------------

def test_drop_stale_noop(tmp_path):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    table = pq.read_table(path)
    df = table.to_pandas()
    original_cols = list(df.columns)
    et._drop_stale_gis_columns(df, ["bio1"], path)
    assert list(df.columns) == original_cols


def test_drop_stale_removes_unknown(tmp_path):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path, extra_cols={"old_layer": [1.0, 2.0]})
    table = pq.read_table(path)
    df = table.to_pandas()
    assert "old_layer" in df.columns
    et._drop_stale_gis_columns(df, ["bio1"], path)
    assert "old_layer" not in df.columns
    # file should have been rewritten without the stale column
    reloaded = pq.read_table(path).to_pandas()
    assert "old_layer" not in reloaded.columns


def test_drop_stale_keeps_current_gis_columns(tmp_path):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path, extra_cols={"bio1": [1.0, 2.0], "stale": [3.0, 4.0]})
    df = pq.read_table(path).to_pandas()
    et._drop_stale_gis_columns(df, ["bio1"], path)
    assert "bio1" in df.columns
    assert "stale" not in df.columns


# ---------------------------------------------------------------------------
# _missing_rows_for_taxon
# ---------------------------------------------------------------------------

def test_missing_rows_no_parquet(tmp_path):
    taxon = {**FAKE_TAXON, "path": "Plantae_6/Ghost_0"}
    with patch.object(et, "TREE_ROOT", tmp_path):
        result = et._missing_rows_for_taxon(taxon, ["bio1"])
    assert result is None


def test_missing_rows_empty_parquet(tmp_path):
    path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"decimalLatitude": pa.array([], type=pa.float64())}), path)
    with patch.object(et, "TREE_ROOT", tmp_path):
        result = et._missing_rows_for_taxon(FAKE_TAXON, ["bio1"])
    assert result is None


def test_missing_rows_missing_required_col(tmp_path):
    path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"decimalLatitude": [1.0]}), path)
    with patch.object(et, "TREE_ROOT", tmp_path):
        result = et._missing_rows_for_taxon(FAKE_TAXON, ["bio1"])
    assert result is None


def test_missing_rows_nothing_missing(tmp_path):
    path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    _make_occurrence_parquet(path, extra_cols={"bio1": [1.0, 2.0]})
    with patch.object(et, "TREE_ROOT", tmp_path):
        result = et._missing_rows_for_taxon(FAKE_TAXON, ["bio1"])
    assert result is None


def test_missing_rows_returns_chunk(tmp_path):
    path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    _make_occurrence_parquet(path)
    with patch.object(et, "TREE_ROOT", tmp_path):
        result = et._missing_rows_for_taxon(FAKE_TAXON, ["bio1", "bio2"])
    assert result is not None
    assert result.num_rows == 2
    assert result.schema.field("hilbertIdx").type == pa.int32()
    missing = result.column("missingLayers").to_pylist()
    assert missing[0] == ["bio1", "bio2"]
    assert result.column("taxonKey").to_pylist()[0] == "2923970"


def test_missing_rows_partial_layers(tmp_path):
    path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    _make_occurrence_parquet(path, extra_cols={"bio1": [1.0, 2.0]})
    with patch.object(et, "TREE_ROOT", tmp_path):
        result = et._missing_rows_for_taxon(FAKE_TAXON, ["bio1", "bio2"])
    assert result is not None
    missing = result.column("missingLayers").to_pylist()[0]
    assert missing == ["bio2"]


# ---------------------------------------------------------------------------
# _iter_leaf_taxa
# ---------------------------------------------------------------------------

def test_iter_leaf_taxa_unknown_root():
    with patch.object(et, "load_catalog", return_value={}):
        results = list(et._iter_leaf_taxa("999"))
    assert results == []


def test_iter_leaf_taxa_yields_leaf_ranks():
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        results = list(et._iter_leaf_taxa("6"))
    taxon_keys = {t["taxon_key"] for t in results}
    assert "2923970" in taxon_keys   # SPECIES under Plantae_6
    assert "9999" not in taxon_keys  # under Fungi, not Plantae
    assert "6" not in taxon_keys     # KINGDOM rank


def test_iter_leaf_taxa_ignores_non_descendants():
    with patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        results = list(et._iter_leaf_taxa("9999"))
    # Fungi_9999 is the root; 9999 is SPECIES so it matches its own prefix
    taxon_keys = {t["taxon_key"] for t in results}
    assert "9999" in taxon_keys
    assert "2923970" not in taxon_keys


# ---------------------------------------------------------------------------
# _iter_worklist_batches
# ---------------------------------------------------------------------------

def test_worklist_batches_empty_tree(tmp_path):
    with patch.object(et, "TREE_ROOT", tmp_path), \
         patch.object(et, "load_catalog", return_value={}):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=100))
    assert batches == []


def test_worklist_batches_yields_sorted_batch(tmp_path):
    path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    # hilbertIdx intentionally out of order
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "decimalLatitude": [40.0, 41.0],
        "decimalLongitude": [-105.0, -106.0],
        "catalogNumber": ["obs1", "obs2"],
        "hilbertIdx": [2000, 1000],
        "eventTimestamp": pa.array([None, None], type=pa.int64()),
        "coordinateUncertaintyInMeters": [10.0, 20.0],
        "obscured": ["No", "No"],
        "gbifRegion": ["NORTH_AMERICA", "NORTH_AMERICA"],
        "level0Gid": ["USA", "USA"],
        "level1Gid": ["USA.5", "USA.5"],
        "level2Gid": ["USA.5.1", "USA.5.2"],
        "dp": ["", ""],
        "vitality": ["", ""],
        "rcs": ["", ""],
    }), path)
    with patch.object(et, "TREE_ROOT", tmp_path), \
         patch.object(et, "load_catalog", return_value=FAKE_CATALOG):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=1000))
    assert len(batches) == 1
    hilbert_vals = batches[0].column("hilbertIdx").to_pylist()
    assert hilbert_vals == sorted(hilbert_vals)


def test_worklist_batches_splits_on_row_limit(tmp_path):
    # Two taxa each with 2 rows; limit=2 → should yield 2 batches
    for suffix in ["A_1", "B_2"]:
        path = tmp_path / f"Plantae_6/{suffix}" / et.OCCURRENCE_FILE
        _make_occurrence_parquet(path)
    taxa = {
        "6": FAKE_CATALOG["6"],
        "1": {"taxon_key": "1", "path": "Plantae_6/A_1", "rank": "SPECIES",
              "scientific_name": "A", "common_name": "a"},
        "2": {"taxon_key": "2", "path": "Plantae_6/B_2", "rank": "SPECIES",
              "scientific_name": "B", "common_name": "b"},
    }
    with patch.object(et, "TREE_ROOT", tmp_path), \
         patch.object(et, "load_catalog", return_value=taxa):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=2))
    assert len(batches) == 2


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
        result = et._sample_cog(Path("kg0.tif"), "kg0", lats, lons, 1.0, 0.0)
    assert result == [5.0]


def test_sample_cog_nominal_no_transform():
    mock_ds = _mock_rasterio_open([15.0], nodata=65535.0)
    lats = np.array([40.0])
    lons = np.array([-105.0])
    with patch("rasterio.open", return_value=mock_ds):
        result = et._sample_cog(Path("kg0.tif"), "kg0", lats, lons, 1.0, 0.0)
    assert result == [15.0]


# ---------------------------------------------------------------------------
# _flush_taxon_updates
# ---------------------------------------------------------------------------

def test_flush_missing_key():
    pending = {}
    et._flush_taxon_updates("nope", "/dev/null", pending)
    assert pending == {}


def test_flush_file_not_exists(tmp_path):
    pending = {"tk1": {"bio1": [("obs1", 5.0)]}}
    et._flush_taxon_updates("tk1", str(tmp_path / "missing.parquet"), pending)
    assert "tk1" not in pending


def test_flush_empty_dataframe(tmp_path):
    path = tmp_path / "occ.parquet"
    pq.write_table(pa.table({"catalogNumber": pa.array([], type=pa.string())}), path)
    pending = {"tk1": {"bio1": [("obs1", 5.0)]}}
    et._flush_taxon_updates("tk1", str(path), pending)
    assert "tk1" not in pending


def test_flush_writes_values(tmp_path):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    pending = {"tk1": {"bio1": [("obs1", 12.3), ("obs2", 7.8)]}}
    et._flush_taxon_updates("tk1", str(path), pending)
    df = pq.read_table(path).to_pandas()
    assert "bio1" in df.columns
    assert pytest.approx(df.loc[df["catalogNumber"] == "obs1", "bio1"].iloc[0]) == 12.3
    assert pytest.approx(df.loc[df["catalogNumber"] == "obs2", "bio1"].iloc[0]) == 7.8
    assert "tk1" not in pending


def test_flush_skips_unknown_catalog(tmp_path):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    pending = {"tk1": {"bio1": [("ghost", 99.9)]}}
    et._flush_taxon_updates("tk1", str(path), pending)
    df = pq.read_table(path).to_pandas()
    # bio1 column created but only for known catalog numbers; ghost skipped
    assert "bio1" in df.columns
    assert df["bio1"].isna().all()


# ---------------------------------------------------------------------------
# _process_batch
# ---------------------------------------------------------------------------

def _make_worklist(
    taxon_key: str,
    data_path: str,
    missing_layers: list[str],
    hilbert_vals: list[int] | None = None,
) -> pa.Table:
    n = 2
    if hilbert_vals is None:
        hilbert_vals = [1000, 1001]
    return pa.table({
        "catalogNumber":    pa.array(["obs1", "obs2"],              type=pa.string()),
        "hilbertIdx":       pa.array(hilbert_vals,                  type=pa.int32()),
        "decimalLatitude":  pa.array([40.0, 41.0],                  type=pa.float64()),
        "decimalLongitude": pa.array([-105.0, -106.0],              type=pa.float64()),
        "missingLayers":    pa.array([missing_layers] * n,          type=pa.list_(pa.string())),
        "taxonKey":         pa.array([taxon_key] * n,               type=pa.string()),
        "dataPath":         pa.array([data_path] * n,               type=pa.string()),
    })


def test_process_batch_empty():
    worklist = pa.table({
        "catalogNumber": pa.array([], type=pa.string()),
        "hilbertIdx": pa.array([], type=pa.int32()),
        "decimalLatitude": pa.array([], type=pa.float64()),
        "decimalLongitude": pa.array([], type=pa.float64()),
        "missingLayers": pa.array([], type=pa.list_(pa.string())),
        "taxonKey": pa.array([], type=pa.string()),
        "dataPath": pa.array([], type=pa.string()),
    })
    et._process_batch(worklist, [])  # should not raise


def test_process_batch_unknown_layer(tmp_path, capsys):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    worklist = _make_worklist("tk1", str(path), ["ghost_layer"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": 0.0}]
    et._process_batch(worklist, layers)
    out = capsys.readouterr().out
    assert "unknown layer" in out


def test_process_batch_missing_file(tmp_path, capsys):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    worklist = _make_worklist("tk1", str(path), ["bio1"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": 0.0}]
    with patch.object(et, "LAYERS_DIR", tmp_path / "nonexistent"):
        et._process_batch(worklist, layers)
    out = capsys.readouterr().out
    assert "not found" in out


def test_process_batch_full_flow(tmp_path):
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    worklist = _make_worklist("tk1", str(path), ["bio1"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": -273.15}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    mock_ds = _mock_rasterio_open([2731.0, 2830.0], nodata=65535.0)
    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch("rasterio.open", return_value=mock_ds):
        # Make the file "exist" by creating it
        (layers_dir / "bio1.tif").touch()
        et._process_batch(worklist, layers)
    df = pq.read_table(path).to_pandas()
    assert "bio1" in df.columns
    assert pytest.approx(df.loc[df["catalogNumber"] == "obs1", "bio1"].iloc[0], abs=0.01) == 2731.0 * 0.1 - 273.15


def test_process_batch_none_scale_offset(tmp_path):
    # nominal layer: scale_factor=None, add_offset=None → defaults to 1.0, 0.0
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    worklist = _make_worklist("tk1", str(path), ["kg0"])
    layers = [{"id": "kg0", "filename": "kg0.tif", "scale_factor": None, "add_offset": None}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "kg0.tif").touch()
    mock_ds = _mock_rasterio_open([15.0, 3.0], nodata=65535.0)
    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch("rasterio.open", return_value=mock_ds):
        et._process_batch(worklist, layers)
    df = pq.read_table(path).to_pandas()
    assert pytest.approx(df.loc[df["catalogNumber"] == "obs1", "kg0"].iloc[0]) == 15.0


def test_process_batch_nodata_not_written(tmp_path):
    # All points are nodata → nothing written for that layer
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    worklist = _make_worklist("tk1", str(path), ["bio1"])
    layers = [{"id": "bio1", "filename": "bio1.tif", "scale_factor": 0.1, "add_offset": 0.0}]
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "bio1.tif").touch()
    mock_ds = _mock_rasterio_open([65535.0, 65535.0], nodata=65535.0)
    with patch.object(et, "LAYERS_DIR", layers_dir), \
         patch("rasterio.open", return_value=mock_ds):
        et._process_batch(worklist, layers)
    df = pq.read_table(path).to_pandas()
    assert "bio1" not in df.columns


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_nothing_to_do(tmp_path, capsys):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    # Empty tree → no worklist batches
    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "TREE_ROOT", tmp_path / "tree"), \
         patch.object(et, "load_catalog", return_value={}):
        et.main()
    out = capsys.readouterr().out
    assert "already populated" in out


def test_main_processes_batch(tmp_path, capsys):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "bio1.tif").touch()

    occ_path = tmp_path / FAKE_TAXON["path"] / et.OCCURRENCE_FILE
    _make_occurrence_parquet(occ_path)

    # Patch _iter_worklist_batches directly so CONFIG.plantae_key (which varies
    # by environment via PLANTAE_KEY env var) doesn't affect the test.
    worklist = _make_worklist("2923970", str(occ_path), ["bio1"])
    mock_ds = _mock_rasterio_open([2731.0, 2830.0], nodata=65535.0)
    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "LAYERS_DIR", layers_dir), \
         patch.object(et, "_iter_worklist_batches", return_value=iter([worklist])), \
         patch("rasterio.open", return_value=mock_ds):
        et.main()
    out = capsys.readouterr().out
    assert "processing batch" in out
    assert "Completed" in out


def test_worklist_batches_skips_none_chunks(tmp_path):
    # Taxon with no parquet → _missing_rows_for_taxon returns None → line 120 continue
    taxa = {
        "6": FAKE_CATALOG["6"],
        "1": {"taxon_key": "1", "path": "Plantae_6/Ghost_1", "rank": "SPECIES",
              "scientific_name": "Ghost", "common_name": "ghost"},
    }
    with patch.object(et, "TREE_ROOT", tmp_path), \
         patch.object(et, "load_catalog", return_value=taxa):
        batches = list(et._iter_worklist_batches(["bio1"], "6", row_limit=100))
    assert batches == []


def test_worklist_batches_progress_print(tmp_path, capsys):
    # Build 1001 taxa to trigger the idx % 1000 == 0 progress print (line 125)
    taxa = {"6": FAKE_CATALOG["6"]}
    for i in range(1001):
        key = str(1000 + i)
        path = f"Plantae_6/Species_{key}"
        taxa[key] = {"taxon_key": key, "path": path, "rank": "SPECIES",
                     "scientific_name": f"Species_{key}", "common_name": ""}
        occ = tmp_path / path / et.OCCURRENCE_FILE
        _make_occurrence_parquet(occ)
    with patch.object(et, "TREE_ROOT", tmp_path), \
         patch.object(et, "load_catalog", return_value=taxa):
        list(et._iter_worklist_batches(["bio1"], "6", row_limit=10_000_000))
    out = capsys.readouterr().out
    assert "scanned 1000 taxa" in out


def test_process_batch_empty_missing_layers(tmp_path):
    # Row with missingLayers=[] → line 215 continue, nothing written
    path = tmp_path / "occ.parquet"
    _make_occurrence_parquet(path)
    worklist = pa.table({
        "catalogNumber":    pa.array(["obs1", "obs2"],      type=pa.string()),
        "hilbertIdx":       pa.array([1000, 1001],          type=pa.int32()),
        "decimalLatitude":  pa.array([40.0, 41.0],          type=pa.float64()),
        "decimalLongitude": pa.array([-105.0, -106.0],      type=pa.float64()),
        "missingLayers":    pa.array([[], []],              type=pa.list_(pa.string())),
        "taxonKey":         pa.array(["tk1", "tk1"],        type=pa.string()),
        "dataPath":         pa.array([str(path)] * 2,       type=pa.string()),
    })
    et._process_batch(worklist, [])
    # parquet unchanged (no layers sampled)
    reloaded = pq.read_table(path)
    assert reloaded.num_rows == 2


def test_main_skips_empty_batch(tmp_path, capsys):
    # Inject an empty batch to cover line 254 continue
    empty_batch = pa.table({
        "catalogNumber":    pa.array([], type=pa.string()),
        "hilbertIdx":       pa.array([], type=pa.int32()),
        "decimalLatitude":  pa.array([], type=pa.float64()),
        "decimalLongitude": pa.array([], type=pa.float64()),
        "missingLayers":    pa.array([], type=pa.list_(pa.string())),
        "taxonKey":         pa.array([], type=pa.string()),
        "dataPath":         pa.array([], type=pa.string()),
    })
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "_iter_worklist_batches", return_value=iter([empty_batch])):
        et.main()
    out = capsys.readouterr().out
    assert "already populated" in out


def test_main_vars_to_enrich_filters_layers(tmp_path, capsys):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    captured_layer_ids = []

    def fake_iter_batches(layer_ids, root_key, *, row_limit):
        captured_layer_ids.extend(layer_ids)
        return iter([])

    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "VARS_TO_ENRICH", ["bio1"]), \
         patch.object(et, "_iter_worklist_batches", side_effect=fake_iter_batches):
        et.main()

    assert captured_layer_ids == ["bio1"]


def test_main_vars_to_enrich_none_uses_all_layers(tmp_path, capsys):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps(FAKE_CATALOG_JSON))
    captured_layer_ids = []

    def fake_iter_batches(layer_ids, root_key, *, row_limit):
        captured_layer_ids.extend(layer_ids)
        return iter([])

    with patch.object(et, "CATALOG_PATH", cat_path), \
         patch.object(et, "VARS_TO_ENRICH", None), \
         patch.object(et, "_iter_worklist_batches", side_effect=fake_iter_batches):
        et.main()

    assert set(captured_layer_ids) == {"bio1", "swe", "kg0"}

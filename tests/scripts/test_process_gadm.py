# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import csv
import io
import sqlite3
import zipfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.gis.process_gadm as pg
from config.config import load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "GIS_DIR", tmp_path)
    monkeypatch.setattr(pg, "GADM_PATH", tmp_path / "gadm.gpkg")
    monkeypatch.setattr(pg, "_GADM_ZIP", tmp_path / "gadm_410-gpkg.zip")
    monkeypatch.setattr(pg, "LOCATIONS_DIR", tmp_path / "locations")
    monkeypatch.setattr(pg, "TREE_ROOT", tmp_path / "tree")


def _make_gpkg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT)")
    conn.execute("INSERT INTO gpkg_contents VALUES ('gadm_0', 'features')")
    conn.execute("INSERT INTO gpkg_contents VALUES ('gadm_1', 'features')")
    conn.execute("INSERT INTO gpkg_contents VALUES ('gadm_2', 'features')")
    conn.execute("CREATE TABLE gadm_0 (GID_0 TEXT, NAME_0 TEXT)")
    conn.execute("INSERT INTO gadm_0 VALUES ('USA', 'United States')")
    conn.execute("INSERT INTO gadm_0 VALUES ('CAN', 'Canada')")
    conn.execute("CREATE TABLE gadm_1 (GID_0 TEXT, GID_1 TEXT, NAME_0 TEXT, NAME_1 TEXT)")
    conn.execute("INSERT INTO gadm_1 VALUES ('USA', 'USA.1_1', 'United States', 'California')")
    conn.execute("INSERT INTO gadm_1 VALUES ('CAN', 'CAN.1_1', 'Canada', 'Ontario')")
    conn.execute("CREATE TABLE gadm_2 (GID_0 TEXT, GID_1 TEXT, GID_2 TEXT, NAME_0 TEXT, NAME_1 TEXT, NAME_2 TEXT)")
    conn.execute("INSERT INTO gadm_2 VALUES ('USA', 'USA.1_1', 'USA.1.1_1', 'United States', 'California', 'Los Angeles')")
    conn.commit()
    conn.close()
    return path


def _make_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "gadm_410-gpkg.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("gadm_410.gpkg", b"GPKG_FAKE_CONTENT")
    zip_path.write_bytes(buf.getvalue())
    return zip_path


def _make_occ_parquet(path: Path, **cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 5
    data = {
        "catalogNumber": [f"obs{i}" for i in range(n)],
        "level0Gid": ["USA"] * n,
        "level1Gid": ["USA.1_1"] * n,
        "level2Gid": ["USA.1.1_1"] * n,
        "gbifRegion": ["NORTH_AMERICA"] * n,
        **cols,
    }
    pq.write_table(pa.Table.from_pandas(__import__("pandas").DataFrame(data), preserve_index=False), path)


_TAXON_A = {"taxon_key": "1", "path": "Root_1", "scientific_name": "Root", "common_name": "", "rank": "KINGDOM"}
_TAXON_B = {"taxon_key": "2", "path": "Root_1/Sp_2", "scientific_name": "Species A", "common_name": "", "rank": "SPECIES"}
_TAXON_C = {"taxon_key": "3", "path": "Root_1/Sp_2/Sub_3", "scientific_name": "Sub A", "common_name": "", "rank": "SUBSPECIES"}

_FAKE_CATALOG = {"1": _TAXON_A, "2": _TAXON_B, "3": _TAXON_C}


# ---------------------------------------------------------------------------
# Download phase
# ---------------------------------------------------------------------------

def test_download_skips_when_gpkg_exists(tmp_path, capsys):
    (tmp_path / "gadm.gpkg").write_bytes(b"existing")
    pg._download()
    assert "[skip]" in capsys.readouterr().out


def test_download_extracts_and_cleans_zip(tmp_path, capsys):
    def _fake_download():
        _make_zip(tmp_path)

    with patch.object(pg, "_download_zip", side_effect=_fake_download):
        pg._download()

    assert (tmp_path / "gadm.gpkg").exists()
    assert (tmp_path / "gadm.gpkg").read_bytes() == b"GPKG_FAKE_CONTENT"
    assert not (tmp_path / "gadm_410-gpkg.zip").exists()
    out = capsys.readouterr().out
    assert "[extract]" in out
    assert "Saved" in out


def test_download_skips_download_when_zip_exists(tmp_path, capsys):
    _make_zip(tmp_path)

    with patch.object(pg, "_download_zip") as mock_dl:
        pg._download()

    mock_dl.assert_not_called()
    assert (tmp_path / "gadm.gpkg").exists()


def test_extract_writes_gpkg(tmp_path):
    zip_path = _make_zip(tmp_path)
    dest = tmp_path / "out.gpkg"
    pg._extract(zip_path, dest)
    assert dest.read_bytes() == b"GPKG_FAKE_CONTENT"


def test_download_zip_calls_aria2c():
    with patch("subprocess.run") as mock_run:
        pg._download_zip()
    args = mock_run.call_args[0][0]
    assert "aria2c" in args[0]
    assert pg.GADM_URL in args


# ---------------------------------------------------------------------------
# GADM sqlite helpers
# ---------------------------------------------------------------------------

def test_list_feature_tables(tmp_path):
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    conn = sqlite3.connect(gpkg)
    tables = pg._list_feature_tables(conn)
    conn.close()
    assert "gadm_0" in tables
    assert "gadm_1" in tables


def test_table_columns(tmp_path):
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    conn = sqlite3.connect(gpkg)
    cols = pg._table_columns(conn, "gadm_0")
    conn.close()
    assert "GID_0" in cols
    assert "NAME_0" in cols


def test_find_table_for_level(tmp_path):
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    conn = sqlite3.connect(gpkg)
    assert pg._find_table_for_level(conn, 0) == "gadm_0"
    assert pg._find_table_for_level(conn, 1) == "gadm_1"
    conn.close()


def test_find_table_for_level_missing_raises(tmp_path):
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    conn = sqlite3.connect(gpkg)
    with pytest.raises(RuntimeError, match="No feature table found"):
        pg._find_table_for_level(conn, 9)
    conn.close()


# ---------------------------------------------------------------------------
# CSV / hierarchy writers
# ---------------------------------------------------------------------------

def test_export_level(tmp_path):
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    conn = sqlite3.connect(gpkg)
    rows = pg._export_level(conn, 0)
    conn.close()
    assert ("USA", "United States") in rows
    assert ("CAN", "Canada") in rows
    csv_path = tmp_path / "locations" / "level0.csv"
    assert csv_path.exists()
    content = list(csv.reader(csv_path.open()))
    assert content[0] == ["gid", "name"]
    gids = {row[0] for row in content[1:]}
    assert "USA" in gids


def test_export_level_deduplicates(tmp_path):
    gpkg = tmp_path / "gadm.gpkg"
    conn = sqlite3.connect(gpkg)
    conn.execute("CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT)")
    conn.execute("INSERT INTO gpkg_contents VALUES ('t', 'features')")
    conn.execute("CREATE TABLE t (GID_0 TEXT, NAME_0 TEXT)")
    conn.execute("INSERT INTO t VALUES ('USA', 'United States')")
    conn.execute("INSERT INTO t VALUES ('USA', 'United States')")
    conn.commit()
    conn.close()
    conn2 = sqlite3.connect(gpkg)
    rows = pg._export_level(conn2, 0)
    conn2.close()
    assert len(rows) == 1


def test_parent_gid_level0():
    assert pg._parent_gid("USA", 0) is None


def test_parent_gid_level1():
    assert pg._parent_gid("USA.1_1", 1) == "USA"


def test_parent_gid_level2():
    assert pg._parent_gid("USA.1.1_1", 2) == "USA.1_1"


def test_parent_gid_no_dot():
    assert pg._parent_gid("NODOT_1", 1) is None


def test_parent_gid_level2_no_suffix():
    assert pg._parent_gid("USA.1.1", 2) == "USA.1"


def test_write_hierarchy(tmp_path):
    level_rows = {
        0: [("USA", "United States"), ("CAN", "Canada")],
        1: [("USA.1_1", "California")],
    }
    pg._write_hierarchy(level_rows)
    csv_path = tmp_path / "locations" / "hierarchy.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    gids = {r["gid"] for r in rows}
    assert "USA" in gids
    assert "USA.1_1" in gids
    usa1 = next(r for r in rows if r["gid"] == "USA.1_1")
    assert usa1["parent_gid"] == "USA"
    usa = next(r for r in rows if r["gid"] == "USA")
    assert usa["parent_gid"] == ""


def test_write_gbif_regions(tmp_path):
    pg._write_gbif_regions()
    csv_path = tmp_path / "locations" / "gbif_regions.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    regions = {r["gbifRegion"] for r in rows}
    assert "NORTH_AMERICA" in regions
    assert len(rows) == len(load_config("global").gbif_regions)


def test_build_tables(tmp_path, monkeypatch):
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    monkeypatch.setattr(pg, "GADM_PATH", gpkg)
    pg._build_tables()
    assert (tmp_path / "locations" / "level0.csv").exists()
    assert (tmp_path / "locations" / "level1.csv").exists()
    assert (tmp_path / "locations" / "level2.csv").exists()
    assert (tmp_path / "locations" / "hierarchy.csv").exists()
    assert (tmp_path / "locations" / "gbif_regions.csv").exists()


def test_build_tables_missing_gadm(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "GADM_PATH", tmp_path / "missing.gpkg")
    with pytest.raises(FileNotFoundError):
        pg._build_tables()


def test_build_tables_corrupt_gadm(tmp_path, monkeypatch):
    bad = tmp_path / "bad.gpkg"
    bad.write_bytes(b"not a sqlite db")
    monkeypatch.setattr(pg, "GADM_PATH", bad)
    with pytest.raises(RuntimeError, match="Failed to read"):
        pg._build_tables()


# ---------------------------------------------------------------------------
# Location catalog helpers
# ---------------------------------------------------------------------------

def test_iter_taxa_with_occurrences(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "load_catalog", lambda: _FAKE_CATALOG)
    occ = tmp_path / "tree" / _TAXON_B["path"] / pg.OCCURRENCE_FILE
    _make_occ_parquet(occ)
    result = pg._iter_taxa_with_occurrences()
    keys = {k for k, _ in result}
    assert "2" in keys
    assert "1" not in keys
    assert "3" not in keys


def test_collect_gid_counts(tmp_path):
    occ = tmp_path / "occ.parquet"
    _make_occ_parquet(occ)
    result = pg._collect_gid_counts(occ)
    assert result["gadm_level0"]["USA"] == 5
    assert result["gadm_level1"]["USA.1_1"] == 5
    assert result["gbif_region"]["NORTH_AMERICA"] == 5


def test_collect_gid_counts_skips_null_column(tmp_path):
    occ = tmp_path / "occ.parquet"
    df = __import__("pandas").DataFrame({
        "level0Gid": [None] * 5,
        "level1Gid": [None] * 5,
        "level2Gid": [None] * 5,
        "gbifRegion": [None] * 5,
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), occ)
    result = pg._collect_gid_counts(occ)
    for scope in result.values():
        assert len(scope) == 0


def test_collect_gid_counts_skips_non_string(tmp_path):
    import pandas as pd
    occ = tmp_path / "occ.parquet"
    df = pd.DataFrame({
        "level0Gid": ["USA", None, ""],
        "level1Gid": [None] * 3,
        "level2Gid": [None] * 3,
        "gbifRegion": [None] * 3,
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), occ)
    result = pg._collect_gid_counts(occ)
    assert result["gadm_level0"].get("USA") == 1
    assert "" not in result["gadm_level0"]


def test_build_parent_map(monkeypatch):
    monkeypatch.setattr(pg, "load_catalog", lambda: _FAKE_CATALOG)
    parent_map = pg._build_parent_map()
    assert parent_map.get("2") == "1"
    assert parent_map.get("3") == "2"
    assert "1" not in parent_map


def test_ancestor_keys():
    parent_map = {"3": "2", "2": "1"}
    assert pg._ancestor_keys("3", parent_map) == ["2", "1"]
    assert pg._ancestor_keys("2", parent_map) == ["1"]
    assert pg._ancestor_keys("1", parent_map) == []
    assert pg._ancestor_keys("99", parent_map) == []


# ---------------------------------------------------------------------------
# _build_catalog integration
# ---------------------------------------------------------------------------

def test_build_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "load_catalog", lambda: _FAKE_CATALOG)
    monkeypatch.setattr(pg, "_build_parent_map", lambda: {"3": "2", "2": "1"})
    occ = tmp_path / "tree" / _TAXON_C["path"] / pg.OCCURRENCE_FILE
    _make_occ_parquet(occ)
    pg._build_catalog()
    out = tmp_path / "locations" / "location_taxa.parquet"
    assert out.exists()
    df = pq.read_table(out).to_pandas()
    taxon_keys = set(df["taxon_key"])
    assert "3" in taxon_keys
    assert "2" in taxon_keys
    assert "1" in taxon_keys


def test_build_catalog_no_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pg, "load_catalog", lambda: _FAKE_CATALOG)
    monkeypatch.setattr(pg, "_iter_taxa_with_occurrences", lambda: [])
    pg._build_catalog()
    assert "No location mappings" in capsys.readouterr().out
    assert not (tmp_path / "locations" / "location_taxa.parquet").exists()


def test_build_catalog_counts_deduplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "load_catalog", lambda: _FAKE_CATALOG)
    monkeypatch.setattr(pg, "_build_parent_map", lambda: {"3": "2", "2": "1"})
    occ = tmp_path / "tree" / _TAXON_C["path"] / pg.OCCURRENCE_FILE
    _make_occ_parquet(occ)
    pg._build_catalog()
    df = pq.read_table(tmp_path / "locations" / "location_taxa.parquet").to_pandas()
    leaf_count = df[(df["taxon_key"] == "3") & (df["scope"] == "gadm_level0")]["count"].iloc[0]
    anc_count = df[(df["taxon_key"] == "1") & (df["scope"] == "gadm_level0")]["count"].iloc[0]
    assert anc_count == leaf_count


def test_build_catalog_progress_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pg, "_LOG_INTERVAL", 1)
    parent_map: dict[str, str] = {}
    catalog: dict[str, dict] = {}
    taxa_with_occ: list[tuple[str, Path]] = []
    for i in range(3):
        key = str(i)
        path = f"Root_{i}"
        catalog[key] = {"taxon_key": key, "path": path, "scientific_name": f"Sp {i}", "common_name": "", "rank": "SPECIES"}
        occ = tmp_path / "tree" / path / pg.OCCURRENCE_FILE
        _make_occ_parquet(occ)
        taxa_with_occ.append((key, occ))
    monkeypatch.setattr(pg, "load_catalog", lambda: catalog)
    monkeypatch.setattr(pg, "_iter_taxa_with_occurrences", lambda: taxa_with_occ)
    monkeypatch.setattr(pg, "_build_parent_map", lambda: parent_map)
    pg._build_catalog()
    out = capsys.readouterr().out
    assert "Processed" in out
    assert "Rolled up" in out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "load_catalog", lambda: _FAKE_CATALOG)
    monkeypatch.setattr(pg, "_build_parent_map", lambda: {})
    gpkg = _make_gpkg(tmp_path / "gadm.gpkg")
    monkeypatch.setattr(pg, "GADM_PATH", gpkg)
    monkeypatch.setattr(pg, "_iter_taxa_with_occurrences", lambda: [])
    pg.main()
    assert (tmp_path / "locations" / "hierarchy.csv").exists()

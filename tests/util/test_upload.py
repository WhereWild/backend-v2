# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import util.upload as up

# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------

def test_norm_strips_special_chars():
    assert up._norm("Hello World!") == "helloworld"


def test_norm_preserves_alphanumeric():
    assert up._norm("decimalLatitude123") == "decimallatitude123"


def test_norm_empty():
    assert up._norm("") == ""


# ---------------------------------------------------------------------------
# _find_column
# ---------------------------------------------------------------------------

def test_find_column_exact_match():
    assert up._find_column(["latitude", "longitude"], ("latitude",)) == "latitude"


def test_find_column_normalized_match():
    assert up._find_column(["Decimal_Latitude"], ("decimallatitude",)) == "Decimal_Latitude"


def test_find_column_first_alias_wins():
    assert up._find_column(["lat", "latitude"], ("lat", "latitude")) == "lat"


def test_find_column_no_match_returns_none():
    assert up._find_column(["x", "y"], ("lat", "longitude")) is None


def test_find_column_empty_columns():
    assert up._find_column([], ("lat",)) is None


# ---------------------------------------------------------------------------
# normalize_coordinate_columns
# ---------------------------------------------------------------------------

def test_normalize_coordinate_columns_already_named():
    df = pd.DataFrame({"decimalLatitude": [1.0], "decimalLongitude": [2.0]})
    result = up.normalize_coordinate_columns(df)
    assert "decimalLatitude" in result.columns
    assert "decimalLongitude" in result.columns


def test_normalize_coordinate_columns_aliases():
    df = pd.DataFrame({"lat": [1.0], "lng": [2.0]})
    result = up.normalize_coordinate_columns(df)
    assert "decimalLatitude" in result.columns
    assert "decimalLongitude" in result.columns


def test_normalize_coordinate_columns_fallback_contains():
    df = pd.DataFrame({"my_latitude_col": [1.0], "my_longitude_col": [2.0]})
    result = up.normalize_coordinate_columns(df)
    assert "decimalLatitude" in result.columns
    assert "decimalLongitude" in result.columns


def test_normalize_coordinate_columns_no_match_unchanged():
    df = pd.DataFrame({"foo": [1.0], "bar": [2.0]})
    result = up.normalize_coordinate_columns(df)
    assert "decimalLatitude" not in result.columns
    assert "decimalLongitude" not in result.columns


def test_normalize_coordinate_columns_does_not_mutate():
    df = pd.DataFrame({"lat": [1.0], "lon": [2.0]})
    original_cols = list(df.columns)
    up.normalize_coordinate_columns(df)
    assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# ensure_catalog_numbers
# ---------------------------------------------------------------------------

def test_ensure_catalog_numbers_already_present():
    df = pd.DataFrame({"catalogNumber": ["A", "B"]})
    result = up.ensure_catalog_numbers(df)
    assert list(result["catalogNumber"]) == ["A", "B"]


def test_ensure_catalog_numbers_alias():
    df = pd.DataFrame({"gbifID": ["10", "20"], "x": [1, 2]})
    result = up.ensure_catalog_numbers(df)
    assert "catalogNumber" in result.columns
    assert list(result["catalogNumber"]) == ["10", "20"]


def test_ensure_catalog_numbers_generated_when_no_alias():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = up.ensure_catalog_numbers(df)
    assert list(result["catalogNumber"]) == ["Observation #1", "Observation #2", "Observation #3"]


# ---------------------------------------------------------------------------
# ensure_observation_names
# ---------------------------------------------------------------------------

def test_ensure_observation_names_already_present():
    df = pd.DataFrame({"observationName": ["Redwood", "Oak"]})
    result = up.ensure_observation_names(df)
    assert list(result["observationName"]) == ["Redwood", "Oak"]


def test_ensure_observation_names_alias():
    df = pd.DataFrame({"name": ["Spruce"]})
    result = up.ensure_observation_names(df)
    assert list(result["observationName"]) == ["Spruce"]


def test_ensure_observation_names_generated():
    df = pd.DataFrame({"x": [1, 2]})
    result = up.ensure_observation_names(df)
    assert list(result["observationName"]) == ["Observation #1", "Observation #2"]


def test_ensure_observation_names_fills_missing_and_blank():
    df = pd.DataFrame({"observationName": [None, "Cedar", "  "]})
    result = up.ensure_observation_names(df)
    assert result["observationName"].iloc[0] == "Observation #1"
    assert result["observationName"].iloc[1] == "Cedar"
    assert result["observationName"].iloc[2] == "Observation #3"


# ---------------------------------------------------------------------------
# validate_coordinates
# ---------------------------------------------------------------------------

def test_validate_coordinates_valid():
    df = pd.DataFrame({"decimalLatitude": [45.0], "decimalLongitude": [-120.0]})
    result = up.validate_coordinates(df)
    assert result["decimalLatitude"].iloc[0] == pytest.approx(45.0)


def test_validate_coordinates_missing_column_raises():
    df = pd.DataFrame({"decimalLatitude": [45.0]})
    with pytest.raises(HTTPException) as exc:
        up.validate_coordinates(df)
    assert exc.value.status_code == 422


def test_validate_coordinates_out_of_range_raises():
    df = pd.DataFrame({"decimalLatitude": [200.0], "decimalLongitude": [0.0]})
    with pytest.raises(HTTPException) as exc:
        up.validate_coordinates(df)
    assert exc.value.status_code == 422


def test_validate_coordinates_non_numeric_raises():
    df = pd.DataFrame({"decimalLatitude": ["abc"], "decimalLongitude": [0.0]})
    with pytest.raises(HTTPException) as exc:
        up.validate_coordinates(df)
    assert exc.value.status_code == 422


def test_validate_coordinates_converts_strings():
    df = pd.DataFrame({"decimalLatitude": ["45.0"], "decimalLongitude": ["-120.0"]})
    result = up.validate_coordinates(df)
    assert result["decimalLatitude"].iloc[0] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# check_reserved_columns
# ---------------------------------------------------------------------------

def test_check_reserved_columns_no_conflict():
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    up.check_reserved_columns(df, {"bio1", "bio2"})  # should not raise


def test_check_reserved_columns_conflict_raises():
    df = pd.DataFrame({"bio1": [1.0], "x": [2.0]})
    with pytest.raises(HTTPException) as exc:
        up.check_reserved_columns(df, {"bio1", "bio2"})
    assert exc.value.status_code == 422
    assert "bio1" in exc.value.detail


# ---------------------------------------------------------------------------
# _sample_layer
# ---------------------------------------------------------------------------

def _make_rasterio_sample_mock(values: list[float], nodata=None):
    mock_ds = MagicMock()
    mock_ds.nodata = nodata
    mock_ds.sample.return_value = [[v] for v in values]
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_ds
    mock_open.return_value.__exit__.return_value = False
    return mock_open, mock_ds


def test_sample_layer_returns_scaled_values():
    lats = np.array([45.0, 46.0])
    lons = np.array([-120.0, -121.0])
    mock_open, _ = _make_rasterio_sample_mock([10.0, 20.0])
    with patch("util.upload.rasterio.open", mock_open):
        result = up._sample_layer(Path("x.tif"), lats, lons, 2.0, 1.0, None)
    assert result == [pytest.approx(21.0), pytest.approx(41.0)]


def test_sample_layer_nodata_returns_none():
    lats = np.array([45.0])
    lons = np.array([-120.0])
    mock_open, _ = _make_rasterio_sample_mock([-9999.0], nodata=-9999.0)
    with patch("util.upload.rasterio.open", mock_open):
        result = up._sample_layer(Path("x.tif"), lats, lons, 1.0, 0.0, None)
    assert result == [None]


def test_sample_layer_explicit_nodata_overrides_ds_nodata():
    lats = np.array([45.0])
    lons = np.array([-120.0])
    mock_open, mock_ds = _make_rasterio_sample_mock([0.0], nodata=99.0)
    with patch("util.upload.rasterio.open", mock_open):
        result = up._sample_layer(Path("x.tif"), lats, lons, 1.0, 0.0, nodata=0.0)
    assert result == [None]


# ---------------------------------------------------------------------------
# _load_legend
# ---------------------------------------------------------------------------

def test_load_legend_missing_file_returns_empty(tmp_path):
    with patch("util.upload._LEGEND_DIR", tmp_path):
        assert up._load_legend("nonexistent") == []


def test_load_legend_returns_classes(tmp_path):
    legend = {"classes": [{"id": 1, "name": "Tropical"}, {"id": 2, "name": "Arid"}]}
    (tmp_path / "kg2_legend.json").write_text(json.dumps(legend))
    with patch("util.upload._LEGEND_DIR", tmp_path):
        result = up._load_legend("kg2")
    assert len(result) == 2
    assert result[0]["name"] == "Tropical"


def test_load_legend_missing_classes_key_returns_empty(tmp_path):
    legend = {"source": "CHELSA"}
    (tmp_path / "kg2_legend.json").write_text(json.dumps(legend))
    with patch("util.upload._LEGEND_DIR", tmp_path):
        result = up._load_legend("kg2")
    assert result == []


# ---------------------------------------------------------------------------
# _build_layer_meta
# ---------------------------------------------------------------------------

def test_build_layer_meta_embeds_category():
    fake = [
        ({"id": "bio1", "filename": "bio1.tif"}, {"id": "bioclimate", "display_name": "Bioclimatic"}),
    ]
    with patch("util.upload.load_layers_with_category", return_value=fake):
        meta = up._build_layer_meta()
    assert "bio1" in meta
    assert meta["bio1"]["category_id"] == "bioclimate"
    assert meta["bio1"]["category_display_name"] == "Bioclimatic"


def test_build_layer_meta_skips_temporal():
    fake = [
        ({"id": "t_1h", "filename": "t.tif", "window_hours": 1}, {"id": "temporal"}),
    ]
    with patch("util.upload.load_layers_with_category", return_value=fake):
        meta = up._build_layer_meta()
    assert meta == {}


def test_build_layer_meta_skips_no_filename():
    fake = [
        ({"id": "virtual"}, {"id": "bioclimate", "display_name": "Bio"}),
    ]
    with patch("util.upload.load_layers_with_category", return_value=fake):
        meta = up._build_layer_meta()
    assert meta == {}


def test_build_layer_meta_category_display_name_fallback():
    fake = [
        ({"id": "bio1", "filename": "bio1.tif"}, {"id": "bioclimate"}),
    ]
    with patch("util.upload.load_layers_with_category", return_value=fake):
        meta = up._build_layer_meta()
    assert meta["bio1"]["category_display_name"] == "bioclimate"


# ---------------------------------------------------------------------------
# enrich_with_gis
# ---------------------------------------------------------------------------

def test_enrich_with_gis_empty_df_returns_copy():
    df = pd.DataFrame({"decimalLatitude": pd.Series([], dtype=float),
                       "decimalLongitude": pd.Series([], dtype=float)})
    with patch("util.upload.load_layers_with_category", return_value=[]):
        result = up.enrich_with_gis(df)
    assert result.empty


def test_enrich_with_gis_adds_layer_column():
    df = pd.DataFrame({"decimalLatitude": [45.0, 46.0], "decimalLongitude": [-120.0, -121.0]})
    fake = [({"id": "bio1", "filename": "bio1.tif", "scale_factor": 1.0, "add_offset": 0.0},
             {"id": "bioclimate", "display_name": "Bioclimatic"})]
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    with patch("util.upload.load_layers_with_category", return_value=fake), \
         patch("util.upload.LAYERS_DIR") as mock_dir, \
         patch("util.upload._sample_layer", return_value=[1.0, 2.0]), \
         patch("util.upload.hilbert_index", return_value=0):
        mock_dir.__truediv__ = lambda _self, x: mock_path
        result = up.enrich_with_gis(df)
    assert "bio1" in result.columns


def test_enrich_with_gis_skips_missing_cog():
    df = pd.DataFrame({"decimalLatitude": [45.0], "decimalLongitude": [-120.0]})
    fake = [({"id": "bio1", "filename": "bio1.tif", "scale_factor": 1.0, "add_offset": 0.0},
             {"id": "bioclimate", "display_name": "Bioclimatic"})]
    mock_path = MagicMock()
    mock_path.exists.return_value = False
    with patch("util.upload.load_layers_with_category", return_value=fake), \
         patch("util.upload.LAYERS_DIR") as mock_dir, \
         patch("util.upload.hilbert_index", return_value=0):
        mock_dir.__truediv__ = lambda _self, x: mock_path
        result = up.enrich_with_gis(df)
    assert "bio1" not in result.columns


def test_enrich_with_gis_sample_exception_skipped():
    df = pd.DataFrame({"decimalLatitude": [45.0], "decimalLongitude": [-120.0]})
    fake = [({"id": "bio1", "filename": "bio1.tif", "scale_factor": 1.0, "add_offset": 0.0},
             {"id": "bioclimate", "display_name": "Bioclimatic"})]
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    with patch("util.upload.load_layers_with_category", return_value=fake), \
         patch("util.upload.LAYERS_DIR") as mock_dir, \
         patch("util.upload._sample_layer", side_effect=RuntimeError("oops")), \
         patch("util.upload.hilbert_index", return_value=0):
        mock_dir.__truediv__ = lambda _self, x: mock_path
        result = up.enrich_with_gis(df)
    assert "bio1" not in result.columns


# ---------------------------------------------------------------------------
# build_archive
# ---------------------------------------------------------------------------

def _make_minimal_df():
    return pd.DataFrame({
        "catalogNumber": ["OBS1", "OBS2"],
        "decimalLatitude": [45.0, 46.0],
        "decimalLongitude": [-120.0, -121.0],
    })


def test_build_archive_returns_zip_path():
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"):
        archive_path, archive_name, work_dir = up.build_archive(df)
    try:
        assert archive_name == "processed_observations.zip"
        assert archive_path.exists()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_includes_occurrence_parquet():
    import zipfile
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
        assert "occurrence.parquet" in names
        assert "occurrence.csv" in names
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_generates_categorical_value_lookup():
    import zipfile
    df = pd.DataFrame({
        "catalogNumber": ["OBS1"],
        "decimalLatitude": [45.0],
        "decimalLongitude": [-120.0],
        "kg2": [15.0],
    })
    fake_meta = {
        "kg2": {"id": "kg2", "display_name": "Köppen-Geiger", "value_type": "nominal",
                "category_display_name": "Bioclimatic", "source": "chelsa_v2_1"},
    }
    fake_legend = [{"id": 15, "name": "Temperate, humid subtropical",
                    "group": "temperate", "group_label": "Temperate"}]
    with patch("util.upload._build_layer_meta", return_value=fake_meta), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch("util.upload._load_legend", return_value=fake_legend):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
        assert "categorical_value_lookup.parquet" in names
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_includes_variable_metadata():
    import io
    import zipfile

    import pyarrow.parquet as pq
    df = _make_minimal_df()
    fake_meta = {
        "bio1": {
            "id": "bio1",
            "display_name": "Annual Mean Temperature",
            "units": "°C",
            "value_type": "interval",
            "category_display_name": "Bioclimatic",
            "source": "chelsa_v2_1",
        },
    }
    with patch("util.upload._build_layer_meta", return_value=fake_meta), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            assert "variable_metadata.parquet" in names
            raw = zf.read("variable_metadata.parquet")
        table = pq.read_table(io.BytesIO(raw))
        row = table.to_pydict()
        assert row["id"] == ["bio1"]
        assert row["name"] == ["Annual Mean Temperature"]
        assert row["units"] == ["°C"]
        assert row["value_type"] == ["interval"]
        assert row["category"] == ["Bioclimatic"]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_csv_conversion_exception_silenced():
    import zipfile
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch("util.upload.pq.read_table", side_effect=RuntimeError("bad parquet")):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
        assert "occurrence.parquet" in names
        assert "occurrence.csv" not in names
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_no_lookup_when_no_nominal_layers():
    import zipfile
    df = _make_minimal_df()
    fake_meta = {
        "bio1": {"id": "bio1", "display_name": "Temp", "value_type": "interval",
                 "category_display_name": "Bioclimatic"},
    }
    with patch("util.upload._build_layer_meta", return_value=fake_meta), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
        assert "categorical_value_lookup.parquet" not in names
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_http_exception_reraises_and_cleans_up():
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=HTTPException(status_code=422, detail="bad")):
        with pytest.raises(HTTPException) as exc:
            up.build_archive(df)
    assert exc.value.status_code == 422


def test_build_archive_generic_exception_wraps_as_500():
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=RuntimeError("crash")):
        with pytest.raises(HTTPException) as exc:
            up.build_archive(df)
    assert exc.value.status_code == 500
    assert "crash" in exc.value.detail

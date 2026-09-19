# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
# normalize_image_column
# ---------------------------------------------------------------------------

def test_normalize_image_column_already_present():
    df = pd.DataFrame({"imageUrl": ["https://example.com/a.jpg"]})
    result = up.normalize_image_column(df)
    assert list(result["imageUrl"]) == ["https://example.com/a.jpg"]


def test_normalize_image_column_alias():
    df = pd.DataFrame({"photo_url": ["https://example.com/b.jpg"], "x": [1]})
    result = up.normalize_image_column(df)
    assert "imageUrl" in result.columns
    assert "photo_url" not in result.columns
    assert list(result["imageUrl"]) == ["https://example.com/b.jpg"]


def test_normalize_image_column_absent_when_no_alias_matches():
    df = pd.DataFrame({"x": [1, 2]})
    result = up.normalize_image_column(df)
    assert "imageUrl" not in result.columns


def test_normalize_image_column_blanks_become_none():
    # pandas normalizes an assigned None to NaN on an object column -- both
    # serialize identically to a parquet/JSON null downstream, so check
    # missing-ness (pd.isna) rather than exact identity to None.
    df = pd.DataFrame({"imageUrl": ["https://example.com/c.jpg", "", "  ", None]})
    result = up.normalize_image_column(df)
    assert result["imageUrl"].iloc[0] == "https://example.com/c.jpg"
    assert pd.isna(result["imageUrl"].iloc[1])
    assert pd.isna(result["imageUrl"].iloc[2])
    assert pd.isna(result["imageUrl"].iloc[3])


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


def test_build_archive_categorical_value_lookup_uses_inline_legend_for_custom_layer():
    """A custom layer has no on-disk legend file for _load_legend() to find
    -- its classes travel as an inline JSON string on the layer_meta entry
    instead (see parse_custom_layer_metadata), same as a temporal layer's
    own synthetic legend_classes."""
    import zipfile
    df = pd.DataFrame({
        "catalogNumber": ["OBS1"],
        "decimalLatitude": [45.0],
        "decimalLongitude": [-120.0],
        "my_layer": [3.0],
    })
    fake_meta = {
        "my_layer": {
            "id": "my_layer", "value_type": "nominal",
            "legend_classes": '[{"id": 3, "name": "Wetland", "color": "#00f"}]',
            "_legend_key": "my_layer",
        },
    }
    import io
    # _load_legend forced to return [] for everything -- if the code fell
    # back to it instead of using the inline legend_classes JSON, the
    # lookup row below would come back empty and the assertion would fail.
    with patch("util.upload._build_layer_meta", return_value=fake_meta), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch("util.upload._load_legend", return_value=[]):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            table = pq.read_table(io.BytesIO(zf.read("categorical_value_lookup.parquet")))
        rows = table.to_pylist()
        assert rows == [{
            "variable": "my_layer", "code": "3", "metric": "class_3",
            "label": "Wetland", "group": "", "groupLabel": "",
        }]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# parse_custom_layer_metadata
# ---------------------------------------------------------------------------

def test_parse_custom_layer_metadata_none_returns_empty():
    assert up.parse_custom_layer_metadata(None) == []


def test_parse_custom_layer_metadata_empty_string_returns_empty():
    assert up.parse_custom_layer_metadata("") == []


def test_parse_custom_layer_metadata_valid_continuous_layer():
    raw = json.dumps([{"id": "my_layer", "name": "My Layer", "valueType": "ratio", "units": "mm"}])
    rows = up.parse_custom_layer_metadata(raw)
    assert rows == [{
        "id": "my_layer", "name": "My Layer", "units": "mm", "imperial_unit": None,
        "value_type": "ratio", "domain": "continuous", "category": "Custom Layers",
        "group": None, "group_label": None, "sort_order": 20000,
        "render_min": None, "render_max": None, "legend_classes": None,
        "_legend_key": "my_layer",
    }]


def test_parse_custom_layer_metadata_valid_categorical_layer_with_legend():
    raw = json.dumps([{
        "id": "my_layer", "name": "My Layer", "valueType": "nominal",
        "legendClasses": [{"id": 1, "name": "Forest", "color": "#0f0"}],
    }])
    rows = up.parse_custom_layer_metadata(raw)
    assert rows[0]["domain"] == "discrete"
    assert json.loads(rows[0]["legend_classes"]) == [
        {"id": 1, "name": "Forest", "color": "#0f0"},
    ]


def test_parse_custom_layer_metadata_ordinal_render_min_max_spans_class_ids():
    """Ordinal coloring is a gradient keyed to (classId - render_min) /
    (render_max - render_min) -- render_min/max must span the class id
    range (matching every built-in ordinal layer's own catalog.json
    convention, e.g. salinity: render_min=0/render_max=4 for classes 0..4),
    not stay None, or the gradient collapses to a single color."""
    raw = json.dumps([{
        "id": "my_layer", "name": "My Layer", "valueType": "ordinal",
        "legendClasses": [
            {"id": 2, "name": "Medium", "color": "#ff0"},
            {"id": 0, "name": "Low", "color": "#0f0"},
            {"id": 4, "name": "High", "color": "#f00"},
        ],
    }])
    rows = up.parse_custom_layer_metadata(raw)
    assert rows[0]["render_min"] == 0.0
    assert rows[0]["render_max"] == 4.0


def test_parse_custom_layer_metadata_nominal_also_gets_render_min_max():
    """Nominal doesn't need render_min/max for its own pixel coloring (it
    uses legendClasses.color directly), but setting it anyway is harmless
    and keeps the two categorical value types consistent."""
    raw = json.dumps([{
        "id": "my_layer", "name": "My Layer", "valueType": "nominal",
        "legendClasses": [
            {"id": 5, "name": "Forest", "color": "#0f0"},
            {"id": 9, "name": "Water", "color": "#00f"},
        ],
    }])
    rows = up.parse_custom_layer_metadata(raw)
    assert rows[0]["render_min"] == 5.0
    assert rows[0]["render_max"] == 9.0


def test_parse_custom_layer_metadata_no_legend_leaves_render_min_max_none():
    raw = json.dumps([{"id": "my_layer", "valueType": "ratio"}])
    rows = up.parse_custom_layer_metadata(raw)
    assert rows[0]["render_min"] is None
    assert rows[0]["render_max"] is None


def test_parse_custom_layer_metadata_invalid_json_raises_422():
    with pytest.raises(HTTPException) as exc:
        up.parse_custom_layer_metadata("not json")
    assert exc.value.status_code == 422


def test_parse_custom_layer_metadata_non_array_raises_422():
    with pytest.raises(HTTPException) as exc:
        up.parse_custom_layer_metadata(json.dumps({"id": "x"}))
    assert exc.value.status_code == 422


def test_parse_custom_layer_metadata_missing_id_raises_422():
    with pytest.raises(HTTPException) as exc:
        up.parse_custom_layer_metadata(json.dumps([{"valueType": "ratio"}]))
    assert exc.value.status_code == 422


def test_parse_custom_layer_metadata_duplicate_id_raises_422():
    raw = json.dumps([
        {"id": "my_layer", "valueType": "ratio"},
        {"id": "my_layer", "valueType": "ratio"},
    ])
    with pytest.raises(HTTPException) as exc:
        up.parse_custom_layer_metadata(raw)
    assert exc.value.status_code == 422


def test_parse_custom_layer_metadata_unsupported_value_type_raises_422():
    raw = json.dumps([{"id": "my_layer", "valueType": "circular"}])
    with pytest.raises(HTTPException) as exc:
        up.parse_custom_layer_metadata(raw)
    assert exc.value.status_code == 422


def test_parse_custom_layer_metadata_invalid_legend_classes_raises_422():
    raw = json.dumps([{
        "id": "my_layer", "valueType": "nominal",
        "legendClasses": [{"name": "missing id"}],
    }])
    with pytest.raises(HTTPException) as exc:
        up.parse_custom_layer_metadata(raw)
    assert exc.value.status_code == 422


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


def test_package_archive_includes_relative_ranks_when_present(tmp_path):
    """Relative-rank positions are download-only: _copy_taxon_stats (in
    util/download.py) writes relative_ranks_positions.parquet into work_dir
    before _package_archive runs, exactly like the other stats files, so
    this only asserts _package_archive's own files_to_zip wiring picks it up
    when present -- a plain upload never writes this file, so it's absent
    there (see the next test)."""
    df = _make_minimal_df()
    pq.write_table(
        pa.Table.from_pylist([
            {"variable": "bio1", "metric": "mean", "position": 4, "count": 10,
             "sampleCount": 25, "contextLabel": "Testaceae"},
        ]),
        tmp_path / up.POSITION_FILE,
    )
    import io
    archive_path = up._package_archive(tmp_path, df, {}, "a.zip")
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        assert up.POSITION_FILE in names
        table = pq.read_table(io.BytesIO(zf.read(up.POSITION_FILE)))
    assert table.to_pylist()[0]["contextLabel"] == "Testaceae"


def test_build_archive_omits_relative_ranks_for_plain_upload():
    """A custom CSV upload has no tree ancestors to rank against, so nothing
    ever writes relative_ranks_positions.parquet into its work_dir -- the
    archive simply doesn't have the entry, same as every other file here
    that _package_archive skips when missing."""
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"):
        archive_path, _, work_dir = up.build_archive(df)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
        assert up.POSITION_FILE not in names
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


# ---------------------------------------------------------------------------
# _resolve_own_rankable_values / compute_relative_ranks_for_upload
# (custom-upload "extra options" parent-taxon ranking)
# ---------------------------------------------------------------------------

_RATIO_LAYER_META = {
    "bio1": {"id": "bio1", "value_type": "ratio"},
    "aspect_deg": {"id": "aspect_deg", "value_type": "circular"},
}


def test_resolve_own_rankable_values_numeric_variable(tmp_path):
    pq.write_table(pa.table({
        "variable": ["bio1"],
        "count": [42],
        "mean": [5.0],
        "min": [1.0],
        "max": [9.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    result = up._resolve_own_rankable_values(tmp_path, _RATIO_LAYER_META)

    assert result[("bio1", "mean")] == (5.0, 42)
    assert result[("bio1", "min")] == (1.0, 42)
    assert result[("bio1", "max")] == (9.0, 42)


def test_resolve_own_rankable_values_skips_variable_below_sample_threshold(tmp_path):
    pq.write_table(pa.table({
        "variable": ["bio1"],
        "count": [5],
        "mean": [5.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    result = up._resolve_own_rankable_values(tmp_path, _RATIO_LAYER_META)

    assert result == {}


def test_resolve_own_rankable_values_skips_unknown_variable(tmp_path):
    pq.write_table(pa.table({
        "variable": ["not_in_layer_meta"],
        "count": [42],
        "mean": [5.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    result = up._resolve_own_rankable_values(tmp_path, _RATIO_LAYER_META)

    assert result == {}


def test_resolve_own_rankable_values_includes_circular_metrics(tmp_path):
    pq.write_table(pa.table({
        "variable": ["aspect_deg"],
        "count": [42],
        "circular_mean": [180.0],
        "rbar": [0.8],
    }), tmp_path / up.CIRCULAR_STATS_FILE)

    result = up._resolve_own_rankable_values(tmp_path, _RATIO_LAYER_META)

    assert result[("aspect_deg", "circular_mean")] == (180.0, 42)
    assert result[("aspect_deg", "rbar")] == (0.8, 42)


def test_resolve_own_rankable_values_no_files(tmp_path):
    assert up._resolve_own_rankable_values(tmp_path, _RATIO_LAYER_META) == {}


_NOMINAL_LAYER_META = {"kg2": {"id": "kg2", "value_type": "nominal"}}
_ORDINAL_LAYER_META = {"salinity": {"id": "salinity", "value_type": "ordinal"}}


def _write_tall_stats(path, rows):
    pq.write_table(pa.table({
        "variable": [r[0] for r in rows],
        "metric": [r[1] for r in rows],
        "value": [r[2] for r in rows],
    }), path)


def test_resolve_own_rankable_values_includes_nominal_metrics(tmp_path):
    _write_tall_stats(tmp_path / up.NOMINAL_STATS_FILE, [
        ("kg2", "total_samples", 50.0),
        ("kg2", "unique_classes", 3.0),
        ("kg2", "entropy", 0.9),
        ("kg2", "mode", 5.0),  # excluded -- a class id, not comparable across taxa
        ("kg2", "class_5", 0.6),
        ("kg2", "class_2", 0.0),  # excluded -- zero presence, no row at all
    ])

    result = up._resolve_own_rankable_values(tmp_path, _NOMINAL_LAYER_META)

    assert result == {
        ("kg2", "total_samples"): (50.0, 50),
        ("kg2", "unique_classes"): (3.0, 50),
        ("kg2", "entropy"): (0.9, 50),
        ("kg2", "class_5"): (0.6, 50),
    }


def test_resolve_own_rankable_values_nominal_below_sample_threshold(tmp_path):
    _write_tall_stats(tmp_path / up.NOMINAL_STATS_FILE, [
        ("kg2", "total_samples", 5.0),
        ("kg2", "class_5", 0.6),
    ])

    assert up._resolve_own_rankable_values(tmp_path, _NOMINAL_LAYER_META) == {}


def test_resolve_own_rankable_values_ordinal_prefers_count_over_total_samples(tmp_path):
    _write_tall_stats(tmp_path / up.ORDINAL_STATS_FILE, [
        ("salinity", "count", 42.0),
        ("salinity", "total_samples", 999.0),
        ("salinity", "median", 3.0),  # excluded -- an ordinal class id
        ("salinity", "unique_classes", 4.0),
        ("salinity", "class_1", 0.3),
    ])

    result = up._resolve_own_rankable_values(tmp_path, _ORDINAL_LAYER_META)

    # sample_count comes from "count" (42), not "total_samples" (999) --
    # matches _write_rank_positions' own count-then-total_samples priority.
    assert result == {
        ("salinity", "count"): (42.0, 42),
        ("salinity", "total_samples"): (999.0, 42),
        ("salinity", "unique_classes"): (4.0, 42),
        ("salinity", "class_1"): (0.3, 42),
    }


def test_resolve_own_rankable_values_categorical_skips_wrong_value_type(tmp_path):
    _write_tall_stats(tmp_path / up.NOMINAL_STATS_FILE, [
        ("salinity", "total_samples", 50.0),
        ("salinity", "class_1", 0.6),
    ])
    # salinity is declared ordinal in the layer meta, not nominal.
    assert up._resolve_own_rankable_values(tmp_path, _ORDINAL_LAYER_META) == {}


def test_compute_relative_ranks_for_upload_returns_none_for_unknown_taxon(tmp_path):
    with patch("util.upload.get_taxon_by_id", return_value=None):
        result = up.compute_relative_ranks_for_upload(tmp_path, {}, "999")
    assert result is None
    assert not (tmp_path / up.POSITION_FILE).exists()


def test_compute_relative_ranks_for_upload_returns_none_with_no_rankable_stats(tmp_path):
    ancestor = {"taxon_key": "42", "scientific_name": "Testaceae"}
    with patch("util.upload.get_taxon_by_id", return_value=ancestor):
        result = up.compute_relative_ranks_for_upload(tmp_path, {}, "42")
    assert result is None
    assert not (tmp_path / up.POSITION_FILE).exists()


def test_compute_relative_ranks_for_upload_writes_position_file(tmp_path):
    pq.write_table(pa.table({
        "variable": ["bio1"],
        "count": [42],
        "mean": [5.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    ancestor = {"taxon_key": "42", "scientific_name": "Testaceae"}
    group = pd.DataFrame({"value": [1.0, 3.0, 9.0], "count": [3, 3, 3]})

    with patch("util.upload.get_taxon_by_id", return_value=ancestor), \
         patch("util.upload.get_ancestors", return_value=[]), \
         patch("util.upload.read_rank_context_groups", return_value={("bio1", "mean"): group}):
        result = up.compute_relative_ranks_for_upload(tmp_path, _RATIO_LAYER_META, "42")

    assert result == [{
        "variable": "bio1",
        "metric": "mean",
        "position": 2,  # 5.0 slots after 1.0 and 3.0, before 9.0
        "count": 4,
        "sampleCount": 42,
        "contextLabel": "Testaceae",
    }]

    written = pq.read_table(tmp_path / up.POSITION_FILE).to_pylist()
    assert written == result


def test_compute_relative_ranks_for_upload_ranks_against_every_ancestor_up_the_tree(tmp_path):
    """A selected genus parent must also produce context rows for its own
    ancestors (family, order, ...), same as a real taxon's own lineage."""
    pq.write_table(pa.table({
        "variable": ["bio1"],
        "count": [42],
        "mean": [5.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    genus = {"taxon_key": "42", "scientific_name": "Testus"}
    family = {"taxon_key": "43", "scientific_name": "Testaceae"}
    order = {"taxon_key": "44", "scientific_name": "Testales"}
    groups_by_context = {
        "42": {("bio1", "mean"): pd.DataFrame({"value": [1.0, 3.0], "count": [2, 2]})},
        "43": {("bio1", "mean"): pd.DataFrame({"value": [1.0, 3.0, 9.0], "count": [3, 3, 3]})},
        "44": {},  # no siblings ranked at this level -- contributes nothing
    }

    with patch("util.upload.get_taxon_by_id", return_value=genus), \
         patch("util.upload.get_ancestors", return_value=[family, order]), \
         patch(
             "util.upload.read_rank_context_groups",
             side_effect=lambda context_id, _rank: groups_by_context[context_id],
         ):
        result = up.compute_relative_ranks_for_upload(tmp_path, _RATIO_LAYER_META, "42")

    assert result == [
        {
            "variable": "bio1", "metric": "mean", "position": 2, "count": 3,
            "sampleCount": 42, "contextLabel": "Testus",
        },
        {
            "variable": "bio1", "metric": "mean", "position": 2, "count": 4,
            "sampleCount": 42, "contextLabel": "Testaceae",
        },
    ]


def test_compute_relative_ranks_for_upload_ranks_nominal_class_metric(tmp_path):
    """End-to-end: a nominal class_ fraction gets ranked with the implicit-
    zero offset applied, not just searched against the group's own (nonzero-
    only) members."""
    _write_tall_stats(tmp_path / up.NOMINAL_STATS_FILE, [
        ("kg2", "total_samples", 50.0),
        ("kg2", "class_5", 0.4),
    ])

    ancestor = {"taxon_key": "42", "scientific_name": "Testaceae"}
    # 5 total in the population, only 2 have nonzero class_5 presence.
    group = pd.DataFrame({"value": [0.1, 0.6], "count": [5, 5]})

    with patch("util.upload.get_taxon_by_id", return_value=ancestor), \
         patch("util.upload.get_ancestors", return_value=[]), \
         patch("util.upload.read_rank_context_groups", return_value={("kg2", "class_5"): group}):
        result = up.compute_relative_ranks_for_upload(tmp_path, _NOMINAL_LAYER_META, "42")

    assert result == [{
        "variable": "kg2",
        "metric": "class_5",
        "position": 4,  # 3 implicit zeros + 0.1 below 0.4
        "count": 6,
        "sampleCount": 50,
        "contextLabel": "Testaceae",
    }]


def test_compute_relative_ranks_for_upload_skips_metrics_with_no_sibling_group(tmp_path):
    pq.write_table(pa.table({
        "variable": ["bio1"],
        "count": [42],
        "mean": [5.0],
        "min": [1.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    ancestor = {"taxon_key": "42", "scientific_name": "Testaceae"}
    # Only "mean" has a sibling group under this ancestor -- "min" has none.
    group = pd.DataFrame({"value": [1.0, 9.0], "count": [2, 2]})

    with patch("util.upload.get_taxon_by_id", return_value=ancestor), \
         patch("util.upload.get_ancestors", return_value=[]), \
         patch("util.upload.read_rank_context_groups", return_value={("bio1", "mean"): group}):
        result = up.compute_relative_ranks_for_upload(tmp_path, _RATIO_LAYER_META, "42")

    assert len(result) == 1
    assert result[0]["metric"] == "mean"


def test_compute_relative_ranks_for_upload_returns_none_when_ancestor_has_no_groups(tmp_path):
    pq.write_table(pa.table({
        "variable": ["bio1"],
        "count": [42],
        "mean": [5.0],
    }), tmp_path / up.NUMERICAL_STATS_FILE)

    ancestor = {"taxon_key": "42", "scientific_name": "Testaceae"}
    with patch("util.upload.get_taxon_by_id", return_value=ancestor), \
         patch("util.upload.get_ancestors", return_value=[]), \
         patch("util.upload.read_rank_context_groups", return_value={}):
        result = up.compute_relative_ranks_for_upload(tmp_path, _RATIO_LAYER_META, "42")

    assert result is None
    assert not (tmp_path / up.POSITION_FILE).exists()


def test_build_archive_writes_relative_ranks_when_parent_taxon_id_given():
    df = _make_minimal_df()
    fake_rows = [{"variable": "bio1", "metric": "mean", "position": 1,
                  "count": 2, "sampleCount": 42, "contextLabel": "Testaceae"}]
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch(
             "util.upload.compute_relative_ranks_for_upload",
             return_value=fake_rows,
         ) as mock_compute:
        archive_path, _, work_dir = up.build_archive(df, parent_taxon_id="42")
    try:
        mock_compute.assert_called_once_with(work_dir, {}, "42")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_parent_taxon_filename_suffix_uses_same_slug_convention_as_download():
    taxon = {"taxon_key": "42", "scientific_name": "Testus taxus"}
    with patch("util.upload.get_taxon_by_id", return_value=taxon):
        assert up._parent_taxon_filename_suffix("42") == "testus-taxus-42"


def test_parent_taxon_filename_suffix_none_for_unknown_taxon():
    with patch("util.upload.get_taxon_by_id", return_value=None):
        assert up._parent_taxon_filename_suffix("999") is None


def test_build_archive_names_zip_with_parent_taxon_suffix():
    df = _make_minimal_df()
    taxon = {"taxon_key": "42", "scientific_name": "Testus taxus"}
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch("util.upload.get_taxon_by_id", return_value=taxon), \
         patch("util.upload.compute_relative_ranks_for_upload", return_value=None):
        archive_path, archive_name, work_dir = up.build_archive(
            df, parent_taxon_id="42",
        )
    try:
        assert archive_name == "processed_observations-testus-taxus-42.zip"
        assert archive_path.name == archive_name
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_keeps_plain_name_for_unresolvable_parent_taxon_id():
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch("util.upload.get_taxon_by_id", return_value=None):
        archive_path, archive_name, work_dir = up.build_archive(
            df, parent_taxon_id="does-not-exist",
        )
    try:
        assert archive_name == "processed_observations.zip"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_merges_custom_layer_metadata_into_layer_meta():
    df = _make_minimal_df()
    custom_rows = [{
        "id": "my_layer", "name": "My Layer", "units": None,
        "imperial_unit": None, "value_type": "ratio", "domain": "continuous",
        "category": "Custom Layers", "group": None, "group_label": None,
        "sort_order": 20000, "render_min": None, "render_max": None,
        "legend_classes": None, "_legend_key": "my_layer",
    }]
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df") as mock_process:
        archive_path, _, work_dir = up.build_archive(
            df, custom_layer_metadata=custom_rows,
        )
    try:
        layer_meta_arg = mock_process.call_args[0][2]
        assert layer_meta_arg["my_layer"] == custom_rows[0]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_archive_skips_ranking_without_parent_taxon_id():
    df = _make_minimal_df()
    with patch("util.upload._build_layer_meta", return_value={}), \
         patch("util.upload._filter_df", side_effect=lambda d: d), \
         patch("util.upload.process_observations_df"), \
         patch("util.upload.compute_relative_ranks_for_upload") as mock_compute:
        archive_path, _, work_dir = up.build_archive(df)
    try:
        mock_compute.assert_not_called()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _load_legend_full
# ---------------------------------------------------------------------------

def test_load_legend_full_missing_file_returns_empty_dict(tmp_path):
    with patch("util.upload._LEGEND_DIR", tmp_path):
        assert up._load_legend_full("nonexistent") == {}


def test_load_legend_full_returns_whole_document(tmp_path):
    legend = {
        "classes": [{"id": 1, "name": "Forest"}],
        "attribute_axes": {"forest": [{"values": ["temperate"]}]},
    }
    (tmp_path / "landcover_legend.json").write_text(json.dumps(legend))
    with patch("util.upload._LEGEND_DIR", tmp_path):
        result = up._load_legend_full("landcover")
    assert result == legend


# ---------------------------------------------------------------------------
# _build_location_counts_table
# ---------------------------------------------------------------------------

def test_build_location_counts_table_tallies_each_level():
    df = pd.DataFrame({
        "level0Gid": ["USA", "USA", "USA"],
        "level1Gid": ["USA.CA", "USA.CA", "USA.OR"],
        "level2Gid": [None, None, None],
    })
    table = up._build_location_counts_table(df)
    rows = {(r["scope"], r["gid"]): r["count"] for r in table.to_pylist()}
    assert rows[("gadm_level0", "USA")] == 3
    assert rows[("gadm_level1", "USA.CA")] == 2
    assert rows[("gadm_level1", "USA.OR")] == 1
    assert ("gadm_level2", None) not in rows


def test_build_location_counts_table_none_when_no_gid_columns():
    df = pd.DataFrame({"catalogNumber": ["A", "B"]})
    assert up._build_location_counts_table(df) is None


# ---------------------------------------------------------------------------
# build_description_profile_for_df
# ---------------------------------------------------------------------------

def test_build_description_profile_for_df_uses_local_stats_and_locations(tmp_path):
    numerical = pa.Table.from_pylist([
        {"variable": "elevation", "min": 100.0, "max": 2000.0, "mean": 900.0},
    ])
    pq.write_table(numerical, tmp_path / up.NUMERICAL_STATS_FILE)
    nominal = pa.Table.from_pylist([
        {"variable": "kg2", "metric": "class_1", "value": 0.9},
    ])
    pq.write_table(nominal, tmp_path / up.NOMINAL_STATS_FILE)

    df = pd.DataFrame({
        "level0Gid": ["USA", "USA"],
        "level1Gid": ["USA.CA", "USA.CA"],
    })

    fake_hierarchy = {
        "USA": {"name": "United States", "level": 0, "parent_gid": None},
        "USA.CA": {"name": "California", "level": 1, "parent_gid": "USA"},
    }
    fake_kg2_legend = [{"id": 1, "name": "Tropical", "group": "tropical", "group_label": "tropical"}]

    with patch("util.upload._load_hierarchy", return_value=fake_hierarchy), \
         patch("util.upload._load_legend", side_effect=lambda lid: fake_kg2_legend if lid == "kg2" else []), \
         patch("util.upload._load_legend_full", return_value={}):
        profile = up.build_description_profile_for_df(tmp_path, df)

    assert "sections" in profile
    section_ids = {s["id"] for s in profile["sections"]}
    assert "locations" in section_ids
    assert "climate" in section_ids
    assert "terrain" in section_ids


def test_build_description_profile_for_df_empty_work_dir_still_returns_profile(tmp_path):
    df = pd.DataFrame({"catalogNumber": ["A"]})
    with patch("util.upload._load_hierarchy", return_value={}):
        profile = up.build_description_profile_for_df(tmp_path, df)
    assert profile == {"sections": []}


# ---------------------------------------------------------------------------
# _add_metadata_to_archive
# ---------------------------------------------------------------------------

def _make_empty_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("occurrence.parquet", b"placeholder")


def test_add_metadata_to_archive_noop_when_nothing_given(tmp_path):
    archive_path = tmp_path / "a.zip"
    _make_empty_zip(archive_path)
    up._add_metadata_to_archive(archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        assert "upload_metadata.json" not in zf.namelist()


def test_add_metadata_to_archive_writes_description_profile(tmp_path):
    archive_path = tmp_path / "a.zip"
    _make_empty_zip(archive_path)
    profile = {"sections": [{"id": "climate", "title": "Climates", "lines": []}]}
    up._add_metadata_to_archive(archive_path, description_profile=profile)
    with zipfile.ZipFile(archive_path) as zf:
        metadata = json.loads(zf.read("upload_metadata.json"))
    assert metadata == {"descriptionProfile": profile}


def test_add_metadata_to_archive_records_parent_taxon_id(tmp_path):
    archive_path = tmp_path / "a.zip"
    _make_empty_zip(archive_path)
    up._add_metadata_to_archive(archive_path, parent_taxon_id="6SRLS")
    with zipfile.ZipFile(archive_path) as zf:
        metadata = json.loads(zf.read("upload_metadata.json"))
    assert metadata == {"parentTaxonId": "6SRLS"}


def test_add_metadata_to_archive_embeds_uploaded_image_bytes(tmp_path):
    archive_path = tmp_path / "a.zip"
    _make_empty_zip(archive_path)
    up._add_metadata_to_archive(
        archive_path,
        image_bytes=b"\xff\xd8\xff\xe0fakejpegbytes",
        image_filename="my photo.JPG",
    )
    with zipfile.ZipFile(archive_path) as zf:
        metadata = json.loads(zf.read("upload_metadata.json"))
        assert metadata["imageFile"] == "taxon_image.JPG"
        assert zf.read("taxon_image.JPG") == b"\xff\xd8\xff\xe0fakejpegbytes"


def test_add_metadata_to_archive_stores_image_url_without_embedding(tmp_path):
    archive_path = tmp_path / "a.zip"
    _make_empty_zip(archive_path)
    up._add_metadata_to_archive(
        archive_path,
        image_url="https://example.com/photo.jpg",
        image_license="CC BY 4.0",
        image_creator="Someone",
    )
    with zipfile.ZipFile(archive_path) as zf:
        metadata = json.loads(zf.read("upload_metadata.json"))
        assert "imageFile" not in metadata
        assert metadata["imageUrl"] == "https://example.com/photo.jpg"
        assert metadata["imageLicense"] == "CC BY 4.0"
        assert metadata["imageCreator"] == "Someone"

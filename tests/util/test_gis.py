# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.ma as ma
import pytest

import util.gis as gis
from util.gis import _HILBERT_ORDER, hilbert_index


def test_hilbert_index_fits_in_int32():
    h = hilbert_index(40.0, -105.0)
    assert 0 <= h < (1 << (2 * _HILBERT_ORDER))


def test_hilbert_nearby_points_same_or_close_index():
    # Two points ~150m apart should map to the same cell
    a = hilbert_index(40.0, -105.0)
    b = hilbert_index(40.001, -105.001)
    assert a == b


def test_hilbert_distant_points_differ():
    denver = hilbert_index(40.0, -105.0)
    sydney = hilbert_index(-33.0, 151.0)
    assert abs(denver - sydney) > 1_000_000


def test_hilbert_poles_and_antimeridian():
    # Edge coordinates should not raise and stay in range
    max_val = (1 << (2 * _HILBERT_ORDER)) - 1
    for lat, lon in [(-90, -180), (90, 180), (0, 0), (-90, 180), (90, -180)]:
        h = hilbert_index(lat, lon)
        assert 0 <= h <= max_val


def test_hilbert_deterministic():
    assert hilbert_index(51.5, -0.1) == hilbert_index(51.5, -0.1)


# ---------------------------------------------------------------------------
# sample_point dispatch
# ---------------------------------------------------------------------------

_STATIC_LAYER = {
    "id": "bio1",
    "filename": "bio1.tif",
    "scale_factor": 0.1,
    "add_offset": -273.15,
    "units": "°C",
    "value_type": "interval",
}

_TEMPORAL_LAYER = {
    "id": "temperature_2m_avg_1h",
    "var_id": "temperature_2m",
    "window_hours": 1,
    "window_label": "1h",
    "model": "copernicus_era5",
    "units": "°C",
    "value_type": "interval",
}


def test_sample_point_dispatches_to_cog():
    with patch.object(gis, "_sample_cog_point", return_value=7.5) as mock:
        result = gis.sample_point(_STATIC_LAYER, 40.0, -105.0)
    mock.assert_called_once_with(_STATIC_LAYER, 40.0, -105.0)
    assert result == 7.5


def test_sample_point_dispatches_to_temporal():
    with patch.object(gis, "_sample_temporal_point", return_value=15.0) as mock:
        result = gis.sample_point(_TEMPORAL_LAYER, 40.0, -105.0)
    mock.assert_called_once_with(_TEMPORAL_LAYER, 40.0, -105.0, "")
    assert result == 15.0


# ---------------------------------------------------------------------------
# _sample_cog_point
# ---------------------------------------------------------------------------

def _make_rasterio_mock(row=50, col=50, height=100, width=100, value=2731.5,
                        dtype=np.float32, nodata=None):
    """Return a mock rasterio dataset context manager."""
    data = ma.array(np.array([[value]], dtype=dtype), mask=[[False]])
    mock_ds = MagicMock()
    mock_ds.height = height
    mock_ds.width = width
    mock_ds.nodata = nodata
    mock_ds.index.return_value = (row, col)
    mock_ds.read.return_value = data
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_ds
    mock_open.return_value.__exit__.return_value = False
    return mock_open, mock_ds


def test_cog_point_returns_scaled_value():
    mock_open, _ = _make_rasterio_mock(value=2731.5)
    with patch("util.gis.rasterio.open", mock_open):
        result = gis._sample_cog_point(_STATIC_LAYER, 40.0, -105.0)
    # scale_factor=0.1, add_offset=-273.15 → 2731.5 * 0.1 + (-273.15) = 0.0
    assert result == pytest.approx(0.0, abs=1e-6)


def test_cog_point_out_of_bounds_returns_none():
    mock_open, _ = _make_rasterio_mock(row=-1)
    with patch("util.gis.rasterio.open", mock_open):
        result = gis._sample_cog_point(_STATIC_LAYER, 91.0, 0.0)
    assert result is None


def test_cog_point_all_masked_returns_none():
    mock_ds = MagicMock()
    mock_ds.height = 100
    mock_ds.width = 100
    mock_ds.nodata = None
    mock_ds.index.return_value = (50, 50)
    mock_ds.read.return_value = ma.array(np.array([[0.0]]), mask=[[True]])
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_ds
    mock_open.return_value.__exit__.return_value = False
    with patch("util.gis.rasterio.open", mock_open):
        result = gis._sample_cog_point(_STATIC_LAYER, 40.0, -105.0)
    assert result is None


def test_cog_point_integer_nodata_sentinel_returns_none():
    mock_open, _ = _make_rasterio_mock(value=255, dtype=np.uint8, nodata=255.0)
    with patch("util.gis.rasterio.open", mock_open):
        result = gis._sample_cog_point(_STATIC_LAYER, 40.0, -105.0)
    assert result is None


def test_cog_point_integer_near_dtype_max_returns_none():
    # dtype_max=255, raw=253 → 253 >= 252 (255-3) → nodata sentinel
    mock_open, _ = _make_rasterio_mock(value=253, dtype=np.uint8, nodata=0.0)
    with patch("util.gis.rasterio.open", mock_open):
        result = gis._sample_cog_point(_STATIC_LAYER, 40.0, -105.0)
    assert result is None


def test_cog_point_exception_returns_none():
    mock_open = MagicMock(side_effect=Exception("disk read error"))
    with patch("util.gis.rasterio.open", mock_open):
        result = gis._sample_cog_point(_STATIC_LAYER, 40.0, -105.0)
    assert result is None


# ---------------------------------------------------------------------------
# _sample_temporal_point
# ---------------------------------------------------------------------------

def _era5_grid():
    return {"ny": 721, "nx": 1440, "lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0}


def _make_era5_arr(lat=40.0, lon=-105.0, value=15.0):
    arr = np.full((721, 1440), np.nan, dtype=np.float32)
    grid = _era5_grid()
    row = round((lat - grid["lat_min"]) / (grid["lat_max"] - grid["lat_min"]) * (grid["ny"] - 1))
    col = round((lon - grid["lon_min"]) / (grid["lon_max"] - grid["lon_min"]) * (grid["nx"] - 1))
    arr[row, col] = value
    return arr


def test_temporal_point_missing_npy_returns_none():
    with patch("util.gis._load_temporal_npy", return_value=None):
        result = gis._sample_temporal_point(_TEMPORAL_LAYER, 40.0, -105.0)
    assert result is None


def test_temporal_point_returns_grid_value():
    arr = _make_era5_arr(lat=40.0, lon=-105.0, value=15.0)
    with patch("util.gis._load_temporal_npy", return_value=arr), \
         patch("util.gis._apply_point_elevation_correction", side_effect=lambda val, *a, **kw: val):
        result = gis._sample_temporal_point(_TEMPORAL_LAYER, 40.0, -105.0)
    assert result == pytest.approx(15.0, abs=1e-5)


def test_temporal_point_nan_returns_none():
    arr = np.full((721, 1440), np.nan, dtype=np.float32)
    with patch("util.gis._load_temporal_npy", return_value=arr):
        result = gis._sample_temporal_point(_TEMPORAL_LAYER, 40.0, -105.0)
    assert result is None


def test_temporal_point_out_of_bounds_returns_none():
    # lat=91.0 maps to a row > 720, outside the 721-row ERA5 grid
    arr = np.full((721, 1440), 15.0, dtype=np.float32)
    with patch("util.gis._load_temporal_npy", return_value=arr):
        result = gis._sample_temporal_point(_TEMPORAL_LAYER, 91.0, 0.0)
    assert result is None


def test_temporal_point_era5_land_grid():
    arr = np.full((1801, 3600), 20.0, dtype=np.float32)
    with patch("util.gis._load_temporal_npy", return_value=arr), \
         patch("util.gis._apply_point_elevation_correction", side_effect=lambda val, *a, **kw: val):
        result = gis._sample_temporal_point(_TEMPORAL_LAYER, 40.0, -105.0)
    assert result == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# sample_point — derived elevation branch
# ---------------------------------------------------------------------------

_SLOPE_LAYER = {"id": "slope", "value_type": "ratio"}


def test_sample_point_dispatches_to_slope():
    with patch.object(gis, "compute_slope_at_point", return_value=12.5) as mock:
        result = gis.sample_point(_SLOPE_LAYER, 40.0, -105.0)
    mock.assert_called_once_with(40.0, -105.0)
    assert result == 12.5


# ---------------------------------------------------------------------------
# _apply_point_elevation_correction
# ---------------------------------------------------------------------------

def _make_elev_rasterio_mock(obs_elev=1749.9, row=50, col=50, height=100, width=100,
                              masked=False, raise_exc=False):
    import numpy.ma as ma_mod
    mock_ds = MagicMock()
    mock_ds.height = height
    mock_ds.width = width
    mock_ds.index.return_value = (row, col)
    data_val = np.array([[obs_elev]], dtype=np.float32)
    mock_ds.read.return_value = ma_mod.array(data_val, mask=[[masked]])
    if raise_exc:
        mock_ds.__enter__ = MagicMock(side_effect=Exception("io error"))
    else:
        mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_open = MagicMock(return_value=mock_ds)
    return mock_open


def test_elevation_correction_no_file(tmp_path):
    with patch.object(gis, "LAYERS_DIR", tmp_path):
        result = gis._apply_point_elevation_correction(15.0, 40.0, -105.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    assert result == 15.0


def test_elevation_correction_out_of_bounds(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = _make_elev_rasterio_mock(row=-1)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis._apply_point_elevation_correction(15.0, 91.0, 0.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    assert result == 15.0


def test_elevation_correction_masked_data(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = _make_elev_rasterio_mock(masked=True)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis._apply_point_elevation_correction(15.0, 40.0, -105.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    assert result == 15.0


def test_elevation_correction_rasterio_raises(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = MagicMock(side_effect=Exception("disk error"))
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis._apply_point_elevation_correction(15.0, 40.0, -105.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    assert result == 15.0


def test_elevation_correction_obs_below_9000(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = _make_elev_rasterio_mock(obs_elev=-9999.0)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis._apply_point_elevation_correction(15.0, 40.0, -105.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    assert result == 15.0


def test_elevation_correction_model_elev_nan(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = _make_elev_rasterio_mock(obs_elev=1749.9)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open), \
         patch("util.gis._read_model_elevation", return_value=np.array([np.nan])), \
         patch("util.gis.grid_indices", return_value=(280, 300)):
        result = gis._apply_point_elevation_correction(15.0, 40.0, -105.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    assert result == 15.0


def test_elevation_correction_applies_lapse_rate(tmp_path):
    from util.temporal import _LAPSE_RATE
    elev = tmp_path / "elevation.tif"
    elev.touch()
    obs_elev = 1749.9
    model_elev = 1600.0
    mock_open = _make_elev_rasterio_mock(obs_elev=obs_elev)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open), \
         patch("util.gis._read_model_elevation", return_value=np.array([model_elev])), \
         patch("util.gis.grid_indices", return_value=(280, 300)):
        result = gis._apply_point_elevation_correction(15.0, 40.0, -105.0, "copernicus_era5", 0.25, "lat_asc_lon_pm180")
    expected = 15.0 + (model_elev - obs_elev) * _LAPSE_RATE
    assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# _meters_per_degree
# ---------------------------------------------------------------------------

def test_meters_per_degree_equator():
    lat_m, lon_m = gis._meters_per_degree(0.0)
    assert 110_000 < lat_m < 112_000
    assert 110_000 < lon_m < 112_000


def test_meters_per_degree_pole():
    lat_m, lon_m = gis._meters_per_degree(90.0)
    assert lat_m > 0
    assert lon_m == pytest.approx(0.0, abs=1.0)


def test_meters_per_degree_returns_floats():
    lat_m, lon_m = gis._meters_per_degree(45.0)
    assert isinstance(lat_m, float)
    assert isinstance(lon_m, float)


# ---------------------------------------------------------------------------
# _horn_slope
# ---------------------------------------------------------------------------

def test_horn_slope_flat():
    w = np.ones((3, 3), dtype=np.float64) * 100.0
    slope = gis._horn_slope(w, dx_m=30.0, dy_m=30.0)
    assert slope == pytest.approx(0.0, abs=1e-10)


def test_horn_slope_inclined():
    # Linearly rising surface in x direction: 1 m per 30 m → ~1.91°
    w = np.array([[0., 0., 0.],
                  [1., 1., 1.],
                  [2., 2., 2.]], dtype=np.float64).T
    slope = gis._horn_slope(w, dx_m=30.0, dy_m=30.0)
    assert 1.0 < slope < 3.0


# ---------------------------------------------------------------------------
# compute_slope_at_point
# ---------------------------------------------------------------------------

def test_compute_slope_at_point_delegates():
    with patch.object(gis, "sample_slope_batch", return_value=[22.5]) as mock:
        result = gis.compute_slope_at_point(40.0, -105.0)
    mock.assert_called_once()
    assert result == 22.5


# ---------------------------------------------------------------------------
# sample_slope_batch
# ---------------------------------------------------------------------------

def _make_slope_rasterio_mock(patch_data=None, row=50, col=50,
                               height=200, width=200, nodata=None,
                               pixel_deg=0.000277778):

    if patch_data is None:
        patch_data = np.zeros((3, 3), dtype=np.float32)

    mock_ds = MagicMock()
    mock_ds.height = height
    mock_ds.width = width
    mock_ds.nodata = nodata
    mock_ds.transform = MagicMock()
    mock_ds.transform.a = pixel_deg
    mock_ds.index.return_value = (row, col)
    mock_ds.read.return_value = patch_data
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_ds)


def test_sample_slope_batch_empty(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    with patch.object(gis, "LAYERS_DIR", tmp_path):
        result = gis.sample_slope_batch(np.array([]), np.array([]))
    assert result == []


def test_sample_slope_batch_no_file(tmp_path):
    with patch.object(gis, "LAYERS_DIR", tmp_path):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_rasterio_raises(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = MagicMock(side_effect=Exception("disk error"))
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_out_of_bounds(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_open = _make_slope_rasterio_mock(row=0)  # row < 1 → skip
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_nodata_sentinel(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    patch_data = np.full((3, 3), -9999.0, dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data, nodata=-9999.0)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_nonfinite_patch(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    patch_data = np.full((3, 3), np.nan, dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_wrong_patch_shape(tmp_path):
    # read() returns something other than 3×3 (e.g. edge of raster)
    elev = tmp_path / "elevation.tif"
    elev.touch()
    patch_data = np.zeros((2, 3), dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_success(tmp_path):
    elev = tmp_path / "elevation.tif"
    elev.touch()
    # Flat surface → slope = 0
    patch_data = np.zeros((3, 3), dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_sample_slope_batch_zero_pixel_deg(tmp_path):
    # pixel_deg=0 → dx_m=0 → skip
    elev = tmp_path / "elevation.tif"
    elev.touch()
    patch_data = np.zeros((3, 3), dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data, pixel_deg=0.0)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


def test_sample_slope_batch_index_raises(tmp_path):
    # ds.index raises an exception for this point → caught, continue
    elev = tmp_path / "elevation.tif"
    elev.touch()
    mock_ds = MagicMock()
    mock_ds.height = 200
    mock_ds.width = 200
    mock_ds.nodata = None
    mock_ds.transform.a = 0.000277778
    mock_ds.index.side_effect = Exception("bad coord")
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_open = MagicMock(return_value=mock_ds)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result == [None]


# ---------------------------------------------------------------------------
# derive_slope_array
# ---------------------------------------------------------------------------

def test_derive_slope_array_all_nan():
    from rasterio.transform import from_bounds
    dem = np.full((4, 4), np.nan, dtype=np.float32)
    tf = from_bounds(-106.0, 39.0, -105.0, 40.0, 4, 4)
    result = gis.derive_slope_array(dem, tf)
    assert result.shape == (4, 4)
    assert np.all(np.isnan(result))


def test_derive_slope_array_flat():
    from rasterio.transform import from_bounds
    dem = np.full((8, 8), 1000.0, dtype=np.float32)
    tf = from_bounds(-106.0, 39.0, -105.0, 40.0, 8, 8)
    result = gis.derive_slope_array(dem, tf)
    assert result.dtype == np.float32
    assert result.shape == (8, 8)
    assert np.all(result == pytest.approx(0.0, abs=1e-4))


def test_derive_slope_array_nan_propagates():
    from rasterio.transform import from_bounds
    dem = np.full((8, 8), 500.0, dtype=np.float32)
    dem[3, 3] = np.nan
    tf = from_bounds(-106.0, 39.0, -105.0, 40.0, 8, 8)
    result = gis.derive_slope_array(dem, tf)
    assert np.isnan(result[3, 3])

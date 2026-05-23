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
    mock.assert_called_once_with(_TEMPORAL_LAYER, 40.0, -105.0)
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
    with patch("util.gis._load_temporal_npy", return_value=arr):
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
    with patch("util.gis._load_temporal_npy", return_value=arr):
        result = gis._sample_temporal_point(_TEMPORAL_LAYER, 40.0, -105.0)
    assert result == pytest.approx(20.0)

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.ma as ma
import pytest

import util.gis as gis
from util.gis import _HILBERT_ORDER, hilbert_index


@pytest.fixture(autouse=True)
def _clear_dataset_cache():
    """Cached dataset handles (util.gis._cached_open) persist across calls
    within a process, so tests must start each case with an empty cache —
    otherwise a mock installed in one test could leak into the next."""
    gis.clear_dataset_cache()
    yield
    gis.clear_dataset_cache()


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
    """Return a mock rasterio dataset (cached opener uses the handle directly, no context manager)."""
    data = ma.array(np.array([[value]], dtype=dtype), mask=[[False]])
    mock_ds = MagicMock()
    mock_ds.height = height
    mock_ds.width = width
    mock_ds.nodata = nodata
    mock_ds.index.return_value = (row, col)
    mock_ds.read.return_value = data
    mock_open = MagicMock(return_value=mock_ds)
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
    mock_open = MagicMock(return_value=mock_ds)
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


# ---------------------------------------------------------------------------
# Horn gradient — gdaldem parity, orientation, edges
# ---------------------------------------------------------------------------

def test_horn_matches_gdaldem_reference():
    # Values checked against `gdaldem slope/aspect -compute_edges` on a
    # 30 m projected grid (Horn 1981 kernel).
    w = np.array([[10., 12., 15.],
                  [ 9., 11., 14.],
                  [ 7.,  9., 13.]])
    assert gis._horn_slope(w, 30.0, 30.0) == pytest.approx(5.6412, abs=1e-3)
    assert gis._horn_aspect(w, 30.0, 30.0) == pytest.approx(242.354, abs=1e-2)


@pytest.mark.parametrize(("window", "expected"), [
    # Rising toward the east faces west (downslope bearing), etc.
    (np.array([[0., 1., 2.]] * 3), 270.0),
    (np.array([[2., 1., 0.]] * 3), 90.0),
    (np.array([[2.] * 3, [1.] * 3, [0.] * 3]), 180.0),
    (np.array([[0.] * 3, [1.] * 3, [2.] * 3]), 0.0),
])
def test_horn_aspect_cardinal_directions(window, expected):
    aspect = gis._horn_aspect(window, 30.0, 30.0)
    assert (aspect - expected + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)


def test_horn_aspect_flat_is_none():
    assert gis._horn_aspect(np.full((3, 3), 5.0), 30.0, 30.0) is None


def test_sample_slope_batch_nodata_neighbour_uses_centre(tmp_path):
    # A nodata neighbour (e.g. ocean next to a coastal cell) is replaced by the
    # centre value, not dropped and not filled with something far away.
    elev = tmp_path / "elevation.tif"
    elev.touch()
    patch_data = np.array([[-9999., 5., 5.],
                           [5.,     5., 5.],
                           [5.,     5., 5.]], dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data, nodata=-9999.0)
    with patch.object(gis, "LAYERS_DIR", tmp_path), \
         patch("util.gis.rasterio.open", mock_open):
        result = gis.sample_slope_batch(np.array([40.0]), np.array([-105.0]))
    assert result[0] == pytest.approx(0.0, abs=1e-9)


def test_derive_slope_array_no_fake_cliff_at_nodata_edge():
    # Flat coastal plain at 1 m beside nodata ocean: slope must stay 0, even
    # though the tile's other valid cells sit far higher.
    from rasterio.transform import from_bounds
    dem = np.full((8, 8), 1.0, dtype=np.float32)
    dem[:, :3] = np.nan
    dem[:, 6:] = 3000.0
    tf = from_bounds(-106.0, 39.0, -105.0, 40.0, 8, 8)
    result = gis.derive_slope_array(dem, tf)
    assert result[:, 3] == pytest.approx(0.0, abs=1e-6)


def test_derive_slope_array_uses_per_row_latitude():
    # Same east-west ramp in cells at 0° and 70° N: the 70° row has cells about
    # a third as wide, so its slope must be steeper — a single tile-centre
    # latitude would give both rows the same value.
    from rasterio.transform import from_origin
    dem = np.tile(np.arange(5, dtype=np.float32) * 10.0, (80, 1))
    tf = from_origin(0.0, 80.0, 0.001, 1.0)  # rows span 80°N → 0°
    result = gis.derive_slope_array(dem, tf)
    assert result[10, 2] > result[79, 2] * 2


def test_derive_aspect_array_flat_is_nan():
    from rasterio.transform import from_bounds
    dem = np.full((6, 6), 250.0, dtype=np.float32)
    tf = from_bounds(-106.0, 39.0, -105.0, 40.0, 6, 6)
    assert np.all(np.isnan(gis.derive_aspect_array(dem, tf)))


# ---------------------------------------------------------------------------
# heat load (McCune & Keon 2002)
# ---------------------------------------------------------------------------

def _hl_hand(slope, aspect, lat):
    """Independent hand evaluation of McCune & Keon (2002), northern fold."""
    s, la = math.radians(slope), math.radians(lat)
    a = math.radians(abs(180 - abs(aspect - 225)))
    return math.exp(-1.467 + 1.582 * math.cos(la) * math.cos(s)
                    - 1.5 * math.cos(a) * math.sin(s) * math.sin(la)
                    - 0.262 * math.sin(la) * math.sin(s)
                    + 0.607 * math.sin(a) * math.sin(s))


@pytest.mark.parametrize(("slope", "aspect", "lat", "expected"), [
    (0.0, 0.0, 40.0, 0.7752),     # flat
    (45.0, 45.0, 40.0, 0.2440),   # steep NE face: coolest
    (45.0, 225.0, 40.0, 0.9541),  # steep SW face: warmest
])
def test_heat_load_reference_values(slope, aspect, lat, expected):
    got = float(gis._heat_load(slope, aspect, lat))
    assert got == pytest.approx(expected, abs=1e-3)
    assert got == pytest.approx(_hl_hand(slope, aspect, lat), rel=1e-9)


def test_heat_load_flat_ignores_undefined_aspect():
    assert float(gis._heat_load(0.0, np.nan, 40.0)) == pytest.approx(float(gis._heat_load(0.0, 123.0, 40.0)))


def test_heat_load_southern_hemisphere_mirrors_north():
    # NW is the warm face south of the equator, as SW is north of it.
    assert float(gis._heat_load(30.0, 315.0, -40.0)) == pytest.approx(float(gis._heat_load(30.0, 225.0, 40.0)))
    assert float(gis._heat_load(30.0, 135.0, -40.0)) == pytest.approx(float(gis._heat_load(30.0, 45.0, 40.0)))


def test_heat_load_nan_slope_is_nan():
    assert np.isnan(gis._heat_load(np.nan, 90.0, 40.0))


def test_derive_heat_load_array_matches_slope_aspect_and_row_latitude():
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    dem = rng.normal(1000, 50, (6, 5))
    dem[2, 2] = np.nan
    tf = from_origin(-111.0, 61.0, 0.000277778, 4.0)  # rows span 61°N → 37°N
    hl = gis.derive_heat_load_array(dem, tf)
    slope = gis.derive_slope_array(dem, tf)
    aspect = gis.derive_aspect_array(dem, tf)
    row_lat = 61.0 - (np.arange(6) + 0.5) * 4.0
    expected = gis._heat_load(slope.astype(np.float64), aspect.astype(np.float64), row_lat[:, None])
    assert np.isnan(hl[2, 2])
    np.testing.assert_allclose(hl, expected, rtol=1e-5)


def test_sample_heat_load_batch_matches_slope_and_aspect(tmp_path):
    (tmp_path / "elevation.tif").touch()
    patch_data = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 60]], dtype=np.float32)
    mock_open = _make_slope_rasterio_mock(patch_data=patch_data)
    with patch.object(gis, "LAYERS_DIR", tmp_path), patch("util.gis.rasterio.open", mock_open):
        out = gis.sample_elevation_terrain_batch(
            np.array([40.0]), np.array([-105.0]), want_slope=True, want_aspect=True, want_heat_load=True,
        )
        single = gis.compute_heat_load_at_point(40.0, -105.0)
    expected = float(gis._heat_load(out["slope"][0], out["aspect"][0], 40.0))
    assert out["heat_load"][0] == pytest.approx(expected)
    assert single == pytest.approx(expected)


def test_sample_point_dispatches_heat_load():
    with patch.object(gis, "compute_heat_load_at_point", return_value=0.5) as mock:
        assert gis.sample_point({"id": "heat_load"}, 40.0, -105.0) == 0.5
    mock.assert_called_once_with(40.0, -105.0)


def test_heat_load_uses_ground_distance_not_degrees_at_high_latitude():
    # A true 10° east-facing plane on a 1″ geographic grid at 70°N, where a
    # cell is ~3× narrower east-west than north-south. Without per-row metre
    # conversion the slope would read ~27° and heat load would be wrong.
    from rasterio.transform import from_origin

    lat0, px = 70.0, 1 / 3600
    tf = from_origin(20.0, lat0 + 4 * px, px, px)
    row_lat = lat0 + 4 * px - (np.arange(8) + 0.5) * px
    _, m_lon = gis._meters_per_degree_np(row_lat)
    cols = np.arange(8)
    # Elevation falls eastward by tan(10°) per metre of ground distance.
    dem = 500.0 - np.tan(np.radians(10.0)) * (cols[None, :] * px * m_lon[:, None])
    hl = gis.derive_heat_load_array(dem, tf)[2:-2, 2:-2]
    expected = gis._heat_load(10.0, 90.0, row_lat[2:-2])[:, None]
    np.testing.assert_allclose(hl, np.broadcast_to(expected, hl.shape), rtol=1e-4)
    np.testing.assert_allclose(gis.derive_slope_array(dem, tf)[2:-2, 2:-2], 10.0, atol=1e-3)


# ---------------------------------------------------------------------------
# plan / profile curvature
# ---------------------------------------------------------------------------

def _quadratic_dem(fn, lat0=60.0, n=9, px=1 / 3600):
    """Sample z = fn(x_m, y_m) on a 1″ geographic grid centred on lat0, with
    x/y measured in ground metres (east/north) from the centre cell."""
    from rasterio.transform import from_origin

    tf = from_origin(10.0, lat0 + (n / 2) * px, px, px)
    row_lat = lat0 + (n / 2) * px - (np.arange(n) + 0.5) * px
    m_lat, m_lon = gis._meters_per_degree_np(row_lat)
    c = n // 2
    x = (np.arange(n)[None, :] - c) * px * m_lon[:, None]
    y = (c - np.arange(n)[:, None]) * px * m_lat[c]
    return fn(x, y), tf, c


def test_profile_curvature_matches_analytic_on_geographic_grid():
    # z = a·x + b·x²: rises eastward and steepens; flow runs west and
    # flattens out downhill, so the profile is concave (negative).
    a, b = 0.3, 0.002
    dem, tf, c = _quadratic_dem(lambda x, y: a * x + b * x * x)
    expected = -(2 * b) / (1 + a * a) ** 1.5
    assert gis.derive_profile_curvature_array(dem, tf)[c, c] == pytest.approx(expected, rel=1e-3)
    assert gis.derive_plan_curvature_array(dem, tf)[c, c] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("c2", "sign"), [(0.004, -1), (-0.004, 1)])
def test_plan_curvature_matches_analytic_on_geographic_grid(c2, sign):
    # z = a·x + c·y²: an east-west trough (c > 0, flow converges, negative)
    # or spur (c < 0, flow diverges, positive) along y = 0.
    a = 0.3
    dem, tf, c = _quadratic_dem(lambda x, y: a * x + c2 * y * y)
    plan = gis.derive_plan_curvature_array(dem, tf)[c, c]
    assert plan == pytest.approx(-2 * c2 / np.sqrt(1 + a * a), rel=1e-3)
    assert np.sign(plan) == sign
    assert gis.derive_profile_curvature_array(dem, tf)[c, c] == pytest.approx(0.0, abs=1e-9)


def test_curvature_undefined_on_flat_and_nodata():
    from rasterio.transform import from_origin

    dem = np.full((5, 5), 100.0)
    dem[0, 0] = np.nan
    tf = from_origin(0.0, 40.0, 1 / 3600, 1 / 3600)
    for fn in (gis.derive_plan_curvature_array, gis.derive_profile_curvature_array):
        out = fn(dem, tf)
        assert np.isnan(out).all()


def test_sample_curvature_batch_matches_array_path(tmp_path):
    a, b, c2 = 0.3, 0.002, 0.004
    dem, tf, c = _quadratic_dem(lambda x, y: a * x + b * x * x + c2 * y * y, lat0=45.0, n=3)
    (tmp_path / "elevation.tif").touch()
    mock_open = _make_slope_rasterio_mock(patch_data=dem.astype(np.float64), pixel_deg=1 / 3600)
    with patch.object(gis, "LAYERS_DIR", tmp_path), patch("util.gis.rasterio.open", mock_open):
        out = gis.sample_elevation_terrain_batch(
            np.array([45.0]), np.array([10.0]), want_plan_curvature=True, want_profile_curvature=True,
        )
    assert out["plan_curvature"][0] == pytest.approx(float(gis.derive_plan_curvature_array(dem, tf)[1, 1]), rel=1e-6)
    assert out["profile_curvature"][0] == pytest.approx(
        float(gis.derive_profile_curvature_array(dem, tf)[1, 1]), rel=1e-6,
    )


def test_every_elevation_derived_layer_has_an_array_deriver():
    assert gis.ELEVATION_DERIVERS.keys() == gis.DERIVED_FROM_ELEVATION


# ---------------------------------------------------------------------------
# convergence index
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("fn", "expected"), [
    (lambda x, y: (x * x + y * y) / 1000.0, -100.0),   # pit: every neighbour drains in
    (lambda x, y: -(x * x + y * y) / 1000.0, 100.0),   # peak: every neighbour drains out
])
def test_convergence_index_pit_and_peak(fn, expected):
    dem, tf, c = _quadratic_dem(fn)
    assert gis.derive_convergence_array(dem, tf)[c, c] == pytest.approx(expected, abs=1e-3)


def test_convergence_index_uses_ground_geometry_on_non_square_cells():
    # A trough (z = a·x + c·y²) at 60°N, where a 1″ cell is ~2× taller than
    # wide. Checked against an independent evaluation from the exact
    # gradient at each neighbour's true ground position and the true bearing
    # back to the centre. Treating cells as square gives ~-41 instead of ~-33.
    a, c2 = 0.3, 0.004
    dem, tf, c = _quadratic_dem(lambda x, y: a * x + c2 * y * y)
    px = 1 / 3600
    row_lat = 60.0 + 4.5 * px - (np.arange(9) + 0.5) * px
    m_lat, m_lon = gis._meters_per_degree_np(row_lat)
    angles = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            x = dc * px * m_lon[c + dr]
            y = -dr * px * m_lat[c]
            downslope = np.degrees(np.arctan2(-a, -2 * c2 * y)) % 360
            to_centre = np.degrees(np.arctan2(-x, -y)) % 360
            angles.append(abs((downslope - to_centre + 180) % 360 - 180))
    expected = (np.mean(angles) - 90) * 100 / 90
    assert gis.derive_convergence_array(dem, tf)[c, c] == pytest.approx(expected, abs=1e-3)
    assert expected == pytest.approx(-32.94, abs=0.01)


def test_convergence_index_undefined_without_any_neighbour_aspect():
    from rasterio.transform import from_origin

    tf = from_origin(0.0, 40.0, 1 / 3600, 1 / 3600)
    assert np.isnan(gis.derive_convergence_array(np.full((5, 5), 50.0), tf)).all()


def test_sample_convergence_batch_reads_5x5_and_matches_array_path(tmp_path):
    dem, tf, c = _quadratic_dem(lambda x, y: 0.3 * x + 0.004 * y * y, lat0=45.0, n=5)
    (tmp_path / "elevation.tif").touch()
    mock_open = _make_slope_rasterio_mock(patch_data=dem.astype(np.float64), pixel_deg=1 / 3600)
    with patch.object(gis, "LAYERS_DIR", tmp_path), patch("util.gis.rasterio.open", mock_open):
        out = gis.sample_elevation_terrain_batch(
            np.array([45.0]), np.array([10.0]), want_convergence=True, want_slope=True,
        )
    window = mock_open.return_value.read.call_args.kwargs["window"]
    assert (window.width, window.height) == (5, 5)
    assert out["convergence"][0] == pytest.approx(float(gis.derive_convergence_array(dem, tf)[2, 2]), rel=1e-4)
    assert out["slope"][0] == pytest.approx(float(gis.derive_slope_array(dem, tf)[2, 2]), rel=1e-4)


def test_sample_point_dispatches_convergence():
    with patch.object(gis, "compute_convergence_at_point", return_value=-12.0) as mock:
        assert gis.sample_point({"id": "convergence"}, 40.0, -105.0) == -12.0
    mock.assert_called_once_with(40.0, -105.0)

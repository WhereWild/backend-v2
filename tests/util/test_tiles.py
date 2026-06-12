import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import util.tiles as tiles

# Capture the real underlying function before any fixture patches tiles._catalog.
# lru_cache (via update_wrapper) sets __wrapped__ to the original callable.
_real_catalog_fn = tiles._catalog.__wrapped__

# ---------------------------------------------------------------------------
# Minimal fake catalog
# ---------------------------------------------------------------------------

FAKE_CATALOG = {
    "categories": [
        {
            "id": "bioclimate",
            "display_name": "Bioclimatic",
            "layers": [
                {
                    "id": "bio1",
                    "display_name": "Annual Mean Temperature",
                    "filename": "bio1.tif",
                    "source": "chelsa_v2_1",
                    "units": "°C",
                    "value_type": "interval",
                    "scale_factor": 0.1,
                    "add_offset": -273.15,
                    "render_min": -50.0,
                    "render_max": 35.0,
                },
                {
                    "id": "bio2",
                    "display_name": "No render range",
                    "filename": "bio2.tif",
                    "source": "chelsa_v2_1",
                    "units": "°C",
                    "value_type": "ratio",
                    "scale_factor": 0.1,
                    "add_offset": 0.0,
                    "render_min": None,
                    "render_max": None,
                },
                {
                    "id": "kg0",
                    "display_name": "Köppen-Geiger Classification",
                    "filename": "kg0.tif",
                    "source": "chelsa_v2_1",
                    "units": "",
                    "value_type": "nominal",
                    "scale_factor": None,
                    "add_offset": None,
                    "render_min": 1.0,
                    "render_max": 31.0,
                },
            ],
        }
    ]
}


@pytest.fixture
def patch_catalog():
    tiles._catalog.cache_clear()
    with patch.object(tiles, "_catalog", return_value=FAKE_CATALOG):
        yield
    tiles._catalog.cache_clear()


@pytest.fixture(autouse=True)
def _auto_patch_catalog(patch_catalog):
    pass


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def test_load_layers_returns_all():
    layers = tiles.load_layers()
    assert len(layers) == 3
    assert layers[0]["id"] == "bio1"
    assert layers[-1]["id"] == "kg0"


def test_load_layers_with_category():
    pairs = tiles.load_layers_with_category()
    assert len(pairs) == 3
    layer, category = pairs[0]
    assert layer["id"] == "bio1"
    assert category["id"] == "bioclimate"


def test_get_layer_found():
    layer = tiles.get_layer("bio1")
    assert layer["display_name"] == "Annual Mean Temperature"


def test_get_layer_not_found():
    with pytest.raises(KeyError, match="bio99"):
        tiles.get_layer("bio99")


# ---------------------------------------------------------------------------
# Temporal layer expansion
# ---------------------------------------------------------------------------

_TEMPORAL_CATALOG = {
    "categories": [
        {
            "id": "temporal",
            "display_name": "Weather",
            "windows": [24, 168],
            "layers": [
                {
                    "id": "temperature_2m",
                    "display_name": "Air Temperature (2m)",
                    "units": "°C",
                    "value_type": "interval",
                    "agg": "avg",
                },
                {
                    "id": "weather_code_simple",
                    "display_name": "Weather Code",
                    "units": "",
                    "value_type": "nominal",
                    "agg": "mode",
                    "windows": [24],
                },
            ],
        }
    ]
}


@pytest.fixture
def patch_temporal_catalog():
    tiles._catalog.cache_clear()
    with patch.object(tiles, "_catalog", return_value=_TEMPORAL_CATALOG):
        yield
    tiles._catalog.cache_clear()


def test_temporal_layers_expanded(patch_temporal_catalog):
    layers = tiles.load_layers()
    ids = [lay["id"] for lay in layers]
    assert "temperature_2m_avg_24h" in ids
    assert "temperature_2m_avg_168h" in ids
    assert len([lay for lay in layers if lay["id"].startswith("temperature_2m")]) == 2


def test_temporal_layer_display_name(patch_temporal_catalog):
    layers = tiles.load_layers()
    t24 = next(lay for lay in layers if lay["id"] == "temperature_2m_avg_24h")
    assert t24["display_name"] == "Air Temperature (2m)"
    assert t24["window_hours"] == 24
    assert t24["window_label"] == "24h"


def test_temporal_layer_inherits_value_type(patch_temporal_catalog):
    layers = tiles.load_layers()
    t24 = next(lay for lay in layers if lay["id"] == "temperature_2m_avg_24h")
    assert t24["value_type"] == "interval"
    assert t24["units"] == "°C"


def test_temporal_layer_window_override(patch_temporal_catalog):
    layers = tiles.load_layers()
    mode_ids = [lay["id"] for lay in layers if lay["id"].startswith("weather_code_simple")]
    assert mode_ids == ["weather_code_simple_mode_24h"]


def test_temporal_layers_with_category(patch_temporal_catalog):
    pairs = tiles.load_layers_with_category()
    categories = {cat["id"] for _, cat in pairs}
    assert "temporal" in categories


# ---------------------------------------------------------------------------
# Tile bounds
# ---------------------------------------------------------------------------

def test_tile_bounds_mercator_z0():
    half = 2 * math.pi * 6378137 / 2.0
    x0, y0, x1, y1 = tiles.tile_bounds_mercator(0, 0, 0)
    assert pytest.approx(x0, abs=1) == -half
    assert pytest.approx(x1, abs=1) == half
    assert pytest.approx(y0, abs=1) == -half
    assert pytest.approx(y1, abs=1) == half


def test_tile_bounds_mercator_subdivides():
    parent = tiles.tile_bounds_mercator(0, 0, 0)
    tl = tiles.tile_bounds_mercator(1, 0, 0)
    tr = tiles.tile_bounds_mercator(1, 1, 0)
    assert pytest.approx(tl[0]) == parent[0]
    assert pytest.approx(tr[2]) == parent[2]
    assert pytest.approx(tl[2]) == tr[0]


def test_tile_bounds_wgs84_z0_covers_globe():
    lon0, lat0, lon1, lat1 = tiles.tile_bounds_wgs84(0, 0, 0)
    assert lon0 < -170
    assert lon1 > 170
    assert lat0 < -80
    assert lat1 > 80


def test_tile_bounds_wgs84_within_mercator():
    lon0, lat0, lon1, lat1 = tiles.tile_bounds_wgs84(4, 8, 5)
    assert -180 <= lon0 < lon1 <= 180
    assert -90 <= lat0 < lat1 <= 90


# ---------------------------------------------------------------------------
# Colorize
# ---------------------------------------------------------------------------

def test_colorize_all_nan():
    values = np.full((4, 4), np.nan, dtype=np.float32)
    rgba = tiles._colorize(values, 0.0, 1.0)
    assert rgba.shape == (4, 4, 4)
    assert np.all(rgba[:, :, 3] == 0)


def test_colorize_uniform():
    values = np.full((4, 4), 0.5, dtype=np.float32)
    rgba = tiles._colorize(values, 0.0, 1.0)
    assert np.all(rgba[:, :, 3] > 0)
    # All pixels should have identical color
    assert np.all(rgba[:, :, 0] == rgba[0, 0, 0])


def test_colorize_min_max_alpha():
    values = np.array([[0.0, 1.0]], dtype=np.float32)
    rgba = tiles._colorize(values, 0.0, 1.0)
    assert rgba[0, 0, 3] == 255
    assert rgba[0, 1, 3] == 255


def test_colorize_clamps_out_of_range():
    values = np.array([[-1.0, 2.0]], dtype=np.float32)
    rgba = tiles._colorize(values, 0.0, 1.0)
    assert rgba.shape == (1, 2, 4)


def test_colorize_mixed_nan():
    values = np.array([[np.nan, 0.5]], dtype=np.float32)
    rgba = tiles._colorize(values, 0.0, 1.0)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] > 0


# ---------------------------------------------------------------------------
# render_layer_tile_bytes
# ---------------------------------------------------------------------------

def _make_mock_ds(raw: np.ndarray, nodata=65535.0, scales=(0.1,), offsets=(0.0,)):
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    ds = MagicMock()
    ds.__enter__ = lambda s: s
    ds.__exit__ = MagicMock(return_value=False)
    ds.dtypes = ["uint16"]
    ds.nodata = nodata
    ds.scales = scales
    ds.offsets = offsets
    ds.crs = CRS.from_epsg(4326)
    ds.overviews = MagicMock(return_value=[])
    ds.width = raw.shape[1]
    ds.height = raw.shape[0]

    from rasterio.coords import BoundingBox
    ds.bounds = BoundingBox(-180, -90, 180, 90)
    ds.transform = from_bounds(-180, -90, 180, 90, raw.shape[1], raw.shape[0])

    def _read(band, window=None, out_shape=None, resampling=None):
        return raw

    ds.read = _read
    return ds


def test_render_tile_returns_png():
    raw = np.full((4, 4), 2731, dtype=np.uint16)
    mock_ds = _make_mock_ds(raw)
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio1", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_nodata_masked():
    raw = np.full((4, 4), 65535, dtype=np.uint16)
    mock_ds = _make_mock_ds(raw, nodata=65535.0)
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio1", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_nominal_nearest():
    raw = np.arange(1, 17, dtype=np.uint16).reshape(4, 4)
    mock_ds = _make_mock_ds(raw, nodata=65535.0, scales=(1.0,), offsets=(0.0,))
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("kg0", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_out_of_bounds_returns_blank():
    raw = np.full((4, 4), 2731, dtype=np.uint16)
    mock_ds = _make_mock_ds(raw)
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio1", z=0, x=0, y=0, tile_size=32)
    assert result[:4] == b"\x89PNG"


def test_render_tile_nodata_not_ceiling():
    # nd_int != dtype_max branch (line 188)
    raw = np.array([[100, 9999, 200]], dtype=np.uint16)
    mock_ds = _make_mock_ds(raw, nodata=9999.0)
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio1", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_float_dtype():
    # float dtype branch (lines 192-194)
    raw = np.array([[1.5, -9999.0, 2.5]], dtype=np.float32)
    mock_ds = _make_mock_ds(raw, nodata=-9999.0, scales=(1.0,), offsets=(0.0,))
    mock_ds.dtypes = ["float32"]
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio1", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_null_render_range_computed():
    # render_min/max None → auto-computed from data (lines 199, 201)
    raw = np.array([[1000, 2000, 3000, 4000]], dtype=np.uint16)
    mock_ds = _make_mock_ds(raw, nodata=65535.0)
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio2", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_null_render_range_all_nodata():
    # render_min/max None + all pixels NaN → fallback 0.0/1.0 (lines 199, 201 else branch)
    raw = np.full((4, 4), 65535, dtype=np.uint16)
    mock_ds = _make_mock_ds(raw, nodata=65535.0)
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio2", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


def test_render_tile_overview_path():
    # overview branch: large raster width + overviews list (lines 165-168)
    raw = np.full((4, 4), 2731, dtype=np.uint16)
    mock_ds = _make_mock_ds(raw)
    mock_ds.width = 43200
    mock_ds.height = 21600
    mock_ds.overviews = MagicMock(return_value=[2, 4, 8, 16, 32])
    with patch("rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("bio1", z=2, x=2, y=1, tile_size=64)
    assert result[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# _render_derived_elevation_tile_bytes / render_layer_tile_bytes derived path
# ---------------------------------------------------------------------------

_SLOPE_CATALOG = {
    "categories": [
        {
            "id": "terrain",
            "display_name": "Terrain",
            "layers": [
                {
                    "id": "slope",
                    "display_name": "Slope",
                    "filename": None,
                    "derived": True,
                    "units": "°",
                    "value_type": "ratio",
                    "scale_factor": None,
                    "add_offset": None,
                    "render_min": 0.0,
                    "render_max": 90.0,
                }
            ],
        }
    ]
}


@pytest.fixture
def patch_slope_catalog():
    tiles._catalog.cache_clear()
    with patch.object(tiles, "_catalog", return_value=_SLOPE_CATALOG):
        yield
    tiles._catalog.cache_clear()


def test_render_derived_tile_no_elevation_file(patch_slope_catalog, tmp_path):
    # elev_path does not exist → returns transparent PNG without opening any file
    with patch.object(tiles, "LAYERS_DIR", tmp_path):
        result = tiles.render_layer_tile_bytes("slope", z=4, x=3, y=5, tile_size=32)
    assert result[:4] == b"\x89PNG"


def test_render_derived_tile_with_elevation(patch_slope_catalog, tmp_path):
    from rasterio.coords import BoundingBox
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    elev_file = tmp_path / "elevation.tif"
    elev_file.touch()

    raw = np.full((8, 8), 500.0, dtype=np.float32)
    src_tf = from_bounds(-180, -90, 180, 90, 8, 8)

    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.bounds = BoundingBox(-180, -90, 180, 90)
    mock_ds.width = 8
    mock_ds.height = 8
    mock_ds.nodata = None
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = src_tf
    mock_ds.read.return_value = raw

    with patch.object(tiles, "LAYERS_DIR", tmp_path), \
         patch("util.tiles.rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("slope", z=4, x=3, y=5, tile_size=32)
    assert result[:4] == b"\x89PNG"


def test_render_derived_tile_nodata_masked(patch_slope_catalog, tmp_path):
    from rasterio.coords import BoundingBox
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    elev_file = tmp_path / "elevation.tif"
    elev_file.touch()

    raw = np.full((8, 8), -9999.0, dtype=np.float32)
    src_tf = from_bounds(-180, -90, 180, 90, 8, 8)

    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.bounds = BoundingBox(-180, -90, 180, 90)
    mock_ds.width = 8
    mock_ds.height = 8
    mock_ds.nodata = -9999.0
    mock_ds.crs = CRS.from_epsg(4326)
    mock_ds.transform = src_tf
    mock_ds.read.return_value = raw

    with patch.object(tiles, "LAYERS_DIR", tmp_path), \
         patch("util.tiles.rasterio.open", return_value=mock_ds):
        result = tiles.render_layer_tile_bytes("slope", z=4, x=3, y=5, tile_size=32)
    assert result[:4] == b"\x89PNG"


def test_catalog_reads_real_file():
    # Lines 46-47: call the real _catalog body via __wrapped__ (captured before patching).
    import os
    tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
    json.dump(FAKE_CATALOG, tmp)
    tmp.close()
    saved = tiles.CATALOG_PATH
    try:
        tiles.CATALOG_PATH = Path(tmp.name)
        result = _real_catalog_fn()
        assert "categories" in result
    finally:
        tiles.CATALOG_PATH = saved
    os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# _load_temporal_npy  (lines 63-71)
# ---------------------------------------------------------------------------

def test_load_temporal_npy_missing(tmp_path):
    tiles._npy_cache.clear()
    result = tiles._load_temporal_npy(tmp_path / "nonexistent.npy")
    assert result is None


def test_load_temporal_npy_loads_and_caches(tmp_path):
    tiles._npy_cache.clear()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    p = tmp_path / "test.npy"
    np.save(p, arr)
    result = tiles._load_temporal_npy(p)
    assert result is not None
    np.testing.assert_array_almost_equal(result, arr)
    # second call returns cached value
    result2 = tiles._load_temporal_npy(p)
    assert result2 is result


# ---------------------------------------------------------------------------
# render_temporal_tile_bytes  (lines 185-235) and render_layer_tile_bytes
# temporal dispatch (line 247)
# ---------------------------------------------------------------------------

_TEMPORAL_CATALOG_FOR_RENDER = {
    "categories": [
        {
            "id": "temporal",
            "display_name": "Recent Weather",
            "windows": [24],
            "layers": [
                {
                    "id": "temperature_2m",
                    "display_name": "Air Temperature",
                    "units": "°C",
                    "value_type": "interval",
                    "agg": "avg",
                    "render_min": -30.0,
                    "render_max": 40.0,
                },
            ],
        }
    ]
}


@pytest.fixture
def patch_temporal_render_catalog():
    tiles._catalog.cache_clear()
    with patch.object(tiles, "_catalog", return_value=_TEMPORAL_CATALOG_FOR_RENDER):
        yield
    tiles._catalog.cache_clear()


def test_render_temporal_tile_no_npy(patch_temporal_render_catalog, tmp_path, monkeypatch):
    """Missing npy → transparent PNG (arr is None path, lines 227-229)."""
    monkeypatch.setattr(tiles, "TEMPORAL_RASTERS_DIR", tmp_path)
    tiles._npy_cache.clear()
    result = tiles.render_temporal_tile_bytes("temperature_2m_avg_24h", z=2, x=2, y=1)
    assert result[:4] == b"\x89PNG"


def test_render_temporal_tile_with_npy(patch_temporal_render_catalog, tmp_path, monkeypatch):
    """Valid npy → warp + colorize path (lines 197-226)."""
    monkeypatch.setattr(tiles, "TEMPORAL_RASTERS_DIR", tmp_path)
    tiles._npy_cache.clear()
    arr = np.linspace(-10.0, 30.0, 721 * 1440, dtype=np.float32).reshape(721, 1440)
    np.save(tmp_path / "temperature_2m_24h.npy", arr)
    result = tiles.render_temporal_tile_bytes("temperature_2m_avg_24h", z=2, x=2, y=1)
    assert result[:4] == b"\x89PNG"


_TEMPORAL_CATALOG_NO_RANGE = {
    "categories": [
        {
            "id": "temporal",
            "display_name": "Recent Weather",
            "windows": [24],
            "layers": [
                {
                    "id": "temperature_2m",
                    "display_name": "Air Temperature",
                    "units": "°C",
                    "value_type": "interval",
                    "agg": "avg",
                    "render_min": None,
                    "render_max": None,
                },
            ],
        }
    ]
}


def test_render_temporal_tile_auto_range(tmp_path, monkeypatch):
    """render_min/render_max None → auto percentile path (lines 200, 202)."""
    tiles._catalog.cache_clear()
    tiles._npy_cache.clear()
    with patch.object(tiles, "_catalog", return_value=_TEMPORAL_CATALOG_NO_RANGE):
        monkeypatch.setattr(tiles, "TEMPORAL_RASTERS_DIR", tmp_path)
        arr = np.linspace(-10.0, 30.0, 721 * 1440, dtype=np.float32).reshape(721, 1440)
        np.save(tmp_path / "temperature_2m_24h.npy", arr)
        result = tiles.render_temporal_tile_bytes("temperature_2m_avg_24h", z=2, x=2, y=1)
    tiles._catalog.cache_clear()
    assert result[:4] == b"\x89PNG"


def test_render_layer_tile_dispatches_temporal(patch_temporal_render_catalog, tmp_path, monkeypatch):
    """render_layer_tile_bytes routes temporal layers through render_temporal_tile_bytes (line 247)."""
    monkeypatch.setattr(tiles, "TEMPORAL_RASTERS_DIR", tmp_path)
    tiles._npy_cache.clear()
    result = tiles.render_layer_tile_bytes("temperature_2m_avg_24h", z=2, x=2, y=1)
    assert result[:4] == b"\x89PNG"

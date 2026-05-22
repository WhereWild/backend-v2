"""Raster-specific test helpers, layered on top of the temporal conftest."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.temporal.conftest import TEST_LOCATIONS, expected_window
from util.temporal import RASTER_GRIDS, ChunkIndex, ChunkRange, grid_indices


class FakeRasterReader:
    """OmFileReader stand-in backed by a 1-D fixture series.

    Returns the same series for every (lat_idx, lon_idx) — sufficient for
    single-cell correctness tests.  shape is (ny, nx, time_len).
    """

    def __init__(self, series: np.ndarray, model: str = "copernicus_era5") -> None:
        g = RASTER_GRIDS[model]
        ny, nx = g.get("ny", 721), g.get("nx", 1440)
        self._series = series
        self.shape = (ny, nx, len(series))

    def __getitem__(self, key: object) -> np.ndarray:
        # key is (row_slice, col_slice, time_slice) — return series for any cell
        if isinstance(key, tuple) and len(key) == 3:
            _, _, t = key
            return self._series[t][np.newaxis, np.newaxis, :]
        return self._series


def chunk_from_fixture(fixture: dict[str, Any], model: str = "copernicus_era5") -> tuple[ChunkIndex, ChunkRange]:
    """Build a single-chunk ChunkIndex/ChunkRange covering the full fixture range."""
    times = fixture["hourly"]["time_unix"]
    t0, t_end, tlen = float(times[0]), float(times[-1]), len(times)
    entry = ChunkRange(chunk_num=2019, start=t0, end=t_end, time_len=tlen, source="year")
    index = ChunkIndex(latest_end_time=t_end, resolution=3600.0, ranges=[entry])
    return index, entry


def raster_cell(fixture: dict[str, Any], model: str) -> tuple[int, int]:
    """Return (lat_idx, lon_idx) for the fixture location on the given model's grid."""
    loc = next(
        (l for l in TEST_LOCATIONS if l["name"] in fixture.get("_key", "")),
        None,
    )
    if loc is None:
        lat, lon = float(fixture["latitude"]), float(fixture["longitude"])
    else:
        lat, lon = loc["lat"], loc["lon"]
    g = RASTER_GRIDS[model]
    ny, nx = g.get("ny", 721), g.get("nx", 1440)
    return grid_indices(lat, lon, ny, nx, "lat_asc_lon_pm180", g["step"])


def expected_raster_window(
    fixture: dict[str, Any],
    variable: str,
    window_hours: int,
    agg: str,
    obs_hour: int,
) -> float | None:
    """Convenience wrapper: expected_window using an hour-index into the fixture."""
    obs_ts = float(fixture["hourly"]["time_unix"][obs_hour])
    return expected_window(fixture, obs_ts, variable, window_hours, agg)

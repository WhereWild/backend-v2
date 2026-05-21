"""
Phase 5 integration smoke tests.

Each test:
  - Builds a minimal synthetic occurrence parquet at a known fixture location.
  - Mocks _download_chunk and OmFileReader so process_chunk uses fixture
    time series instead of real S3 .om files.
  - Runs process_chunk → write_back (or derive_* for derived vars).
  - Asserts output columns match expected_window() — the same ground-truth
    helper used to verify the fixture data itself.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.temporal.conftest import expected_window
from util.temporal import (
    ChunkIndex,
    ChunkRange,
    derive_vpd,
    derive_weather_code,
    map_to_worklist,
    process_chunk,
    vpd_kpa,
    weather_code_simple,
    window_steps,
    write_back,
)

pytestmark = pytest.mark.usefixtures("require_fixtures")

_BERLIN_LAT = 52.52
_BERLIN_LON = 13.40
_STEP = 0.25
_GRID_MODE = "lat_asc_lon_pm180"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeReader:
    """Minimal OmFileReader stand-in backed by a numpy array."""

    def __init__(self, series: np.ndarray, ny: int = 721, nx: int = 1440) -> None:
        self._series = series
        self.shape = (ny, nx, len(series))

    def __getitem__(self, key: object) -> np.ndarray:
        return self._series

    def close(self) -> None:
        pass


def _write_occ(path, lat: float, lon: float, ts: float) -> None:
    pq.write_table(
        pa.table({
            "decimalLatitude": pa.array([lat]),
            "decimalLongitude": pa.array([lon]),
            "eventTimestamp": pa.array([ts]),
        }),
        path,
    )


def _chunk_from_fixture(fixture: dict) -> tuple[ChunkIndex, ChunkRange]:
    times = fixture["hourly"]["time_unix"]
    t0, t_end, tlen = float(times[0]), float(times[-1]), len(times)
    entry = ChunkRange(chunk_num=2019, start=t0, end=t_end, time_len=tlen, source="year")
    index = ChunkIndex(latest_end_time=t_end, resolution=3600.0, ranges=[entry])
    return index, entry


def _run_process_chunk(
    fixture: dict,
    variable: str,
    obs_hour: int,
    window_hours: tuple[int, ...],
    agg: str,
    occ_path,
    tmp_path,
    monkeypatch,
) -> dict:
    """Wire up a fake .om reader from fixture data, run process_chunk, return parquet columns."""
    series = np.array(fixture["hourly"][variable], dtype=np.float64)
    chunk_index, chunk_entry = _chunk_from_fixture(fixture)
    obs_ts = float(fixture["hourly"]["time_unix"][obs_hour])

    _write_occ(occ_path, _BERLIN_LAT, _BERLIN_LON, obs_ts)

    occ_table = pa.table({
        "taxon_path": pa.array([str(occ_path)]),
        "row_idx": pa.array([0], type=pa.int64()),
        "latitude": pa.array([_BERLIN_LAT]),
        "longitude": pa.array([_BERLIN_LON]),
        "timestamp": pa.array([obs_ts]),
    })
    worklist = map_to_worklist(occ_table, chunk_index, _GRID_MODE, _STEP)
    steps = window_steps(3600.0, window_hours)

    dummy = tmp_path / "dummy.om"
    dummy.write_bytes(b"")
    monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: dummy)
    monkeypatch.setattr("util.temporal.OmFileReader", lambda fh: _FakeReader(series))

    updates, _ = process_chunk(
        chunk_entry, worklist, {}, "copernicus_era5", variable, steps, agg, str(tmp_path),
    )
    write_back(updates)
    return pq.read_table(occ_path).to_pydict()


# ---------------------------------------------------------------------------
# process_chunk correctness
# ---------------------------------------------------------------------------

class TestProcessChunk:
    def test_precipitation_sum_24h(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        result = _run_process_chunk(fix, "precipitation", obs_hour, (24,), "sum",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        expected = expected_window(fix, obs_ts, "precipitation", 24, "sum")
        assert result["precipitation_sum_24h"][0] == pytest.approx(expected, abs=1e-3)

    def test_temperature_avg_24h(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        result = _run_process_chunk(fix, "temperature_2m", obs_hour, (24,), "avg",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        expected = expected_window(fix, obs_ts, "temperature_2m", 24, "avg")
        assert result["temperature_2m_avg_24h"][0] == pytest.approx(expected, abs=1e-3)

    def test_multiple_windows(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        windows = (1, 24, 168)
        result = _run_process_chunk(fix, "precipitation", obs_hour, windows, "sum",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        for h in windows:
            expected = expected_window(fix, obs_ts, "precipitation", h, "sum")
            assert result[f"precipitation_sum_{h}h"][0] == pytest.approx(expected, abs=1e-3), \
                f"window={h}h mismatch"

    def test_partial_window_near_series_start(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 10  # only 11 hours available for a 24h window
        result = _run_process_chunk(fix, "precipitation", obs_hour, (24,), "sum",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        expected = expected_window(fix, obs_ts, "precipitation", 24, "sum")
        assert result["precipitation_sum_24h"][0] == pytest.approx(expected, abs=1e-3)

    def test_dew_point_avg_24h(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        result = _run_process_chunk(fix, "dew_point_2m", obs_hour, (24,), "avg",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        expected = expected_window(fix, obs_ts, "dew_point_2m", 24, "avg")
        assert result["dew_point_2m_avg_24h"][0] == pytest.approx(expected, abs=1e-3)

    def test_soil_temperature_avg_72h(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 1000
        result = _run_process_chunk(fix, "soil_temperature_0_to_7cm", obs_hour, (72,), "avg",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        expected = expected_window(fix, obs_ts, "soil_temperature_0_to_7cm", 72, "avg")
        assert result["soil_temperature_0_to_7cm_avg_72h"][0] == pytest.approx(expected, abs=1e-3)

    def test_snow_depth_avg_1h(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["reykjavik_boundary"]  # Dec–Jan, has snow
        obs_hour = 400
        result = _run_process_chunk(fix, "snow_depth", obs_hour, (1,), "avg",
                                    tmp_path / "occ.parquet", tmp_path, monkeypatch)
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])
        expected = expected_window(fix, obs_ts, "snow_depth", 1, "avg")
        assert result["snow_depth_avg_1h"][0] == pytest.approx(expected, abs=1e-4)

    def test_cross_location_sydney(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["sydney_early"]
        series = np.array(fix["hourly"]["precipitation"], dtype=np.float64)
        chunk_index, chunk_entry = _chunk_from_fixture(fix)
        obs_hour = 300
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])

        occ_path = tmp_path / "occ.parquet"
        _write_occ(occ_path, -33.87, 151.21, obs_ts)

        occ_table = pa.table({
            "taxon_path": pa.array([str(occ_path)]),
            "row_idx": pa.array([0], type=pa.int64()),
            "latitude": pa.array([-33.87]),
            "longitude": pa.array([151.21]),
            "timestamp": pa.array([obs_ts]),
        })
        worklist = map_to_worklist(occ_table, chunk_index, _GRID_MODE, _STEP)
        steps = window_steps(3600.0, (24,))

        dummy = tmp_path / "dummy.om"
        dummy.write_bytes(b"")
        monkeypatch.setattr("util.temporal._download_chunk", lambda *a, **kw: dummy)
        monkeypatch.setattr("util.temporal.OmFileReader", lambda fh: _FakeReader(series))

        updates, _ = process_chunk(
            chunk_entry, worklist, {}, "copernicus_era5", "precipitation",
            steps, "sum", str(tmp_path),
        )
        write_back(updates)

        result = pq.read_table(occ_path).to_pydict()
        expected = expected_window(fix, obs_ts, "precipitation", 24, "sum")
        assert result["precipitation_sum_24h"][0] == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Derived variables
# ---------------------------------------------------------------------------

class TestDerivedVariables:
    def _fake_taxon_node(self, path) -> dict:
        return {"taxon_key": "1", "path": str(path.parent),
                "scientific_name": "Test", "common_name": "", "rank": "SPECIES"}

    def test_derive_vpd_matches_formula(self, require_fixtures, tmp_path, monkeypatch) -> None:
        fix = require_fixtures["berlin_early"]
        obs_hour = 500
        obs_ts = float(fix["hourly"]["time_unix"][obs_hour])

        t_avg = expected_window(fix, obs_ts, "temperature_2m", 24, "avg")
        td_avg = expected_window(fix, obs_ts, "dew_point_2m", 24, "avg")

        occ_path = tmp_path / "occurrence.parquet"
        pq.write_table(
            pa.table({
                "decimalLatitude": pa.array([_BERLIN_LAT]),
                "decimalLongitude": pa.array([_BERLIN_LON]),
                "eventTimestamp": pa.array([obs_ts]),
                "temperature_2m_avg_24h": pa.array([t_avg]),
                "dew_point_2m_avg_24h": pa.array([td_avg]),
            }),
            occ_path,
        )

        node = self._fake_taxon_node(occ_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])

        derive_vpd("1", str(tmp_path), "occurrence.parquet", [24])

        result = pq.read_table(occ_path).to_pydict()
        expected_vpd = float(vpd_kpa(float(t_avg), float(td_avg)))
        assert result["vapor_pressure_deficit_avg_24h"][0] == pytest.approx(expected_vpd, abs=1e-5)

    def test_derive_vpd_nan_when_source_missing(self, tmp_path, monkeypatch) -> None:
        occ_path = tmp_path / "occurrence.parquet"
        pq.write_table(
            pa.table({
                "decimalLatitude": pa.array([_BERLIN_LAT]),
                "decimalLongitude": pa.array([_BERLIN_LON]),
                "eventTimestamp": pa.array([1.0]),
                # temperature_2m_avg_24h intentionally absent
            }),
            occ_path,
        )
        node = self._fake_taxon_node(occ_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])

        derive_vpd("1", str(tmp_path), "occurrence.parquet", [24])

        result = pq.read_table(occ_path)
        assert "vapor_pressure_deficit_avg_24h" not in result.column_names

    def test_derive_weather_code_clear_sky(self, tmp_path, monkeypatch) -> None:
        occ_path = tmp_path / "occurrence.parquet"
        pq.write_table(
            pa.table({
                "decimalLatitude": pa.array([_BERLIN_LAT]),
                "decimalLongitude": pa.array([_BERLIN_LON]),
                "eventTimestamp": pa.array([1.0]),
                "cloud_cover_avg_1h": pa.array([5.0]),
                "precipitation_sum_1h": pa.array([0.0]),
                "snowfall_water_equivalent_sum_1h": pa.array([0.0]),
            }),
            occ_path,
        )
        node = self._fake_taxon_node(occ_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])

        derive_weather_code("1", str(tmp_path), "occurrence.parquet")

        result = pq.read_table(occ_path).to_pydict()
        assert result["weather_code_simple"][0] == weather_code_simple(5.0, 0.0, 0.0, 3600)

    def test_derive_weather_code_heavy_rain(self, tmp_path, monkeypatch) -> None:
        occ_path = tmp_path / "occurrence.parquet"
        pq.write_table(
            pa.table({
                "decimalLatitude": pa.array([_BERLIN_LAT]),
                "decimalLongitude": pa.array([_BERLIN_LON]),
                "eventTimestamp": pa.array([1.0]),
                "cloud_cover_avg_1h": pa.array([90.0]),
                "precipitation_sum_1h": pa.array([10.0]),
                "snowfall_water_equivalent_sum_1h": pa.array([0.0]),
            }),
            occ_path,
        )
        node = self._fake_taxon_node(occ_path)
        monkeypatch.setattr("util.temporal.get_taxon_by_id", lambda _: node)
        monkeypatch.setattr("util.temporal.iter_descendants", lambda r, **kw: [r])

        derive_weather_code("1", str(tmp_path), "occurrence.parquet")

        result = pq.read_table(occ_path).to_pydict()
        assert result["weather_code_simple"][0] == 65  # heavy rain

"""Sanity checks on fixture data — verifies API responses look reasonable."""
from __future__ import annotations

from typing import Any

import pytest

from tests.temporal.conftest import (
    FETCH_RANGES,
    HOURLY_VARS,
    TEST_LOCATIONS,
    expected_window,
    obs_timestamp,
)

pytestmark = pytest.mark.usefixtures("require_fixtures")


def _key(name: str, label: str) -> str:
    return f"{name}_{label}"


class TestFixtureCompleteness:
    def test_all_locations_present(self, require_fixtures: dict[str, Any]) -> None:
        for loc in TEST_LOCATIONS:
            for label, _, _ in FETCH_RANGES:
                key = _key(loc["name"], label)
                assert key in require_fixtures, f"Missing fixture: {key}"

    def test_all_variables_present(self, require_fixtures: dict[str, Any]) -> None:
        for key, data in require_fixtures.items():
            for var in HOURLY_VARS:
                assert var in data["hourly"], f"{key}: missing variable '{var}'"

    def test_time_unix_populated(self, require_fixtures: dict[str, Any]) -> None:
        for key, data in require_fixtures.items():
            times = data["hourly"]["time_unix"]
            assert len(times) > 0, f"{key}: empty time_unix"
            # ERA5 is hourly; confirm ~3600s spacing
            gaps = set(times[i + 1] - times[i] for i in range(min(10, len(times) - 1)))
            assert gaps == {3600}, f"{key}: unexpected time spacing {gaps}"

    def test_early_range_length(self, require_fixtures: dict[str, Any]) -> None:
        # "early" range is 2019-06-01 to 2020-05-31 = ~8784 hours (366-day year)
        for loc in TEST_LOCATIONS:
            key = _key(loc["name"], "early")
            if key not in require_fixtures:
                continue
            hours = len(require_fixtures[key]["hourly"]["time_unix"])
            assert 8700 <= hours <= 8800, f"{key}: unexpected hour count {hours}"

    def test_boundary_range_length(self, require_fixtures: dict[str, Any]) -> None:
        # "boundary" range is 2022-12-15 to 2023-01-15 inclusive = 32 days = 768 hours
        for loc in TEST_LOCATIONS:
            key = _key(loc["name"], "boundary")
            if key not in require_fixtures:
                continue
            hours = len(require_fixtures[key]["hourly"]["time_unix"])
            assert 764 <= hours <= 772, f"{key}: unexpected hour count {hours}"


class TestFixturePlausibility:
    """Spot-check that values are in physically plausible ranges."""

    def _non_null(self, data: dict, var: str) -> list[float]:
        return [v for v in data["hourly"][var] if v is not None]

    def test_temperature_range(self, require_fixtures: dict[str, Any]) -> None:
        for key, data in require_fixtures.items():
            vals = self._non_null(data, "temperature_2m")
            if not vals:
                continue
            assert min(vals) > -90, f"{key}: temperature too low"
            assert max(vals) < 60, f"{key}: temperature too high"

    def test_precipitation_non_negative(self, require_fixtures: dict[str, Any]) -> None:
        for key, data in require_fixtures.items():
            vals = self._non_null(data, "precipitation")
            assert all(v >= 0 for v in vals), f"{key}: negative precipitation"

    def test_cloud_cover_range(self, require_fixtures: dict[str, Any]) -> None:
        for key, data in require_fixtures.items():
            vals = self._non_null(data, "cloud_cover")
            if not vals:
                continue
            assert min(vals) >= 0, f"{key}: cloud cover < 0"
            assert max(vals) <= 100, f"{key}: cloud cover > 100"

    def test_arctic_location_has_snow(self, require_fixtures: dict[str, Any]) -> None:
        # Use boundary fixture (Dec–Jan) — unambiguously snowy at 71°N
        key = _key("reykjavik", "boundary")
        if key not in require_fixtures:
            pytest.skip("reykjavik boundary fixture not loaded")
        vals = self._non_null(require_fixtures[key], "snow_depth")
        assert any(v > 0 for v in vals), "Reykjavik should have snow depth in December/January"

    def test_dubai_minimal_precipitation(self, require_fixtures: dict[str, Any]) -> None:
        key = _key("dubai", "early")
        if key not in require_fixtures:
            pytest.skip("dubai early fixture not loaded")
        total = sum(self._non_null(require_fixtures[key], "precipitation"))
        # Dubai averages ~90mm/year — a full year should be well under 500mm
        assert total < 500, f"Dubai total precip suspiciously high: {total:.1f}mm"


class TestReferenceWindowHelper:
    """Verify the expected_window helper itself behaves correctly."""

    def test_sum_window(self, require_fixtures: dict[str, Any]) -> None:
        key = _key("berlin", "early")
        if key not in require_fixtures:
            pytest.skip("berlin early fixture not loaded")
        data = require_fixtures[key]
        ts = obs_timestamp(data, 500)  # 500 hours into the series
        result = expected_window(data, ts, "precipitation", 24, "sum")
        # Verify manually: sum of 24 values from index 477 to 500 inclusive
        vals = [v for v in data["hourly"]["precipitation"][477:501] if v is not None]
        assert result == pytest.approx(sum(vals), abs=1e-6)

    def test_avg_window(self, require_fixtures: dict[str, Any]) -> None:
        key = _key("berlin", "early")
        if key not in require_fixtures:
            pytest.skip("berlin early fixture not loaded")
        data = require_fixtures[key]
        ts = obs_timestamp(data, 500)
        result = expected_window(data, ts, "temperature_2m", 24, "avg")
        vals = [v for v in data["hourly"]["temperature_2m"][477:501] if v is not None]
        assert result == pytest.approx(sum(vals) / len(vals), abs=1e-6)

    def test_partial_window_at_start(self, require_fixtures: dict[str, Any]) -> None:
        key = _key("berlin", "early")
        if key not in require_fixtures:
            pytest.skip("berlin early fixture not loaded")
        data = require_fixtures[key]
        # Obs at index 5 with 24h window: only 6 hours available (0..5)
        ts = obs_timestamp(data, 5)
        result = expected_window(data, ts, "precipitation", 24, "sum")
        vals = [v for v in data["hourly"]["precipitation"][0:6] if v is not None]
        assert result == pytest.approx(sum(vals), abs=1e-6)

    def test_out_of_range_returns_none(self, require_fixtures: dict[str, Any]) -> None:
        key = _key("berlin", "early")
        if key not in require_fixtures:
            pytest.skip("berlin early fixture not loaded")
        data = require_fixtures[key]
        # Timestamp way before the fixture range
        result = expected_window(data, 0.0, "temperature_2m", 24, "avg")
        assert result is None

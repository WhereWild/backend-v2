"""
Fixture infrastructure for temporal enrichment tests.

Fixture files live in tests/temporal/fixtures/ and are NOT committed to git.
On first run, missing fixtures are fetched automatically from the
Open-Meteo Historical API (~20 requests, ~30s).

To force a full re-fetch (e.g. after adding new locations):

    uv run pytest tests/temporal/ --regenerate-fixtures

Open-Meteo Historical API limit: 600 requests/day. The full fixture set
requires ~20 requests and takes ~30 seconds.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Variables fetched from Open-Meteo Historical API (excludes derived vars)
HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "snowfall_water_equivalent",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "snow_depth",
]

# Two date ranges per location:
#   "early"  — sits firmly in year_*.om territory, gives us data for 90-day window tests
#   "boundary" — spans the year_YYYY.om → year_{YYYY+1}.om boundary
FETCH_RANGES = [
    ("early",    "2019-06-01", "2020-05-31"),
    ("boundary", "2022-12-15", "2023-01-15"),
]

# Chosen to cover: diverse climates, southern hemisphere, ±180° edge, near-polar
TEST_LOCATIONS: list[dict[str, Any]] = [
    {"name": "salt_lake_city", "lat":  40.77,  "lon": -111.89},  # continental USA
    {"name": "london",         "lat":  51.51,  "lon":   -0.13},  # oceanic, near 0° meridian
    {"name": "sydney",         "lat": -33.87,  "lon":  151.21},  # S hemisphere, oceanic
    {"name": "reykjavik",      "lat":  64.13,  "lon":  -21.93},  # subarctic, lots of precip
    {"name": "dubai",          "lat":  25.20,  "lon":   55.27},  # desert, minimal precip
    {"name": "nairobi",        "lat":  -1.29,  "lon":   36.82},  # equatorial
    {"name": "tromsoe",        "lat":  71.17,  "lon":   25.78},  # high-arctic
    {"name": "berlin",         "lat":  52.52,  "lon":   13.40},  # continental Europe
    {"name": "ushuaia",        "lat": -54.80,  "lon":  -68.30},  # subpolar S hemisphere
    {"name": "tuvalu",         "lat":  -8.51,  "lon":  179.20},  # dateline edge (lon ~180°)
]


# ---------------------------------------------------------------------------
# Fixture file helpers
# ---------------------------------------------------------------------------

def _fixture_path(name: str, range_label: str) -> Path:
    return FIXTURE_DIR / f"{name}_{range_label}.json"


def _fetch_from_api(loc: dict[str, Any], start_date: str, end_date: str) -> dict[str, Any]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        # Disable elevation correction so fixtures match raw .om model grid values.
        # Remove this once elevation correction is implemented (needs DEM pipeline).
        "elevation": "NaN",
    }
    resp = httpx.get(url, params=params, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    # Convert ISO time strings to Unix timestamps for easier arithmetic
    times_iso = data["hourly"]["time"]
    times_unix = [
        int(datetime.fromisoformat(t).replace(tzinfo=UTC).timestamp())
        for t in times_iso
    ]
    data["hourly"]["time_unix"] = times_unix
    return data


def load_fixtures() -> dict[str, dict[str, Any]]:
    """Load all fixture files from disk. Keys: '{name}_{range_label}'."""
    result: dict[str, dict] = {}
    for loc in TEST_LOCATIONS:
        for label, _, _ in FETCH_RANGES:
            path = _fixture_path(loc["name"], label)
            if path.exists():
                result[f"{loc['name']}_{label}"] = json.loads(path.read_text())
    return result


def regenerate_fixtures() -> None:
    """Fetch all fixture data from the Open-Meteo API and save to disk."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(TEST_LOCATIONS) * len(FETCH_RANGES)
    done = 0
    for loc in TEST_LOCATIONS:
        for label, start, end in FETCH_RANGES:
            done += 1
            path = _fixture_path(loc["name"], label)
            print(f"[{done}/{total}] {loc['name']} {label} ({start} → {end}) ... ", end="", flush=True)
            data = _fetch_from_api(loc, start, end)
            path.write_text(json.dumps(data, separators=(",", ":")))
            hours = len(data["hourly"]["time_unix"])
            print(f"{hours} hours")
    print("Done.")


# ---------------------------------------------------------------------------
# Reference implementation: compute expected windowed value from raw fixture
# ---------------------------------------------------------------------------

def expected_window(
    fixture: dict[str, Any],
    obs_ts: float,
    variable: str,
    window_hours: int,
    agg: str,
) -> float | None:
    """
    Compute ground-truth windowed value for an observation using raw API data.

    Mimics what enrich_temporal should produce: take the window_hours ending
    at obs_ts (inclusive), apply sum or avg over non-null values.
    Returns None if the timestamp falls outside the fixture's range.
    """
    times = fixture["hourly"]["time_unix"]
    values = fixture["hourly"].get(variable, [])
    if not times or not values:
        return None

    # Floor observation to the hour boundary (ERA5 is hourly)
    obs_hour = int(obs_ts // 3600) * 3600

    # Find the index of the obs hour in the time series
    t0 = times[0]
    resolution = 3600
    end_idx = round((obs_hour - t0) / resolution)
    if end_idx < 0 or end_idx >= len(times):
        return None

    start_idx = max(0, end_idx - window_hours + 1)
    window_vals = [v for v in values[start_idx : end_idx + 1] if v is not None]

    if not window_vals:
        return None
    if agg == "sum":
        return sum(window_vals)
    if agg == "avg":
        return sum(window_vals) / len(window_vals)
    return None


def obs_timestamp(fixture: dict[str, Any], offset_from_start_hours: int) -> float:
    """Return a Unix timestamp offset_from_start_hours into a fixture's time series."""
    return float(fixture["hourly"]["time_unix"][offset_from_start_hours])


# ---------------------------------------------------------------------------
# pytest hooks and fixtures
# ---------------------------------------------------------------------------


def _missing_fixtures() -> bool:
    """Return True if any expected fixture files are absent."""
    return any(
        not _fixture_path(loc["name"], label).exists()
        for loc in TEST_LOCATIONS
        for label, _, _ in FETCH_RANGES
    )


@pytest.fixture(scope="session")
def fixture_store(request: pytest.FixtureRequest) -> dict[str, dict[str, Any]]:
    """
    Session-scoped fixture providing all loaded fixture data.

    Auto-fetches from Open-Meteo API on first run if any fixtures are missing.
    Pass --regenerate-fixtures to force a full re-fetch even when files exist.
    """
    force = request.config.getoption("--regenerate-fixtures")
    if force or _missing_fixtures():
        try:
            regenerate_fixtures()
        except Exception as exc:
            # Network unavailable (e.g. CI runners) — tests that need fixtures
            # will be skipped via require_fixtures.
            print(f"\n[fixtures] fetch failed ({exc}); tests requiring live data will be skipped.")
            return load_fixtures()
    return load_fixtures()


@pytest.fixture(scope="session")
def require_fixtures(fixture_store: dict[str, dict]) -> dict[str, dict]:
    """Like fixture_store but skips the test if any fixtures couldn't be loaded."""
    expected = {
        f"{loc['name']}_{label}"
        for loc in TEST_LOCATIONS
        for label, _, _ in FETCH_RANGES
    }
    missing = expected - fixture_store.keys()
    if missing:
        pytest.skip(f"Fixture data incomplete ({len(missing)} missing); skipping live-data tests.")
    return fixture_store

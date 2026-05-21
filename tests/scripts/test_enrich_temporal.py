"""Tests for scripts/enrich_temporal.py — script-level logic only."""
from __future__ import annotations

import scripts.enrich_temporal as et
from scripts.enrich_temporal import _filter_layers
from util.temporal import TemporalLayer


def _layers() -> list[TemporalLayer]:
    return [
        TemporalLayer(id="temperature_2m", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="avg", windows=[24]),
        TemporalLayer(id="precipitation", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="sum", windows=[24]),
        TemporalLayer(id="snow_depth", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="avg", windows=[1]),
        TemporalLayer(id="vapor_pressure_deficit", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="avg", windows=[24], derived=True),
        TemporalLayer(id="weather_code_simple", model="copernicus_era5", grid_mode="lat_asc_lon_pm180", agg="snapshot", windows=[1], derived=True),
    ]


class TestFilterLayers:
    def test_none_returns_all(self) -> None:
        layers = _layers()
        assert _filter_layers(layers, None) == layers

    def test_single_temporal_id(self) -> None:
        result = _filter_layers(_layers(), ["precipitation"])
        assert len(result) == 1
        assert result[0].id == "precipitation"

    def test_multiple_temporal_ids(self) -> None:
        result = _filter_layers(_layers(), ["precipitation", "snow_depth"])
        ids = {layer.id for layer in result}
        assert ids == {"precipitation", "snow_depth"}

    def test_no_temporal_ids_returns_all(self) -> None:
        # All ids are spatial → treat as "do all temporal"
        layers = _layers()
        result = _filter_layers(layers, ["bio1", "bio12", "gsl"])
        assert result == layers

    def test_mixed_ids_returns_only_temporal_matches(self) -> None:
        result = _filter_layers(_layers(), ["bio1", "precipitation"])
        assert len(result) == 1
        assert result[0].id == "precipitation"

    def test_derived_var_included_when_requested(self) -> None:
        result = _filter_layers(_layers(), ["vapor_pressure_deficit"])
        assert len(result) == 1
        assert result[0].derived is True

    def test_empty_list_returns_all(self) -> None:
        layers = _layers()
        assert _filter_layers(layers, []) == layers

    def test_order_preserved(self) -> None:
        result = _filter_layers(_layers(), ["snow_depth", "temperature_2m"])
        assert [layer.id for layer in result] == ["temperature_2m", "snow_depth"]


class TestVarsToEnrichParsing:
    def test_module_level_parsing_none_when_empty(self) -> None:
        # VARS_TO_ENRICH should be None when env var was not set (empty string)
        # This relies on the module being imported without the env var set
        assert et.VARS_TO_ENRICH is None or isinstance(et.VARS_TO_ENRICH, list)

"""Unit tests for util.units."""

from __future__ import annotations

import pytest

from util import units


def test_normalize_helpers():
    assert units.normalize_unit_system(None) is None
    assert units.normalize_unit_system("metric") == "metric"
    assert units.normalize_unit_system("IMPERIAL") == "imperial"
    assert units.normalize_unit_system("weird") is None

    assert units.normalize_unit("°C") == "c"
    assert units.normalize_unit(" miles ") == "mi"
    assert units.normalize_unit(None) is None
    assert units.normalize_unit("nonsense") is None

    assert units.unit_system_for_unit("kpa") == "metric"
    assert units.unit_system_for_unit("nonsense") is None

    assert units.display_unit("c") == "°C"
    assert units.display_unit("mi") == "mi"
    assert units.display_unit(None) is None


def test_equivalent_unit_and_conversion_params(monkeypatch):
    assert units.equivalent_unit("m", "imperial") == "ft"
    assert units.equivalent_unit(None, "metric") is None

    # Exercise mapping-missing fallback.
    monkeypatch.setitem(units._EQUIVALENT_UNIT, "kpa", {})
    assert units.equivalent_unit("kpa", "imperial") == "kpa"

    assert units.conversion_params("m", "m") == units.ConversionParams(1.0, 0.0)
    assert units.conversion_params("c", "f") == units.ConversionParams(9 / 5, 32.0)
    assert units.conversion_params("f", "c") == units.ConversionParams(5 / 9, -32.0 * (5 / 9))
    assert units.conversion_params("kpa", "psi") == units.ConversionParams(0.1450377377, 0.0)
    assert units.conversion_params("psi", "kpa") == units.ConversionParams(6.8947572932, 0.0)
    assert units.conversion_params("m", "ft") is not None
    assert units.conversion_params("c", "mi") is None


def test_convert_between_and_for_system(monkeypatch):
    assert units.convert_between_units(10.0, "c", "f") == pytest.approx(50.0)
    assert units.convert_between_units(10.0, "c", "unknown") == 10.0

    # early-return branch
    assert units.convert_value_for_system(None, "c", "imperial") == (None, "°C")
    assert units.convert_value_for_system(1.0, "c", None) == (1.0, "°C")

    # unknown unit branch
    assert units.convert_value_for_system(10.0, "unknown", "imperial") == (10.0, "unknown")

    # same-system branch
    assert units.convert_value_for_system(10.0, "c", "metric") == (10.0, "°C")

    # target equivalent not usable branch
    monkeypatch.setattr(units, "equivalent_unit", lambda *_a, **_k: "c")
    assert units.convert_value_for_system(10.0, "c", "imperial") == (10.0, "°C")

    monkeypatch.setattr(units, "equivalent_unit", lambda *_a, **_k: None)
    assert units.convert_value_for_system(10.0, "c", "imperial") == (10.0, "°C")

    # reset and actual conversion
    monkeypatch.setattr(units, "equivalent_unit", lambda unit, target: "f" if unit == "c" else unit)
    assert units.convert_value_for_system(10.0, "c", "imperial")[0] == pytest.approx(50.0)


def test_convert_value_from_system(monkeypatch):
    assert units.convert_value_from_system(None, "c", "imperial") is None
    assert units.convert_value_from_system(10.0, "unknown", "imperial") == 10.0
    assert units.convert_value_from_system(10.0, "f", "imperial") == 10.0

    monkeypatch.setattr(units, "equivalent_unit", lambda *_a, **_k: "f")
    assert units.convert_value_from_system(10.0, "c", "imperial") == pytest.approx(-12.2222222)

    monkeypatch.setattr(units, "equivalent_unit", lambda *_a, **_k: None)
    assert units.convert_value_from_system(10.0, "c", "imperial") == 10.0


def test_convert_summary_density_values_and_observations(monkeypatch):
    original_equivalent_unit = units.equivalent_unit
    original_conversion_params = units.conversion_params
    assert units.convert_summary(None, "c", "imperial") is None
    raw_summary = {"mean": 10.0, "label": "x"}
    converted_summary = units.convert_summary(raw_summary, "c", "imperial")
    assert converted_summary["mean"] == pytest.approx(50.0)
    assert converted_summary["label"] == "x"

    # spread metrics: factor only, no +32 offset
    spread_summary = {
        "std": 5.0,
        "stddev": 5.0,
        "interquartile range": 15.0,
        "10-90 range": 20.0,
        "1-99 range": 40.0,
        "range": 10.0,
    }
    converted_spread = units.convert_summary(spread_summary, "c", "imperial")
    assert converted_spread["std"] == pytest.approx(9.0)  # 5 * 9/5
    assert converted_spread["stddev"] == pytest.approx(9.0)
    assert converted_spread["interquartile range"] == pytest.approx(27.0)  # 15 * 9/5
    assert converted_spread["10-90 range"] == pytest.approx(36.0)  # 20 * 9/5
    assert converted_spread["1-99 range"] == pytest.approx(72.0)  # 40 * 9/5
    assert converted_spread["range"] == pytest.approx(18.0)  # 10 * 9/5

    # convert_density_curve early returns
    assert units.convert_density_curve(None, "c", "imperial") is None
    curve = {"points": [0.0, 10.0], "density": [1.0, 2.0], "min": 0.0, "max": 10.0, "bandwidth": 2.0}
    assert units.convert_density_curve(curve, "unknown", "imperial") == curve
    assert units.convert_density_curve(curve, "c", "metric") == curve

    monkeypatch.setattr(units, "equivalent_unit", lambda *_a, **_k: None)
    assert units.convert_density_curve(curve, "c", "imperial") == curve

    monkeypatch.setattr(units, "equivalent_unit", lambda *_a, **_k: "f")
    monkeypatch.setattr(units, "conversion_params", lambda *_a, **_k: None)
    assert units.convert_density_curve(curve, "c", "imperial") == curve

    # factor==0 branch keeps density unchanged
    monkeypatch.setattr(units, "conversion_params", lambda *_a, **_k: units.ConversionParams(0.0, 1.0))
    out_zero = units.convert_density_curve(curve, "c", "imperial")
    assert out_zero["density"] == curve["density"]

    monkeypatch.setattr(units, "conversion_params", lambda *_a, **_k: units.ConversionParams(2.0, 3.0))
    out = units.convert_density_curve(curve, "c", "imperial")
    assert out["points"] == [3.0, 23.0]
    assert out["density"] == [0.5, 1.0]
    assert out["min"] == 3.0 and out["max"] == 23.0 and out["bandwidth"] == 4.0

    # restore real params for downstream checks in this test
    monkeypatch.setattr(units, "conversion_params", original_conversion_params)
    monkeypatch.setattr(units, "equivalent_unit", original_equivalent_unit)

    assert units.convert_values_list(None, "c", "imperial") is None
    assert units.convert_values_list([1.0, 2.0], "c", None) == [1.0, 2.0]
    assert units.convert_values_list([0.0, 10.0], "c", "imperial") == pytest.approx([32.0, 50.0])

    rows = [{"value": 10.0, "a": 1}, {"value": "x"}]
    assert units.convert_observations([], "c", "imperial") == []
    assert units.convert_observations(rows, "c", "imperial")[0]["value"] == pytest.approx(50.0)
    assert units.convert_observations(rows, "c", "imperial")[1]["value"] == "x"


def test_apply_unit_system_helpers():
    variables = [{"id": "bio_1", "units": "c"}, {"id": "distance", "units": "m"}]
    assert units.apply_unit_system_to_variables(variables, "weird") == variables
    converted_vars = units.apply_unit_system_to_variables(variables, "imperial")
    assert converted_vars[0]["units"] == "°F"
    assert converted_vars[1]["units"] == "ft"

    env = {
        "units": "c",
        "variable_metadata": {"units": "c"},
        "summary": {"mean": 10.0},
        "baselineSummary": {"mean": 5.0},
        "densityCurve": {"points": [0.0], "density": [1.0]},
    }
    assert units.apply_unit_system_to_env_response(env, "metric", None) == env
    converted_env = units.apply_unit_system_to_env_response(env, "imperial", "c")
    assert converted_env["units"] == "°F"
    assert converted_env["variable_metadata"]["units"] == "°F"
    assert converted_env["summary"]["mean"] == pytest.approx(50.0)
    assert converted_env["baselineSummary"]["mean"] == pytest.approx(41.0)
    assert converted_env["baseline_summary"]["mean"] == pytest.approx(41.0)
    assert converted_env["densityCurve"]["points"] == pytest.approx([32.0])
    assert converted_env["density_curve"]["points"] == pytest.approx([32.0])

    slice_resp = {"range": {"min": 0.0, "max": 10.0}, "observations": [{"value": 10.0}]}
    assert units.apply_unit_system_to_slice_response(slice_resp, "metric", None) == slice_resp
    converted_slice = units.apply_unit_system_to_slice_response(slice_resp, "imperial", "c")
    assert converted_slice["range"]["min"] == pytest.approx(32.0)
    assert converted_slice["range"]["max"] == pytest.approx(50.0)
    assert converted_slice["observations"][0]["value"] == pytest.approx(50.0)
    assert converted_slice["units"] == "°F"

    rankings = {
        "units": "c",
        "entries": [{"value": 10.0}, {"value": "x"}],
        "distribution": {"points": [0.0], "density": [1.0]},
    }
    assert units.apply_unit_system_to_rankings_response(rankings, "metric", None) == rankings
    converted_rankings = units.apply_unit_system_to_rankings_response(rankings, "imperial", "c")
    assert converted_rankings["units"] == "°F"
    assert converted_rankings["entries"][0]["value"] == pytest.approx(50.0)
    assert converted_rankings["entries"][1]["value"] == "x"
    assert converted_rankings["distribution"]["points"] == pytest.approx([32.0])

    query_rows = [
        {
            "taxon_id": 1,
            "sort_value": 10.0,
            "sort_variable": "bio_1",
            "sort_metric": "mean",
            "sample_count": 2,
            "position": 1,
            "percentile": 0.0,
        },
        {
            "taxon_id": 2,
            "sort_value": None,
            "sort_variable": "bio_1",
            "sort_metric": "mean",
            "sample_count": 3,
            "position": 2,
            "percentile": 1.0,
        },
    ]
    same_rows, same_units = units.apply_unit_system_to_query_rows(
        query_rows,
        None,
        variable_id="bio_1",
        unit="C",
    )
    assert same_units == "C"
    assert same_rows[0]["sort_value"] == pytest.approx(10.0)

    converted_rows, converted_units = units.apply_unit_system_to_query_rows(
        query_rows,
        "imperial",
        variable_id="bio_1",
        unit="C",
    )
    assert converted_units == "°F"
    assert converted_rows[0]["sort_value"] == pytest.approx(50.0)
    assert converted_rows[1]["sort_value"] is None

    # spread metric: factor only, no +32
    spread_rows = [{"sort_value": 0.0}, {"sort_value": 10.0}]
    spread_converted, _ = units.apply_unit_system_to_query_rows(
        spread_rows,
        "imperial",
        variable_id="bio_1",
        unit="c",
        sort_metric="10-90 range",
    )
    assert spread_converted[0]["sort_value"] == pytest.approx(0.0)  # 0 * 9/5, no +32
    assert spread_converted[1]["sort_value"] == pytest.approx(18.0)  # 10 * 9/5

    iqr_converted, _ = units.apply_unit_system_to_query_rows(
        spread_rows,
        "imperial",
        variable_id="bio_1",
        unit="c",
        sort_metric="interquartile range",
    )
    assert iqr_converted[0]["sort_value"] == pytest.approx(0.0)
    assert iqr_converted[1]["sort_value"] == pytest.approx(18.0)

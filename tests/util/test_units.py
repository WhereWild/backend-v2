# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import pytest

from util import units

_TEMP = {"id": "bio1", "value_type": "interval", "units": "°C", "imperial_unit": "°F"}
_CURVE = {"points": [0.0, 10.0], "density": [0.09, 0.09], "min": 0.0, "max": 10.0, "bandwidth": 1.0}


def test_convert_density_curve_raw_variable_applies_offset():
    result = units.convert_density_curve(_CURVE, _TEMP, "imperial")
    assert result["points"] == pytest.approx([32.0, 50.0])
    assert result["density"] == pytest.approx([0.05, 0.05])
    assert result["bandwidth"] == pytest.approx(1.8)


def test_convert_density_curve_spread_metric_skips_offset():
    result = units.convert_density_curve(_CURVE, _TEMP, "imperial", metric="std")
    assert result["points"] == pytest.approx([0.0, 18.0])
    assert result["max"] == pytest.approx(18.0)


def test_convert_density_curve_variance_squares_factor():
    result = units.convert_density_curve(_CURVE, _TEMP, "imperial", metric="variance")
    assert result["points"] == pytest.approx([0.0, 32.4])
    assert result["density"] == pytest.approx([0.09 / 3.24] * 2)


def test_convert_density_curve_entropy_shifts_only():
    result = units.convert_density_curve(_CURVE, _TEMP, "imperial", metric="entropy")
    shift = math.log(9 / 5)
    assert result["points"] == pytest.approx([shift, 10.0 + shift])
    assert result["density"] == pytest.approx([0.09, 0.09])


def test_convert_density_curve_dimensionless_metric_untouched():
    assert units.convert_density_curve(_CURVE, _TEMP, "imperial", metric="count") is _CURVE


def test_convert_density_curve_converts_small_group_values():
    result = units.convert_density_curve({"count": 2, "values": [0.0, 10.0]}, _TEMP, "imperial", metric="mean")
    assert result["values"] == pytest.approx([32.0, 50.0])

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Any

# (from_unit, to_unit) → (factor, offset)
# Applied as: result = value * factor + offset
# Offset is only meaningful for interval-scale conversions (e.g. °C→°F).
# For ratio-scale variables and spread statistics, offset must not be applied.
_CONVERSION: dict[tuple[str, str], tuple[float, float]] = {
    ("°C", "°F"):         (9 / 5, 32.0),
    ("°F", "°C"):         (5 / 9, -32 * 5 / 9),
    ("°C·days", "°F·days"): (9 / 5, 0.0),
    ("°F·days", "°C·days"): (5 / 9, 0.0),
    ("mm", "in"):         (1 / 25.4, 0.0),
    ("in", "mm"):         (25.4, 0.0),
    ("m", "ft"):          (1 / 0.3048, 0.0),
    ("ft", "m"):          (0.3048, 0.0),
    ("m s⁻¹", "mph"):     (2.2369362921, 0.0),
    ("mph", "m s⁻¹"):     (1 / 2.2369362921, 0.0),
}

# Metrics whose values are dimensionless (counts, fractions, angular stats).
# These are never converted and never carry a unit label.
_DIMENSIONLESS_METRICS = frozenset({
    "count", "unique_samples", "total_samples", "unique_classes",
    "entropy",
    "rbar", "circular_var", "circular_std", "circular_mean",
})

# Summary metrics that measure spread rather than position.
# For these, the conversion offset is never applied even on interval-scale variables
# (e.g. stddev of temperature in °C converts to °F by ×9/5 only, not +32).
_SPREAD_METRICS = frozenset({"std", "stddev", "variance", "range", "iqr", "10_90_range", "bandwidth"})

# Variance is in squared units: Var(T_°F) = (9/5)² × Var(T_°C).
# The base conversion factor must be squared for these metrics.
_SQUARED_FACTOR_METRICS = frozenset({"variance"})


def _is_interval(layer: dict) -> bool:
    return layer.get("value_type") == "interval"


def _apply_offset(layer: dict, metric: str | None) -> bool:
    """True if the unit-conversion offset should be added for this layer+metric combo."""
    return _is_interval(layer) and (metric is None or metric not in _SPREAD_METRICS)


def _convert(value: float, from_unit: str, to_unit: str, *, offset: bool, metric: str | None = None) -> float:
    if from_unit == to_unit:
        return value
    params = _CONVERSION.get((from_unit, to_unit))
    if params is None:
        return value
    factor, off = params
    if metric is not None and metric in _SQUARED_FACTOR_METRICS:
        factor = factor ** 2
    return value * factor + (off if offset else 0.0)


def display_units(layer: dict, unit_system: str | None, *, metric: str | None = None) -> str | None:
    """Return the unit label that should be shown for the given system.

    Returns None for dimensionless metrics regardless of the layer's declared units.
    """
    if metric is not None and metric in _DIMENSIONLESS_METRICS:
        return None
    if unit_system == "imperial":
        return layer.get("imperial_unit") or layer.get("units") or None
    return layer.get("units") or None


def convert_value_from_display(
    value: float,
    layer: dict,
    unit_system: str | None,
    *,
    metric: str | None = None,
) -> float:
    """Convert a value FROM display units BACK TO raw (metric) units."""
    if unit_system != "imperial":
        return value
    if metric is not None and metric in _DIMENSIONLESS_METRICS:
        return value
    from_unit = layer.get("imperial_unit") or ""
    to_unit = layer.get("units") or ""
    if not from_unit or not to_unit or from_unit == to_unit:
        return value
    return _convert(float(value), from_unit, to_unit, offset=_apply_offset(layer, metric), metric=metric)


def convert_value(
    value: float | None,
    layer: dict,
    unit_system: str | None,
    *,
    metric: str | None = None,
) -> float | None:
    """Convert a single scalar value to the target unit system."""
    if value is None or unit_system != "imperial":
        return value
    if metric is not None and metric in _DIMENSIONLESS_METRICS:
        return value
    from_unit = layer.get("units") or ""
    to_unit = layer.get("imperial_unit") or ""
    if not from_unit or not to_unit or from_unit == to_unit:
        return value
    return _convert(float(value), from_unit, to_unit, offset=_apply_offset(layer, metric), metric=metric)


def convert_summary(
    summary: dict[str, Any] | None,
    layer: dict,
    unit_system: str | None,
) -> dict[str, Any] | None:
    if not summary or unit_system != "imperial":
        return summary
    from_unit = layer.get("units") or ""
    to_unit = layer.get("imperial_unit") or ""
    if not from_unit or not to_unit or from_unit == to_unit:
        return summary

    result: dict[str, Any] = {}
    for key, val in summary.items():
        if isinstance(val, (int, float)) and key not in _DIMENSIONLESS_METRICS:
            result[key] = _convert(float(val), from_unit, to_unit, offset=_apply_offset(layer, key), metric=key)
        else:
            result[key] = val
    return result


def convert_density_curve(
    curve: dict[str, Any] | None,
    layer: dict,
    unit_system: str | None,
) -> dict[str, Any] | None:
    if not curve or unit_system != "imperial":
        return curve
    from_unit = layer.get("units") or ""
    to_unit = layer.get("imperial_unit") or ""
    if not from_unit or not to_unit or from_unit == to_unit:
        return curve
    params = _CONVERSION.get((from_unit, to_unit))
    if params is None:
        return curve
    factor, off = params
    use_offset = _apply_offset(layer, None)

    result = dict(curve)
    if "points" in result:
        result["points"] = [v * factor + (off if use_offset else 0.0) for v in result["points"]]
    if "density" in result and factor:
        result["density"] = [v / abs(factor) for v in result["density"]]
    if isinstance(result.get("bandwidth"), (int, float)):
        result["bandwidth"] = result["bandwidth"] * abs(factor)
    for key in ("min", "max"):
        if isinstance(result.get(key), (int, float)):
            result[key] = result[key] * factor + (off if use_offset else 0.0)
    return result

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Iterable, Any

UnitSystem = Literal["metric", "imperial"]


@dataclass(frozen=True)
class ConversionParams:
    factor: float
    offset: float = 0.0


_UNIT_ALIASES: dict[str, str] = {
    "c": "c",
    "celsius": "c",
    "degc": "c",
    "°c": "c",
    "f": "f",
    "fahrenheit": "f",
    "degf": "f",
    "°f": "f",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "mi": "mi",
    "mile": "mi",
    "miles": "mi",
    "kpa": "kpa",
    "kilopascal": "kpa",
    "kilopascals": "kpa",
    "psi": "psi",
}

_UNIT_SYSTEM_BY_UNIT: dict[str, UnitSystem] = {
    "c": "metric",
    "mm": "metric",
    "cm": "metric",
    "m": "metric",
    "km": "metric",
    "f": "imperial",
    "in": "imperial",
    "ft": "imperial",
    "mi": "imperial",
    "kpa": "metric",
    "psi": "imperial",
}

_LENGTH_FACTORS_M: dict[str, float] = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "mi": 1609.344,
}

_EQUIVALENT_UNIT: dict[str, dict[UnitSystem, str]] = {
    "mm": {"metric": "mm", "imperial": "in"},
    "cm": {"metric": "cm", "imperial": "in"},
    "m": {"metric": "m", "imperial": "ft"},
    "km": {"metric": "km", "imperial": "mi"},
    "in": {"metric": "mm", "imperial": "in"},
    "ft": {"metric": "m", "imperial": "ft"},
    "mi": {"metric": "km", "imperial": "mi"},
    "c": {"metric": "c", "imperial": "f"},
    "f": {"metric": "c", "imperial": "f"},
    "kpa": {"metric": "kpa", "imperial": "psi"},
    "psi": {"metric": "kpa", "imperial": "psi"},
}

_DISPLAY_UNIT: dict[str, str] = {
    "c": "°C",
    "f": "°F",
    "kpa": "kPa",
    "psi": "psi",
}


def normalize_unit_system(value: Optional[str]) -> Optional[UnitSystem]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in ("metric", "imperial"):
        return normalized  # type: ignore[return-value]
    return None


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    cleaned = unit.strip().lower().replace("°", "")
    cleaned = cleaned.replace(" ", "")
    return _UNIT_ALIASES.get(cleaned, cleaned if cleaned in _UNIT_SYSTEM_BY_UNIT else None)


def unit_system_for_unit(unit: Optional[str]) -> Optional[UnitSystem]:
    canonical = normalize_unit(unit)
    if not canonical:
        return None
    return _UNIT_SYSTEM_BY_UNIT.get(canonical)


def display_unit(unit: Optional[str]) -> Optional[str]:
    canonical = normalize_unit(unit)
    if not canonical:
        return unit
    return _DISPLAY_UNIT.get(canonical, canonical)


def equivalent_unit(unit: Optional[str], target_system: UnitSystem) -> Optional[str]:
    canonical = normalize_unit(unit)
    if not canonical:
        return None
    mapping = _EQUIVALENT_UNIT.get(canonical)
    if not mapping:
        return canonical
    return mapping.get(target_system, canonical)


def conversion_params(from_unit: str, to_unit: str) -> Optional[ConversionParams]:
    if from_unit == to_unit:
        return ConversionParams(1.0, 0.0)
    if {from_unit, to_unit} == {"c", "f"}:
        if from_unit == "c":
            return ConversionParams(9 / 5, 32.0)
        return ConversionParams(5 / 9, -32.0 * (5 / 9))
    if {from_unit, to_unit} == {"kpa", "psi"}:
        if from_unit == "kpa":
            return ConversionParams(0.1450377377, 0.0)
        return ConversionParams(6.8947572932, 0.0)
    if from_unit in _LENGTH_FACTORS_M and to_unit in _LENGTH_FACTORS_M:
        factor = _LENGTH_FACTORS_M[from_unit] / _LENGTH_FACTORS_M[to_unit]
        return ConversionParams(factor, 0.0)
    return None


def convert_between_units(value: float, from_unit: str, to_unit: str) -> float:
    params = conversion_params(from_unit, to_unit)
    if not params:
        return value
    return value * params.factor + params.offset


def convert_value_for_system(
    value: float | None,
    unit: Optional[str],
    target_system: Optional[UnitSystem],
) -> tuple[float | None, Optional[str]]:
    if value is None or target_system is None:
        return value, display_unit(unit)
    canonical = normalize_unit(unit)
    if not canonical:
        return value, unit
    unit_system = _UNIT_SYSTEM_BY_UNIT.get(canonical)
    if unit_system == target_system:
        return value, display_unit(canonical)
    target_unit = equivalent_unit(canonical, target_system)
    if not target_unit or target_unit == canonical:
        return value, display_unit(canonical)
    converted = convert_between_units(float(value), canonical, target_unit)
    return converted, display_unit(target_unit)


def convert_value_from_system(
    value: float | None,
    unit: Optional[str],
    source_system: Optional[UnitSystem],
) -> float | None:
    if value is None or source_system is None:
        return value
    canonical = normalize_unit(unit)
    if not canonical:
        return value
    unit_system = _UNIT_SYSTEM_BY_UNIT.get(canonical)
    if unit_system == source_system:
        return value
    source_unit = equivalent_unit(canonical, source_system)
    if not source_unit or source_unit == canonical:
        return value
    return convert_between_units(float(value), source_unit, canonical)


def convert_summary(
    summary: Optional[dict[str, Any]],
    unit: Optional[str],
    target_system: Optional[UnitSystem],
) -> Optional[dict[str, Any]]:
    if not summary or not target_system or not unit:
        return summary
    converted: dict[str, Any] = dict(summary)
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            converted[key], _ = convert_value_for_system(float(value), unit, target_system)
    return converted


def convert_density_curve(
    curve: Optional[dict[str, Any]],
    unit: Optional[str],
    target_system: Optional[UnitSystem],
) -> Optional[dict[str, Any]]:
    if not curve or not target_system or not unit:
        return curve
    canonical = normalize_unit(unit)
    if not canonical:
        return curve
    unit_system = _UNIT_SYSTEM_BY_UNIT.get(canonical)
    if unit_system == target_system:
        return curve
    target_unit = equivalent_unit(canonical, target_system)
    if not target_unit or target_unit == canonical:
        return curve
    params = conversion_params(canonical, target_unit)
    if not params:
        return curve
    points = curve.get("points") or []
    density = curve.get("density") or []
    factor = params.factor
    offset = params.offset
    adjusted_density = density
    if factor:
        adjusted_density = [float(value) / abs(factor) for value in density]
    converted = dict(curve)
    converted["points"] = [float(value) * factor + offset for value in points]
    converted["density"] = adjusted_density
    if "min" in converted and isinstance(converted["min"], (int, float)):
        converted["min"] = float(converted["min"]) * factor + offset
    if "max" in converted and isinstance(converted["max"], (int, float)):
        converted["max"] = float(converted["max"]) * factor + offset
    if "bandwidth" in converted and isinstance(converted["bandwidth"], (int, float)):
        converted["bandwidth"] = float(converted["bandwidth"]) * abs(factor)
    return converted


def convert_values_list(
    values: Optional[Iterable[float]],
    unit: Optional[str],
    target_system: Optional[UnitSystem],
) -> Optional[list[float]]:
    if values is None or target_system is None or not unit:
        return None if values is None else list(values)
    converted: list[float] = []
    for value in values:
        new_value, _ = convert_value_for_system(float(value), unit, target_system)
        converted.append(float(new_value) if new_value is not None else float(value))
    return converted


def convert_observations(
    observations: list[dict[str, Any]],
    unit: Optional[str],
    target_system: Optional[UnitSystem],
    value_key: str = "value",
) -> list[dict[str, Any]]:
    if not observations or not unit or not target_system:
        return observations
    converted_rows: list[dict[str, Any]] = []
    for row in observations:
        value = row.get(value_key)
        if isinstance(value, (int, float)):
            converted_value, _ = convert_value_for_system(float(value), unit, target_system)
            updated = dict(row)
            updated[value_key] = converted_value
            converted_rows.append(updated)
        else:
            converted_rows.append(row)
    return converted_rows


def apply_unit_system_to_variables(
    variables: list[dict[str, Any]],
    unit_system: Optional[str],
) -> list[dict[str, Any]]:
    resolved = normalize_unit_system(unit_system)
    if not resolved:
        return variables
    converted: list[dict[str, Any]] = []
    for entry in variables:
        unit_label = entry.get("units")
        target_unit = equivalent_unit(unit_label, resolved) or unit_label
        converted.append({**entry, "units": display_unit(target_unit)})
    return converted


def apply_unit_system_to_env_response(
    response: dict[str, Any],
    unit_system: Optional[str],
    unit: Optional[str],
) -> dict[str, Any]:
    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return response
    updated = dict(response)
    target_unit = equivalent_unit(unit, resolved) or unit
    display = display_unit(target_unit)
    updated["units"] = display
    if isinstance(updated.get("variable_metadata"), dict):
        metadata = dict(updated["variable_metadata"])
        metadata["units"] = display
        updated["variable_metadata"] = metadata
    summary = convert_summary(updated.get("summary"), unit, resolved)
    updated["summary"] = summary
    baseline_summary = convert_summary(updated.get("baselineSummary"), unit, resolved)
    updated["baselineSummary"] = baseline_summary
    updated["baseline_summary"] = baseline_summary
    density_curve = convert_density_curve(updated.get("densityCurve"), unit, resolved)
    updated["densityCurve"] = density_curve
    updated["density_curve"] = density_curve
    return updated


def apply_unit_system_to_slice_response(
    response: dict[str, Any],
    unit_system: Optional[str],
    unit: Optional[str],
) -> dict[str, Any]:
    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return response
    updated = dict(response)
    updated["observations"] = convert_observations(
        updated.get("observations", []), unit, resolved
    )
    min_value = updated.get("range", {}).get("min")
    max_value = updated.get("range", {}).get("max")
    min_value, display = convert_value_for_system(min_value, unit, resolved)
    max_value, _display = convert_value_for_system(max_value, unit, resolved)
    updated["range"] = {"min": min_value, "max": max_value}
    updated["units"] = display
    return updated


def apply_unit_system_to_rankings_response(
    response: dict[str, Any],
    unit_system: Optional[str],
    unit: Optional[str],
) -> dict[str, Any]:
    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return response
    updated = dict(response)
    target_unit = equivalent_unit(unit, resolved) or unit
    updated["units"] = display_unit(target_unit)
    entries = []
    for entry in updated.get("entries", []):
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
            converted_value, _ = convert_value_for_system(
                float(entry["value"]), unit, resolved
            )
            updated_entry = dict(entry)
            updated_entry["value"] = converted_value
            entries.append(updated_entry)
        else:
            entries.append(entry)
    updated["entries"] = entries
    updated["distribution"] = convert_density_curve(
        updated.get("distribution"), unit, resolved
    )
    return updated

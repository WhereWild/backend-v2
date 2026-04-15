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
    "pa": "pa",
    "pascal": "pa",
    "pascals": "pa",
    "psi": "psi",
    "m/s": "mps",
    "ms-1": "mps",
    "ms^-1": "mps",
    "mps": "mps",
    "mph": "mph",
    "milesperhour": "mph",
    "kg/m^2/year": "mm_per_year",
    "kg/m2/year": "mm_per_year",
    "kgm-2year-1": "mm_per_year",
    "mm/year": "mm_per_year",
    "mm/yr": "mm_per_year",
    "in/year": "in_per_year",
    "in/yr": "in_per_year",
    "g/kg": "g_per_kg",
    "gkg": "g_per_kg",
    "gramperkilogram": "g_per_kg",
    "gramsperkilogram": "g_per_kg",
    "lb/ton": "lb_per_ton",
    "lbs/ton": "lb_per_ton",
    "lbton": "lb_per_ton",
    "poundperton": "lb_per_ton",
    "poundsperton": "lb_per_ton",
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
    "pa": "metric",
    "kpa": "metric",
    "psi": "imperial",
    "mps": "metric",
    "mph": "imperial",
    "mm_per_year": "metric",
    "in_per_year": "imperial",
    "g_per_kg": "metric",
    "lb_per_ton": "imperial",
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
    "pa": {"metric": "pa", "imperial": "psi"},
    "kpa": {"metric": "kpa", "imperial": "psi"},
    "psi": {"metric": "kpa", "imperial": "psi"},
    "mps": {"metric": "mps", "imperial": "mph"},
    "mph": {"metric": "mps", "imperial": "mph"},
    "mm_per_year": {"metric": "mm_per_year", "imperial": "in_per_year"},
    "in_per_year": {"metric": "mm_per_year", "imperial": "in_per_year"},
    "g_per_kg": {"metric": "g_per_kg", "imperial": "lb_per_ton"},
    "lb_per_ton": {"metric": "g_per_kg", "imperial": "lb_per_ton"},
}

_DISPLAY_UNIT: dict[str, str] = {
    "c": "°C",
    "f": "°F",
    "kpa": "kPa",
    "pa": "Pa",
    "psi": "psi",
    "mps": "m/s",
    "mph": "mph",
    "mm_per_year": "mm/year",
    "in_per_year": "in/year",
    "g_per_kg": "g/kg",
    "lb_per_ton": "lb/ton",
}

_VARIABLE_DISPLAY_SCALE: dict[str, float] = {
    # Cloud area fraction is stored as hundredths of percent in source rasters.
    "clt": 0.01,
    # SWE is stored as kg/m²/year * 10 in CHELSA source rasters.
    "swe": 0.1,
    # Near-surface wind speed is stored as m/s * 100 in CHELSA source rasters.
    "sfc": 0.01,
    # Vapor pressure deficit is stored as Pa * 10 in CHELSA source rasters.
    "vpd": 0.1,
    # Coarse fragments are stored as cm^3/dm^3 (x10 of percent by volume).
    "cfvo": 0.1,
    # Soil texture fractions are stored as g/kg; display as percent by mass.
    "clay": 0.1,
    "sand": 0.1,
    "silt": 0.1,
    # Nitrogen is stored as cg/kg in source rasters; display as percent.
    "nitrogen": 0.001,
    # Soil organic carbon is stored as dg/kg in source rasters; display as percent.
    "soc": 0.01,
    # pH is stored as pH*10 in source rasters.
    "phh2o": 0.1,
}

_SUMMARY_CONVERTIBLE_KEYS = {
    "min",
    "max",
    "mean",
    "median",
    "std",
    "stddev",
    "q01",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q99",
    "1st percentile",
    "10th percentile",
    "25th percentile",
    "75th percentile",
    "90th percentile",
    "99th percentile",
    "10-90 range",
    "1-99 range",
    "range",
}


def variable_display_scale(variable_id: Optional[str]) -> float:
    if not variable_id:
        return 1.0
    return float(_VARIABLE_DISPLAY_SCALE.get(str(variable_id).strip().lower(), 1.0))


def _scale_number(value: Any, factor: float) -> Any:
    if factor == 1.0:
        return value
    if isinstance(value, (int, float)):
        return float(value) * factor
    return value


def _scale_summary(summary: Optional[dict[str, Any]], factor: float) -> Optional[dict[str, Any]]:
    if not summary or factor == 1.0:
        return summary
    scaled = dict(summary)
    for key, value in summary.items():
        if isinstance(value, (int, float)) and str(key).strip().lower() in _SUMMARY_CONVERTIBLE_KEYS:
            scaled[key] = float(value) * factor
    return scaled


def _scale_density_curve(
    curve: Optional[dict[str, Any]],
    factor: float,
) -> Optional[dict[str, Any]]:
    if not curve or factor == 1.0:
        return curve
    adjusted = dict(curve)
    points = adjusted.get("points") or []
    density = adjusted.get("density") or []
    adjusted["points"] = [float(value) * factor for value in points]
    if factor:
        adjusted["density"] = [float(value) / abs(factor) for value in density]
    for key in ("min", "max", "bandwidth"):
        if isinstance(adjusted.get(key), (int, float)):
            adjusted[key] = float(adjusted[key]) * abs(factor) if key == "bandwidth" else float(adjusted[key]) * factor
    return adjusted


def _scale_observations(
    observations: list[dict[str, Any]],
    factor: float,
    *,
    value_key: str = "value",
) -> list[dict[str, Any]]:
    if factor == 1.0 or not observations:
        return observations
    scaled: list[dict[str, Any]] = []
    for row in observations:
        value = row.get(value_key)
        if isinstance(value, (int, float)):
            updated = dict(row)
            updated[value_key] = float(value) * factor
            scaled.append(updated)
        else:
            scaled.append(row)
    return scaled


def _scale_rank_entries(entries: list[dict[str, Any]], factor: float) -> list[dict[str, Any]]:
    if factor == 1.0 or not entries:
        return entries
    scaled: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
            updated = dict(entry)
            updated["value"] = float(entry["value"]) * factor
            scaled.append(updated)
        else:
            scaled.append(entry)
    return scaled


def convert_value_from_display(value: float | None, variable_id: Optional[str]) -> float | None:
    if value is None:
        return value
    factor = variable_display_scale(variable_id)
    if factor == 1.0:
        return value
    if factor == 0:
        return value
    return float(value) / factor


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
    if {from_unit, to_unit} == {"pa", "kpa"}:
        if from_unit == "pa":
            return ConversionParams(0.001, 0.0)
        return ConversionParams(1000.0, 0.0)
    if {from_unit, to_unit} == {"pa", "psi"}:
        if from_unit == "pa":
            return ConversionParams(0.0001450377377, 0.0)
        return ConversionParams(6894.7572932, 0.0)
    if {from_unit, to_unit} == {"mps", "mph"}:
        if from_unit == "mps":
            return ConversionParams(2.2369362921, 0.0)
        return ConversionParams(0.44704, 0.0)
    if {from_unit, to_unit} == {"mm_per_year", "in_per_year"}:
        if from_unit == "mm_per_year":
            return ConversionParams(1.0 / 25.4, 0.0)
        return ConversionParams(25.4, 0.0)
    if {from_unit, to_unit} == {"g_per_kg", "lb_per_ton"}:
        if from_unit == "g_per_kg":
            return ConversionParams(2.0, 0.0)
        return ConversionParams(0.5, 0.0)
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
        if isinstance(value, (int, float)) and str(key).strip().lower() in _SUMMARY_CONVERTIBLE_KEYS:
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


def apply_unit_system_to_query_rows(
    rows: list[dict[str, Any]],
    unit_system: Optional[str],
    *,
    variable_id: Optional[str],
    unit: Optional[str],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    converted_rows = [dict(row) for row in rows]
    scale = variable_display_scale(variable_id)
    if scale != 1.0:
        for row in converted_rows:
            sort_value = row.get("sort_value")
            if isinstance(sort_value, (int, float)):
                row["sort_value"] = float(sort_value) * scale

    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return converted_rows, unit

    target_unit = equivalent_unit(unit, resolved) or unit
    display = display_unit(target_unit)
    for row in converted_rows:
        sort_value = row.get("sort_value")
        if isinstance(sort_value, (int, float)):
            converted_value, _ = convert_value_for_system(float(sort_value), unit, resolved)
            row["sort_value"] = converted_value
    return converted_rows, display


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
    updated = dict(response)
    scale = variable_display_scale(updated.get("variable"))
    if scale != 1.0:
        updated["summary"] = _scale_summary(updated.get("summary"), scale)
        updated["baselineSummary"] = _scale_summary(updated.get("baselineSummary"), scale)
        updated["baseline_summary"] = _scale_summary(updated.get("baseline_summary"), scale)
        scaled_curve = _scale_density_curve(updated.get("densityCurve"), scale)
        updated["densityCurve"] = scaled_curve
        updated["density_curve"] = _scale_density_curve(updated.get("density_curve"), scale) or scaled_curve
        updated["binSamples"] = _scale_observations(updated.get("binSamples", []), scale)
        updated["bin_samples"] = _scale_observations(updated.get("bin_samples", []), scale)

    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return updated
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
    updated = dict(response)
    scale = variable_display_scale(updated.get("variable"))
    if scale != 1.0:
        updated["observations"] = _scale_observations(updated.get("observations", []), scale)
        range_obj = dict(updated.get("range") or {})
        range_obj["min"] = _scale_number(range_obj.get("min"), scale)
        range_obj["max"] = _scale_number(range_obj.get("max"), scale)
        updated["range"] = range_obj

    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return updated
    updated["observations"] = convert_observations(updated.get("observations", []), unit, resolved)
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
    updated = dict(response)
    scale = variable_display_scale(updated.get("variable"))
    if scale != 1.0:
        updated["entries"] = _scale_rank_entries(updated.get("entries", []), scale)
        updated["distribution"] = _scale_density_curve(updated.get("distribution"), scale)

    resolved = normalize_unit_system(unit_system)
    if not resolved or not unit:
        return updated
    target_unit = equivalent_unit(unit, resolved) or unit
    updated["units"] = display_unit(target_unit)
    entries = []
    for entry in updated.get("entries", []):
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
            converted_value, _ = convert_value_for_system(float(entry["value"]), unit, resolved)
            updated_entry = dict(entry)
            updated_entry["value"] = converted_value
            entries.append(updated_entry)
        else:
            entries.append(entry)
    updated["entries"] = entries
    updated["distribution"] = convert_density_curve(updated.get("distribution"), unit, resolved)
    return updated

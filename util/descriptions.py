"""Simple rule-based taxon descriptions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional
import math

from util import summary_stats, units
from util.config import load_config

_LOCATION_CATEGORY_SAMPLE_LIMIT = 500

# ---------------------------------------------------------------------------
# Shared qualifier comparison table (used by categorical outlier functions)
# ---------------------------------------------------------------------------

_QUALIFIER_COMPARISON = {
    "extremely": ("much more common", "much less common"),
    "very": ("more common", "less common"),
    "quite": ("a bit more common", "a bit less common"),
}

_CATEGORICAL_FALLBACK_LABELS: dict[str, str] = {
    "landcover": "these habitats",
    "koppen_geiger": "these climates",
}

# ---------------------------------------------------------------------------
# Label conversions
# ---------------------------------------------------------------------------


def _winter_coldness_label(celsius: float) -> str:
    if celsius < -40:
        return "extremely cold"
    if celsius < -30:
        return "incredibly cold"
    if celsius < -20:
        return "very cold"
    if celsius < -10:
        return "quite cold"
    if celsius < 0:
        return "cold"
    if celsius < 10:
        return "cool"
    if celsius < 20:
        return "temperate"
    if celsius < 30:
        return "warm"
    return "hot"


def _summer_heat_label(celsius: float) -> str:
    if celsius > 40:
        return "scorching"
    if celsius > 35:
        return "very hot"
    if celsius > 30:
        return "hot"
    if celsius > 20:
        return "warm"
    if celsius > 10:
        return "temperate"
    return "cool"


def _annual_precip_label(mm: float) -> str:
    if mm < 50:
        return "extremely xeric"
    if mm < 150:
        return "xeric"
    if mm < 250:
        return "arid"
    if mm < 400:
        return "semi-arid"
    if mm < 500:
        return "dry"
    if mm < 800:
        return "subhumid"
    if mm < 1000:
        return "moderately wet"
    if mm < 1200:
        return "wet"
    if mm < 1500:
        return "very wet"
    if mm < 2000:
        return "incredibly wet"
    if mm < 3000:
        return "extremely wet"
    return "torrential"


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sentence_case(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _capitalize_leading_the(text: Optional[str]) -> Optional[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return text
    if cleaned.lower().startswith("the "):
        return f"The {cleaned[4:]}"
    return cleaned


def _to_natural_habitat_name(text: str) -> str:
    normalized = text.strip().lower()
    replacements = (
        (r"\bgrassland\b", "grasslands"),
        (r"\bshrubland\b", "shrublands"),
        (r"\bwetland\b", "wetlands"),
        (r"\bforest\b", "forests"),
        (r"\bdesert\b", "deserts"),
        (r"\bsavanna\b", "savannas"),
        (r"\btundra\b", "tundras"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _to_natural_climate_name(text: str) -> str:
    stripped = re.sub(r"\([^)]*\)\s*", "", text).strip().lower()
    return stripped


def _strip_phrase(phrase: str) -> tuple[str, str]:
    for prefix in ("often in ", "primarily in ", "across a broad range of "):
        if phrase.startswith(prefix):
            return prefix, phrase[len(prefix) :].strip()
    return "", phrase.strip()


def _ensure_climate_suffix(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if " and " in lowered:
        parts = [part.strip() for part in cleaned.split(" and ")]
        stripped: list[str] = []
        for part in parts:
            lower_part = part.lower()
            if lower_part.endswith("climates"):
                stripped.append(part[: -len("climates")].rstrip())
            elif lower_part.endswith("climate"):
                stripped.append(part[: -len("climate")].rstrip())
            else:
                stripped.append(part)
        return f"{' and '.join(stripped)} climates"
    if "climate" in lowered:
        return cleaned
    return f"{cleaned} climates"


def _format_categorical_phrase(phrase: str, *, label: str) -> str:
    prefix, remainder = _strip_phrase(phrase.strip())
    if label == "habitat":
        remainder = _to_natural_habitat_name(remainder)
    if label == "climate":
        remainder = _to_natural_climate_name(remainder)
        remainder = _ensure_climate_suffix(remainder)
    return f"{prefix}{remainder}".strip()


def _parse_class_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.search(r"\bclass[_\s-]*(\d+)\b", text, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _extract_koppen_code(*values: Any) -> Optional[str]:
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        match = re.search(r"\(([A-Za-z]{2,3})\)", text)
        if match:
            return match.group(1).upper()
        token_match = re.search(r"\b([ABCDE][A-Za-z]{1,2})\b", text)
        if token_match:
            return token_match.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Group / trait helpers
# ---------------------------------------------------------------------------


def _landcover_forest_openness(name: str) -> str:
    lowered = name.lower()
    if "open" in lowered:
        return "sparse"
    if "closed" in lowered:
        return "dense"
    return "other"


def _landcover_forest_phenology(name: str) -> str:
    lowered = name.lower()
    if "evergreen" in lowered:
        return "evergreen"
    if "deciduous" in lowered:
        return "deciduous"
    return "generic"


def _normalized_group_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "none", "null", "nan"}:
        return ""
    return token


def _infer_landcover_group(name: str, class_id: Optional[int]) -> tuple[str, str]:
    if class_id in {10, 11, 12, 20}:
        return "cropland", "Cropland"
    if class_id in {51, 52, 61, 62, 71, 72, 81, 82, 91, 92}:
        return "forest", "Forest"
    if class_id in {120, 121, 122}:
        return "shrubland", "Shrubland"
    if class_id == 130:
        return "grassland", "Grassland"
    if class_id == 140:
        return "lichens_mosses", "Lichens and Mosses"
    if class_id in {150, 152, 153}:
        return "sparse_vegetation", "Sparse Vegetation"
    if class_id == 180:
        return "wetlands", "Wetlands"
    if class_id == 190:
        return "urban", "Urban"
    if class_id in {200, 201, 202}:
        return "bare_areas", "Bare Areas"
    if class_id == 210:
        return "water", "Water"
    if class_id == 220:
        return "ice_snow", "Ice and Snow"
    if class_id == 250:
        return "filled", "Filled"

    lowered = name.lower()
    if "forest" in lowered:
        return "forest", "Forest"
    if "cropland" in lowered or "orchard" in lowered:
        return "cropland", "Cropland"
    if "shrubland" in lowered:
        return "shrubland", "Shrubland"
    if "grassland" in lowered:
        return "grassland", "Grassland"
    if "wetland" in lowered:
        return "wetlands", "Wetlands"
    if "water" in lowered:
        return "water", "Water"
    if "bare" in lowered:
        return "bare_areas", "Bare Areas"
    if "impervious" in lowered or "urban" in lowered:
        return "urban", "Urban"
    if "ice" in lowered or "snow" in lowered:
        return "ice_snow", "Ice and Snow"
    if "lichen" in lowered or "moss" in lowered:
        return "lichens_mosses", "Lichens and Mosses"
    if "sparse" in lowered:
        return "sparse_vegetation", "Sparse Vegetation"
    return "", ""


def _landcover_group_label(group: str, group_label: str) -> str:
    normalized = group.strip().lower()
    mapping = {
        "cropland": "croplands",
        "forest": "forests",
        "shrubland": "shrublands",
        "grassland": "grasslands",
        "lichens_mosses": "lichens and mosses",
        "sparse_vegetation": "sparse vegetation",
        "wetlands": "wetlands",
        "urban": "urban areas",
        "bare_areas": "bare areas",
        "water": "water bodies",
        "ice_snow": "ice and snow",
        "filled": "filled areas",
    }
    if normalized in mapping:
        return mapping[normalized]
    cleaned_label = group_label.strip().lower()
    if cleaned_label:
        return cleaned_label
    return normalized.replace("_", " ") if normalized else "landcover classes"


# ---------------------------------------------------------------------------
# Layer rules + semantic labels
# ---------------------------------------------------------------------------

_CATEGORICAL_LAYER_RULES: dict[str, dict[str, Any]] = {
    "landcover": {
        "default_style": "group_map",
        "split_groups": {
            "forest": {
                "base_label": "forests",
                "dimensions": [
                    {"key": "openness", "order": ["sparse", "dense"], "combine": "join"},
                    {"key": "phenology", "order": ["evergreen", "deciduous"], "combine": "drop"},
                ],
            },
        },
    },
    "koppen_geiger": {
        "default_style": "climate_suffix",
        "split_groups": {
            "desert": {
                "base_label": "desert climates",
                "dimensions": [
                    {
                        "key": "thermal",
                        "order": ["hot", "warm", "cold", "severe-winter"],
                        "combine": "join",
                    }
                ],
            },
            "steppe": {
                "base_label": "steppe climates",
                "dimensions": [
                    {
                        "key": "thermal",
                        "order": ["hot", "warm", "cold", "severe-winter"],
                        "combine": "join",
                    }
                ],
            },
            "mediterranean": {
                "base_label": "mediterranean climates",
                "dimensions": [
                    {
                        "key": "thermal",
                        "order": ["hot", "warm", "cool", "severe-winter"],
                        "combine": "join",
                    }
                ],
            },
            "continental": {
                "base_label": "continental climates",
                "dimensions": [
                    {
                        "key": "thermal",
                        "order": ["hot", "warm", "cool", "severe-winter"],
                        "combine": "join",
                    }
                ],
            },
            "subpolar": {
                "base_label": "subpolar climates",
                "dimensions": [
                    {
                        "key": "thermal",
                        "order": ["hot", "warm", "cool", "severe-winter"],
                        "combine": "join",
                    }
                ],
            },
        },
    },
}


def _semantic_default_label(variable_id: str, group: str, group_label: str) -> str:
    style = str((_CATEGORICAL_LAYER_RULES.get(variable_id) or {}).get("default_style") or "").strip()
    if style == "group_map":
        return _landcover_group_label(group, group_label)
    if style == "climate_suffix":
        base = group_label or group
        return _ensure_climate_suffix(_to_natural_climate_name(base))
    return str(group_label or group).replace("_", " ").strip()


def _trait_string(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "none", "null", "nan"}:
        return ""
    return token


def _derive_koppen_thermal(code: str) -> str:
    normalized = str(code or "").strip().upper()
    if not normalized:
        return ""
    if normalized.startswith("B") and len(normalized) >= 3:
        if normalized[2] == "H":
            return "hot"
        if normalized[2] == "K":
            return "cold"
        return ""
    if len(normalized) >= 3:
        return {"A": "hot", "B": "warm", "C": "cool", "D": "severe-winter"}.get(
            normalized[2], ""
        ) or ""
    if normalized == "ET":
        return "cold"
    if normalized == "EF":
        return "severe-winter"
    return ""


def _extract_legend_traits(legend_entry: Optional[dict[str, Any]]) -> dict[str, str]:
    traits: dict[str, str] = {}
    if not isinstance(legend_entry, dict):
        return traits
    legend_traits = legend_entry.get("traits")
    if not isinstance(legend_traits, dict):
        return traits
    for key, raw_value in legend_traits.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        value_text = _trait_string(raw_value)
        if value_text:
            traits[key_text] = value_text
    return traits


def _semantic_label_from_group(
    *,
    variable_id: str,
    group: str,
    group_label: str,
    traits: dict[str, str],
) -> tuple[str, Optional[dict[str, Any]]]:
    layer_rules = _CATEGORICAL_LAYER_RULES.get(variable_id) or {}
    split_rules = layer_rules.get("split_groups") or {}
    split_rule = split_rules.get(group)
    if not split_rule:
        return _semantic_default_label(variable_id, group, group_label), None

    base_label = str(split_rule.get("base_label") or "").strip()
    if not base_label:
        base_label = _semantic_default_label(variable_id, group, group_label)
    dims = split_rule.get("dimensions") or []
    values: dict[str, str] = {}
    label_parts: list[str] = []
    normalized_dims: list[dict[str, Any]] = []
    for dim in dims:
        key = str(dim.get("key") or "").strip()
        if not key:
            continue
        order = [str(v).strip().lower() for v in (dim.get("order") or []) if str(v).strip()]
        combine = str(dim.get("combine") or "join").strip().lower()
        value = _trait_string(traits.get(key))
        if value and (not order or value in order):
            values[key] = value
            label_parts.append(value)
        normalized_dims.append({"key": key, "order": order, "combine": combine})

    label = f"{' '.join(label_parts)} {base_label}".strip() if label_parts else base_label
    semantic = {
        "group": group,
        "base_label": base_label,
        "dimensions": normalized_dims,
        "values": values,
    }
    return label, semantic


def _combine_semantic_entries(first: dict[str, Any], second: dict[str, Any]) -> Optional[str]:
    first_meta = first.get("_semantic")
    second_meta = second.get("_semantic")
    if not isinstance(first_meta, dict) or not isinstance(second_meta, dict):
        return None
    if first_meta.get("group") != second_meta.get("group"):
        return None
    if first_meta.get("base_label") != second_meta.get("base_label"):
        return None

    dims = first_meta.get("dimensions")
    second_dims = second_meta.get("dimensions")
    if not isinstance(dims, list) or not isinstance(second_dims, list) or dims != second_dims:
        return None

    first_values = first_meta.get("values")
    second_values = second_meta.get("values")
    if not isinstance(first_values, dict) or not isinstance(second_values, dict):
        return None

    dim_keys = [str(dim.get("key") or "").strip() for dim in dims if str(dim.get("key") or "").strip()]
    differing: list[str] = []
    for key in dim_keys:
        if _trait_string(first_values.get(key)) != _trait_string(second_values.get(key)):
            differing.append(key)
    if len(differing) != 1:
        return None
    varying = differing[0]
    dim_config = next((dim for dim in dims if str(dim.get("key") or "") == varying), {})
    mode = str(dim_config.get("combine") or "join").strip().lower()
    order = [str(v).strip().lower() for v in (dim_config.get("order") or []) if str(v).strip()]
    first_variant = _trait_string(first_values.get(varying))
    second_variant = _trait_string(second_values.get(varying))
    if not first_variant or not second_variant:
        return None

    variant_values = [first_variant, second_variant]
    if order:
        order_idx = {value: idx for idx, value in enumerate(order)}
        variant_values = sorted(variant_values, key=lambda value: order_idx.get(value, 10_000))
    if variant_values[0] == variant_values[1]:
        variant_values = [variant_values[0]]

    composed_parts: list[str] = []
    for dim in dims:
        key = str(dim.get("key") or "").strip()
        if not key:
            continue
        if key == varying:
            if mode == "drop":
                continue
            if len(variant_values) == 1:
                composed_parts.append(variant_values[0])
            else:
                composed_parts.append(f"{variant_values[0]} and {variant_values[1]}")
            continue
        shared_first = _trait_string(first_values.get(key))
        shared_second = _trait_string(second_values.get(key))
        if shared_first and shared_first == shared_second:
            composed_parts.append(shared_first)

    base_label = str(first_meta.get("base_label") or "").strip()
    if not base_label:
        return None
    if composed_parts:
        return f"{' '.join(composed_parts)} {base_label}".strip()
    return base_label


# ---------------------------------------------------------------------------
# Extracted helpers (were nested inside _top_categorical_phrase_from_payload)
# ---------------------------------------------------------------------------


def _sanitize_label(value: str) -> str:
    cleaned = str(value or "").strip()
    lowered = cleaned.lower()
    if lowered in {"none", "null", "nan"}:
        return ""
    if lowered.startswith("none "):
        cleaned = cleaned[5:].strip()
    return cleaned


def _legend_for_entry(entry: dict[str, Any], legend_lookup: dict[str, Any]) -> Optional[dict[str, Any]]:
    class_id = _parse_class_id(entry.get("value"))
    if class_id is None:
        class_id = _parse_class_id(entry.get("class_name"))
    if class_id is not None:
        return legend_lookup.get(str(class_id))
    slug = str(entry.get("slug") or "").strip()
    if slug:
        return legend_lookup.get(slug)
    return None


def _resolve_entry_name(entry: dict[str, Any], legend_lookup: dict[str, Any]) -> str:
    legend_entry = _legend_for_entry(entry, legend_lookup)
    short_name = str(entry.get("short_name") or "").strip()
    class_name = str(entry.get("class_name") or "").strip()
    generic_pattern = re.compile(r"^class[_\s-]*\d+$", flags=re.IGNORECASE)
    short_is_generic = bool(short_name and generic_pattern.match(short_name))
    class_is_generic = bool(class_name and generic_pattern.match(class_name))
    if legend_entry:
        if short_name and not short_is_generic:
            return short_name
        if class_name and not class_is_generic:
            return class_name
        legend_short = legend_entry.get("short_name")
        if legend_short:
            return str(legend_short).strip()
        legend_name = legend_entry.get("name")
        if legend_name:
            return str(legend_name).strip()
    return str(short_name or class_name or entry.get("value") or "").strip()


def _combine_parallel_labels(first: str, second: str) -> str:
    first_clean = _sanitize_label(first)
    second_clean = _sanitize_label(second)
    if not first_clean and not second_clean:
        return ""
    if not first_clean:
        return second_clean
    if not second_clean:
        return first_clean
    first_parts = first_clean.split()
    second_parts = second_clean.split()
    if (
        len(first_parts) >= 3
        and len(second_parts) >= 3
        and len(first_parts) == len(second_parts)
        and first_parts[0] == second_parts[0]
        and first_parts[-1] == second_parts[-1]
    ):
        first_middle = " ".join(first_parts[1:-1]).strip()
        second_middle = " ".join(second_parts[1:-1]).strip()
        if first_middle and second_middle:
            return (
                f"{first_parts[0]} {first_middle} and {second_middle} "
                f"{first_parts[-1]}"
            )
    if (
        len(first_parts) >= 2
        and len(second_parts) >= 2
        and len(first_parts) == len(second_parts)
        and first_parts[-1] == second_parts[-1]
    ):
        return f"{' '.join(first_parts[:-1])} and {' '.join(second_parts[:-1])} {first_parts[-1]}"
    return f"{first_clean} and {second_clean}"


def _combine_parallel_entries(first: dict[str, Any], second: dict[str, Any]) -> str:
    semantic_combined = _combine_semantic_entries(first, second)
    if semantic_combined:
        return semantic_combined
    return _combine_parallel_labels(
        str(first.get("name") or "").strip(),
        str(second.get("name") or "").strip(),
    )


def _is_semantically_subsumed_by_primary(
    candidate: dict[str, Any],
    primaries: list[dict[str, Any]],
) -> bool:
    candidate_meta = candidate.get("_semantic")
    if not isinstance(candidate_meta, dict) or len(primaries) < 2:
        return False
    primary_meta = [entry.get("_semantic") for entry in primaries]
    if not all(isinstance(meta, dict) for meta in primary_meta):
        return False

    group = candidate_meta.get("group")
    base_label = candidate_meta.get("base_label")
    dims = candidate_meta.get("dimensions")
    candidate_values = candidate_meta.get("values")
    if not group or not base_label or not isinstance(dims, list) or not isinstance(candidate_values, dict):
        return False

    for meta in primary_meta:
        if meta.get("group") != group or meta.get("base_label") != base_label:
            return False
        if meta.get("dimensions") != dims:
            return False
        values = meta.get("values")
        if not isinstance(values, dict):
            return False
        for key, value in candidate_values.items():
            if _trait_string(values.get(key)) != _trait_string(value):
                return False
    dim_keys = [str(dim.get("key") or "").strip() for dim in dims if str(dim.get("key") or "").strip()]
    return any(_trait_string(candidate_values.get(key)) == "" for key in dim_keys)


def _frequency_verb(frac: float) -> Optional[str]:
    if frac >= 1.0:
        return "always"
    if frac > 0.80:
        return "almost always"
    if frac > 0.50:
        return "primarily"
    if frac > 0.30:
        return "often"
    if frac > 0.20:
        return "sometimes"
    if frac > 0.10:
        return "rarely"
    return None


def _secondary_frequency_verb(frac: float) -> str:
    if frac < 0.10:
        return "rarely"
    if frac < 0.25:
        return "sometimes"
    return "often"


def _combine_entry_pair_or_single(entries: list[dict[str, Any]]) -> str:
    def _name(entry: dict[str, Any]) -> str:
        return str(entry.get("name") or "").strip()

    cleaned = [entry for entry in entries if _name(entry)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return _name(cleaned[0])
    return _combine_parallel_entries(cleaned[0], cleaned[1])


# ---------------------------------------------------------------------------
# Categorical phrase building
# ---------------------------------------------------------------------------


def _top_categorical_phrase_from_payload(
    *,
    variable_id: str,
    label: str,
    payload: dict[str, Any],
) -> Optional[str]:
    from util import gis_lookup

    distribution = payload.get("distribution") or []
    if not distribution:
        return None
    legend_lookup = gis_lookup.load_layer_legend(variable_id)

    # --- Aggregation phase ---
    grouped_counts: dict[str, int] = {}
    ungrouped_count = 0
    aggregated_totals: dict[str, float] = {}
    aggregated_meta: dict[str, dict[str, Any]] = {}

    for entry in distribution:
        fraction = float(entry.get("fraction") or 0.0)
        if fraction <= 0:
            continue
        legend_entry = _legend_for_entry(entry, legend_lookup)
        name = _resolve_entry_name(entry, legend_lookup)
        class_id = _parse_class_id(entry.get("value"))
        if class_id is None:
            class_id = _parse_class_id(entry.get("class_name"))

        group_value = (
            _normalized_group_token(entry.get("group"))
            or _normalized_group_token(legend_entry.get("group") if legend_entry else "")
        )
        group_label = str(
            entry.get("group_label")
            or (legend_entry.get("group_label") if legend_entry else "")
            or group_value
        ).strip()
        if not group_value and variable_id == "landcover":
            inferred_group, inferred_group_label = _infer_landcover_group(name, class_id)
            group_value = inferred_group
            if not group_label:
                group_label = inferred_group_label

        traits = _extract_legend_traits(legend_entry)

        if variable_id == "landcover" and group_value == "forest":
            openness_fallback = _landcover_forest_openness(name)
            if openness_fallback in {"sparse", "dense"} and not traits.get("openness"):
                traits["openness"] = openness_fallback
            phenology_fallback = _landcover_forest_phenology(name)
            if phenology_fallback and phenology_fallback != "generic" and not traits.get("phenology"):
                traits["phenology"] = phenology_fallback

        if variable_id == "koppen_geiger":
            code = _trait_string(traits.get("code"))
            if not code:
                code = _extract_koppen_code(
                    entry.get("short_name"),
                    entry.get("class_name"),
                    entry.get("value"),
                    legend_entry.get("name") if legend_entry else None,
                ) or ""
                code = code.upper()
                if code:
                    traits["code"] = code
            if not traits.get("thermal"):
                thermal = _derive_koppen_thermal(code)
                if thermal:
                    traits["thermal"] = thermal

        if group_value:
            grouped_counts[group_value] = grouped_counts.get(group_value, 0) + 1
        else:
            ungrouped_count += 1

        semantic: Optional[dict[str, Any]] = None
        if group_value:
            label_name, semantic = _semantic_label_from_group(
                variable_id=variable_id,
                group=group_value,
                group_label=group_label,
                traits=traits,
            )
        else:
            label_name = str(name).strip()
            if variable_id == "koppen_geiger":
                label_name = _ensure_climate_suffix(_to_natural_climate_name(label_name))

        normalized_name = _sanitize_label(label_name)
        if not normalized_name:
            continue
        aggregated_totals[normalized_name] = aggregated_totals.get(normalized_name, 0.0) + fraction
        if semantic and normalized_name not in aggregated_meta:
            aggregated_meta[normalized_name] = semantic

    aggregated: list[dict[str, Any]] = []
    for label_name, fraction in aggregated_totals.items():
        row = {"name": label_name, "fraction": fraction}
        semantic = aggregated_meta.get(label_name)
        if semantic:
            row["_semantic"] = semantic
        aggregated.append(row)
    aggregated = [
        entry
        for entry in aggregated
        if str(entry.get("name") or "").strip().lower() not in {"", "none", "null", "nan"}
    ]
    ranking_entries = aggregated
    if not ranking_entries:
        return None

    # --- Ranking phase ---
    ranked = sorted(ranking_entries, key=lambda row: float(row.get("fraction") or 0.0), reverse=True)
    top = ranked[0]
    top_name = str(top.get("name") or "").strip()
    top_frac = float(top.get("fraction") or 0.0)
    if not top_name:
        return None

    def _name(entry: dict[str, Any]) -> str:
        return str(entry.get("name") or "").strip()

    top_entry_verb = _frequency_verb(top_frac) or "sometimes"

    paired_primary_candidate: Optional[dict[str, Any]] = None
    for candidate in ranked[1:]:
        candidate_name = _name(candidate)
        candidate_frac = float(candidate.get("fraction") or 0.0)
        if not candidate_name or candidate_frac < 0.02:
            continue
        if _combine_semantic_entries(top, candidate):
            paired_primary_candidate = candidate
            break

    if paired_primary_candidate is not None:
        selected_primary_candidates = [top, paired_primary_candidate]
    else:
        primary_candidates: list[dict[str, Any]] = [
            entry
            for entry in ranked
            if _frequency_verb(float(entry.get("fraction") or 0.0)) == top_entry_verb
        ]
        selected_primary_candidates = primary_candidates[:2]
    primary_labels = [_name(entry) for entry in selected_primary_candidates if _name(entry)]
    primary_label_text = _combine_entry_pair_or_single(selected_primary_candidates)
    if not primary_label_text:
        return None
    used_label_set = set(primary_labels)
    remaining_name_set = {
        _name(entry)
        for entry in ranked
        if _name(entry) and _name(entry) not in used_label_set
    }
    if len(selected_primary_candidates) == 2 and primary_label_text in remaining_name_set:
        explicit_pair_text = _combine_parallel_labels(
            _name(selected_primary_candidates[0]),
            _name(selected_primary_candidates[1]),
        )
        if explicit_pair_text and explicit_pair_text != primary_label_text:
            primary_label_text = explicit_pair_text
    primary_total_frac = sum(float(entry.get("fraction") or 0.0) for entry in selected_primary_candidates)
    top_verb = _frequency_verb(primary_total_frac) or top_entry_verb

    remaining_entries = [
        entry
        for entry in ranked
        if _name(entry)
        and _name(entry) not in used_label_set
        and not _is_semantically_subsumed_by_primary(entry, selected_primary_candidates)
    ]

    secondary_entries = [
        entry
        for entry in remaining_entries
        if float(entry.get("fraction") or 0.0) >= 0.02
    ]
    if secondary_entries:
        second = secondary_entries[0]
        second_name = _name(second)
        second_frac = float(second.get("fraction") or 0.0)
        second_verb = _secondary_frequency_verb(second_frac)
        if second_name:
            if abs(top_frac - second_frac) <= 0.10 and second_verb == top_verb:
                same_band_candidates = [
                    entry
                    for entry in secondary_entries
                    if abs(top_frac - float(entry.get("fraction") or 0.0)) <= 0.10
                ]
                combined_name = _combine_entry_pair_or_single(same_band_candidates[:2])
                if combined_name:
                    if combined_name == primary_label_text:
                        return f"{top_verb} in {primary_label_text}"
                    return f"{top_verb} in {primary_label_text}, as well as {combined_name}"
            else:
                same_secondary_candidates = [
                    entry
                    for entry in secondary_entries
                    if _secondary_frequency_verb(float(entry.get("fraction") or 0.0)) == second_verb
                ]
                secondary_label_text = _combine_entry_pair_or_single(same_secondary_candidates[:2])
                second_label_final = secondary_label_text or second_name
                if second_verb == top_verb:
                    return f"{top_verb} in {primary_label_text}, as well as {second_label_final}"
                top_text = f"{top_verb} in {primary_label_text}"
                second_text = f"{second_verb} in {second_label_final}"
                return f"{top_text}, and {second_text}"

    return f"{top_verb} in {primary_label_text}"


def _top_categorical_phrase(
    taxon_dir: Path,
    *,
    variable_id: str,
    label: str,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
) -> Optional[str]:
    payload: Optional[dict[str, Any]] = None
    if location_gid and taxon_id is not None:
        try:
            payload = summary_stats.build_categorical_stats_for_location(
                taxon_id,
                variable_id,
                location_gid,
                sample_limit=_LOCATION_CATEGORY_SAMPLE_LIMIT,
            )
        except Exception:
            payload = None
    if not payload:
        payload = summary_stats.load_categorical_distribution(taxon_dir, variable_id)
    if not payload:
        return None
    return _top_categorical_phrase_from_payload(
        variable_id=variable_id,
        label=label,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Location text
# ---------------------------------------------------------------------------


def _find_ancestor_by_rank(taxon: dict[str, Any], rank: str) -> Optional[dict[str, Any]]:
    from util import taxa_navigation

    target = taxa_navigation.canonical_rank(rank)
    current = taxon
    while current is not None:
        if taxa_navigation.canonical_rank(current.get("rank")) == target:
            return current
        current = taxa_navigation.get_parent_taxon(current)
    return None


def _with_definite_article(name: str) -> str:
    normalized = name.strip()
    lower = normalized.lower()
    if lower.startswith("the "):
        return normalized
    needs_article = {"united states", "united kingdom", "netherlands", "philippines", "gambia"}
    if lower in needs_article:
        return f"the {normalized}"
    return normalized


def _join_names(names: list[str], *, use_and: bool = True) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if use_and:
        return ", ".join(names[:-1]) + f", and {names[-1]}"
    return ", ".join(names)


def _combine_label_pair(left: str, right: str) -> str:
    left_clean = str(left or "").strip()
    right_clean = str(right or "").strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if left_clean == right_clean:
        return left_clean
    left_parts = left_clean.split()
    right_parts = right_clean.split()
    common_suffix: list[str] = []
    li = len(left_parts) - 1
    ri = len(right_parts) - 1
    while li >= 0 and ri >= 0 and left_parts[li].lower() == right_parts[ri].lower():
        common_suffix.insert(0, left_parts[li])
        li -= 1
        ri -= 1
    if common_suffix and li >= 0 and ri >= 0:
        left_stem = " ".join(left_parts[: li + 1]).strip()
        right_stem = " ".join(right_parts[: ri + 1]).strip()
        suffix = " ".join(common_suffix).strip()
        if left_stem and right_stem and suffix:
            return f"{left_stem} and {right_stem} {suffix}"
    return f"{left_clean} and {right_clean}"


def _build_location_text(
    taxon_id: int,
    *,
    location_gid: Optional[str] = None,
    min_fraction: float = 0.0,
    limit: int = 3,
) -> str:
    from util import gis_lookup

    config = load_config("global")
    scope0 = config.location_scope_by_level.get(0, "gadm_level0")
    scope1 = config.location_scope_by_level.get(1, "gadm_level1")
    scope2 = config.location_scope_by_level.get(2, "gadm_level2")
    by_location = gis_lookup.location_counts_for_taxon(taxon_id)
    if not by_location:
        return ""
    _, mapping = gis_lookup.load_location_catalog()
    by_scope: dict[str, list[tuple[str, int]]] = {}
    for (scope, gid), count in by_location.items():
        if not count:
            continue
        by_scope.setdefault(scope, []).append((gid, int(count)))

    def names_for_scope(scope: str) -> list[tuple[str, str, int]]:
        entries: list[tuple[str, str, int]] = []
        for gid, count in by_scope.get(scope, []):
            record = mapping.get(gid)
            if record and record.name:
                entries.append((gid, record.name, count))
        return entries

    def filtered_entries(
        scope: str,
        *,
        parent_gid: Optional[str] = None,
        apply_article: bool = False,
    ) -> tuple[list[tuple[str, str, int]], bool]:
        entries = names_for_scope(scope)
        if parent_gid:
            entries = [
                (gid, name, count)
                for gid, name, count in entries
                if mapping.get(gid) and mapping[gid].parent_gid == parent_gid
            ]
        if not entries:
            return [], False
        total = sum(count for _gid, _name, count in entries)
        if total <= 0:
            return [], False
        filtered = [
            (gid, name, count)
            for gid, name, count in entries
            if (count / total) >= min_fraction
        ]
        if not filtered:
            return [], False
        filtered.sort(key=lambda row: row[2], reverse=True)
        deduped: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for gid, name, count in filtered:
            if name in seen:
                continue
            seen.add(name)
            if apply_article:
                name = _with_definite_article(name)
            deduped.append((gid, name, count))
            if limit > 0 and len(deduped) >= limit:
                break
        has_more = len(filtered) > len(deduped)
        return deduped, has_more

    def format_with_label(
        names: list[str],
        *,
        parent: str,
        has_more: bool,
        more_label: str,
    ) -> str:
        if not names:
            return ""
        text = _join_names(names, use_and=not has_more)
        if has_more:
            return f"{text} and other {more_label} in {parent}"
        return f"{text} in {parent}"

    if location_gid:
        try:
            _column, scope, target = gis_lookup.location_lookup_for_gid(location_gid)
        except ValueError:
            return ""
        if scope == scope1:
            parent_name = mapping.get(target).name if mapping.get(target) else target
            entries, has_more = filtered_entries(scope2, parent_gid=target)
            if entries:
                names = [name for _gid, name, _count in entries]
                return format_with_label(
                    names,
                    parent=parent_name,
                    has_more=has_more,
                    more_label="subregions",
                )
            return parent_name
        if scope == scope0:
            country = mapping.get(target).name if mapping.get(target) else target
            entries, has_more = filtered_entries(scope1, parent_gid=target)
            if entries:
                names = [name for _gid, name, _count in entries]
                return format_with_label(
                    names,
                    parent=_with_definite_article(country),
                    has_more=has_more,
                    more_label="regions",
                )
            return _with_definite_article(country)
        if scope == scope2:
            record = mapping.get(target)
            if record and record.parent_gid:
                parent = mapping.get(record.parent_gid)
                parent_name = parent.name if parent else record.parent_gid
                return f"{record.name} in {parent_name}"
            return record.name if record else target
        if scope == "gbif_region":
            return target.replace("_", " ").title()
        return target

    country_entries, has_more = filtered_entries(scope0, apply_article=True)
    if not country_entries:
        return ""
    if len(country_entries) == 1:
        country_gid, country_name, _count = country_entries[0]
        state_entries, has_more_states = filtered_entries(scope1, parent_gid=country_gid)
        if state_entries:
            state_names = [name for _gid, name, _count in state_entries]
            return format_with_label(
                state_names,
                parent=country_name,
                has_more=has_more_states,
                more_label="regions",
            )
    country_names = [name for _gid, name, _count in country_entries]
    text = _join_names(country_names, use_and=not has_more)
    if has_more:
        return f"{text} and other countries"
    return text


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------


def _format_terrain_value(
    value: Any,
    *,
    unit: Optional[str] = None,
    unit_system: Optional[units.UnitSystem] = None,
) -> Optional[str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    converted, _display = units.convert_value_for_system(numeric, unit, unit_system)
    if converted is None:
        return None
    numeric = float(converted)
    rounded = int(round(numeric / 100.0) * 100)
    return str(rounded)


def _extract_elevation_range_values(
    elevation: dict[str, Any],
    *,
    unit: Optional[str] = None,
    unit_system: Optional[units.UnitSystem] = None,
) -> tuple[Optional[str], Optional[str]]:
    range_value = elevation.get("range")
    if isinstance(range_value, dict):
        min_value = _format_terrain_value(
            range_value.get("min"),
            unit=unit,
            unit_system=unit_system,
        )
        max_value = _format_terrain_value(
            range_value.get("max"),
            unit=unit,
            unit_system=unit_system,
        )
        if min_value and max_value:
            return min_value, max_value
    min_value = _format_terrain_value(
        elevation.get("min"),
        unit=unit,
        unit_system=unit_system,
    )
    max_value = _format_terrain_value(
        elevation.get("max"),
        unit=unit,
        unit_system=unit_system,
    )
    return min_value, max_value


def _slope_grade_percent(mean_slope_degrees: Any) -> Optional[float]:
    try:
        degrees = float(mean_slope_degrees)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(degrees):
        return None
    return math.tan(math.radians(degrees)) * 100.0


def _slope_band_from_grade(grade_percent: float) -> str:
    if grade_percent < 5.0:
        return "flat"
    if grade_percent < 10.0:
        return "mild"
    if grade_percent < 15.0:
        return "gentle"
    if grade_percent < 20.0:
        return "moderate"
    if grade_percent < 30.0:
        return "steep"
    return "very steep"


def _slope_phrase_for_band(band: str) -> str:
    if band == "flat":
        return "flat areas"
    return f"{band} slopes"


def _slope_range_phrase(low_band: str, high_band: str) -> str:
    if low_band == high_band:
        return _slope_phrase_for_band(high_band)
    if low_band == "flat":
        return f"flat areas to {high_band} slopes"
    if high_band == "flat":
        return f"{low_band} slopes to flat areas"
    return f"{low_band} to {high_band} slopes"


def _aspect_cardinal_masses(taxon_dir: Path) -> tuple[dict[str, float], float]:
    payload = summary_stats.load_categorical_distribution(taxon_dir, "aspect") or {}
    distribution = payload.get("distribution") or []
    totals = payload.get("totals") or {}
    try:
        total_samples = float(totals.get("total_samples") or 0.0)
    except (TypeError, ValueError):
        total_samples = 0.0

    masses = {"north": 0.0, "east": 0.0, "south": 0.0, "west": 0.0}
    for entry in distribution:
        try:
            frac = float(entry.get("fraction") or 0.0)
        except (TypeError, ValueError):
            continue
        if frac <= 0:
            continue
        class_id = _parse_class_id(entry.get("value"))
        if class_id is None:
            class_id = _parse_class_id(entry.get("class_name"))
        if class_id == 1:
            masses["north"] += frac
        elif class_id == 2:
            masses["north"] += frac
            masses["east"] += frac
        elif class_id == 3:
            masses["east"] += frac
        elif class_id == 4:
            masses["south"] += frac
            masses["east"] += frac
        elif class_id == 5:
            masses["south"] += frac
        elif class_id == 6:
            masses["south"] += frac
            masses["west"] += frac
        elif class_id == 7:
            masses["west"] += frac
        elif class_id == 8:
            masses["north"] += frac
            masses["west"] += frac
    return masses, total_samples


def _aspect_preference_text(taxon_dir: Path) -> Optional[str]:
    masses, total_samples = _aspect_cardinal_masses(taxon_dir)
    if total_samples <= 100:
        return None
    preferred = [direction for direction, value in masses.items() if value > 0.50]
    if not preferred:
        return None
    preferred_facing = [f"{direction}-facing" for direction in preferred]
    facing_text = _join_names(preferred_facing)
    return f"Aspect: prefers {facing_text} slopes"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def _outlier_severity_from_strength(
    strength: float,
    *,
    thresholds: tuple[float, float, float] = (0.99, 0.95, 0.90),
) -> tuple[int, str]:
    extreme_cutoff, very_cutoff, quite_cutoff = thresholds
    if strength > extreme_cutoff:
        return 4, "extremely"
    if strength > very_cutoff:
        return 3, "very"
    if strength > quite_cutoff:
        return 2, "quite"
    return 0, ""


def _select_variable_outlier_text(
    *,
    variable_id: str,
    taxon: dict[str, Any],
    taxon_dir: Path,
    preferred_metrics: tuple[str, ...],
    location_gid: Optional[str] = None,
) -> Optional[str]:
    candidate = _select_variable_outlier_candidate(
        variable_id=variable_id,
        taxon=taxon,
        taxon_dir=taxon_dir,
        preferred_metrics=preferred_metrics,
        location_gid=location_gid,
    )
    if not candidate:
        return None
    metric_name = str(candidate.get("metric") or "").strip().lower()
    metric_tag = "avg" if metric_name == "mean" else metric_name
    phrase = str(candidate.get("phrase") or "").strip()
    context = str(candidate.get("context") or "").strip()
    if not phrase or not context:
        return None
    return f"{metric_tag} {phrase} for {context}"


def _select_variable_outlier_candidate(
    *,
    variable_id: str,
    taxon: dict[str, Any],
    taxon_dir: Path,
    preferred_metrics: tuple[str, ...],
    metric_names: Optional[tuple[str, ...]] = None,
    severity_thresholds: tuple[float, float, float] = (0.99, 0.95, 0.90),
    max_ancestor_rank: Optional[str] = None,
    location_gid: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    from util import indexing, taxa_navigation

    config = load_config("global")
    if bool(getattr(config, "skip_description_outliers", False)):
        return None

    entries = indexing.load_relative_ranks(
        taxon_dir,
        variable_id,
        metric_names=metric_names,
        location_gid=location_gid,
    )
    if not entries:
        return None

    depth_by_id: dict[str, int] = {}
    current = taxon
    depth = 0
    while current is not None:
        parent = taxa_navigation.get_parent_taxon(current)
        if parent is None:
            break
        depth += 1
        parent_id = str(parent.get("taxon_key") or "").strip()
        if parent_id:
            depth_by_id[parent_id] = depth
        current = parent

    ancestor_cache: dict[str, Optional[dict[str, Any]]] = {}
    rank_order = (
        "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY",
        "GENUS", "SPECIES", "SUBSPECIES",
    )
    max_rank_index: Optional[int] = None
    if max_ancestor_rank:
        canonical_max = taxa_navigation.canonical_rank(max_ancestor_rank)
        if canonical_max in rank_order:
            max_rank_index = rank_order.index(canonical_max)

    def _ancestor_for_id(ancestor_id: str) -> Optional[dict[str, Any]]:
        if ancestor_id not in ancestor_cache:
            ancestor_cache[ancestor_id] = taxa_navigation.get_taxon_by_id(ancestor_id)
        return ancestor_cache[ancestor_id]

    for metric_name in preferred_metrics:
        metric_entries = []
        for entry in entries:
            if str(entry.get("metric") or "").strip().lower() != metric_name:
                continue
            try:
                context_count = int(entry.get("count") or 0)
            except (TypeError, ValueError):
                context_count = 0
            if context_count < 10:
                continue
            metric_entries.append(entry)
        if not metric_entries:
            continue

        best: Optional[dict[str, Any]] = None
        for entry in metric_entries:
            try:
                percentile = float(entry.get("percentile"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(percentile):
                continue
            percentile = max(0.0, min(1.0, percentile))

            low_level, low_word = _outlier_severity_from_strength(
                1.0 - percentile, thresholds=severity_thresholds,
            )
            high_level, high_word = _outlier_severity_from_strength(
                percentile, thresholds=severity_thresholds,
            )
            if low_level == 0 and high_level == 0:
                continue

            if low_level >= high_level:
                level = low_level
                qualifier = "moderately" if low_word == "moderate" else low_word
                polarity = "low"
                strength = 1.0 - percentile
            else:
                level = high_level
                qualifier = "moderately" if high_word == "moderate" else high_word
                polarity = "high"
                strength = percentile

            ancestor_id = str(entry.get("ancestorTaxonId") or "").strip()
            depth_score = depth_by_id.get(ancestor_id, -1)

            context_label = str(entry.get("label") or entry.get("context") or "").strip()
            ancestor_rank = ""
            if ancestor_id:
                ancestor = _ancestor_for_id(ancestor_id)
                if ancestor:
                    ancestor_rank_canonical = str(
                        taxa_navigation.canonical_rank(ancestor.get("rank")) or ""
                    ).strip()
                    if max_rank_index is not None and ancestor_rank_canonical in rank_order:
                        if rank_order.index(ancestor_rank_canonical) < max_rank_index:
                            continue
                    ancestor_rank = ancestor_rank_canonical.lower()
                    if not context_label:
                        context_label = str(
                            ancestor.get("scientific_name")
                            or ancestor.get("common_name")
                            or ancestor.get("taxon_key")
                            or ""
                        ).replace("_", " ").strip()
            if not context_label:
                continue
            context_text = f"{ancestor_rank} {context_label}".strip() if ancestor_rank else context_label

            candidate = {
                "metric": metric_name,
                "level": level,
                "depth": depth_score,
                "strength": strength,
                "qualifier": qualifier,
                "polarity": polarity,
                "phrase": f"{qualifier} {polarity}",
                "context": context_text,
                "ancestor_taxon_id": ancestor_id,
            }
            if best is None:
                best = candidate
                continue
            if candidate["level"] > best["level"]:
                best = candidate
                continue
            if candidate["level"] == best["level"] and candidate["depth"] > best["depth"]:
                best = candidate
                continue
            if (
                candidate["level"] == best["level"]
                and candidate["depth"] == best["depth"]
                and candidate["strength"] > best["strength"]
            ):
                best = candidate

        if best:
            return best
    return None


# ---------------------------------------------------------------------------
# Categorical metric helpers
# ---------------------------------------------------------------------------


def _build_metric_candidates(
    class_name: str,
    class_id: Optional[int],
    entry: dict[str, Any],
) -> list[str]:
    fallback_metric = f"class_{class_id}" if class_id is not None else "class_unknown"
    raw_metric = summary_stats._slugify_metric(class_name, fallback_metric)
    candidates: list[str] = [raw_metric]
    raw_value_token = str(entry.get("value") or "").strip()
    if raw_value_token:
        raw_value_slug = re.sub(r"[^a-z0-9]+", "_", raw_value_token.lower()).strip("_")
        if raw_value_slug:
            if raw_value_slug.startswith("class_"):
                candidates.append(raw_value_slug)
            else:
                candidates.append(f"class_{raw_value_slug}")
    if class_id is not None:
        candidates.append(f"class_{class_id}")
    return candidates


def _categorical_metric_fraction_for_aliases(
    taxon_dir: Path,
    *,
    variable_id: str,
    aliases: tuple[str, ...],
    default: Optional[float] = None,
) -> Optional[float]:
    metrics = summary_stats._load_categorical_stats(str(taxon_dir)) or {}
    variable_metrics = metrics.get(variable_id) or {}
    if not variable_metrics:
        return default
    by_lower = {
        str(metric_name).strip().lower(): metric_name
        for metric_name in variable_metrics.keys()
        if str(metric_name).strip()
    }
    for alias in aliases:
        key = str(alias or "").strip().lower()
        if not key:
            continue
        resolved = by_lower.get(key)
        if resolved is None:
            continue
        value = variable_metrics.get(resolved)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return default


def _resolve_metric_name_for_variable(
    taxon_dir: Path,
    *,
    variable_id: str,
    candidates: tuple[str, ...],
) -> str:
    cleaned_candidates = [str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()]
    if not cleaned_candidates:
        return ""
    metrics = summary_stats._load_categorical_stats(str(taxon_dir)) or {}
    variable_metrics = metrics.get(variable_id) or {}
    if not variable_metrics:
        return cleaned_candidates[0]
    by_lower = {
        str(metric_name).strip().lower(): str(metric_name).strip()
        for metric_name in variable_metrics.keys()
        if str(metric_name).strip()
    }
    for candidate in cleaned_candidates:
        resolved = by_lower.get(candidate.lower())
        if resolved:
            return resolved
    return cleaned_candidates[0]


def _delta_adjusted_qualifier(
    qualifier: str,
    *,
    abs_delta: float,
) -> Optional[str]:
    normalized = str(qualifier or "").strip().lower()
    if normalized not in {"quite", "very", "extremely"}:
        return None
    if abs_delta < 0.10:
        return None
    if abs_delta < 0.20 and normalized == "extremely":
        return "very"
    return normalized


def _pick_best_outlier_candidate(candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not candidates:
        return None

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(item.get("level") or 0),
            float(item.get("depth") or 0),
            float(item.get("strength") or 0.0),
        )

    return max(candidates, key=_sort_key)


def _join_outlier_labels(labels: list[str], fallback: str) -> str:
    cleaned = [str(label or "").strip() for label in labels if str(label or "").strip()]
    if not cleaned:
        return fallback
    if len(cleaned) == 1:
        return cleaned[0]
    return _combine_label_pair(cleaned[0], cleaned[1])


def _load_categorical_payload_for_context(
    taxon_dir: Path,
    *,
    variable_id: str,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    payload: Optional[dict[str, Any]] = None
    if location_gid and taxon_id is not None:
        try:
            payload = summary_stats.build_categorical_stats_for_location(
                taxon_id,
                variable_id,
                location_gid,
                sample_limit=_LOCATION_CATEGORY_SAMPLE_LIMIT,
            )
        except Exception:
            payload = None
    if not payload:
        payload = summary_stats.load_categorical_distribution(taxon_dir, variable_id)
    return payload


def _entry_metric_aliases(
    taxon_dir: Path,
    *,
    variable_id: str,
    entry: dict[str, Any],
    legend: dict[str, Any],
) -> tuple[str, ...]:
    class_id = _parse_class_id(entry.get("value"))
    if class_id is None:
        class_id = _parse_class_id(entry.get("class_name"))
    legend_entry = legend.get(str(class_id)) if class_id is not None else None
    class_name = str(
        (legend_entry or {}).get("name")
        or entry.get("class_name")
        or entry.get("short_name")
        or entry.get("value")
        or ""
    ).strip()
    if not class_name:
        return ()
    metric_candidates = _build_metric_candidates(class_name, class_id, entry)
    resolved = _resolve_metric_name_for_variable(
        taxon_dir,
        variable_id=variable_id,
        candidates=tuple(metric_candidates),
    )
    if resolved:
        metric_candidates.append(resolved)
    return tuple(dict.fromkeys(candidate for candidate in metric_candidates if candidate))


def _fraction_for_aliases_from_payload(
    taxon_dir: Path,
    *,
    variable_id: str,
    payload: Optional[dict[str, Any]],
    aliases: tuple[str, ...],
) -> float:
    if not payload or not aliases:
        return 0.0
    from util import gis_lookup

    alias_set = {str(alias or "").strip().lower() for alias in aliases if str(alias or "").strip()}
    if not alias_set:
        return 0.0
    legend = gis_lookup.load_layer_legend(variable_id)
    total = 0.0
    distribution = payload.get("distribution") or []
    for entry in distribution:
        try:
            frac = float(entry.get("fraction") or 0.0)
        except (TypeError, ValueError):
            continue
        if frac <= 0:
            continue
        entry_aliases = _entry_metric_aliases(
            taxon_dir,
            variable_id=variable_id,
            entry=entry,
            legend=legend,
        )
        if any(str(name).strip().lower() in alias_set for name in entry_aliases):
            total += frac
    return total


def _outlier_qualifier_level(qualifier: str) -> int:
    normalized = str(qualifier or "").strip().lower()
    if normalized == "extremely":
        return 3
    if normalized == "very":
        return 2
    if normalized == "quite":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Unified categorical class metrics + outlier text
# ---------------------------------------------------------------------------


def _categorical_display_label(
    *,
    variable_id: str,
    style: str,
    class_name: str,
    class_id: Optional[int],
    group_value: str,
    group_label: str,
    legend_entry: Optional[dict[str, Any]],
) -> str:
    if style == "group_map":
        if group_value == "forest":
            traits = _extract_legend_traits(legend_entry)
            openness_fallback = _landcover_forest_openness(class_name)
            if openness_fallback in {"sparse", "dense"} and not traits.get("openness"):
                traits["openness"] = openness_fallback
            phenology_fallback = _landcover_forest_phenology(class_name)
            if phenology_fallback and phenology_fallback != "generic" and not traits.get("phenology"):
                traits["phenology"] = phenology_fallback
            label, _semantic = _semantic_label_from_group(
                variable_id=variable_id,
                group=group_value,
                group_label=group_label,
                traits=traits,
            )
            return label
        if group_value:
            return _landcover_group_label(group_value, group_label)
        return _to_natural_habitat_name(class_name)
    if style == "climate_suffix":
        if group_value or group_label:
            return _ensure_climate_suffix(
                _to_natural_climate_name(group_label or group_value)
            )
        return _ensure_climate_suffix(_to_natural_climate_name(class_name))
    return str(class_name).strip()


def _top_categorical_class_metrics(
    taxon_dir: Path,
    *,
    variable_id: str,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
    limit: int = 2,
) -> list[dict[str, Any]]:
    from util import gis_lookup

    payload = _load_categorical_payload_for_context(
        taxon_dir,
        variable_id=variable_id,
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    if not payload:
        return []
    distribution = payload.get("distribution") or []
    if not distribution:
        return []
    ranked = sorted(
        distribution,
        key=lambda entry: float(entry.get("fraction") or 0.0),
        reverse=True,
    )
    legend = gis_lookup.load_layer_legend(variable_id)
    style = str((_CATEGORICAL_LAYER_RULES.get(variable_id) or {}).get("default_style") or "").strip()
    is_landcover = (style == "group_map")
    selected: list[dict[str, Any]] = []
    seen_metrics: set[str] = set()
    for entry in ranked:
        fraction = float(entry.get("fraction") or 0.0)
        if fraction <= 0:
            continue
        class_id = _parse_class_id(entry.get("value"))
        if class_id is None and is_landcover:
            class_id = _parse_class_id(entry.get("class_name"))
        legend_entry = legend.get(str(class_id)) if class_id is not None else None
        class_name = str(
            (legend_entry or {}).get("name")
            or entry.get("class_name")
            or entry.get("short_name")
            or entry.get("value")
            or ""
        ).strip()
        if not class_name:
            continue
        group_value = _normalized_group_token(
            entry.get("group")
            or ((legend_entry or {}).get("group") if isinstance(legend_entry, dict) else "")
        )
        group_label = str(
            entry.get("group_label")
            or ((legend_entry or {}).get("group_label") if isinstance(legend_entry, dict) else "")
            or ""
        ).strip()
        if not group_value and is_landcover:
            inferred_group, inferred_group_label = _infer_landcover_group(class_name, class_id)
            group_value = inferred_group
            if not group_label:
                group_label = inferred_group_label
        metric_candidates = _build_metric_candidates(class_name, class_id, entry)
        metric_name = _resolve_metric_name_for_variable(
            taxon_dir,
            variable_id=variable_id,
            candidates=tuple(metric_candidates),
        )
        if metric_name in seen_metrics:
            continue
        seen_metrics.add(metric_name)
        display_label = _categorical_display_label(
            variable_id=variable_id,
            style=style,
            class_name=class_name,
            class_id=class_id,
            group_value=group_value,
            group_label=group_label,
            legend_entry=legend_entry,
        )
        selected.append(
            {
                "metric": metric_name,
                "aliases": tuple(dict.fromkeys(metric_candidates)),
                "label": display_label,
                "fraction": fraction,
            }
        )
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _location_delta_outlier_text(
    taxon_dir: Path,
    *,
    variable_id: str,
    top_metrics: list[dict[str, Any]],
    taxon_id: Optional[int],
    location_gid: Optional[str],
) -> Optional[str]:
    if not location_gid or taxon_id is None:
        return None
    if not top_metrics:
        return None
    local_payload = _load_categorical_payload_for_context(
        taxon_dir,
        variable_id=variable_id,
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    if not local_payload:
        return None
    qualified: list[dict[str, Any]] = []
    for row in top_metrics[:2]:
        aliases = tuple(str(alias or "").strip() for alias in (row.get("aliases") or ()) if str(alias or "").strip())
        label = str(row.get("label") or "").strip()
        global_fraction = float(row.get("fraction") or 0.0)
        local_fraction = _fraction_for_aliases_from_payload(
            taxon_dir,
            variable_id=variable_id,
            payload=local_payload,
            aliases=aliases,
        )
        delta = local_fraction - global_fraction
        abs_delta = abs(delta)
        if abs_delta < 0.10:
            level = 0
        elif abs_delta >= 0.30:
            level = 3
        elif abs_delta >= 0.20:
            level = 2
        else:
            level = 1
        polarity = "high" if delta > 0 else "low"
        if level <= 0 or not label:
            continue
        qualified.append(
            {
                "label": label,
                "level": level,
                "polarity": polarity,
                "delta": abs_delta,
            }
        )
    if not qualified:
        return None
    best = max(qualified, key=lambda item: (int(item.get("level") or 0), float(item.get("delta") or 0.0)))
    best_level = int(best.get("level") or 0)
    best_polarity = str(best.get("polarity") or "").strip()
    kept_labels = [
        str(item.get("label") or "").strip()
        for item in qualified
        if int(item.get("level") or 0) == best_level
        and str(item.get("polarity") or "").strip() == best_polarity
    ]
    class_label = _join_outlier_labels(kept_labels, "these categories")
    location_name = _location_label(location_gid)
    if best_level >= 3:
        more_phrase = "much more common"
        less_phrase = "much less common"
    elif best_level == 2:
        more_phrase = "more common"
        less_phrase = "less common"
    else:
        more_phrase = "a bit more common"
        less_phrase = "a bit less common"
    if best_polarity == "high":
        phrase = f"{more_phrase} in {class_label} when in {location_name}"
    else:
        phrase = f"{less_phrase} in {class_label} when in {location_name}"
    return _sentence_case(phrase)


def _categorical_outlier_text(
    taxon: dict[str, Any],
    taxon_dir: Path,
    *,
    variable_id: str,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
) -> Optional[str]:
    from util import taxa_navigation

    if location_gid and taxon_id is not None:
        global_top_metrics = _top_categorical_class_metrics(
            taxon_dir,
            variable_id=variable_id,
            taxon_id=taxon_id,
            location_gid=None,
            limit=4,
        )
        location_text = _location_delta_outlier_text(
            taxon_dir,
            variable_id=variable_id,
            top_metrics=global_top_metrics,
            taxon_id=taxon_id,
            location_gid=location_gid,
        )
        if location_text:
            return location_text
    top_metrics = _top_categorical_class_metrics(
        taxon_dir,
        variable_id=variable_id,
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    if not top_metrics:
        return None
    labels = [str(row.get("label") or "").strip() for row in top_metrics if str(row.get("label") or "").strip()]
    fallback_label = _CATEGORICAL_FALLBACK_LABELS.get(variable_id, "these categories")
    if len(labels) >= 2:
        class_label = _combine_label_pair(labels[0], labels[1])
    elif labels:
        class_label = labels[0]
    else:
        class_label = fallback_label
    ranked_rows = [row for row in top_metrics if str(row.get("metric") or "").strip()]
    evaluated: list[dict[str, Any]] = []
    for row in ranked_rows:
        aliases = tuple(str(alias or "").strip() for alias in (row.get("aliases") or ()) if str(alias or "").strip())
        if not aliases:
            continue
        candidate = _select_variable_outlier_candidate(
            variable_id=variable_id,
            taxon=taxon,
            taxon_dir=taxon_dir,
            preferred_metrics=aliases,
            metric_names=aliases,
            severity_thresholds=(0.95, 0.90, 0.85),
            max_ancestor_rank="FAMILY",
            location_gid=location_gid,
        )
        if not candidate:
            continue
        qualifier = str(candidate.get("qualifier") or "").strip()
        context = str(candidate.get("context") or "").strip()
        ancestor_taxon_id = str(candidate.get("ancestor_taxon_id") or "").strip()
        if not qualifier or not context:
            continue
        taxon_fraction = 0.0
        context_fraction = 0.0
        has_taxon_fraction = False
        has_context_fraction = False
        local_value = _categorical_metric_fraction_for_aliases(
            taxon_dir,
            variable_id=variable_id,
            aliases=aliases,
            default=float(row.get("fraction") or 0.0),
        )
        if local_value is not None:
            taxon_fraction += float(local_value)
            has_taxon_fraction = True
        if ancestor_taxon_id:
            ancestor = taxa_navigation.get_taxon_by_id(ancestor_taxon_id)
            if ancestor is not None:
                ancestor_value = _categorical_metric_fraction_for_aliases(
                    Path(ancestor["path"]),
                    variable_id=variable_id,
                    aliases=aliases,
                    default=0.0,
                )
                if ancestor_value is not None:
                    context_fraction += float(ancestor_value)
                    has_context_fraction = True
        if not has_taxon_fraction or not has_context_fraction:
            continue
        delta = abs(taxon_fraction - context_fraction)
        adjusted_qualifier = _delta_adjusted_qualifier(qualifier, abs_delta=delta)
        if not adjusted_qualifier:
            continue
        evaluated.append(
            {
                "row": row,
                "candidate": candidate,
                "adjusted_qualifier": adjusted_qualifier,
                "adjusted_level": _outlier_qualifier_level(adjusted_qualifier),
            }
        )
    if not evaluated:
        return None
    best_item = max(
        evaluated,
        key=lambda item: (
            int(item.get("adjusted_level") or 0),
            float((item.get("candidate") or {}).get("depth") or 0),
            float((item.get("candidate") or {}).get("strength") or 0.0),
        ),
    )
    best = best_item["candidate"]
    if not best:
        return None
    best_adjusted_level = int(best_item.get("adjusted_level") or 0)
    best_polarity = str(best.get("polarity") or "").strip()
    best_context = str(best.get("context") or "").strip()
    best_adjusted = str(best_item.get("adjusted_qualifier") or "").strip()
    best_labels: list[str] = []
    for item in evaluated:
        cand = item["candidate"]
        if (
            int(item.get("adjusted_level") or 0) == best_adjusted_level
            and str(cand.get("polarity") or "").strip() == best_polarity
            and str(cand.get("context") or "").strip() == best_context
        ):
            best_labels.append(str(item["row"].get("label") or "").strip())
    if not best_adjusted:
        return None
    class_label = _join_outlier_labels(best_labels, class_label)
    comparison = _QUALIFIER_COMPARISON.get(best_adjusted)
    if not comparison:
        return None
    higher_phrase, lower_phrase = comparison
    if best_polarity == "high":
        phrase = higher_phrase
    elif best_polarity == "low":
        phrase = lower_phrase
    else:
        return None
    return _sentence_case(f"{phrase} in {class_label} compared to others in {best_context}")


# ---------------------------------------------------------------------------
# Status rows helpers
# ---------------------------------------------------------------------------


def _elevation_outlier_text(
    taxon: dict[str, Any],
    taxon_dir: Path,
    *,
    location_gid: Optional[str] = None,
) -> Optional[str]:
    return _select_variable_outlier_text(
        variable_id="elevation",
        taxon=taxon,
        taxon_dir=taxon_dir,
        preferred_metrics=("mean", "max"),
        location_gid=location_gid,
    )


def _numeric_summary_for_context(
    *,
    taxon_id: Optional[int],
    taxon_dir: Path,
    variable_id: str,
    location_gid: Optional[str] = None,
) -> dict[str, Any]:
    if location_gid and taxon_id is not None:
        samples = summary_stats.gather_numeric_records(
            taxon_id,
            taxon_dir,
            variable_id,
            location_gid=location_gid,
        )
        values = [
            float(sample.get("value"))
            for sample in samples
            if isinstance(sample, dict) and isinstance(sample.get("value"), (int, float))
        ]
        if values:
            return summary_stats.summarize_values(values)
        return {}
    return summary_stats.load_numeric_summary(str(taxon_dir), variable_id) or {}


def _format_scalar_value(value: Any) -> Optional[str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return str(int(rounded))
    return f"{numeric:.1f}".rstrip("0").rstrip(".")


def _format_scalar_value_for_system(
    value: Any,
    *,
    unit: Optional[str] = None,
    unit_system: Optional[units.UnitSystem] = None,
) -> Optional[str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    converted, _display = units.convert_value_for_system(numeric, unit, unit_system)
    return _format_scalar_value(converted)


def _location_label(location_gid: Optional[str]) -> str:
    if not location_gid:
        return "this location"
    from util import gis_lookup

    token = str(location_gid).strip()
    if not token:
        return "this location"
    try:
        _column, scope, target = gis_lookup.location_lookup_for_gid(token)
    except Exception:
        return token
    if scope == "gbif_region":
        return str(target).replace("_", " ").title()
    _entries, by_gid = gis_lookup.load_location_catalog()
    record = by_gid.get(str(target))
    if record and record.name:
        return record.name
    return str(target)


def _temperature_location_compare_text(
    *,
    local_mean: Any,
    global_mean: Any,
    location_name: str,
) -> Optional[str]:
    try:
        local_value = float(local_mean)
        global_value = float(global_mean)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(local_value) and math.isfinite(global_value)):
        return None
    delta = local_value - global_value
    magnitude = abs(delta)
    if magnitude < 0.5:
        return f"about the same in {location_name}"
    if magnitude < 1.5:
        degree = "slightly"
    elif magnitude < 3.0:
        degree = "a bit"
    elif magnitude < 5.0:
        degree = "noticeably"
    else:
        degree = "much"
    direction = "warmer" if delta > 0 else "cooler"
    return f"{degree} {direction} in {location_name}"


def _precip_location_compare_text(
    *,
    local_mean: Any,
    global_mean: Any,
    location_name: str,
) -> Optional[str]:
    try:
        local_value = float(local_mean)
        global_value = float(global_mean)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(local_value) and math.isfinite(global_value)):
        return None
    if global_value <= 0:
        diff = abs(local_value - global_value)
        if diff < 25:
            return f"about the same in {location_name}"
        degree = "much" if diff >= 200 else "noticeably" if diff >= 100 else "a bit"
        direction = "wetter" if local_value > global_value else "drier"
        return f"{degree} {direction} in {location_name}"
    ratio_delta = (local_value - global_value) / abs(global_value)
    magnitude = abs(ratio_delta)
    if magnitude < 0.10:
        return f"about the same in {location_name}"
    if magnitude < 0.25:
        degree = "slightly"
    elif magnitude < 0.50:
        degree = "a bit"
    elif magnitude < 0.80:
        degree = "noticeably"
    else:
        degree = "much"
    direction = "wetter" if ratio_delta > 0 else "drier"
    return f"{degree} {direction} in {location_name}"


# ---------------------------------------------------------------------------
# Status rows
# ---------------------------------------------------------------------------


def _terrain_status_rows(
    taxon: dict[str, Any],
    taxon_dir: Path,
    *,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
    unit_system: Optional[units.UnitSystem] = None,
) -> list[dict[str, Any]]:
    from util import gis_lookup

    elevation = _numeric_summary_for_context(
        taxon_id=taxon_id,
        taxon_dir=taxon_dir,
        variable_id="elevation",
        location_gid=location_gid,
    )
    raw_units: Optional[str] = None
    display_units = ""
    min_value: Optional[str]
    max_value: Optional[str]
    slope = _numeric_summary_for_context(
        taxon_id=taxon_id,
        taxon_dir=taxon_dir,
        variable_id="slope",
        location_gid=location_gid,
    )
    slope_grade = _slope_grade_percent(slope.get("mean"))
    slope_p10_grade = _slope_grade_percent(slope.get("10th percentile"))
    try:
        _entries, variable_lookup = gis_lookup.load_variable_metadata()
        raw_units = str((variable_lookup.get("elevation") or {}).get("units") or "").strip() or None
    except Exception:
        raw_units = None
    target_unit = units.equivalent_unit(raw_units, unit_system) if raw_units else raw_units
    display_units = str(units.display_unit(target_unit) or "").strip()
    min_value, max_value = _extract_elevation_range_values(
        elevation,
        unit=raw_units,
        unit_system=unit_system,
    )

    detail_parts: list[str] = []
    elevation_outlier_text = (
        _elevation_outlier_text(
            taxon,
            taxon_dir,
            location_gid=location_gid,
        )
        if not location_gid
        else None
    )
    if min_value and max_value:
        if display_units:
            elevation_text = f"Elevation: {min_value}-{max_value} {display_units}"
        else:
            elevation_text = f"Elevation: {min_value}-{max_value}"
        if elevation_outlier_text:
            elevation_text += f" ({elevation_outlier_text})"
        detail_parts.append(elevation_text)
    elif elevation_outlier_text:
        detail_parts.append(f"Elevation: {elevation_outlier_text}")
    if slope_grade is not None:
        if slope_p10_grade is not None:
            low_grade = min(slope_p10_grade, slope_grade)
            high_grade = max(slope_p10_grade, slope_grade)
            low_band = _slope_band_from_grade(low_grade)
            high_band = _slope_band_from_grade(high_grade)
            low_pct = int(round(low_grade))
            high_pct = int(round(high_grade))
            slope_text = (
                f"Slope: often {_slope_range_phrase(low_band, high_band)} "
                f"({low_pct} to {high_pct} percent grade)"
            )
        else:
            slope_band = _slope_band_from_grade(slope_grade)
            slope_pct = int(round(slope_grade))
            slope_text = f"Slope: often {_slope_phrase_for_band(slope_band)} ({slope_pct} percent grade)"
        detail_parts.append(slope_text)
    aspect_text = _aspect_preference_text(taxon_dir)
    if aspect_text:
        detail_parts.append(aspect_text)
    detail: Optional[str] = "\n".join(detail_parts) if detail_parts else None
    return [
        {
            "category": "terrain",
            "notable": False,
            "level": None,
            "detail": detail,
        }
    ]


def _temperature_status_rows(
    taxon: dict[str, Any],
    taxon_dir: Path,
    *,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
    unit_system: Optional[units.UnitSystem] = None,
) -> list[dict[str, Any]]:
    from util import gis_lookup

    bio6 = _numeric_summary_for_context(
        taxon_id=taxon_id,
        taxon_dir=taxon_dir,
        variable_id="bio_6",
        location_gid=location_gid,
    )
    bio5 = _numeric_summary_for_context(
        taxon_id=taxon_id,
        taxon_dir=taxon_dir,
        variable_id="bio_5",
        location_gid=location_gid,
    )
    bio1_local = (
        _numeric_summary_for_context(
            taxon_id=taxon_id,
            taxon_dir=taxon_dir,
            variable_id="bio_1",
            location_gid=location_gid,
        )
        if location_gid
        else {}
    )
    bio1_global = (
        _numeric_summary_for_context(
            taxon_id=taxon_id,
            taxon_dir=taxon_dir,
            variable_id="bio_1",
            location_gid=None,
        )
        if location_gid
        else {}
    )
    coldest_winter_raw = bio6.get("min")
    hottest_raw = bio5.get("max")
    raw_units: Optional[str] = None
    display_units = ""
    try:
        _entries, variable_lookup = gis_lookup.load_variable_metadata()
        raw_units = str((variable_lookup.get("bio_6") or {}).get("units") or "").strip() or None
    except Exception:
        raw_units = None
    target_unit = units.equivalent_unit(raw_units, unit_system) if raw_units else raw_units
    display_units = str(units.display_unit(target_unit) or "").strip()
    coldest_winter = _format_scalar_value_for_system(
        coldest_winter_raw,
        unit=raw_units,
        unit_system=unit_system,
    )
    hottest = _format_scalar_value_for_system(
        hottest_raw,
        unit=raw_units,
        unit_system=unit_system,
    )

    detail: Optional[str] = None
    detail_parts: list[str] = []
    if coldest_winter is not None:
        label = _winter_coldness_label(float(coldest_winter_raw))
        low_outlier_text = (
            _select_variable_outlier_text(
                variable_id="bio_6",
                taxon=taxon,
                taxon_dir=taxon_dir,
                preferred_metrics=("min", "mean"),
                location_gid=location_gid,
            )
            if not location_gid
            else None
        )
        if display_units:
            line = f"Lowest temperature: {label} ({coldest_winter} {display_units})"
        else:
            line = f"Lowest temperature: {label} ({coldest_winter})"
        if low_outlier_text:
            line += f" ({low_outlier_text})"
        detail_parts.append(line)
    if hottest is not None:
        label = _summer_heat_label(float(hottest_raw))
        high_outlier_text = (
            _select_variable_outlier_text(
                variable_id="bio_5",
                taxon=taxon,
                taxon_dir=taxon_dir,
                preferred_metrics=("max", "mean"),
                location_gid=location_gid,
            )
            if not location_gid
            else None
        )
        if display_units:
            line = f"Hottest temperature: {label} ({hottest} {display_units})"
        else:
            line = f"Hottest temperature: {label} ({hottest})"
        if high_outlier_text:
            line += f" ({high_outlier_text})"
        detail_parts.append(line)
    if detail_parts:
        if location_gid:
            location_name = _location_label(location_gid)
            comparison = _temperature_location_compare_text(
                local_mean=bio1_local.get("mean"),
                global_mean=bio1_global.get("mean"),
                location_name=location_name,
            )
            local_mean_value = _format_scalar_value_for_system(
                bio1_local.get("mean"),
                unit=raw_units,
                unit_system=unit_system,
            )
            global_mean_value = _format_scalar_value_for_system(
                bio1_global.get("mean"),
                unit=raw_units,
                unit_system=unit_system,
            )
            if comparison and local_mean_value and global_mean_value:
                if display_units:
                    detail_parts.append(
                        f"Mean temperature: {comparison} ({local_mean_value} {display_units} vs {global_mean_value} {display_units})."
                    )
                else:
                    detail_parts.append(
                        f"Mean temperature: {comparison} ({local_mean_value} vs {global_mean_value})."
                    )
        detail = "\n".join(detail_parts)
    return [
        {
            "category": "temperature",
            "notable": False,
            "level": None,
            "detail": detail,
        }
    ]


def _precipitation_status_rows(
    taxon: dict[str, Any],
    taxon_dir: Path,
    *,
    taxon_id: Optional[int] = None,
    location_gid: Optional[str] = None,
    unit_system: Optional[units.UnitSystem] = None,
) -> list[dict[str, Any]]:
    from util import gis_lookup

    bio12 = _numeric_summary_for_context(
        taxon_id=taxon_id,
        taxon_dir=taxon_dir,
        variable_id="bio_12",
        location_gid=location_gid,
    )
    driest_raw = bio12.get("min")
    wettest_raw = bio12.get("max")
    average_raw = bio12.get("mean")
    bio12_global = (
        _numeric_summary_for_context(
            taxon_id=taxon_id,
            taxon_dir=taxon_dir,
            variable_id="bio_12",
            location_gid=None,
        )
        if location_gid
        else {}
    )

    raw_units: Optional[str] = None
    try:
        _entries, variable_lookup = gis_lookup.load_variable_metadata()
        raw_units = str((variable_lookup.get("bio_12") or {}).get("units") or "").strip() or None
    except Exception:
        raw_units = None
    target_unit = units.equivalent_unit(raw_units, unit_system) if raw_units else raw_units
    display_units = str(units.display_unit(target_unit) or "").strip()
    driest = _format_scalar_value_for_system(
        driest_raw,
        unit=raw_units,
        unit_system=unit_system,
    )
    wettest = _format_scalar_value_for_system(
        wettest_raw,
        unit=raw_units,
        unit_system=unit_system,
    )
    average = _format_scalar_value_for_system(
        average_raw,
        unit=raw_units,
        unit_system=unit_system,
    )

    def _with_units(value: str) -> str:
        return f"{value} {display_units}" if display_units else value

    def _range_with_units(low: str, high: str) -> str:
        if display_units:
            return f"{low}-{high} {display_units}"
        return f"{low}-{high}"

    detail: Optional[str] = None
    detail_lines: list[str] = []
    if average is not None:
        mean_label = _annual_precip_label(float(average_raw))
        preference_line = f"Prefers {mean_label} ({_with_units(average)})"
        if not location_gid:
            outlier = _select_variable_outlier_candidate(
                variable_id="bio_12",
                taxon=taxon,
                taxon_dir=taxon_dir,
                preferred_metrics=("mean",),
                location_gid=location_gid,
            )
            if outlier:
                qualifier = str(outlier.get("qualifier") or "").strip()
                polarity = str(outlier.get("polarity") or "").strip()
                context = str(outlier.get("context") or "").strip()
                if qualifier and polarity and context:
                    condition = "dry" if polarity == "low" else "wet"
                    preference_line += f", which is {qualifier} {condition} for {context}"
        else:
            location_name = _location_label(location_gid)
            comparison = _precip_location_compare_text(
                local_mean=average_raw,
                global_mean=bio12_global.get("mean"),
                location_name=location_name,
            )
            global_mean_value = _format_scalar_value_for_system(
                bio12_global.get("mean"),
                unit=raw_units,
                unit_system=unit_system,
            )
            if comparison and global_mean_value:
                if display_units:
                    preference_line += (
                        f", {comparison} "
                        f"({_with_units(average)} vs {_with_units(global_mean_value)})"
                    )
                else:
                    preference_line += f", {comparison} ({average} vs {global_mean_value})"
        detail_lines.append(preference_line)
        if driest is not None and wettest is not None:
            low_label = _annual_precip_label(float(driest_raw))
            high_label = _annual_precip_label(float(wettest_raw))
            if low_label == high_label:
                detail_lines.append(
                    f"Can tolerate {low_label} ({_range_with_units(driest, wettest)})"
                )
            else:
                detail_lines.append(
                    f"Can tolerate {low_label} ({_with_units(driest)}) to {high_label} ({_with_units(wettest)})"
                )
    elif driest is not None and wettest is not None:
        low_label = _annual_precip_label(float(driest_raw))
        high_label = _annual_precip_label(float(wettest_raw))
        if low_label == high_label:
            detail_lines.append(
                f"Can tolerate {low_label} ({_range_with_units(driest, wettest)})"
            )
        else:
            detail_lines.append(
                f"Can tolerate {low_label} ({_with_units(driest)}) to {high_label} ({_with_units(wettest)})"
            )
    if detail_lines:
        detail = "\n".join(detail_lines)
    return [
        {
            "category": "precipitation",
            "notable": False,
            "level": None,
            "detail": detail,
        }
    ]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _title_case_words(value: str) -> str:
    return " ".join(
        token[:1].upper() + token[1:].lower()
        for token in str(value or "").split()
        if token
    )


# ---------------------------------------------------------------------------
# Rendering + main entry point
# ---------------------------------------------------------------------------


def _lines_from_categorical_phrase(
    text: Optional[str],
) -> list[dict[str, Any]]:
    styled_prefix_starts = {
        "always",
        "almost always",
        "primarily",
        "often",
        "sometimes",
        "rarely",
    }
    raw_lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not raw_lines:
        return []

    lines: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        cleaned = raw_line.strip().strip(".")
        if not cleaned:
            continue
        parts: list[str]
        if ", and " in cleaned and " in " in cleaned.split(", and ", 1)[1]:
            first, second = cleaned.split(", and ", 1)
            parts = [first.strip(), second.strip()]
        else:
            parts = [cleaned]

        for part in parts:
            clause = part.strip().strip(".")
            if not clause:
                continue
            lower_clause = clause.lower()
            if " in " in clause and any(
                lower_clause == prefix or lower_clause.startswith(f"{prefix} in ")
                for prefix in styled_prefix_starts
            ):
                prefix_raw, body_raw = clause.split(" in ", 1)
                prefix_text = _title_case_words(prefix_raw)
                body_text = body_raw.strip()
                if prefix_text and body_text:
                    lines.append({"prefix": f"{prefix_text} in:", "body": body_text})
                    continue
            lines.append({"body": clause})
    return lines


def _build_profile_sections(profile: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    habitat = profile.get("habitat")
    if habitat:
        lines = _lines_from_categorical_phrase(habitat)
        if lines:
            sections.append({"id": "habitat", "title": "Habitat", "lines": lines})

    climate = profile.get("climate")
    if climate:
        lines = _lines_from_categorical_phrase(climate)
        if lines:
            sections.append({"id": "climate", "title": "Climates", "lines": lines})

    locations = profile.get("locations")
    if locations:
        sections.append(
            {
                "id": "locations",
                "title": "Locations",
                "lines": [{"body": str(locations).strip()}],
            }
        )

    for row in profile.get("categories", []):
        title = str(row.get("category") or "other").replace("_", " ").title()
        detail = str(row.get("detail") or "").strip()
        category_id = str(row.get("category") or "other")
        if detail:
            raw_lines = [line.strip() for line in detail.splitlines() if line.strip()]
            if not raw_lines:
                raw_lines = [detail]
            lines = [{"body": line_body} for line_body in raw_lines]
        else:
            lines = [{"body": "Not notable."}]
        sections.append(
            {
                "id": category_id,
                "title": title,
                "lines": lines,
            }
        )

    return sections


def _render_profile_text(profile: dict[str, Any]) -> str:
    lines: list[str] = [f"Summary: {profile['summary']}"]
    habitat = profile.get("habitat")
    if habitat:
        lines.append(f"Habitat: {habitat}")
    climate = profile.get("climate")
    if climate:
        lines.append(f"Climate: {climate}")
    locations = profile.get("locations")
    if locations:
        lines.append(f"Locations: {locations}")

    for row in profile.get("categories", []):
        title = str(row.get("category") or "other").replace("_", " ").title()
        detail = str(row.get("detail") or "").strip()
        if detail:
            lines.append(f"{title}: {detail}.")
        else:
            lines.append(f"{title}: Not notable.")
    return "\n".join(lines)


def build_taxon_description(
    taxon: dict[str, Any],
    *,
    location_gid: Optional[str] = None,
    unit_system: Optional[str] = None,
) -> dict[str, Any]:
    """Builds a structured description and plain text rendering for a taxon."""
    from util import taxa_navigation

    scientific_name = (taxon.get("scientific_name") or "").replace("_", " ").strip()
    if not scientific_name:
        return {
            "summary": "",
            "habitat": None,
            "climate": None,
            "locations": None,
            "categories": [],
            "sections": [],
            "text": "",
        }

    common_names = taxa_navigation.extract_common_names_for_language(
        taxon, language=taxa_navigation.CONFIG.common_name_language
    )
    common_name = common_names[0] if common_names else None
    subject = f"The {common_name} ({scientific_name})" if common_name else scientific_name

    family = _find_ancestor_by_rank(taxon, "FAMILY")
    family_name = None
    family_common = None
    if family:
        family_name = (family.get("scientific_name") or "").replace("_", " ").strip()
        family_common_names = taxa_navigation.extract_common_names_for_language(
            family, language=taxa_navigation.CONFIG.common_name_language
        )
        if family_common_names:
            family_common = family_common_names[0]

    if family_common:
        summary = f"{subject} is a species of {family_common}."
    elif family_name:
        summary = f"{subject} is a species in the family {family_name}."
    else:
        summary = f"{subject} is a species."

    taxon_id = _to_int(taxon.get("taxon_key"))
    taxon_dir = Path(taxon["path"])
    habitat_raw = _top_categorical_phrase(
        taxon_dir,
        variable_id="landcover",
        label="habitats",
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    climate_raw = _top_categorical_phrase(
        taxon_dir,
        variable_id="koppen_geiger",
        label="climates",
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    habitat = _format_categorical_phrase(habitat_raw, label="habitat") if habitat_raw else None
    habitat_outlier = _categorical_outlier_text(
        taxon,
        taxon_dir,
        variable_id="landcover",
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    if habitat and habitat_outlier:
        habitat = f"{habitat}\n{habitat_outlier}."
    climate = _format_categorical_phrase(climate_raw, label="climate") if climate_raw else None
    climate_outlier = _categorical_outlier_text(
        taxon,
        taxon_dir,
        variable_id="koppen_geiger",
        taxon_id=taxon_id,
        location_gid=location_gid,
    )
    if climate and climate_outlier:
        climate = f"{climate}\n{climate_outlier}."

    location_text = None
    if taxon_id is not None:
        location_text = _build_location_text(
            taxon_id,
            location_gid=location_gid,
            min_fraction=0.0,
            limit=3,
        )
        if location_text == "":
            location_text = None
    location_text = _capitalize_leading_the(location_text)

    resolved_unit_system = units.normalize_unit_system(unit_system)

    categories = (
        _terrain_status_rows(
            taxon,
            taxon_dir,
            taxon_id=taxon_id,
            location_gid=location_gid,
            unit_system=resolved_unit_system,
        )
        + _temperature_status_rows(
            taxon,
            taxon_dir,
            taxon_id=taxon_id,
            location_gid=location_gid,
            unit_system=resolved_unit_system,
        )
        + _precipitation_status_rows(
            taxon,
            taxon_dir,
            taxon_id=taxon_id,
            location_gid=location_gid,
            unit_system=resolved_unit_system,
        )
    )

    profile: dict[str, Any] = {
        "summary": summary,
        "habitat": habitat,
        "climate": climate,
        "locations": location_text,
        "categories": categories,
    }
    profile["sections"] = _build_profile_sections(profile)
    profile["text"] = _render_profile_text(profile)
    return profile

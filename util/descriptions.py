# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rule-based natural language descriptions for taxa."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_NEEDS_ARTICLE: frozenset[str] = frozenset({
    "united states", "united kingdom", "netherlands", "philippines", "gambia",
})


def _with_definite_article(name: str) -> str:
    s = name.strip()
    if s.lower().startswith("the "):
        return s
    if s.lower() in _NEEDS_ARTICLE:
        return f"the {s}"
    return s


def _capitalize_leading_the(text: str) -> str:
    s = text.strip()
    if s.lower().startswith("the "):
        return f"The {s[4:]}"
    return s


def _join_names(names: list[str], *, use_and: bool = True) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if use_and:
        return ", ".join(names[:-1]) + f", and {names[-1]}"
    return ", ".join(names)


# ---------------------------------------------------------------------------
# Location text
# ---------------------------------------------------------------------------


def build_location_text(
    taxon_key: str | int,
    *,
    hierarchy: dict[str, dict],
    storage,
    loc_taxa_path: Path,
    scope_by_level: dict[int, str],
    location_gid: str | None = None,
    limit: int = 3,
) -> str:
    """Return a natural language location string for a taxon.

    Examples:
        "California, Oregon, and Washington in the United States"
        "the United States, Canada, and Mexico"
        "California and other regions in the United States"
    """
    try:
        table = storage.read_table(loc_taxa_path, filters=[("taxon_key", "=", str(taxon_key))])
    except Exception:
        return ""
    if table.num_rows == 0:
        return ""

    by_scope: dict[str, list[tuple[str, int]]] = {}
    for scope, gid, count in zip(
        table.column("scope").to_pylist(),
        table.column("gid").to_pylist(),
        table.column("count").to_pylist(),
    ):
        if count:
            by_scope.setdefault(str(scope), []).append((str(gid), int(count)))

    scope0 = scope_by_level.get(0, "gadm_level0")
    scope1 = scope_by_level.get(1, "gadm_level1")
    scope2 = scope_by_level.get(2, "gadm_level2")

    def _entries_for_scope(scope: str, parent_gid: str | None = None) -> list[tuple[str, str, int]]:
        result = []
        for gid, count in by_scope.get(scope, []):
            rec = hierarchy.get(gid)
            if not rec or not rec.get("name"):
                continue
            if parent_gid and rec.get("parent_gid") != parent_gid:
                continue
            result.append((gid, rec["name"], count))
        result.sort(key=lambda r: r[2], reverse=True)
        return result

    def _top_names(scope: str, parent_gid: str | None = None) -> tuple[list[tuple[str, str, int]], bool]:
        entries = _entries_for_scope(scope, parent_gid)
        seen: set[str] = set()
        deduped = []
        for entry in entries:
            if entry[1] not in seen:
                seen.add(entry[1])
                deduped.append(entry)
        has_more = len(deduped) > limit
        return deduped[:limit], has_more

    def _format(names: list[str], *, parent: str, has_more: bool, more_label: str) -> str:
        text = _join_names(names, use_and=not has_more)
        if has_more:
            return f"{text} and other {more_label} in {parent}"
        return f"{text} in {parent}"

    # --- Location-scoped: drill down into the given location ---
    if location_gid:
        rec = hierarchy.get(location_gid)
        if rec is None:
            return ""
        level = rec.get("level")
        name = rec.get("name", location_gid)
        if level == 0:
            entries, has_more = _top_names(scope1, parent_gid=location_gid)
            state_names = [e[1] for e in entries]
            if state_names:
                return _format(state_names, parent=_with_definite_article(name), has_more=has_more, more_label="regions")
            return _with_definite_article(name)
        if level == 1:
            entries, has_more = _top_names(scope2, parent_gid=location_gid)
            county_names = [e[1] for e in entries]
            if county_names:
                return _format(county_names, parent=name, has_more=has_more, more_label="subregions")
            return name
        if level == 2:
            parent = hierarchy.get(rec.get("parent_gid", ""))
            parent_name = parent["name"] if parent else ""
            return f"{name} in {parent_name}" if parent_name else name
        return name

    # --- Global: countries, drilling into states if there is only one ---
    country_entries, has_more = _top_names(scope0)
    if not country_entries:
        return ""

    if len(country_entries) == 1 and not has_more:
        country_gid, country_name, _ = country_entries[0]
        state_entries, has_more_states = _top_names(scope1, parent_gid=country_gid)
        state_names = [e[1] for e in state_entries]
        if state_names:
            return _format(state_names, parent=_with_definite_article(country_name), has_more=has_more_states, more_label="regions")
        return _with_definite_article(country_name)

    display_names = [_with_definite_article(e[1]) for e in country_entries]
    text = _join_names(display_names, use_and=not has_more)
    if has_more:
        return f"{text} and other countries"
    return text


# ---------------------------------------------------------------------------
# Terrain text
# ---------------------------------------------------------------------------

def _slope_band(grade: float) -> str:
    if grade < 3:
        return "flat"
    if grade < 8:
        return "gentle"
    if grade < 12:
        return "moderate"
    if grade < 18:
        return "moderately steep"
    if grade < 28:
        return "steep"
    return "very steep"


def _compass_direction(degrees: float) -> str:
    d = degrees % 360
    idx = int((d + 22.5) / 45) % 8
    return ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"][idx]


def build_terrain_lines(
    numerical_stats: dict[str, dict],
    circular_stats: dict[str, dict],
    *,
    unit_system: Optional[str] = None,
) -> list[dict]:
    lines: list[dict] = []

    elev = numerical_stats.get("elevation") or {}
    p10 = elev.get("min")
    p90 = elev.get("max")
    if p10 is not None and p90 is not None:
        if unit_system == "imperial":
            p10 = p10 / 0.3048
            p90 = p90 / 0.3048
            unit_label = "ft"
        else:
            unit_label = "m"
        elev_text: str | None = None
        for step in (100, 10, 1):
            lo = round(p10 / step) * step
            hi = round(p90 / step) * step
            if lo != hi:
                elev_text = f"Found from {lo:,} to {hi:,} {unit_label} elevation"
                break
        if elev_text is None:
            elev_text = f"Found at {round(p10):,} {unit_label} elevation"
        lines.append({"body": elev_text})

    slope = numerical_stats.get("slope") or {}
    slope_mean = slope.get("mean")
    slope_p10 = slope.get("10th_percentile")
    if slope_mean is not None:
        mean_band = _slope_band(slope_mean)
        if slope_p10 is not None:
            p10_band = _slope_band(slope_p10)
        else:
            p10_band = mean_band
        if p10_band == mean_band:
            band_text = "flat areas" if mean_band == "flat" else f"{mean_band} slopes"
        elif p10_band == "flat":
            band_text = f"flat areas to {mean_band} slopes"
        else:
            band_text = f"{p10_band} to {mean_band} slopes"
        lines.append({"body": f"Often on {band_text}"})

    aspect = circular_stats.get("aspect") or {}
    rbar = aspect.get("rbar")
    mean_dir = aspect.get("circular_mean")
    count = aspect.get("count") or 0
    median_slope = (numerical_stats.get("slope") or {}).get("median") or 0
    if rbar is not None and mean_dir is not None and count > 100 and rbar > 0.15 and median_slope > 5:
        direction = _compass_direction(mean_dir)
        qualifier = "Strongly prefers" if rbar > 0.35 else "Prefers"
        lines.append({"body": f"{qualifier} {direction}-facing slopes"})

    return lines


# ---------------------------------------------------------------------------
# Climate text
# ---------------------------------------------------------------------------


# Ordered highest → lowest. _VERB_RANK and _frequency_verb both derive from this.
_FREQ_THRESHOLDS: list[tuple[float, str]] = [
    (1.00, "almost always"),
    (0.80, "primarily"),
    (0.60, "commonly"),
    (0.40, "often"),
    (0.20, "sometimes"),
    (0.10, "uncommonly"),
    (0.05, "rarely"),
]
_VERB_RANK: dict[str, int] = {
    verb: len(_FREQ_THRESHOLDS) - i for i, (_, verb) in enumerate(_FREQ_THRESHOLDS)
}


def _frequency_verb(frac: float) -> str | None:
    for threshold, verb in _FREQ_THRESHOLDS:
        if frac >= threshold:
            return verb
    return None


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _build_nominal_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    *,
    attribute_axes: dict[str, list[dict]] | None = None,
    body_suffix: str = "",
) -> list[dict]:
    agg: dict[tuple, dict] = {}
    for cls in legend_classes:
        cid = cls.get("id")
        if cid is None:
            continue
        frac = float(class_fractions.get(cid, 0.0))
        if frac <= 0:
            continue
        group = str(cls.get("group") or "").strip().lower()
        if not group:
            continue
        group_label = str(cls.get("group_label") or group).strip().lower()
        attrs = sorted(str(a).strip().lower() for a in (cls.get("attributes") or []) if str(a).strip())
        key = (group, tuple(attrs))
        if key not in agg:
            agg[key] = {"group": group, "group_label": group_label, "attrs": attrs, "fraction": 0.0}
        agg[key]["fraction"] += frac

    if not agg:
        return []

    group_key_count: dict[str, int] = {}
    for k in agg:
        group_key_count[k[0]] = group_key_count.get(k[0], 0) + 1

    ranked = sorted(agg.values(), key=lambda e: e["fraction"], reverse=True)

    def _entry_key(e: dict) -> tuple:
        return (e["group"], tuple(e["attrs"]))

    def _make_stem(g: str, g_entries: list, group_label: str) -> str:
        axes = (attribute_axes or {}).get(g)
        if axes:
            kept: list[str] = []
            for axis in axes:
                axis_vals = set(axis["values"])
                per_entry = [next((a for a in e["attrs"] if a in axis_vals), None) for e in g_entries]
                distinct = set(per_entry)
                if len(distinct) == 1:
                    val = next(iter(distinct))
                    if val is not None:
                        kept.append(val)
            return f"{' '.join(kept)} {group_label}" if kept else group_label
        else:
            all_attrs = [a for e in g_entries for a in e["attrs"]]
            all_variants_present = len(g_entries) == group_key_count[g] and len(g_entries) > 1
            if all_attrs and not all_variants_present:
                return f"{_join_labels(all_attrs)} {group_label}"
            return group_label

    def _build_from_band(band: list) -> tuple[str, str]:
        by_group: dict[str, list] = {}
        for e in band:
            by_group.setdefault(e["group"], []).append(e)
        group_order = sorted(
            by_group.keys(),
            key=lambda g: sum(e["fraction"] for e in by_group[g]),
            reverse=True,
        )
        stems = [
            _make_stem(g, sorted(by_group[g], key=lambda e: e["fraction"], reverse=True), by_group[g][0]["group_label"])
            for g in group_order
        ]
        combined_frac = sum(e["fraction"] for e in band)
        verb = _frequency_verb(combined_frac) or top_verb
        return verb, _join_labels(stems) + body_suffix

    result: list[dict] = []
    used: set[tuple] = set()

    while len(result) < 2:
        remaining = [e for e in ranked if _entry_key(e) not in used]
        if not remaining:
            break
        top_verb = _frequency_verb(remaining[0]["fraction"])
        if top_verb is None:
            break

        band = [e for e in remaining if _frequency_verb(e["fraction"]) == top_verb]
        for e in band:
            used.add(_entry_key(e))

        verb, body = _build_from_band(band)

        if result and _VERB_RANK.get(verb, 0) >= _VERB_RANK.get(result[-1]["verb"], 0):
            merged_band = result[-1]["band"] + band
            verb, body = _build_from_band(merged_band)
            result[-1] = {"verb": verb, "body": body, "band": merged_band}
        else:
            result.append({"verb": verb, "body": body, "band": band})

    return [{"prefix": f"{r['verb'].capitalize()} in", "body": r["body"]} for r in result]


def build_climate_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
) -> list[dict]:
    return _build_nominal_lines(class_fractions, legend_classes, body_suffix=" climates")


def build_habitat_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    attribute_axes: dict[str, list[dict]] | None = None,
) -> list[dict]:
    return _build_nominal_lines(class_fractions, legend_classes, attribute_axes=attribute_axes)


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------


def build_description_profile(
    taxon_key: str | int,
    *,
    hierarchy: dict[str, dict],
    storage,
    loc_taxa_path: Path,
    scope_by_level: dict[int, str],
    location_gid: str | None = None,
    kg2_class_fractions: dict[int, float] | None = None,
    kg2_legend_classes: list[dict] | None = None,
    lc_class_fractions: dict[int, float] | None = None,
    lc_legend: dict | None = None,
    numerical_stats: dict[str, dict] | None = None,
    circular_stats: dict[str, dict] | None = None,
    unit_system: str | None = None,
) -> dict:
    """Return a description_profile dict with structured sections for the frontend."""
    location_text = build_location_text(
        taxon_key,
        hierarchy=hierarchy,
        storage=storage,
        loc_taxa_path=loc_taxa_path,
        scope_by_level=scope_by_level,
        location_gid=location_gid,
    )
    location_text = _capitalize_leading_the(location_text) if location_text else ""

    sections = []

    if location_text:
        sections.append({"id": "locations", "title": "Locations", "lines": [{"body": location_text}]})

    if kg2_class_fractions and kg2_legend_classes:
        climate_lines = build_climate_lines(kg2_class_fractions, kg2_legend_classes)
        if climate_lines:
            sections.append({"id": "climate", "title": "Climates", "lines": climate_lines})

    if lc_class_fractions and lc_legend:
        lc_classes = lc_legend.get("classes") or []
        lc_axes = lc_legend.get("attribute_axes") or {}
        habitat_lines = build_habitat_lines(lc_class_fractions, lc_classes, attribute_axes=lc_axes)
        if habitat_lines:
            sections.append({"id": "habitat", "title": "Habitat", "lines": habitat_lines})

    if numerical_stats or circular_stats:
        terrain_lines = build_terrain_lines(numerical_stats or {}, circular_stats or {}, unit_system=unit_system)
        if terrain_lines:
            sections.append({"id": "terrain", "title": "Terrain", "lines": terrain_lines})

    return {"sections": sections}

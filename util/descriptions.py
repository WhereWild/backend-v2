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
    if len(names) == 2 and use_and:
        return f"{names[0]} and {names[1]}"
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
            if len(entries) == 1 and not has_more:
                state_gid, state_name, _ = entries[0]
                county_entries, has_more_counties = _top_names(scope2, parent_gid=state_gid)
                county_names = [e[1] for e in county_entries]
                if county_names:
                    return _format(county_names, parent=state_name, has_more=has_more_counties, more_label="counties")
                return state_name
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
        if len(state_entries) == 1 and not has_more_states:
            state_gid, state_name, _ = state_entries[0]
            county_entries, has_more_counties = _top_names(scope2, parent_gid=state_gid)
            county_names = [e[1] for e in county_entries]
            if county_names:
                return _format(county_names, parent=state_name, has_more=has_more_counties, more_label="counties")
            return f"{state_name} in {_with_definite_article(country_name)}"
        state_names = [e[1] for e in state_entries]
        if state_names:
            return _format(state_names, parent=_with_definite_article(country_name), has_more=has_more_states, more_label="regions")
        return _with_definite_article(country_name)

    display_names = [_with_definite_article(e[1]) for e in country_entries]
    text = _join_names(display_names, use_and=not has_more)
    if has_more:
        return f"{text} and other countries"
    return text


_ECOREGION_DISPLAY_LIMIT = 3
_ECOREGION_MIN_FRACTION = 0.05


def _top_ecoregion_names(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    *,
    limit: int = _ECOREGION_DISPLAY_LIMIT,
    min_fraction: float = _ECOREGION_MIN_FRACTION,
) -> tuple[list[str], bool]:
    """Return (top ecoregion names by fraction, has_more).

    Only ecoregions at or above min_fraction are shown (a single stray
    observation shouldn't clutter the sentence), but has_more counts every
    ecoregion with any presence at all — including ones hidden by the
    threshold or past the display limit — so "and other ecoregions" stays
    accurate either way.
    """
    qualifying: list[tuple[float, str]] = []
    total_present = 0
    for cls in legend_classes:
        cid = cls.get("id")
        name = cls.get("name")
        if cid is None or not name:
            continue
        frac = float(class_fractions.get(cid, 0.0))
        if frac <= 0:
            continue
        total_present += 1
        if frac >= min_fraction:
            qualifying.append((frac, str(name)))

    if not qualifying:
        return [], False

    qualifying.sort(key=lambda e: -e[0])
    shown = qualifying[:limit]
    has_more = total_present > len(shown)
    return [name for _, name in shown], has_more


def build_ecoregion_text(
    class_fractions: dict[int, float] | None,
    legend_classes: list[dict] | None,
) -> str:
    """Return a standalone named-ecoregions line, e.g.

        "Colorado Rockies forests, Western shortgrass prairie, and Colorado
        Plateau shrublands and other ecoregions"

    Empty string if there's no qualifying ecoregion data.
    """
    if not class_fractions or not legend_classes:
        return ""
    names, has_more = _top_ecoregion_names(class_fractions, legend_classes)
    if not names:
        return ""
    text = _join_names(names, use_and=not has_more)
    if has_more:
        text = f"{text} and other ecoregions"
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
    unit_system: str | None = None,
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
    # Some legend names (e.g. WWF biomes like "Grasslands, Savannas &
    # Shrublands") — or, since clustering, a whole factored phrase like
    # "grasslands, shrublands, and savannas" — already contain a comma, so a
    # plain "X and Y" becomes ambiguous (reads as one run-on list with two
    # "and"s). Fall back to semicolons whenever any candidate label has one.
    if len(labels) == 2:
        sep = "; and " if any("," in label for label in labels) else " and "
        return f"{labels[0]}{sep}{labels[1]}"
    if any("," in label for label in labels):
        return "; ".join(labels[:-1]) + f"; and {labels[-1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _summer_heat_label(celsius: float) -> str:
    if celsius > 34:
        return "scorching"
    if celsius > 31:
        return "very hot"
    if celsius > 29:
        return "hot"
    if celsius > 27.5:
        return "warm"
    if celsius > 25:
        return "temperate"
    if celsius > 10:
        return "cool"
    return "cold"


def _winter_cold_label(celsius: float) -> str:
    if celsius < -15:
        return "extremely cold"
    if celsius < -10:
        return "incredibly cold"
    if celsius < -5:
        return "very cold"
    if celsius < 0:
        return "cold"
    if celsius < 5:
        return "cool"
    if celsius < 10:
        return "temperate"
    return "warm"


def _precip_label(mm: float) -> str:
    if mm < 150:
        return "extremely xeric"
    if mm < 300:
        return "xeric"
    if mm < 450:
        return "arid"
    if mm < 600:
        return "semi-arid"
    if mm < 800:
        return "subhumid"
    if mm < 1100:
        return "moderately wet"
    if mm < 1500:
        return "wet"
    if mm < 2200:
        return "very wet"
    if mm < 3000:
        return "extremely wet"
    return "torrential"


def _seasonal_precip_label(mm_quarter: float) -> str:
    return _precip_label(mm_quarter * 4)


def _ph_label(ph: float) -> str:
    if ph < 3.5:
        return "ultra acidic"
    if ph < 4.5:
        return "extremely acidic"
    if ph < 5.1:
        return "very strongly acidic"
    if ph < 5.6:
        return "strongly acidic"
    if ph < 6.1:
        return "moderately acidic"
    if ph < 6.6:
        return "slightly acidic"
    if ph < 7.4:
        return "neutral"
    if ph < 7.9:
        return "slightly alkaline"
    if ph < 8.5:
        return "moderately alkaline"
    if ph <= 9.0:
        return "strongly alkaline"
    return "very strongly alkaline"


def _swe_tier(swe_mm: float) -> str:
    if swe_mm < 5:
        return "snow-free"
    if swe_mm < 50:
        return "slightly snowy"
    if swe_mm < 100:
        return "moderately snowy"
    if swe_mm < 200:
        return "snowy"
    if swe_mm < 300:
        return "very snowy"
    return "incredibly snowy"


def _coarse_fragment_label(cfvo: float | None) -> str | None:
    if cfvo is None:
        return None
    if cfvo < 2:
        return "very fine"
    if cfvo < 5:
        return "fine"
    if cfvo >= 25:
        return "very coarse"
    if cfvo >= 15:
        return "coarse"
    return None


def _salinity_phrase(median_class: float | None, legend_classes: list[dict] | None) -> str | None:
    """Median salinity class name as a lowercase adjective phrase, e.g. "slightly saline".

    Dropped entirely for "Non saline" (class 0) — only worth calling out when the
    taxon's typical soil actually registers as saline.
    """
    if median_class is None or not legend_classes:
        return None
    cid = int(round(median_class))
    if cid <= 0:
        return None
    for cls in legend_classes:
        if cls.get("id") == cid:
            name = cls.get("name")
            return str(name).strip().lower() if name else None
    return None


def _build_nominal_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    *,
    attribute_axes: dict[str, list[dict]] | None = None,
    body_suffix: str = "",
    factor_shared_modifiers: bool = False,
) -> list[dict]:
    agg: dict[tuple, dict] = {}
    # True once any class fans its fraction out to more than one group (see
    # below) — the moment that happens, group totals can overlap (the same
    # underlying occurrences get full credit in more than one group), so
    # re-summing fractions *across* groups to sharpen a combined verb (done
    # below in _build_from_band) would double-count and must be skipped.
    has_fanout = False
    for cls in legend_classes:
        cid = cls.get("id")
        if cid is None:
            continue
        frac = float(class_fractions.get(cid, 0.0))
        if frac <= 0:
            continue
        # Normally one class belongs to exactly one group (a plain
        # group/group_label/attributes triple). A few source schemes bundle
        # multiple conceptually distinct things into one compound class (e.g.
        # a WWF biome like "Grasslands, Savannas & Shrublands" isn't really
        # one land-cover type — it's three, inseparable in the source pixel
        # data). "memberships" lets such a class fan its full fraction out to
        # several groups at once (each gets full credit, not a split share) —
        # see biome_legend.json for the concrete case.
        memberships = cls.get("memberships") or [
            {
                "group": cls.get("group"),
                "group_label": cls.get("group_label"),
                "solo_group_label": cls.get("solo_group_label"),
                "attributes": cls.get("attributes"),
            }
        ]
        if len(memberships) > 1:
            has_fanout = True
        for m in memberships:
            group = str(m.get("group") or "").strip().lower()
            if not group:
                continue
            group_label = str(m.get("group_label") or group).strip().lower()
            solo_group_label = m.get("solo_group_label")
            solo_group_label = str(solo_group_label).strip().lower() if solo_group_label else None
            attrs = sorted(str(a).strip().lower() for a in (m.get("attributes") or []) if str(a).strip())
            key = (group, tuple(attrs))
            if key not in agg:
                agg[key] = {
                    "group": group,
                    "group_label": group_label,
                    "solo_group_label": solo_group_label,
                    "attrs": attrs,
                    "fraction": 0.0,
                }
            agg[key]["fraction"] += frac

    if not agg:
        return []

    group_key_count: dict[str, int] = {}
    for k in agg:
        group_key_count[k[0]] = group_key_count.get(k[0], 0) + 1

    ranked = sorted(agg.values(), key=lambda e: e["fraction"], reverse=True)

    def _entry_key(e: dict) -> tuple:
        return (e["group"], tuple(e["attrs"]))

    def _group_modifier(g: str, g_entries: list) -> str:
        """The adjective phrase for group g's currently-active entries, e.g.
        "temperate and montane" — without the group_label attached, so
        sibling groups' modifiers can be compared for exact equality (see
        factor_shared_modifiers below) before the label gets glued on."""
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
            return " ".join(kept)
        else:
            all_attrs = [a for e in g_entries for a in e["attrs"]]
            all_variants_present = len(g_entries) == group_key_count[g] and len(g_entries) > 1
            if all_attrs and not all_variants_present:
                return _join_labels(all_attrs)
            return ""

    def _stem_parts(g: str, g_entries: list) -> tuple[str, str]:
        """(label, modifier) for group g's currently-active entries. A
        group's label is usually fixed, but a member can register a distinct
        "solo_group_label" (e.g. "Taiga") to use only when it's the sole
        representative of its group in this output — the moment a sibling
        entry (e.g. another conifer forest) joins it, the generic
        group_label takes over instead. The solo label already carries
        whatever specificity the modifier would add (Taiga *is* boreal), so
        the modifier is suppressed whenever the solo label is used."""
        if len(g_entries) == 1 and g_entries[0].get("solo_group_label"):
            return g_entries[0]["solo_group_label"], ""
        return g_entries[0]["group_label"], _group_modifier(g, g_entries)

    def _make_stem(g: str, g_entries: list) -> str:
        label, modifier = _stem_parts(g, g_entries)
        return f"{modifier} {label}" if modifier else label

    def _build_from_band(band: list) -> tuple[str, str]:
        by_group: dict[str, list] = {}
        for e in band:
            by_group.setdefault(e["group"], []).append(e)
        group_order = sorted(
            by_group.keys(),
            key=lambda g: sum(e["fraction"] for e in by_group[g]),
            reverse=True,
        )
        sorted_entries = {
            g: sorted(by_group[g], key=lambda e: e["fraction"], reverse=True) for g in group_order
        }

        if factor_shared_modifiers:
            # Two or more sibling groups landing on the exact same modifier
            # (e.g. grassland and shrubland both "temperate and montane")
            # only need to say it once — cluster by modifier equality and
            # hoist it in front of the joined bare labels, instead of each
            # group repeating it independently.
            clusters: dict[str, list[str]] = {}
            for g in group_order:
                label, modifier = _stem_parts(g, sorted_entries[g])
                clusters.setdefault(modifier, []).append(label)
            stems = [
                f"{modifier} {_join_labels(labels)}" if modifier else _join_labels(labels)
                for modifier, labels in clusters.items()
            ]
        else:
            stems = [_make_stem(g, sorted_entries[g]) for g in group_order]

        if has_fanout:
            # Bandmates' fractions may share underlying occurrences (fanned
            # out from the same class into different groups), so summing
            # them would overstate prevalence — the shared tier that put
            # them in this band together is the only sound label.
            verb = top_verb
        else:
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


def build_biome_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    attribute_axes: dict[str, list[dict]] | None = None,
) -> list[dict]:
    # No suffix — unlike kg2's short climate labels, biome names already end
    # in their own descriptive noun ("...grasslands, savannas & shrublands"),
    # so appending anything reads redundant.
    #
    # factor_shared_modifiers=True is biome-specific: several biomes fan out
    # into sibling concept groups (grassland/savanna/shrubland — see
    # biome_legend.json's "memberships") that often end up sharing the exact
    # same climate-zone modifier in the same sentence, which reads as
    # redundant repetition ("tropical & subtropical grasslands, tropical &
    # subtropical savannas..."). Climate (build_climate_lines) deliberately
    # keeps its own per-group repetition (e.g. "cold desert and cold steppe")
    # since that distinction was an intentional design choice there, not a
    # side effect of decomposing one compound noun — so this flag must NOT
    # default to on in _build_nominal_lines.
    return _build_nominal_lines(
        class_fractions, legend_classes, attribute_axes=attribute_axes, factor_shared_modifiers=True,
    )


def build_habitat_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    attribute_axes: dict[str, list[dict]] | None = None,
) -> list[dict]:
    return _build_nominal_lines(class_fractions, legend_classes, attribute_axes=attribute_axes)


# Pure single-word texture classes read better as adjectives than as bare nouns
# ("clay-rich soil" vs "clay soil"); compound classes (e.g. "Sandy Loam") already
# read correctly as-is, with the "loam"/"clay"/etc. head noun naturally last.
_PURE_TEXTURE_ADJECTIVES: dict[str, str] = {
    "clay": "clay-rich", "loam": "loamy", "silt": "silty", "sand": "sandy",
}


def _texture_phrase(name: str) -> str:
    lname = name.strip().lower()
    return _PURE_TEXTURE_ADJECTIVES.get(lname, lname)


def build_soil_texture_lines(
    class_fractions: dict[int, float],
    legend_classes: list[dict],
    *,
    coarse_part: str | None = None,
) -> list[dict]:
    entries: list[tuple[float, str]] = []
    for cls in legend_classes:
        cid = cls.get("id")
        name = cls.get("name")
        if cid is None or not name:
            continue
        frac = float(class_fractions.get(cid, 0.0))
        if frac <= 0:
            continue
        entries.append((frac, _texture_phrase(str(name))))

    texture_phrase: str | None = None
    if entries:
        entries.sort(key=lambda e: e[0], reverse=True)
        top_verb = _frequency_verb(entries[0][0])
        if top_verb is not None:
            band = [phrase for frac, phrase in entries if _frequency_verb(frac) == top_verb]
            seen: set[str] = set()
            ordered = [p for p in band if not (p in seen or seen.add(p))]
            texture_phrase = _join_labels(ordered)

    if not texture_phrase and not coarse_part:
        return []
    if texture_phrase and coarse_part:
        body = f"Prefers {coarse_part}, {texture_phrase} soil"
    elif texture_phrase:
        body = f"Prefers {texture_phrase} soil"
    else:
        body = f"Prefers {coarse_part} soil"
    return [{"body": body}]


def build_soil_lines(numerical_stats: dict[str, dict], *, salinity_phrase: str | None = None) -> list[dict]:
    lines: list[dict] = []

    def _get(var: str, metric: str) -> float | None:
        v = (numerical_stats.get(var) or {}).get(metric)
        return float(v) if v is not None else None

    # pH — SoilGrids stores phh2o as pH × 10
    phh2o_raw = _get("phh2o", "mean")
    nitrogen = _get("nitrogen", "mean")
    ph_phrase: str | None = None
    nutrient_phrase: str | None = None
    if phh2o_raw is not None:
        ph_phrase = _ph_label(phh2o_raw)
    if nitrogen is not None:
        if nitrogen < 1.5:
            nutrient_phrase = "very nutrient poor"
        elif nitrogen < 2:
            nutrient_phrase = "nutrient poor"
        elif nitrogen >= 10:
            nutrient_phrase = "very nutrient rich"
        elif nitrogen >= 7:
            nutrient_phrase = "nutrient rich"

    parts = [p for p in (nutrient_phrase, ph_phrase, salinity_phrase) if p]
    if parts:
        lines.append({"body": f"Usually {_join_labels(parts)} soil"})

    return lines


def build_weather_lines(numerical_stats: dict[str, dict]) -> list[dict]:
    lines: list[dict] = []

    def _get(var: str, metric: str) -> float | None:
        v = (numerical_stats.get(var) or {}).get(metric)
        return float(v) if v is not None else None

    hottest = _get("bio5", "mean")
    coldest = _get("bio6", "mean")
    summer_precip = _get("bio18", "mean")
    swe_median = _get("swe", "median")
    winter_precip = _get("bio19", "median")

    summer_parts: list[str] = []
    if hottest is not None:
        summer_parts.append(_summer_heat_label(hottest))
    if summer_precip is not None:
        summer_parts.append(_seasonal_precip_label(summer_precip))

    winter_parts: list[str] = []
    if coldest is not None:
        winter_parts.append(_winter_cold_label(coldest))
    snow_tier = _swe_tier(swe_median) if swe_median is not None else None
    if snow_tier and snow_tier != "snow-free":
        winter_parts.append(snow_tier)
    elif winter_precip is not None:
        winter_parts.append(_seasonal_precip_label(winter_precip))

    if summer_parts or winter_parts:
        season_parts = []
        if summer_parts:
            season_parts.append(f"{', '.join(summer_parts)} summers")
        if winter_parts:
            season_parts.append(f"{', '.join(winter_parts)} winters")
        lines.append({"body": f"Prefers {' and '.join(season_parts)}"})

    avg_precip = _get("bio12", "median")
    if avg_precip is not None:
        lines.append({"body": f"Typically {_precip_label(avg_precip)} locations overall"})

    return lines


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
    soil_texture_class_fractions: dict[int, float] | None = None,
    soil_texture_legend: dict | None = None,
    eco_class_fractions: dict[int, float] | None = None,
    eco_legend_classes: list[dict] | None = None,
    biome_class_fractions: dict[int, float] | None = None,
    biome_legend: dict | None = None,
    salinity_median: float | None = None,
    salinity_legend_classes: list[dict] | None = None,
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
    ecoregion_text = build_ecoregion_text(eco_class_fractions, eco_legend_classes)

    sections = []

    location_lines = []
    if location_text:
        location_lines.append({"body": location_text})
    if ecoregion_text:
        location_lines.append({"body": ecoregion_text})
    if location_lines:
        sections.append({"id": "locations", "title": "Locations", "lines": location_lines})

    if kg2_class_fractions and kg2_legend_classes:
        climate_lines = build_climate_lines(kg2_class_fractions, kg2_legend_classes)
        if climate_lines:
            sections.append({"id": "climate", "title": "Climates", "lines": climate_lines})

    if biome_class_fractions and biome_legend:
        biome_classes = biome_legend.get("classes") or []
        biome_axes = biome_legend.get("attribute_axes") or {}
        biome_lines = build_biome_lines(biome_class_fractions, biome_classes, biome_axes)
        if biome_lines:
            sections.append({"id": "biomes", "title": "Biomes", "lines": biome_lines})

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

    if numerical_stats:
        weather_lines = build_weather_lines(numerical_stats)
        if weather_lines:
            sections.append({"id": "weather", "title": "Weather", "lines": weather_lines})

    soil_lines: list[dict] = []
    coarse_part = _coarse_fragment_label((numerical_stats or {}).get("cfvo", {}).get("mean"))
    if soil_texture_class_fractions and soil_texture_legend:
        soil_classes = soil_texture_legend.get("classes") or []
        soil_lines.extend(build_soil_texture_lines(soil_texture_class_fractions, soil_classes, coarse_part=coarse_part))
    elif coarse_part:
        soil_lines.append({"body": f"Prefers {coarse_part} soil"})
    if numerical_stats:
        salinity_phrase = _salinity_phrase(salinity_median, salinity_legend_classes)
        soil_lines.extend(build_soil_lines(numerical_stats, salinity_phrase=salinity_phrase))
    if soil_lines:
        sections.append({"id": "soil", "title": "Soil", "lines": soil_lines})

    return {"sections": sections}

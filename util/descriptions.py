# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rule-based natural language descriptions for taxa."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


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
    location_gid: Optional[str] = None,
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

    def _entries_for_scope(scope: str, parent_gid: Optional[str] = None) -> list[tuple[str, str, int]]:
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

    def _top_names(scope: str, parent_gid: Optional[str] = None) -> tuple[list[tuple[str, str, int]], bool]:
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
# Profile assembly
# ---------------------------------------------------------------------------


def build_description_profile(
    taxon_key: str | int,
    *,
    hierarchy: dict[str, dict],
    storage,
    loc_taxa_path: Path,
    scope_by_level: dict[int, str],
    location_gid: Optional[str] = None,
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
        sections.append({
            "id": "locations",
            "title": "Locations",
            "lines": [{"body": location_text}],
        })

    return {"sections": sections}

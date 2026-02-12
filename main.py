from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter

from util.config import load_config
from util import gis_lookup, indexing, summary_stats, taxa_navigation, units

CONFIG = load_config("global")

api_title = "WhereWild API"

api_version = "0.2.0"

category_sample_limit = 500

cors_allow_headers = ("*",)

cors_allow_methods = ("GET",)

cors_allow_origins = ("*",)

density_points = 128

forced_categorical_variables = frozenset({"landcover"})

default_species_limit = 12

max_species_limit = 100



app = FastAPI(title=api_title, version=api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cors_allow_origins),
    allow_methods=list(cors_allow_methods),
    allow_headers=list(cors_allow_headers),
)


@app.get("/health", summary="Simple liveness probe")
def health_check() -> dict[str, str]:
    """Returns a simple liveness payload.
    
    Returns:
        A status string and UTC timestamp.
    """
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/variables")
def list_environment_variables(
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> List[dict[str, Any]]:
    """Lists available environmental variables.
    
    Returns:
        A list of variable metadata entries.
    """
    return units.apply_unit_system_to_variables(
        gis_lookup.load_variable_metadata()[0],
        unit_system,
    )


@app.get("/api/species")
def list_species(
    q: str = Query(..., min_length=1, description="Search term (scientific name or common name)"),
    limit: int = Query(default_species_limit, ge=1, le=max_species_limit),
) -> List[dict[str, Any]]:
    """Searches taxa by name and returns serialized results.
    
    Args:
        q: Search term for scientific or common names.
        limit: Maximum number of matches to return.
    
    Returns:
        A list of serialized taxon payloads.
    """
    records = taxa_navigation.search_taxa_by_name(q, limit=limit)
    payloads: list[dict[str, Any]] = []
    for record, _score, matched_name in records:
        payload = taxa_navigation.serialize_taxon(record)
        if payload:
            common_names = payload.get("common_names") or []
            matched_common_name = taxa_navigation.resolve_matched_common_name(
                common_names,
                matched_name,
            )
            payload["matched_common_name"] = matched_common_name
            payloads.append(payload)
    return payloads


@app.get("/api/species/{taxon_id}")
def get_species_detail(taxon_id: int) -> dict[str, Any]:
    """Loads a single taxon record by id.
    
    Args:
        taxon_id: Taxon id to look up.
    
    Returns:
        A serialized taxon payload.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    payload = taxa_navigation.serialize_taxon(taxon) if taxon else None
    if not payload:
        raise HTTPException(
            status_code=404,
            detail=f"Species with taxon_id {taxon_id} not found",
        )
    return payload


@app.get("/locations/search")
def search_locations_endpoint(
    q: str = Query(..., min_length=1, description="Location name or partial match"),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Searches locations by name substring.
    
    Args:
        q: Search term for location names.
        limit: Maximum number of matches to return.
    
    Returns:
        A dict containing location match results.
    """
    matches = gis_lookup.search_locations(q, limit)
    return {"results": matches}


@app.get("/species/{taxon_id}/occurrences")
def species_occurrences(
    taxon_id: int,
    location: Optional[str] = Query(None, description="Filter observations by location gid"),
) -> dict[str, Any]:
    """Returns occurrence points for a taxon, optionally filtered by location.
    
    Args:
        taxon_id: Taxon id to query.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict with occurrence count and point records.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    if not Path(taxon["path"]).exists():
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    rows = taxa_navigation.load_occurrence_points(
        taxon_id,
        location.strip() if location else None,
    )
    return {
        "speciesId": taxon_id,
        "count": len(rows),
        "occurrences": rows,
    }

@app.get("/species/{taxon_id}/locations")
def species_locations(
    taxon_id: int,
    level: Optional[str] = Query(None, description="continent|country|state|county"),
    parent: Optional[str] = Query(None, description="Parent location GID (optional)"),
    limit: int = Query(500, ge=1, le=5000),
) -> List[dict[str, Any]]:
    """
    Returns location GIDs for a species based on occurrence data.
    """
    print(f"[DEBUG] species_locations called with taxon_id={taxon_id}, level={level}")
    
    try:
        taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
        print(f"[DEBUG] Taxon retrieved: {taxon}")
    except Exception as e:
        print(f"[ERROR] Failed to get taxon: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get taxon: {str(e)}")
    
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    
    taxon_dir = Path(taxon["path"])
    if not taxon_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")

    # Load all occurrence points - match the working occurrences endpoint
    print(f"[DEBUG] Loading occurrence points for taxon_id={taxon_id}")
    try:
        rows = taxa_navigation.load_occurrence_points(
            taxon_id,  # Try with int first
            None,      # location parameter
        )
        print(f"[DEBUG] Loaded {len(rows) if rows else 0} occurrence rows")
    except Exception as e:
        print(f"[ERROR] Failed to load occurrences with int, trying string: {e}")
        try:
            # Try with string version like the working endpoint does
            rows = taxa_navigation.load_occurrence_points(
                str(taxon_id),
                None,
            )
            print(f"[DEBUG] Loaded {len(rows) if rows else 0} occurrence rows with string")
        except Exception as e2:
            print(f"[ERROR] Failed to load occurrences: {e2}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load occurrences: {str(e2)}"
            )
    
    if not rows:
        print("[DEBUG] No occurrence rows found, returning empty list")
        return []

    # Rest of the function stays the same...
    gid_candidate_fields = [
        "location_gid", "loc_gid", "location", "gid", "gadm_gid",
        "admin_gid", "country_gid", "country", "admin0_gid", 
        "admin1_gid", "admin2_gid",
    ]

    gids: list[str] = []
    for r in rows:
        found = None
        for key in gid_candidate_fields:
            if key in r and r.get(key):
                found = r.get(key)
                break
        if found:
            try:
                gids.append(str(found))
            except Exception:
                continue

    print(f"[DEBUG] Found {len(gids)} location gids")
    
    if not gids:
        return []

    counts = Counter(gids)
    results: list[dict[str, Any]] = []
    processed_gids = set()
    
    for gid, cnt in counts.most_common(limit * 3):
        if gid in processed_gids:
            continue
        processed_gids.add(gid)
        
        loc_info = None
        try:
            if hasattr(gis_lookup, "get_location_by_gid"):
                loc_info = gis_lookup.get_location_by_gid(gid)
            elif hasattr(gis_lookup, "search_locations"):
                matches = gis_lookup.search_locations(gid, 1)
                if matches:
                    loc_info = matches[0]
        except Exception:
            pass
        
        if not loc_info:
            continue
        
        loc_level = loc_info.get("level", -999)
        
        if level:
            level_map = {
                "continent": -1,
                "country": 0,
                "state": 1,
                "county": 2,
            }
            expected_level = level_map.get(level.lower())
            if expected_level is not None and loc_level != expected_level:
                continue
        
        results.append({
            "gid": gid,
            "name": loc_info.get("name") or str(gid),
            "level": loc_level,
            "hierarchy": loc_info.get("hierarchy") or [],
            "count": int(cnt),
        })
        
        if len(results) >= limit:
            break

    print(f"[DEBUG] Returning {len(results)} location results")
    return results

@app.get("/locations/search_hierarchy")
def search_locations_by_hierarchy(
    q: str = Query("", description="Location name or partial match (optional if parent provided)"),
    level: Optional[str] = Query(None, description="continent|country|state|county or numeric level code"),
    parent: Optional[str] = Query(None, description="Parent name or gid. For counties pass 'United States|Utah' or a gid."),
    limit: int = Query(50, ge=1, le=1000),
) -> dict[str, Any]:

    q = (q or "").strip()

    level_map = {"continent": -1, "country": 0, "state": 1, "county": 2}

    expected_level = None
    if level is not None:
        try:
            expected_level = int(level)
        except Exception:
            expected_level = level_map.get(level.lower())

    parents_raw = (parent or "").strip()
    parent_tokens = [p.strip() for p in parents_raw.split("|") if p.strip()]

    resolved_parent_names: list[str] = []
    resolved_parent_gids: list[str] = []
    for tok in parent_tokens:
        resolved_name = tok
        resolved_gid = tok
        try:
            if hasattr(gis_lookup, "get_location_by_gid"):
                maybe = gis_lookup.get_location_by_gid(tok)
                if maybe:
                    resolved_name = maybe.get("name", tok)
                    resolved_gid = maybe.get("gid", tok)
        except Exception:
            pass
        resolved_parent_names.append(str(resolved_name).lower())
        resolved_parent_gids.append(str(resolved_gid).lower())

    if not q and not parent_tokens and expected_level is None:
        return {"results": []}

    candidates: list[dict[str, Any]] = []
    seen_gids = set()

    def matches_parent(cand: dict[str, Any]) -> bool:
        # if no parent requested, everything matches
        if not resolved_parent_names:
            return True
        cand_hierarchy = [str(x).lower() for x in (cand.get("hierarchy") or []) if x is not None]
        cand_name = str(cand.get("name") or "").lower()
        cand_gid = str(cand.get("gid") or "").lower()
        for pname, pgid in zip(resolved_parent_names, resolved_parent_gids):
            if pname in cand_hierarchy or pname == cand_name or pgid == cand_gid or pgid in cand_hierarchy:
                continue
            return False
        return True

    def push_candidate_if_valid(cand: dict[str, Any]):
        gid = str(cand.get("gid") or "")
        if not gid or gid in seen_gids:
            return
        # enforce parent matching here (critical fix)
        if not matches_parent(cand):
            return
        seen_gids.add(gid)
        candidates.append(cand)

    try:
        if q:
            raw = gis_lookup.search_locations(q, limit)
            for cand in raw:
                push_candidate_if_valid(cand)

        else:
            # 1) catalog-based enumeration (fast)
            if expected_level is not None and hasattr(gis_lookup, "load_location_catalog"):
                try:
                    entries, mapping = gis_lookup.load_location_catalog()
                    for rec in entries:
                        if getattr(rec, "level", None) != expected_level:
                            continue

                        # build hierarchy names
                        hierarchy = []
                        parent_gid = getattr(rec, "parent_gid", None)
                        while parent_gid:
                            parent_rec = mapping.get(parent_gid)
                            if not parent_rec:
                                break
                            hierarchy.append(parent_rec.name)
                            parent_gid = parent_rec.parent_gid

                        cand = {
                            "gid": rec.gid,
                            "name": rec.name,
                            "level": rec.level,
                            "hierarchy": list(reversed(hierarchy)),
                        }
                        push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break
                except Exception:
                    pass

            # 2) list_children if available
            if not candidates and hasattr(gis_lookup, "list_children"):
                for parent_tok in parent_tokens or []:
                    try:
                        parent_gid = None
                        if hasattr(gis_lookup, "get_location_by_gid"):
                            maybe = gis_lookup.get_location_by_gid(parent_tok)
                            if maybe:
                                parent_gid = maybe.get("gid")
                        raw = gis_lookup.list_children(parent_gid or parent_tok, level=expected_level, limit=limit * 3)
                        for cand in raw:
                            push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break
                    except Exception:
                        continue

            # 3) letter-scan fallback — keep scanning letters until we have enough valid matches
            if not candidates:
                letters = "abcdefghijklmnopqrstuvwxyz"
                per_letter_limit = max(50, min(200, limit))
                for ch in letters:
                    if len(candidates) >= limit:
                        break
                    try:
                        partial = gis_lookup.search_locations(ch, per_letter_limit)
                    except Exception:
                        continue
                    for cand in partial:
                        push_candidate_if_valid(cand)
                        if len(candidates) >= limit:
                            break

    except Exception:
        return {"results": []}

    # final strict filter by level (redundant but safe)
    results: list[dict[str, Any]] = []
    for cand in candidates:
        if expected_level is not None and cand.get("level") != expected_level:
            continue
        results.append({
            "gid": str(cand.get("gid") or ""),
            "name": cand.get("name") or "",
            "level": cand.get("level", -999),
            "hierarchy": cand.get("hierarchy") or [],
        })
        if len(results) >= limit:
            break

    return {"results": results}

@app.get("/species/{taxon_id}/environment/{variable_id}")
def species_environment_stats(
    taxon_id: int,
    variable_id: str,
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Returns environment stats for a taxon and variable.
    
    Args:
        taxon_id: Taxon id to query.
        variable_id: Environmental variable id.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict containing summary stats, distributions, and rankings.
    """
    variable_id = variable_id.strip()
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable_id)
    if not variable_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_id}' is not available.",
        )
    raw_units = variable_entry.get("units")
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    taxon_dir = Path(taxon["path"])
    if not taxon_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    location_gid = location.strip() if location else None
    value_type = str(variable_entry.get("value_type") or "").lower() or "numeric"
    forced_categorical = variable_id.lower() in forced_categorical_variables
    categorical_payload = None
    category_samples: list[dict[str, Any]] = []
    if forced_categorical or value_type == "categorical":
        if location_gid:
            categorical_payload = summary_stats.build_categorical_stats_for_location(
                taxon_id,
                variable_id,
                location_gid,
                sample_limit=category_sample_limit,
            )
            if categorical_payload is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No samples available for taxon {taxon_id}, "
                        f"variable '{variable_id}' and location '{location_gid}'."
                    ),
                )
            if categorical_payload:
                category_samples = categorical_payload.get("samples", [])
            value_type = "categorical"
        else:
            categorical_payload = summary_stats.load_categorical_distribution(taxon_dir, variable_id)
            if categorical_payload is None and forced_categorical:
                value_type = "categorical"
            elif categorical_payload is not None:
                value_type = "categorical"
                category_samples = summary_stats.build_categorical_samples(
                    taxon_dir, variable_id, categorical_payload.get("distribution", [])
                )
    generated_at = datetime.now(timezone.utc).isoformat()

    baseline_numeric_summary = None
    baseline_categorical_distribution: list[dict[str, Any]] = []
    baseline_categorical_totals: dict[str, Any] = {}

    if categorical_payload:
        if location_gid:
            baseline_stats = summary_stats.load_categorical_distribution(taxon_dir, variable_id)
            if baseline_stats:
                baseline_categorical_distribution = baseline_stats.get("distribution", [])
                baseline_categorical_totals = baseline_stats.get("totals", {})
        totals = categorical_payload.get("totals", {})
        total_samples = totals.get("total_samples") or 0
        summary = {
            "count": int(total_samples),
            "min": None,
            "mean": None,
            "max": None,
            "stddev": None,
            "q01": None,
            "q10": None,
            "q90": None,
            "q99": None,
        }
        ranks = indexing.load_relative_ranks(taxon_dir, variable_id, location_gid=location_gid)
        if not location_gid:
            category_samples = summary_stats.build_categorical_samples(
                taxon_dir, variable_id, categorical_payload.get("distribution", [])
            )
        response = {
            "speciesId": taxon_id,
            "species_id": taxon_id,
            "variable": variable_id,
            "variableName": variable_entry.get("name"),
            "variable_metadata": {
                "name": variable_entry.get("name"),
                "units": raw_units,
                "value_type": "categorical",
            },
            "units": raw_units,
            "variableType": "categorical",
            "generatedAt": generated_at,
            "generated_at": generated_at,
            "summary": summary,
            "histogram": None,
            "densityCurve": None,
            "binSamples": [],
            "bin_samples": [],
            "density_curve": None,
            "categoricalDistribution": categorical_payload.get("distribution", []),
            "categorical_distribution": categorical_payload.get("distribution", []),
            "dominantCategories": categorical_payload.get("dominant", []),
            "dominant_categories": categorical_payload.get("dominant", []),
            "categoricalSamples": category_samples,
            "categorical_samples": category_samples,
            "baselineCategoricalDistribution": baseline_categorical_distribution,
            "baseline_categorical_distribution": baseline_categorical_distribution,
            "baselineCategoricalTotals": baseline_categorical_totals,
            "baseline_categorical_totals": baseline_categorical_totals,
            "baselineSummary": baseline_numeric_summary,
            "baseline_summary": baseline_numeric_summary,
            "relativeRanks": ranks,
            "relative_ranks": ranks,
        }
        return response

    samples = summary_stats.gather_numeric_records(
        taxon_id,
        taxon_dir,
        variable_id,
        location_gid=location_gid,
    )
    values = [sample["value"] for sample in samples]
    if not values:
        raise HTTPException(
            status_code=404,
            detail=f"No samples available for taxon {taxon_id} and variable '{variable_id}'.",
        )
    if location_gid:
        baseline_samples = summary_stats.gather_numeric_records(
            taxon_id,
            taxon_dir,
            variable_id,
            location_gid=None,
        )
        baseline_values = [sample["value"] for sample in baseline_samples]
        if baseline_values:
            baseline_numeric_summary = summary_stats.summarize_values(baseline_values)
    summary = summary_stats.summarize_values(values)
    density_curve = indexing.build_density_curve(values, point_count=density_points)
    ranks = indexing.load_relative_ranks(taxon_dir, variable_id, location_gid=location_gid)
    response = {
        "speciesId": taxon_id,
        "species_id": taxon_id,
        "variable": variable_id,
        "variableName": variable_entry.get("name"),
        "variable_metadata": {
            "name": variable_entry.get("name"),
            "units": raw_units,
            "value_type": value_type or "numeric",
        },
        "units": raw_units,
        "variableType": value_type or "numeric",
        "generatedAt": generated_at,
        "generated_at": generated_at,
        "summary": summary,
        "histogram": None,
        "densityCurve": density_curve,
        "binSamples": [],
        "bin_samples": [],
        "density_curve": density_curve,
        "baselineSummary": baseline_numeric_summary,
        "baseline_summary": baseline_numeric_summary,
        "baselineCategoricalDistribution": [],
        "baseline_categorical_distribution": [],
        "baselineCategoricalTotals": {},
        "baseline_categorical_totals": {},
        "categoricalDistribution": [],
        "categorical_distribution": [],
        "dominantCategories": [],
        "dominant_categories": [],
        "categoricalSamples": [],
        "categorical_samples": [],
        "relativeRanks": ranks,
        "relative_ranks": ranks,
    }
    return units.apply_unit_system_to_env_response(response, unit_system, raw_units)


@app.get("/species/{taxon_id}/environment/{variable_id}/class/{class_value}/samples")
def species_environment_class_samples(
    taxon_id: int,
    variable_id: str,
    class_value: str,
    limit: int | None = Query(None, ge=1, le=10000),
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
) -> dict[str, Any]:
    """Returns categorical class samples for a taxon and variable.
    
    Args:
        taxon_id: Taxon id to query.
        variable_id: Categorical variable id.
        class_value: Class value to match.
        limit: Maximum number of samples to return.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict containing matching observation samples.
    """
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    taxon_dir = Path(taxon["path"])
    if not taxon_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    try:
        parsed_value: float | int | str
        parsed_value = float(class_value)
        if parsed_value.is_integer():
            parsed_value = int(parsed_value)
    except ValueError:
        parsed_value = class_value
    location_gid = location.strip() if location else None
    observations: list[dict[str, Any]] = []
    if location_gid:
        observations = summary_stats.categorical_class_samples_for_location(
            taxon_id,
            variable_id,
            parsed_value,
            location_gid=location_gid,
            limit=limit,
        )
    else:
        index_path = taxon_dir / "occurrence_index.parquet"
        if not index_path.exists():
            raise HTTPException(
                status_code=503,
                detail="GIS lookup utilities are unavailable on this server.",
            )
        try:
            rows = summary_stats.get_layer_records_for_class(index_path, variable_id, parsed_value)
        except Exception as exc:  # pragma: no cover - passthrough
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if limit is not None and limit > 0:
            rows = rows[:limit]
        observations = [
            {
                "catalogNumber": row[0],
                "latitude": row[1],
                "longitude": row[2],
                "value": row[3],
            }
            for row in rows
        ]
    return {
        "speciesId": taxon_id,
        "variable": variable_id,
        "classValue": parsed_value,
        "observations": observations,
        "count": len(observations),
    }


@app.get("/species/{taxon_id}/environment/{variable_id}/slice")
def species_environment_slice(
    taxon_id: int,
    variable_id: str,
    min_value: float = Query(..., alias="min"),
    max_value: float = Query(..., alias="max"),
    limit: int | None = Query(None, ge=1, le=10000),
    location: Optional[str] = Query(
        None, description="Optional location gid (GADM or GBIF region) to filter observations."
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Returns numeric samples within a value range for a taxon/variable.
    
    Args:
        taxon_id: Taxon id to query.
        variable_id: Numeric variable id.
        min_value: Minimum value to include.
        max_value: Maximum value to include.
        limit: Maximum number of samples to return.
        location: Optional location GID to filter observations.
    
    Returns:
        A dict containing range parameters and matching observations.
    """
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise HTTPException(status_code=400, detail="min and max must be finite numbers")
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable_id)
    if not variable_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_id}' is not available.",
        )
    value_type = str(variable_entry.get("value_type") or "").lower() or "numeric"
    raw_units = variable_entry.get("units")
    resolved_unit_system = units.normalize_unit_system(unit_system)
    if resolved_unit_system and raw_units:
        min_value = units.convert_value_from_system(min_value, raw_units, resolved_unit_system)
        max_value = units.convert_value_from_system(max_value, raw_units, resolved_unit_system)
    if value_type == "categorical" or variable_id.lower() in forced_categorical_variables:
        raise HTTPException(
            status_code=400,
            detail="Categorical layers must be queried via the class samples endpoint.",
        )
    taxon = taxa_navigation.get_taxon_by_id(str(taxon_id))
    if taxon is None:
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    taxon_dir = Path(taxon["path"])
    if not taxon_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown taxon {taxon_id}")
    index_path = taxon_dir / "occurrence_index.parquet"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index parquet missing for taxon {taxon_id}",
        )
    location_gid = location.strip() if location else None
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    if location_gid:
        rows = summary_stats.numeric_range_samples_for_location(
            taxon_id,
            variable_id,
            min_value,
            max_value,
            location_gid=location_gid,
            limit=limit,
        )
    else:
        try:
            rows = summary_stats.get_sorted_layer_records_in_value_range(
                index_path,
                variable_id,
                value_min=min_value,
                value_max=max_value,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    observations: list[dict[str, Any]] = []
    for catalog, lat, lon, value in rows:
        observations.append(
            {
                "catalogNumber": catalog,
                "value": float(value) if isinstance(value, (int, float)) else value,
                "latitude": lat,
                "longitude": lon,
            }
        )
    response = {
        "speciesId": taxon_id,
        "variable": variable_id,
        "range": {"min": min_value, "max": max_value},
        "units": raw_units,
        "limit": limit,
        "count": len(observations),
        "observations": observations,
    }
    return units.apply_unit_system_to_slice_response(response, unit_system, raw_units)


@app.get("/relative-rankings/{taxon_id}")
def get_relative_rankings(
    taxon_id: int,
    rank: str = Query(..., description="Descendant rank to include (e.g., SPECIES)"),
    variable: str = Query(..., description="Environmental variable / layer id"),
    metric: str = Query(..., description="Metric to rank by (min, mean, max, std, 1-99 range)"),
    limit: int = Query(50, ge=1, le=200),
    order: str = Query("asc", description="Sort order: asc or desc"),
    min_samples: int = Query(0, ge=0, description="Minimum samples required to appear"),
    include_species_like: bool = Query(
        False, description="When rank=SPECIES, include subspecies/varieties/forms"
    ),
    include_distribution: bool = Query(
        False,
        description=(
            "Include the kernel density distribution for all eligible descendants. "
            "This can be expensive for large taxa."
        ),
    ),
    location: Optional[str] = Query(
        None,
        description="Optional location GID (GADM) or GBIF region to filter descendants by",
    ),
    unit_system: Optional[str] = Query(
        None, description="Unit system for response values (metric or imperial)"
    ),
) -> dict[str, Any]:
    """Returns descendant rankings for a taxon by variable/metric.
    
    Args:
        taxon_id: Ancestor taxon id to rank descendants under.
        rank: Descendant rank to include.
        variable: Environmental variable id to rank by.
        metric: Metric name to rank by.
        limit: Maximum number of results to return.
        order: Sort order ("asc" or "desc").
        min_samples: Minimum sample count required to appear.
        include_species_like: Whether to include subspecies-like ranks for species.
        include_distribution: Whether to return raw values for density curves.
        location: Optional location GID to filter descendants by occurrence membership.
    
    Returns:
        A dict containing ranking entries and optional distribution data.
    """
    location_gid = location.strip() if location else None
    try:
        entries, distribution_values = indexing.child_relative_rankings(
            str(taxon_id),
            rank,
            variable,
            metric,
            limit=limit,
            order=order,
            min_samples=min_samples,
            include_species_like=include_species_like,
            return_distribution=include_distribution,
            location_gid=location_gid,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    total = entries[0]["count"] if entries else 0
    distribution_curve = None
    if include_distribution and distribution_values:
        distribution_curve = indexing.build_density_curve(
            distribution_values,
            point_count=density_points,
        )
    raw_units = None
    variable_entry = gis_lookup.load_variable_metadata()[1].get(variable)
    if variable_entry:
        raw_units = variable_entry.get("units")
    response = {
        "ancestor_taxon_id": taxon_id,
        "rank": rank.upper(),
        "variable": variable,
        "metric": metric,
        "units": raw_units,
        "total": total,
        "limit": limit,
        "order": order.lower(),
        "min_samples": min_samples,
        "include_species_like": include_species_like,
        "entries": entries,
        "distribution": distribution_curve,
    }
    return units.apply_unit_system_to_rankings_response(response, unit_system, raw_units)


@app.get("/relative-rankings/{taxon_id}/options")
def list_relative_ranking_options(
    taxon_id: int,
    rank: str = Query(..., description="Descendant rank to inspect (e.g., SPECIES)"),
) -> dict[str, Any]:
    """Lists available ranking metrics for an ancestor/rank.
    
    Args:
        taxon_id: Ancestor taxon id to inspect.
        rank: Descendant rank to inspect.
    
    Returns:
        A dict containing available variable/metric options.
    """
    try:
        options = indexing.list_rank_metric_options(str(taxon_id), rank)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ancestor_taxon_id": taxon_id,
        "rank": rank.upper(),
        "options": options,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

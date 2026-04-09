"""Tests for GET /species/{taxon_id}/environment/{variable_id} and sample sub-endpoints"""

import pytest
from fastapi import HTTPException

import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_url(taxon_id, variable_id, **params):
    base = f"/species/{taxon_id}/environment/{variable_id}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{qs}"
    return base


# ---------------------------------------------------------------------------
# /species/{taxon_id}/environment/{variable_id} — numeric variable
# ---------------------------------------------------------------------------


def test_env_numeric_returns_200(client, known_taxon_id, known_numeric_var):
    r = client.get(_env_url(known_taxon_id, known_numeric_var))
    assert r.status_code == 200


def test_env_numeric_top_level_fields(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    required = {
        "speciesId",
        "variable",
        "variableName",
        "variableType",
        "summary",
        "densityCurve",
    }
    missing = required - body.keys()
    assert not missing, f"Response missing fields: {missing}"


def test_env_numeric_species_id_matches(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    assert body["speciesId"] == known_taxon_id


def test_env_numeric_variable_matches(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    assert body["variable"] == known_numeric_var


def test_env_numeric_type_is_numeric(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    assert body["variableType"] == "numeric"


def test_env_numeric_summary_has_count(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    summary = body["summary"]
    assert "count" in summary
    assert summary["count"] > 0


def test_env_numeric_summary_min_lte_max(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    summary = body["summary"]
    mn = summary.get("min")
    mx = summary.get("max")
    if mn is not None and mx is not None:
        assert mn <= mx, f"summary min ({mn}) > max ({mx})"


def test_env_numeric_density_curve_present(client, known_taxon_id, known_numeric_var):
    body = client.get(_env_url(known_taxon_id, known_numeric_var)).json()
    dc = body.get("densityCurve")
    assert dc is not None, "densityCurve should not be None for numeric variable"


def test_env_numeric_unit_system_metric(client, known_taxon_id, known_numeric_var):
    r = client.get(_env_url(known_taxon_id, known_numeric_var, unit_system="metric"))
    assert r.status_code == 200


def test_env_numeric_unit_system_imperial(client, known_taxon_id, known_numeric_var):
    r = client.get(_env_url(known_taxon_id, known_numeric_var, unit_system="imperial"))
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /species/{taxon_id}/environment/{variable_id} — categorical variable
# ---------------------------------------------------------------------------


def test_env_categorical_returns_200(client, known_taxon_id, known_categorical_var):
    r = client.get(_env_url(known_taxon_id, known_categorical_var))
    assert r.status_code == 200


def test_env_categorical_type_is_categorical(client, known_taxon_id, known_categorical_var):
    body = client.get(_env_url(known_taxon_id, known_categorical_var)).json()
    assert body["variableType"] == "categorical"


def test_env_categorical_has_distribution(client, known_taxon_id, known_categorical_var):
    body = client.get(_env_url(known_taxon_id, known_categorical_var)).json()
    dist = body.get("categoricalDistribution")
    assert isinstance(dist, list), "categoricalDistribution should be a list"
    assert len(dist) > 0, "categoricalDistribution should not be empty"


def test_env_categorical_density_curve_is_none(client, known_taxon_id, known_categorical_var):
    body = client.get(_env_url(known_taxon_id, known_categorical_var)).json()
    assert body.get("densityCurve") is None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_env_invalid_variable_returns_404(client, known_taxon_id):
    r = client.get(_env_url(known_taxon_id, "not_a_real_variable"))
    assert r.status_code == 404


def test_env_invalid_taxon_returns_404(client, known_numeric_var):
    r = client.get(_env_url(999999999, known_numeric_var))
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /species/{taxon_id}/environment/{variable_id}/slice
# ---------------------------------------------------------------------------


def test_env_slice_returns_200(client, known_taxon_id, known_numeric_var):
    r = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=-10&max=25")
    assert r.status_code == 200


def test_env_slice_shape(client, known_taxon_id, known_numeric_var):
    body = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=-10&max=25").json()
    required = {"speciesId", "variable", "range", "count", "observations"}
    missing = required - body.keys()
    assert not missing, f"Slice response missing fields: {missing}"


def test_env_slice_count_matches_observations(client, known_taxon_id, known_numeric_var):
    body = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=-10&max=25").json()
    assert body["count"] == len(body["observations"])


def test_env_slice_observation_fields(client, known_taxon_id, known_numeric_var):
    body = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=-10&max=25").json()
    required = {"catalogNumber", "latitude", "longitude", "value"}
    for obs in body["observations"][:20]:
        missing = required - obs.keys()
        assert not missing, f"Observation missing fields: {missing}"


def test_env_slice_values_within_range(client, known_taxon_id, known_numeric_var):
    mn, mx = -10.0, 25.0
    body = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min={mn}&max={mx}").json()
    for obs in body["observations"]:
        v = obs["value"]
        assert mn <= v <= mx, f"Observation value {v} outside requested range [{mn}, {mx}]"


def test_env_slice_missing_min_returns_422(client, known_taxon_id, known_numeric_var):
    r = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?max=25")
    assert r.status_code == 422


def test_env_slice_categorical_variable_returns_400(client, known_taxon_id, known_categorical_var):
    r = client.get(f"/species/{known_taxon_id}/environment/{known_categorical_var}/slice?min=0&max=10")
    assert r.status_code == 400


def test_env_slice_invalid_variable_returns_404(client, known_taxon_id):
    """Unknown variable in slice endpoint returns 404 (line 845)."""
    r = client.get(f"/species/{known_taxon_id}/environment/not_a_real_var/slice?min=-10&max=25")
    assert r.status_code == 404


def test_env_slice_non_finite_min_returns_400(client, known_taxon_id, known_numeric_var):
    """Non-finite min triggers 400 (line 840)."""
    r = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=inf&max=25")
    assert r.status_code == 400


def test_env_slice_non_finite_max_returns_400(client, known_taxon_id, known_numeric_var):
    """Non-finite max triggers 400 (line 840)."""
    r = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=-10&max=-inf")
    assert r.status_code == 400


def test_env_slice_swapped_min_max_auto_corrects(client, known_taxon_id, known_numeric_var):
    """Swapped min/max are auto-corrected (line 842). Should return 200, not error."""
    r = client.get(f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=25&max=-10")
    assert r.status_code == 200
    body = r.json()
    assert body["range"]["min"] <= body["range"]["max"]


def test_env_slice_with_unit_system_imperial(client, known_taxon_id, known_numeric_var):
    """unit_system param triggers value conversion (lines 853-854)."""
    r = client.get(
        f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice?min=-10&max=25&unit_system=imperial"
    )
    assert r.status_code == 200


def test_variable_tiles_accept_circular_aspect_deg(client, monkeypatch):
    monkeypatch.setattr(
        main.gis_lookup,
        "load_layer_metadata",
        lambda: {
            "aspect_deg": {
                "value_type": "circular",
                "derived": True,
                "region_root": "regions",
                "filename_template": "dem.tif",
            }
        },
    )
    monkeypatch.setattr(main.tiles, "render_variable_tile_bytes", lambda **_kwargs: b"png")
    main._map_enabled_variables.cache_clear()

    try:
        response = client.get("/api/variables/aspect_deg/tiles/1/0/0.png")
    finally:
        main._map_enabled_variables.cache_clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png"


def test_env_circular_with_location_uses_circular_summary(client, monkeypatch):
    captured = {}

    monkeypatch.setattr(
        main.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [],
            {
                "aspect_deg": {
                    "name": "Aspect",
                    "units": "degrees",
                    "value_type": "circular",
                }
            },
        ),
    )
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _taxon_id: {"path": "/tmp/fake-taxon"})
    monkeypatch.setattr(main, "_path_exists", lambda _path: True)
    monkeypatch.setattr(
        main.summary_stats,
        "gather_numeric_records",
        lambda *_args, **_kwargs: [{"value": 359.0}, {"value": 1.0}],
    )

    def _fake_summarize(values, *, circular=False):
        captured["summary_circular"] = circular
        return {
            "count": len(values),
            "min": 359.0,
            "1st percentile": 359.02,
            "10th percentile": 359.2,
            "25th percentile": 359.5,
            "median": 0.0,
            "75th percentile": 0.5,
            "90th percentile": 0.8,
            "99th percentile": 0.98,
            "max": 1.0,
            "mean": 0.0,
            "std": 1.0,
            "interquartile range": 1.0,
            "10-90 range": 1.6,
            "1-99 range": 1.96,
            "range": 2.0,
        }

    def _fake_density(values, *, point_count, circular=False):
        captured["density_circular"] = circular
        return {"points": [0.0], "density": [1.0], "min": 0.0, "max": 360.0, "bandwidth": 1.0}

    monkeypatch.setattr(main.summary_stats, "summarize_values", _fake_summarize)
    monkeypatch.setattr(main.indexing, "build_density_curve", _fake_density)
    monkeypatch.setattr(main.units, "apply_unit_system_to_env_response", lambda response, *_args: response)

    response = client.get(_env_url(123, "aspect_deg", location="gadm.1"))

    assert response.status_code == 200
    body = response.json()
    assert body["variableType"] == "circular"
    assert body["summary"]["mean"] == pytest.approx(0.0)
    assert captured["summary_circular"] is True
    assert captured["density_circular"] is True


def test_env_slice_with_location(client, known_taxon_id, known_numeric_var, known_species_location_gid):
    """?location filter uses numeric_range_samples_for_location (line 875)."""
    r = client.get(
        f"/species/{known_taxon_id}/environment/{known_numeric_var}/slice"
        f"?min=-50&max=50&location={known_species_location_gid}"
    )
    assert r.status_code == 200
    body = r.json()
    assert "observations" in body
    assert "count" in body


# ---------------------------------------------------------------------------
# /species/{taxon_id}/environment/{variable_id} — with location filter
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_env_categorical_with_location(client, known_taxon_id, known_categorical_var, known_species_location_gid):
    """Categorical env with location calls build_categorical_stats_for_location (lines 544-558)."""
    r = client.get(_env_url(known_taxon_id, known_categorical_var, location=known_species_location_gid))
    assert r.status_code == 200
    body = r.json()
    assert body["variableType"] == "categorical"
    assert "categoricalDistribution" in body


@pytest.mark.slow
def test_env_categorical_with_location_has_baseline(
    client, known_taxon_id, known_categorical_var, known_species_location_gid
):
    """When location filter is used, baseline distribution is loaded (lines 573-576)."""
    body = client.get(_env_url(known_taxon_id, known_categorical_var, location=known_species_location_gid)).json()
    # baseline fields are always present in response (may be empty if no baseline data)
    assert "baselineCategoricalDistribution" in body
    assert "baselineCategoricalTotals" in body


@pytest.mark.slow
def test_env_numeric_with_location(client, known_taxon_id, known_numeric_var, known_species_location_gid):
    """Numeric env with location calls gather_numeric_records (lines 675-733)."""
    r = client.get(_env_url(known_taxon_id, known_numeric_var, location=known_species_location_gid))
    assert r.status_code == 200
    body = r.json()
    assert body["variableType"] == "numeric"
    summary = body["summary"]
    assert summary["count"] > 0
    assert "densityCurve" in body


def test_env_numeric_with_location_has_baseline(client, known_taxon_id, known_numeric_var, known_species_location_gid):
    """When location filter is used, baseline summary is computed (line 696)."""
    body = client.get(_env_url(known_taxon_id, known_numeric_var, location=known_species_location_gid)).json()
    # baselineSummary may be populated if baseline data exists
    assert "baselineSummary" in body


def test_env_categorical_with_empty_location_returns_404(client, known_taxon_id, known_categorical_var):
    """location GID with zero matching samples raises 404 (line 551)."""
    # XYZ has no occurrence data — categorical build returns None → 404
    r = client.get(_env_url(known_taxon_id, known_categorical_var, location="XYZ"))
    assert r.status_code == 404


def test_env_numeric_with_empty_location_returns_404(client, known_taxon_id, known_numeric_var):
    """location GID with zero matching samples raises 404 (line 683)."""
    r = client.get(_env_url(known_taxon_id, known_numeric_var, location="XYZ"))
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /species/{taxon_id}/environment/{variable_id}/class/{class_value}/samples
# ---------------------------------------------------------------------------


def _class_url(taxon_id, variable_id, class_value, **params):
    base = f"/species/{taxon_id}/environment/{variable_id}/class/{class_value}/samples"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{qs}"
    return base


def test_env_class_samples_returns_200(client, known_taxon_id, known_categorical_var, known_categorical_class_value):
    """class_samples endpoint returns 200 for valid taxon + categorical variable (lines 758-803)."""
    r = client.get(_class_url(known_taxon_id, known_categorical_var, known_categorical_class_value))
    assert r.status_code == 200


def test_env_class_samples_response_shape(client, known_taxon_id, known_categorical_var, known_categorical_class_value):
    """Response has required top-level fields."""
    body = client.get(_class_url(known_taxon_id, known_categorical_var, known_categorical_class_value)).json()
    required = {"speciesId", "variable", "classValue", "observations", "count"}
    missing = required - body.keys()
    assert not missing, f"class_samples response missing fields: {missing}"


def test_env_class_samples_count_matches_observations(
    client, known_taxon_id, known_categorical_var, known_categorical_class_value
):
    body = client.get(_class_url(known_taxon_id, known_categorical_var, known_categorical_class_value)).json()
    assert body["count"] == len(body["observations"])


def test_env_class_samples_string_class_value(client, known_taxon_id, known_categorical_var):
    """String class value that can't be float-parsed falls back to string (line 770)."""
    r = client.get(_class_url(known_taxon_id, known_categorical_var, "Cfb"))
    assert r.status_code == 200
    body = r.json()
    assert body["classValue"] == "Cfb"


def test_env_class_samples_numeric_class_value(client, known_taxon_id, known_categorical_var):
    """Integer-valued float class (e.g. '1') is parsed to int (lines 766-768)."""
    r = client.get(_class_url(known_taxon_id, known_categorical_var, "1"))
    assert r.status_code == 200
    body = r.json()
    assert body["classValue"] == 1


def test_env_class_samples_with_limit(client, known_taxon_id, known_categorical_var, known_categorical_class_value):
    """limit parameter is respected (line 792-793)."""
    body = client.get(_class_url(known_taxon_id, known_categorical_var, known_categorical_class_value, limit=2)).json()
    assert body["count"] <= 2
    assert len(body["observations"]) <= 2


def test_env_class_samples_with_location(
    client, known_taxon_id, known_categorical_var, known_categorical_class_value, known_species_location_gid
):
    """location filter calls categorical_class_samples_for_location (lines 773-780)."""
    r = client.get(
        _class_url(
            known_taxon_id,
            known_categorical_var,
            known_categorical_class_value,
            location=known_species_location_gid,
        )
    )
    assert r.status_code == 200
    body = r.json()
    assert "observations" in body
    assert "count" in body


def test_env_class_samples_invalid_taxon_returns_404(client, known_categorical_var, known_categorical_class_value):
    r = client.get(_class_url(999999999, known_categorical_var, known_categorical_class_value))
    assert r.status_code == 404


def test_environment_stats_and_class_samples_missing_paths(monkeypatch):
    monkeypatch.setattr(main.gis_lookup, "load_variable_metadata", lambda: ([], {"bio_1": {"value_type": "numeric"}}))
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _tid: {"path": "/tmp/nope", "taxon_key": "1"})
    monkeypatch.setattr(main, "_path_exists", lambda _p: False)

    with pytest.raises(HTTPException) as exc1:
        main.species_environment_stats(1, "bio_1", location=None, unit_system=None)
    assert exc1.value.status_code == 404

    with pytest.raises(HTTPException) as exc2:
        main.species_environment_class_samples(1, "bio_1", "1", limit=None, location=None)
    assert exc2.value.status_code == 404


def test_environment_forced_categorical_and_missing_numeric_precompute(monkeypatch):
    variable = {
        "landcover": {"value_type": "numeric", "name": "Landcover", "units": None},
        "bio_1": {"value_type": "numeric", "name": "Bio 1", "units": "c"},
    }
    monkeypatch.setattr(main.gis_lookup, "load_variable_metadata", lambda: ([], variable))
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _tid: {"path": "/tmp/ok", "taxon_key": "1"})
    monkeypatch.setattr(main, "_path_exists", lambda _p: True)
    monkeypatch.setattr(main.summary_stats, "load_categorical_distribution", lambda *_a, **_k: None)
    monkeypatch.setattr(main.summary_stats, "load_numeric_summary", lambda *_a, **_k: {"count": 1})
    monkeypatch.setattr(main.summary_stats, "load_density_graph", lambda *_a, **_k: {"points": [0], "density": [1]})
    monkeypatch.setattr(main.indexing, "load_relative_ranks", lambda *_a, **_k: [])
    out = main.species_environment_stats(1, "landcover", location=None, unit_system=None)
    assert out["variableType"] == "categorical"

    monkeypatch.setattr(main.summary_stats, "load_numeric_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(main.summary_stats, "load_density_graph", lambda *_a, **_k: None)
    with pytest.raises(HTTPException) as exc:
        main.species_environment_stats(1, "bio_1", location=None, unit_system=None)
    assert exc.value.status_code == 503


def test_class_samples_missing_index_raises_503(monkeypatch):
    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _tid: {"path": "/tmp/ok", "taxon_key": "1"})
    monkeypatch.setattr(main, "_path_exists", lambda p: str(p).endswith("/tmp/ok"))
    with pytest.raises(HTTPException) as exc:
        main.species_environment_class_samples(1, "koppen_geiger", "1", limit=None, location=None)
    assert exc.value.status_code == 503


def test_slice_not_found_and_passthrough_errors(monkeypatch):
    monkeypatch.setattr(
        main.gis_lookup,
        "load_variable_metadata",
        lambda: ([], {"bio_1": {"value_type": "numeric", "units": None}}),
    )

    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _tid: None)
    with pytest.raises(HTTPException) as exc1:
        main.species_environment_slice(
            1, "bio_1", min_value=0, max_value=1, limit=None, location=None, unit_system=None
        )
    assert exc1.value.status_code == 404

    monkeypatch.setattr(main.taxa_navigation, "get_taxon_by_id", lambda _tid: {"path": "/tmp/ok", "taxon_key": "1"})
    monkeypatch.setattr(main, "_path_exists", lambda p: not str(p).endswith("occurrence_index.parquet"))
    with pytest.raises(HTTPException) as exc2:
        main.species_environment_slice(
            1, "bio_1", min_value=0, max_value=1, limit=None, location=None, unit_system=None
        )
    assert exc2.value.status_code == 404

    monkeypatch.setattr(main, "_path_exists", lambda p: str(p) != "/tmp/ok")
    with pytest.raises(HTTPException) as exc_taxon_path:
        main.species_environment_slice(
            1, "bio_1", min_value=0, max_value=1, limit=None, location=None, unit_system=None
        )
    assert exc_taxon_path.value.status_code == 404

    monkeypatch.setattr(main, "_path_exists", lambda _p: True)
    monkeypatch.setattr(
        main.summary_stats,
        "get_sorted_layer_records_in_value_range",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("missing file")),
    )
    with pytest.raises(HTTPException) as exc3:
        main.species_environment_slice(
            1, "bio_1", min_value=0, max_value=1, limit=None, location=None, unit_system=None
        )
    assert exc3.value.status_code == 404

    monkeypatch.setattr(
        main.summary_stats,
        "get_sorted_layer_records_in_value_range",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad value")),
    )
    with pytest.raises(HTTPException) as exc4:
        main.species_environment_slice(
            1, "bio_1", min_value=0, max_value=1, limit=None, location=None, unit_system=None
        )
    assert exc4.value.status_code == 400

from __future__ import annotations

from types import SimpleNamespace

import pytest

from util import descriptions as desc
from util import gis_lookup


@pytest.mark.parametrize(
    ("func", "value", "expected"),
    [
        (desc._winter_coldness_label, -45, "extremely cold"),
        (desc._winter_coldness_label, 5, "cool"),
        (desc._summer_heat_label, 41, "scorching"),
        (desc._summer_heat_label, 15, "temperate"),
        (desc._annual_precip_label, 40, "extremely xeric"),
        (desc._annual_precip_label, 3200, "torrential"),
    ],
)
def test_threshold_label_helpers(func, value, expected):
    assert func(value) == expected


def test_text_and_parse_helpers():
    assert desc._to_int("12") == 12
    assert desc._to_int("bad") is None
    assert desc._sentence_case("hello world") == "Hello world"
    assert desc._capitalize_leading_the("the netherlands") == "The netherlands"
    assert desc._to_natural_habitat_name("Open Grassland and Forest") == "open grasslands and forests"
    assert desc._to_natural_climate_name("Arid (BWh)") == "arid"
    assert desc._strip_phrase("often in forests") == ("often in ", "forests")
    assert desc._strip_phrase("mixed habitats") == ("", "mixed habitats")
    assert desc._ensure_climate_suffix("arid and polar climate") == "arid and polar climates"
    assert desc._format_categorical_phrase("often in Open Grassland", label="habitat") == "often in open grasslands"
    assert desc._format_categorical_phrase("primarily in Desert (BWh)", label="climate") == "primarily in desert climates"
    assert desc._parse_class_id("class_9") == 9
    assert desc._parse_class_id("nope") is None
    assert desc._extract_koppen_code("Desert climate (BWh)") == "BWH"
    assert desc._extract_koppen_code("Dfb") == "DFB"


def test_group_semantic_and_parallel_helpers():
    assert desc._landcover_forest_openness("Open forest") == "sparse"
    assert desc._landcover_forest_phenology("Deciduous forest") == "deciduous"
    assert desc._normalized_group_token("NaN") == ""
    assert desc._infer_landcover_group("Urban area", None) == ("urban", "Urban")
    assert desc._landcover_group_label("forest", "") == "forests"
    assert desc._semantic_default_label("koppen_geiger", "desert", "Desert") == "desert climates"
    assert desc._trait_string("None") == ""
    assert desc._derive_koppen_thermal("Cfa") == "hot"
    assert desc._extract_legend_traits({"traits": {"thermal": "warm", "none": "null"}}) == {"thermal": "warm"}

    label, semantic = desc._semantic_label_from_group(
        variable_id="landcover",
        group="forest",
        group_label="Forest",
        traits={"openness": "dense", "phenology": "evergreen"},
    )
    assert label == "dense evergreen forests"
    assert isinstance(semantic, dict)

    first = {"_semantic": semantic}
    second = {
        "_semantic": {
            "group": "forest",
            "base_label": "forests",
            "dimensions": semantic["dimensions"],
            "values": {"openness": "sparse", "phenology": "evergreen"},
        }
    }
    assert desc._combine_semantic_entries(first, second) == "sparse and dense evergreen forests"
    assert desc._sanitize_label("none grasslands") == "grasslands"
    assert desc._combine_parallel_labels("cold desert climates", "hot desert climates") == "cold desert and hot desert climates"
    assert desc._frequency_verb(0.55) == "primarily"
    assert desc._secondary_frequency_verb(0.2) == "sometimes"
    assert desc._combine_entry_pair_or_single([{"name": "A"}, {"name": "B"}]) == "A and B"


def test_legend_and_entry_name_helpers():
    legend = {"1": {"name": "Evergreen Needleleaf Forest", "short_name": "Evergreen forest"}}
    entry = {"value": "class_1", "class_name": "class_1", "short_name": "class_1", "slug": "x"}
    legend_entry = desc._legend_for_entry(entry, legend)
    assert legend_entry == legend["1"]
    assert desc._resolve_entry_name(entry, legend) == "Evergreen forest"


def test_top_categorical_phrase_from_payload(monkeypatch):
    monkeypatch.setattr(
        gis_lookup,
        "load_layer_legend",
        lambda _v: {
            "1": {"name": "Open evergreen forest", "group": "forest", "group_label": "Forest"},
            "2": {"name": "Closed deciduous forest", "group": "forest", "group_label": "Forest"},
            "3": {"name": "Urban", "group": "urban", "group_label": "Urban"},
        },
    )
    payload = {
        "distribution": [
            {"value": "class_1", "fraction": 0.42},
            {"value": "class_2", "fraction": 0.36},
            {"value": "class_3", "fraction": 0.2},
            {"value": "class_4", "fraction": 0.0},
        ]
    }
    text = desc._top_categorical_phrase_from_payload(variable_id="landcover", label="habitat", payload=payload)
    assert text is not None
    assert "forests" in text
    assert "in " in text


def test_top_categorical_phrase_falls_back_when_location_stats_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(
        desc.summary_stats,
        "build_categorical_stats_for_location",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        desc.summary_stats,
        "load_categorical_distribution",
        lambda *_a, **_k: {"distribution": [{"value": "class_1", "class_name": "grassland", "fraction": 1.0}]},
    )
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    phrase = desc._top_categorical_phrase(
        tmp_path,
        variable_id="landcover",
        label="habitat",
        taxon_id=1,
        location_gid="USA",
    )
    assert phrase == "always in grasslands"


def test_location_text_helpers(monkeypatch):
    monkeypatch.setattr(
        desc,
        "load_config",
        lambda _n: SimpleNamespace(
            location_scope_by_level={0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}
        ),
    )
    monkeypatch.setattr(
        gis_lookup,
        "location_counts_for_taxon",
        lambda _taxon_id: {
            ("gadm_level0", "USA"): 100,
            ("gadm_level1", "USA.45_1"): 55,
            ("gadm_level1", "USA.5_1"): 45,
            ("gadm_level2", "USA.45.1_1"): 35,
            ("gadm_level2", "USA.45.2_1"): 20,
        },
    )
    mapping = {
        "USA": SimpleNamespace(name="United States", parent_gid=None),
        "USA.45_1": SimpleNamespace(name="Utah", parent_gid="USA"),
        "USA.5_1": SimpleNamespace(name="California", parent_gid="USA"),
        "USA.45.1_1": SimpleNamespace(name="Salt Lake", parent_gid="USA.45_1"),
        "USA.45.2_1": SimpleNamespace(name="Summit", parent_gid="USA.45_1"),
    }
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], mapping))
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda gid: ("gid", "gadm_level1", gid))

    assert desc._with_definite_article("United States") == "the United States"
    assert desc._join_names(["a", "b", "c"]) == "a, b, and c"
    assert desc._combine_label_pair("cold desert climates", "hot desert climates") == "cold and hot desert climates"

    text = desc._build_location_text(1, location_gid="USA.45_1", limit=2)
    assert text == "Salt Lake, and Summit in Utah"


def test_terrain_slope_aspect_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda value, _unit, _system: (value, ""))
    assert desc._format_terrain_value(1234) == "1200"
    assert desc._format_terrain_value("bad") is None
    assert desc._extract_elevation_range_values({"range": {"min": 101, "max": 499}}) == ("100", "500")
    assert desc._slope_grade_percent(45) == pytest.approx(100.0, rel=1e-2)
    assert desc._slope_band_from_grade(4.9) == "flat"
    assert desc._slope_phrase_for_band("flat") == "flat areas"
    assert desc._slope_range_phrase("flat", "steep") == "flat areas to steep slopes"

    monkeypatch.setattr(
        desc.summary_stats,
        "load_categorical_distribution",
        lambda *_a, **_k: {
            "distribution": [{"value": "1", "fraction": 0.6}, {"value": "7", "fraction": 0.2}],
            "totals": {"total_samples": 200},
        },
    )
    masses, total = desc._aspect_cardinal_masses(tmp_path)
    assert total == 200.0
    assert masses["north"] == pytest.approx(0.6)
    assert desc._aspect_preference_text(tmp_path) == "Prefers north-facing slopes"


def test_outlier_metric_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(desc.summary_stats, "_slugify_metric", lambda name, _fallback: name.lower().replace(" ", "_"))
    monkeypatch.setattr(
        desc.summary_stats,
        "_load_categorical_stats",
        lambda *_a, **_k: {"landcover": {"class_1": 0.6, "class_2": 0.4}},
    )
    assert desc._build_metric_candidates("Class 1", 1, {"value": "1"})[0] == "class_1"
    assert desc._categorical_metric_fraction_for_aliases(tmp_path, variable_id="landcover", aliases=("CLASS_1",)) == 0.6
    assert desc._resolve_metric_name_for_variable(tmp_path, variable_id="landcover", candidates=("CLASS_2",)) == "class_2"
    assert desc._delta_adjusted_qualifier("extremely", abs_delta=0.15) == "very"
    assert desc._pick_best_outlier_candidate([{"level": 1, "depth": 1, "strength": 0.2}, {"level": 2, "depth": 0, "strength": 0.1}])["level"] == 2
    assert desc._join_outlier_labels(["", "wet forests"], "fallback") == "wet forests"
    assert desc._outlier_qualifier_level("very") == 2

    legend = {"1": {"name": "Class 1"}}
    aliases = desc._entry_metric_aliases(
        tmp_path,
        variable_id="landcover",
        entry={"value": "class_1", "class_name": "Class 1"},
        legend=legend,
    )
    assert "class_1" in aliases


def test_rendering_and_build_taxon_description(monkeypatch, tmp_path):
    profile = {
        "summary": "x",
        "habitat": "always in forests",
        "climate": "often in arid climates, and sometimes in wet climates",
        "locations": "the United States",
        "categories": [{"category": "terrain", "detail": "Steep slopes"}],
    }
    sections = desc._build_profile_sections(profile)
    assert any(section["id"] == "habitat" for section in sections)
    text = desc._render_profile_text(profile)
    assert "Summary: x" in text
    assert "Terrain: Steep slopes." in text
    assert desc._title_case_words("aLmoSt always") == "Almost Always"
    assert desc._lines_from_categorical_phrase("often in forests, and rarely in deserts")[0]["prefix"] == "Often in"

    monkeypatch.setattr(desc.units, "normalize_unit_system", lambda _u: None)
    monkeypatch.setattr(desc, "_find_ancestor_by_rank", lambda *_a, **_k: {"scientific_name": "Felidae", "rank": "FAMILY"})
    monkeypatch.setattr(desc, "_top_categorical_phrase", lambda *_a, **_k: "often in forests")
    monkeypatch.setattr(desc, "_categorical_outlier_text", lambda *_a, **_k: None)
    monkeypatch.setattr(desc, "_build_location_text", lambda *_a, **_k: "the United States")
    monkeypatch.setattr(desc, "_terrain_status_rows", lambda *_a, **_k: [{"category": "terrain", "detail": "Steep"}])
    monkeypatch.setattr(desc, "_weather_status_rows", lambda *_a, **_k: [])

    from util import taxa_navigation

    monkeypatch.setattr(
        taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, language=None: ["cat"] if taxon.get("scientific_name") == "Felis_catus" else ["cats"],
    )
    monkeypatch.setattr(taxa_navigation.CONFIG, "common_name_language", "en")

    taxon = {"scientific_name": "Felis_catus", "path": str(tmp_path), "taxon_key": "1", "rank": "SPECIES"}
    built = desc.build_taxon_description(taxon)
    assert built["summary"].startswith("The cat (Felis catus)")
    assert built["locations"] == "The United States"
    assert built["sections"]


def test_outlier_candidate_and_text(monkeypatch, tmp_path):
    from util import indexing, taxa_navigation

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "percentile": 0.98, "count": 20, "ancestorTaxonId": "10", "label": "Felidae"},
            {"metric": "mean", "percentile": 0.01, "count": 20, "ancestorTaxonId": "20", "label": "Carnivora"},
            {"metric": "max", "percentile": "bad", "count": 20, "ancestorTaxonId": "10", "label": "Felidae"},
        ],
    )
    parent2 = {"taxon_key": "20", "rank": "ORDER", "scientific_name": "Carnivora", "path": str(tmp_path)}
    parent1 = {"taxon_key": "10", "rank": "FAMILY", "scientific_name": "Felidae", "path": str(tmp_path)}
    root = {"taxon_key": "1", "rank": "SPECIES", "scientific_name": "Felis catus", "path": str(tmp_path)}
    parent_map = {"1": parent1, "10": parent2, "20": None}
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda t: parent_map.get(str(t.get("taxon_key"))))
    monkeypatch.setattr(
        taxa_navigation,
        "get_taxon_by_id",
        lambda taxon_id: {"10": parent1, "20": parent2}.get(str(taxon_id)),
    )
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())

    candidate = desc._select_variable_outlier_candidate(
        variable_id="bio_1",
        taxon=root,
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
        max_ancestor_rank="FAMILY",
    )
    assert candidate is not None
    assert candidate["polarity"] in {"high", "low"}
    assert "Felidae" in candidate["context"]

    text = desc._select_variable_outlier_text(
        variable_id="bio_1",
        taxon=root,
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
    )
    assert text is not None
    assert text.startswith("avg ")


def test_categorical_outlier_location_and_context_paths(monkeypatch, tmp_path):
    from util import taxa_navigation

    taxon = {"taxon_key": "1", "rank": "SPECIES", "path": str(tmp_path)}
    ancestor = {"taxon_key": "10", "rank": "FAMILY", "path": str(tmp_path / "anc")}
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: ancestor)

    top_metrics = [
        {"aliases": ("class_1",), "label": "forests", "fraction": 0.3, "metric": "class_1"},
        {"aliases": ("class_2",), "label": "grasslands", "fraction": 0.1, "metric": "class_2"},
    ]
    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: top_metrics)
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [{"value": "class_1", "fraction": 0.8}]})
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.8)
    monkeypatch.setattr(desc, "_location_label", lambda _g: "United States")
    text = desc._location_delta_outlier_text(
        tmp_path,
        variable_id="landcover",
        top_metrics=top_metrics,
        taxon_id=1,
        location_gid="USA",
    )
    assert text is not None
    assert "common" in text.lower()

    monkeypatch.setattr(
        desc,
        "_select_variable_outlier_candidate",
        lambda *_a, **_k: {
            "qualifier": "very",
            "polarity": "high",
            "context": "family Felidae",
            "ancestor_taxon_id": "10",
            "depth": 3,
            "strength": 0.9,
        },
    )
    monkeypatch.setattr(
        desc,
        "_categorical_metric_fraction_for_aliases",
        lambda _p, **_k: 0.7 if str(_p) == str(tmp_path) else 0.4,
    )
    monkeypatch.setattr(desc, "_delta_adjusted_qualifier", lambda q, abs_delta: q)
    context_text = desc._categorical_outlier_text(
        taxon,
        tmp_path,
        variable_id="landcover",
        taxon_id=1,
        location_gid=None,
    )
    assert context_text is not None
    assert "compared to others" in context_text


def test_numeric_and_compare_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        desc.summary_stats,
        "gather_numeric_records",
        lambda *_a, **_k: [{"value": 1.0}, {"value": 3.0}, {"value": "bad"}],
    )
    monkeypatch.setattr(desc.summary_stats, "summarize_values", lambda values: {"mean": sum(values) / len(values)})
    summary = desc._numeric_summary_for_context(
        taxon_id=1,
        taxon_dir=tmp_path,
        variable_id="bio_1",
        location_gid="USA",
    )
    assert summary["mean"] == 2.0
    monkeypatch.setattr(desc.summary_stats, "load_numeric_summary", lambda *_a, **_k: {"mean": 4.0})
    assert desc._numeric_summary_for_context(taxon_id=1, taxon_dir=tmp_path, variable_id="bio_1")["mean"] == 4.0

    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda v, _u, _s: (v, ""))
    assert desc._format_scalar_value(3.0) == "3"
    assert desc._format_scalar_value(3.14) == "3"
    assert desc._format_scalar_value_for_system(12.2, unit="celsius") == "12"
    assert desc._temperature_location_compare_text(local_mean=10, global_mean=10.2, location_name="X") == "about the same in X"
    assert "warmer" in (desc._temperature_location_compare_text(local_mean=14, global_mean=10, location_name="X") or "")
    assert desc._precip_location_compare_text(local_mean=100, global_mean=100, location_name="X") == "about the same in X"
    assert "wetter" in (desc._precip_location_compare_text(local_mean=180, global_mean=100, location_name="X") or "")


def test_location_label_scope_paths(monkeypatch):
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gbif_region", "north_america"))
    assert desc._location_label("north_america") == "North America"
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level0", "USA"))
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], {"USA": SimpleNamespace(name="United States")}))
    assert desc._location_label("USA") == "United States"


def test_status_rows_temperature_precip_terrain(monkeypatch, tmp_path):
    taxon = {"taxon_key": "1", "rank": "SPECIES", "path": str(tmp_path)}
    monkeypatch.setattr(
        desc,
        "_numeric_summary_for_context",
        lambda *, variable_id, location_gid=None, **_k: {
            "elevation": {"range": {"min": 100, "max": 900}},
            "slope": {"mean": 18, "10th percentile": 3},
            "bio_6": {"mean": -8},
            "bio_5": {"mean": 33},
            "bio_18": {"mean": 30},
            "bio_19": {"median": 80},
            "swe": {"median": 0},
            "bio_1": {"mean": 12 if location_gid else 10},
            "bio_12": {"min": 200, "max": 1800, "median": 900 if location_gid else 700},
        }.get(variable_id, {}),
    )
    monkeypatch.setattr(
        gis_lookup,
        "load_variable_metadata",
        lambda: (
            [],
            {
                "elevation": {"units": "m"},
                "bio_6": {"units": "celsius"},
                "bio_12": {"units": "mm"},
            },
        ),
    )
    monkeypatch.setattr(desc.units, "equivalent_unit", lambda unit, _sys: unit)
    monkeypatch.setattr(desc.units, "display_unit", lambda unit: unit or "")
    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda value, _unit, _sys: (value, ""))
    monkeypatch.setattr(desc, "_elevation_outlier_text", lambda *_a, **_k: "avg high for family Felidae")
    monkeypatch.setattr(desc, "_aspect_preference_text", lambda _p: "Aspect: prefers north-facing slopes")
    monkeypatch.setattr(desc, "_select_variable_outlier_text", lambda **_k: "avg very low for family Felidae")
    monkeypatch.setattr(desc, "_location_label", lambda _g: "United States")
    monkeypatch.setattr(desc, "_select_variable_outlier_candidate", lambda **_k: {"qualifier": "very", "polarity": "low", "context": "family Felidae"})

    terrain = desc._terrain_status_rows(taxon, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert terrain is not None and "Found from" in terrain and "slopes" in terrain
    weather = desc._weather_status_rows(taxon, tmp_path, taxon_id=1, location_gid="USA")[0]["detail"]
    assert weather == (
        "Typically hot, xeric summers\n"
        "Typically cold, semi-arid winters\n"
        "Prefers moderately wet areas, but can tolerate arid to incredibly wet"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-35, "incredibly cold"),
        (-25, "very cold"),
        (-15, "quite cold"),
        (-1, "cold"),
        (15, "temperate"),
        (25, "warm"),
        (35, "hot"),
    ],
)
def test_winter_all_branches(value, expected):
    assert desc._winter_coldness_label(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (38, "very hot"),
        (32, "hot"),
        (25, "warm"),
        (11, "temperate"),
        (5, "cool"),
    ],
)
def test_summer_all_branches(value, expected):
    assert desc._summer_heat_label(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (120, "xeric"),
        (220, "arid"),
        (320, "semi-arid"),
        (420, "dry"),
        (700, "subhumid"),
        (900, "moderately wet"),
        (1100, "wet"),
        (1400, "very wet"),
        (1900, "incredibly wet"),
        (2900, "extremely wet"),
    ],
)
def test_precip_all_branches(value, expected):
    assert desc._annual_precip_label(value) == expected


def test_location_build_scope_variants(monkeypatch):
    monkeypatch.setattr(
        desc,
        "load_config",
        lambda _n: SimpleNamespace(
            location_scope_by_level={0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}
        ),
    )
    mapping = {
        "USA": SimpleNamespace(name="United States", parent_gid=None),
        "USA.45_1": SimpleNamespace(name="Utah", parent_gid="USA"),
        "USA.45.1_1": SimpleNamespace(name="Salt Lake", parent_gid="USA.45_1"),
    }
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], mapping))
    monkeypatch.setattr(
        gis_lookup,
        "location_counts_for_taxon",
        lambda _taxon_id: {
            ("gadm_level0", "USA"): 100,
            ("gadm_level1", "USA.45_1"): 100,
            ("gadm_level2", "USA.45.1_1"): 100,
        },
    )

    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level0", "USA"))
    assert "in the United States" in desc._build_location_text(1, location_gid="USA")

    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level2", "USA.45.1_1"))
    assert desc._build_location_text(1, location_gid="USA.45.1_1") == "Salt Lake in Utah"

    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gbif_region", "north_america"))
    assert desc._build_location_text(1, location_gid="north_america") == "North America"

    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: (_ for _ in ()).throw(ValueError("bad")))
    assert desc._build_location_text(1, location_gid="bad") == ""


def test_fraction_aliases_and_display_label_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {"1": {"name": "Open evergreen forest"}})
    monkeypatch.setattr(
        desc,
        "_entry_metric_aliases",
        lambda *_a, **_k: ("class_1",),
    )
    payload = {"distribution": [{"fraction": "bad"}, {"fraction": -1}, {"fraction": 0.6, "value": "class_1"}]}
    assert desc._fraction_for_aliases_from_payload(tmp_path, variable_id="landcover", payload=payload, aliases=("class_1",)) == 0.6
    assert desc._fraction_for_aliases_from_payload(tmp_path, variable_id="landcover", payload=None, aliases=("class_1",)) == 0.0
    assert desc._fraction_for_aliases_from_payload(tmp_path, variable_id="landcover", payload=payload, aliases=("",)) == 0.0

    assert desc._categorical_display_label(
        variable_id="landcover",
        style="group_map",
        class_name="Open evergreen forest",
        class_id=1,
        group_value="forest",
        group_label="Forest",
        legend_entry={"traits": {"phenology": "evergreen"}},
    ).endswith("forests")
    assert desc._categorical_display_label(
        variable_id="landcover",
        style="group_map",
        class_name="Urban",
        class_id=190,
        group_value="urban",
        group_label="Urban",
        legend_entry=None,
    ) == "urban areas"
    assert desc._categorical_display_label(
        variable_id="koppen_geiger",
        style="climate_suffix",
        class_name="Cfb",
        class_id=None,
        group_value="",
        group_label="",
        legend_entry=None,
    ) == "cfb climates"


def test_edge_helpers_invalid_inputs(monkeypatch):
    assert desc._to_int(None) is None
    assert desc._sentence_case("") == ""
    assert desc._capitalize_leading_the("") == ""
    assert desc._join_names([]) == ""
    assert desc._join_names(["one"], use_and=False) == "one"
    assert desc._combine_label_pair("", "x") == "x"
    assert desc._combine_label_pair("x", "") == "x"
    assert desc._combine_label_pair("x", "x") == "x"

    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda _v, _u, _s: (None, ""))
    assert desc._format_terrain_value(100) is None

    assert desc._temperature_location_compare_text(local_mean="bad", global_mean=1, location_name="x") is None
    assert desc._temperature_location_compare_text(local_mean=float("nan"), global_mean=1, location_name="x") is None
    assert desc._precip_location_compare_text(local_mean="bad", global_mean=1, location_name="x") is None
    assert desc._precip_location_compare_text(local_mean=float("nan"), global_mean=1, location_name="x") is None
    assert desc._precip_location_compare_text(local_mean=10, global_mean=0, location_name="x") == "about the same in x"

    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: (_ for _ in ()).throw(RuntimeError("x")))
    assert desc._location_label("USA") == "USA"
    assert desc._location_label("") == "this location"


@pytest.mark.parametrize(
    ("class_id", "name", "expected"),
    [
        (10, "x", ("cropland", "Cropland")),
        (51, "x", ("forest", "Forest")),
        (120, "x", ("shrubland", "Shrubland")),
        (130, "x", ("grassland", "Grassland")),
        (140, "x", ("lichens_mosses", "Lichens and Mosses")),
        (150, "x", ("sparse_vegetation", "Sparse Vegetation")),
        (180, "x", ("wetlands", "Wetlands")),
        (190, "x", ("urban", "Urban")),
        (200, "x", ("bare_areas", "Bare Areas")),
        (210, "x", ("water", "Water")),
        (220, "x", ("ice_snow", "Ice and Snow")),
        (250, "x", ("filled", "Filled")),
        (None, "forest type", ("forest", "Forest")),
        (None, "orchard", ("cropland", "Cropland")),
        (None, "sparse area", ("sparse_vegetation", "Sparse Vegetation")),
        (None, "unknown", ("", "")),
    ],
)
def test_infer_landcover_group_branches(class_id, name, expected):
    assert desc._infer_landcover_group(name, class_id) == expected


def test_semantic_and_combine_negative_paths():
    assert desc._semantic_default_label("unknown", "foo_bar", "") == "foo bar"
    assert desc._derive_koppen_thermal("") == ""
    assert desc._derive_koppen_thermal("BWW") == ""
    assert desc._derive_koppen_thermal("ET") == "cold"
    assert desc._derive_koppen_thermal("EF") == "severe-winter"
    assert desc._extract_legend_traits(None) == {}
    assert desc._extract_legend_traits({"traits": []}) == {}
    assert desc._extract_legend_traits({"traits": {"": "warm"}}) == {}

    a = {"_semantic": {"group": "forest", "base_label": "forests", "dimensions": [], "values": {}}}
    b = {"_semantic": {"group": "urban", "base_label": "forests", "dimensions": [], "values": {}}}
    assert desc._combine_semantic_entries(a, b) is None
    assert desc._combine_semantic_entries({"_semantic": {}}, {"_semantic": []}) is None
    assert desc._is_semantically_subsumed_by_primary({"_semantic": {}}, []) is False


def test_top_categorical_phrase_secondary_band_branches(monkeypatch):
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})

    payload_same_band = {
        "distribution": [
            {"class_name": "alpha", "fraction": 0.24},
            {"class_name": "beta", "fraction": 0.18},
            {"class_name": "gamma", "fraction": 0.17},
        ]
    }
    text1 = desc._top_categorical_phrase_from_payload(
        variable_id="misc",
        label="habitat",
        payload=payload_same_band,
    )
    assert text1 is not None and "as well as" in text1

    payload_else_same_verb = {
        "distribution": [
            {"class_name": "alpha", "fraction": 0.40},
            {"class_name": "beta", "fraction": 0.26},
            {"class_name": "gamma", "fraction": 0.25},
        ]
    }
    text2 = desc._top_categorical_phrase_from_payload(
        variable_id="misc",
        label="habitat",
        payload=payload_else_same_verb,
    )
    assert text2 is not None and "as well as" in text2


def test_find_ancestor_and_outlier_config_guard(monkeypatch, tmp_path):
    from util import taxa_navigation

    chain = {
        "1": {"taxon_key": "1", "rank": "SPECIES"},
        "2": {"taxon_key": "2", "rank": "GENUS"},
        "3": {"taxon_key": "3", "rank": "FAMILY"},
    }
    parents = {"1": chain["2"], "2": chain["3"], "3": None}
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda t: parents.get(str(t.get("taxon_key"))))
    assert desc._find_ancestor_by_rank(chain["1"], "family") == chain["3"]
    assert desc._find_ancestor_by_rank(chain["1"], "order") is None

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=True))
    assert (
        desc._select_variable_outlier_candidate(
            variable_id="bio_1",
            taxon={"taxon_key": "1", "path": str(tmp_path)},
            taxon_dir=tmp_path,
            preferred_metrics=("mean",),
        )
        is None
    )


def test_top_categorical_class_metrics_edge_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: None)
    assert desc._top_categorical_class_metrics(tmp_path, variable_id="landcover") == []

    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": []})
    assert desc._top_categorical_class_metrics(tmp_path, variable_id="landcover") == []

    monkeypatch.setattr(
        desc,
        "_load_categorical_payload_for_context",
        lambda *_a, **_k: {
            "distribution": [
                {"fraction": 0.0, "value": "class_1"},
                {"fraction": 0.5, "value": "class_x", "class_name": "", "short_name": ""},
                {"fraction": 0.5, "value": "class_1", "class_name": "Forest"},
            ]
        },
    )
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    monkeypatch.setattr(desc, "_resolve_metric_name_for_variable", lambda *_a, **_k: "class_1")
    rows = desc._top_categorical_class_metrics(tmp_path, variable_id="landcover", limit=1)
    assert len(rows) == 1
    assert rows[0]["metric"] == "class_1"


def test_location_delta_and_categorical_outlier_guard_paths(monkeypatch, tmp_path):
    assert desc._location_delta_outlier_text(
        tmp_path,
        variable_id="landcover",
        top_metrics=[],
        taxon_id=None,
        location_gid=None,
    ) is None

    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: None)
    assert desc._location_delta_outlier_text(
        tmp_path,
        variable_id="landcover",
        top_metrics=[{"aliases": ("x",), "label": "L", "fraction": 0.3}],
        taxon_id=1,
        location_gid="USA",
    ) is None

    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: [])
    assert desc._categorical_outlier_text({"taxon_key": "1"}, tmp_path, variable_id="landcover") is None

    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: [{"label": "A", "metric": "", "aliases": ()}])
    assert desc._categorical_outlier_text({"taxon_key": "1"}, tmp_path, variable_id="landcover") is None


def test_categorical_outlier_additional_branch_matrix(monkeypatch, tmp_path):
    from util import taxa_navigation

    taxon = {"taxon_key": "1", "path": str(tmp_path)}
    top_metrics = [{"label": "A", "metric": "class_1", "aliases": ("class_1",), "fraction": 0.5}]
    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: top_metrics)
    monkeypatch.setattr(
        desc,
        "_select_variable_outlier_candidate",
        lambda *_a, **_k: {"qualifier": "", "context": "family F", "ancestor_taxon_id": "10"},
    )
    assert desc._categorical_outlier_text(taxon, tmp_path, variable_id="landcover") is None

    monkeypatch.setattr(
        desc,
        "_select_variable_outlier_candidate",
        lambda *_a, **_k: {"qualifier": "very", "context": "family F", "ancestor_taxon_id": "10", "polarity": "high"},
    )
    monkeypatch.setattr(desc, "_categorical_metric_fraction_for_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"path": str(tmp_path / "anc")})
    assert desc._categorical_outlier_text(taxon, tmp_path, variable_id="landcover") is None

    monkeypatch.setattr(desc, "_categorical_metric_fraction_for_aliases", lambda *_a, **_k: 0.5)
    monkeypatch.setattr(desc, "_delta_adjusted_qualifier", lambda *_a, **_k: None)
    assert desc._categorical_outlier_text(taxon, tmp_path, variable_id="landcover") is None

    monkeypatch.setattr(desc, "_delta_adjusted_qualifier", lambda *_a, **_k: "very")
    monkeypatch.setattr(desc, "_outlier_qualifier_level", lambda _q: 2)
    monkeypatch.setattr(desc, "_join_outlier_labels", lambda _l, f: f)
    monkeypatch.setattr(desc, "_sentence_case", lambda t: t)
    monkeypatch.setattr(desc, "_QUALIFIER_COMPARISON", {})
    assert desc._categorical_outlier_text(taxon, tmp_path, variable_id="landcover") is None


def test_status_rows_exception_and_fallback_branches(monkeypatch, tmp_path):
    taxon = {"taxon_key": "1", "rank": "SPECIES", "path": str(tmp_path)}
    monkeypatch.setattr(
        desc,
        "_numeric_summary_for_context",
        lambda *, variable_id, **_k: {
            "elevation": {"min": 100, "max": 120},
            "slope": {"mean": 10},
            "bio_6": {"min": -2},
            "bio_5": {"max": 4},
            "bio_12": {"min": 200, "max": 220, "mean": None},
        }.get(variable_id, {}),
    )
    monkeypatch.setattr(gis_lookup, "load_variable_metadata", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(desc.units, "equivalent_unit", lambda u, _s: u)
    monkeypatch.setattr(desc.units, "display_unit", lambda _u: "")
    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda v, _u, _s: (v, ""))
    monkeypatch.setattr(desc, "_aspect_preference_text", lambda _p: None)
    monkeypatch.setattr(desc, "_elevation_outlier_text", lambda *_a, **_k: "outlier text")
    monkeypatch.setattr(desc, "_select_variable_outlier_text", lambda **_k: None)
    monkeypatch.setattr(desc, "_select_variable_outlier_candidate", lambda **_k: {"qualifier": "very", "polarity": "low", "context": "family F"})
    terrain = desc._terrain_status_rows(taxon, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert terrain is not None and "Found from" in terrain and "slopes" in terrain

    weather = desc._weather_status_rows(taxon, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert weather is not None and "Can tolerate" in weather


def test_precip_compare_degree_branches():
    assert "a bit wetter" in (desc._precip_location_compare_text(local_mean=80, global_mean=0, location_name="x") or "")
    assert "noticeably wetter" in (desc._precip_location_compare_text(local_mean=120, global_mean=0, location_name="x") or "")
    assert "much wetter" in (desc._precip_location_compare_text(local_mean=220, global_mean=0, location_name="x") or "")
    assert "slightly drier" in (desc._precip_location_compare_text(local_mean=90, global_mean=100, location_name="x") or "")
    assert "noticeably wetter" in (desc._precip_location_compare_text(local_mean=170, global_mean=100, location_name="x") or "")
    assert "much wetter" in (desc._precip_location_compare_text(local_mean=190, global_mean=100, location_name="x") or "")


def test_payload_loader_and_parse_edge_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(
        desc.summary_stats,
        "build_categorical_stats_for_location",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(desc.summary_stats, "load_categorical_distribution", lambda *_a, **_k: {"distribution": [1]})
    payload = desc._load_categorical_payload_for_context(
        tmp_path,
        variable_id="landcover",
        taxon_id=1,
        location_gid="USA",
    )
    assert payload == {"distribution": [1]}
    assert desc._parse_class_id(float("nan")) is None
    assert desc._extract_koppen_code(None, "", " ") is None
    assert desc._landcover_forest_openness("woodland") == "other"
    assert desc._landcover_forest_phenology("mixed forest") == "generic"


def test_build_description_remaining_branches(monkeypatch, tmp_path):
    from util import taxa_navigation

    # Empty scientific name short-circuit.
    empty = desc.build_taxon_description({"scientific_name": "", "path": str(tmp_path)})
    assert empty["summary"] == ""
    assert empty["text"] == ""

    monkeypatch.setattr(desc.units, "normalize_unit_system", lambda _u: None)
    monkeypatch.setattr(desc, "_terrain_status_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(desc, "_weather_status_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(desc, "_top_categorical_phrase", lambda *_a, **_k: "often in wetlands")
    monkeypatch.setattr(desc, "_build_location_text", lambda *_a, **_k: "")
    monkeypatch.setattr(
        desc,
        "_categorical_outlier_text",
        lambda *args, **kwargs: "outlier text" if kwargs.get("variable_id") == "landcover" else "climate outlier",
    )
    monkeypatch.setattr(
        taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, language=None: [] if taxon.get("rank") == "FAMILY" else ["cat"],
    )
    monkeypatch.setattr(taxa_navigation.CONFIG, "common_name_language", "en")
    monkeypatch.setattr(
        desc,
        "_find_ancestor_by_rank",
        lambda *_a, **_k: {"rank": "FAMILY", "scientific_name": "Felidae"},
    )

    taxon = {"scientific_name": "Felis_catus", "path": str(tmp_path), "taxon_key": "1", "rank": "SPECIES"}
    profile = desc.build_taxon_description(taxon)
    assert "family Felidae" in profile["summary"]
    assert "wetlands" in (profile["habitat"] or "")
    assert profile["climate"] is not None
    assert profile["locations"] is None


def test_render_and_lines_additional_branches():
    assert desc._lines_from_categorical_phrase(None) == []
    assert desc._lines_from_categorical_phrase(" \n . ") == []
    profile = {
        "summary": "s",
        "categories": [{"category": "terrain", "detail": ""}],
    }
    sections = desc._build_profile_sections(profile)
    assert sections[0]["lines"][0]["body"] == "Not notable."
    text = desc._render_profile_text(profile)
    assert "Terrain: Not notable." in text


def test_outlier_and_compare_remaining_simple_branches(monkeypatch, tmp_path):
    # _elevation_outlier_text wrapper
    monkeypatch.setattr(desc, "_select_variable_outlier_text", lambda **_k: "x")
    assert desc._elevation_outlier_text({"taxon_key": "1"}, tmp_path) == "x"

    # Temperature compare degree labels.
    assert "slightly warmer" in (desc._temperature_location_compare_text(local_mean=11.0, global_mean=10.0, location_name="x") or "")
    assert "much warmer" in (desc._temperature_location_compare_text(local_mean=20.0, global_mean=10.0, location_name="x") or "")

    # Categorical outlier location short-circuit.
    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: [{"label": "A", "metric": "class_1", "aliases": ("class_1",), "fraction": 0.4}])
    monkeypatch.setattr(desc, "_location_delta_outlier_text", lambda *_a, **_k: "Location specific")
    out = desc._categorical_outlier_text({"taxon_key": "1"}, tmp_path, variable_id="landcover", taxon_id=1, location_gid="USA")
    assert out == "Location specific"

    # Build description fallback summary when family missing.
    from util import taxa_navigation

    monkeypatch.setattr(desc.units, "normalize_unit_system", lambda _u: None)
    monkeypatch.setattr(desc, "_find_ancestor_by_rank", lambda *_a, **_k: None)
    monkeypatch.setattr(desc, "_top_categorical_phrase", lambda *_a, **_k: None)
    monkeypatch.setattr(desc, "_categorical_outlier_text", lambda *_a, **_k: None)
    monkeypatch.setattr(desc, "_build_location_text", lambda *_a, **_k: None)
    monkeypatch.setattr(desc, "_terrain_status_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(desc, "_weather_status_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(taxa_navigation.CONFIG, "common_name_language", "en")
    profile = desc.build_taxon_description({"scientific_name": "A_b", "path": str(tmp_path), "taxon_key": "1", "rank": "SPECIES"})
    assert profile["summary"] == "A b is a species."


def test_outlier_selector_additional_guard_paths(monkeypatch, tmp_path):
    from util import indexing, taxa_navigation

    base_taxon = {"taxon_key": "1", "rank": "SPECIES", "path": str(tmp_path)}
    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda _t: None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: None)

    monkeypatch.setattr(indexing, "load_relative_ranks", lambda *_a, **_k: [])
    assert (
        desc._select_variable_outlier_candidate(
            variable_id="bio_1",
            taxon=base_taxon,
            taxon_dir=tmp_path,
            preferred_metrics=("mean",),
        )
        is None
    )

    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "count": "bad", "percentile": 0.9, "ancestorTaxonId": "10", "label": "X"},
            {"metric": "mean", "count": 5, "percentile": 0.9, "ancestorTaxonId": "10", "label": "X"},
            {"metric": "mean", "count": 20, "percentile": "bad", "ancestorTaxonId": "10", "label": "X"},
            {"metric": "mean", "count": 20, "percentile": float("nan"), "ancestorTaxonId": "10", "label": "X"},
            {"metric": "mean", "count": 20, "percentile": 0.51, "ancestorTaxonId": "10", "label": "X"},
        ],
    )
    assert (
        desc._select_variable_outlier_candidate(
            variable_id="bio_1",
            taxon=base_taxon,
            taxon_dir=tmp_path,
            preferred_metrics=("mean",),
        )
        is None
    )

    monkeypatch.setattr(
        desc,
        "_select_variable_outlier_candidate",
        lambda *_a, **_k: {"metric": "mean", "phrase": "", "context": "family X"},
    )
    assert (
        desc._select_variable_outlier_text(
            variable_id="bio_1",
            taxon=base_taxon,
            taxon_dir=tmp_path,
            preferred_metrics=("mean",),
        )
        is None
    )
    monkeypatch.setattr(
        desc,
        "_select_variable_outlier_candidate",
        lambda *_a, **_k: {"metric": "max", "phrase": "very high", "context": ""},
    )
    assert (
        desc._select_variable_outlier_text(
            variable_id="bio_1",
            taxon=base_taxon,
            taxon_dir=tmp_path,
            preferred_metrics=("mean",),
        )
        is None
    )

    assert desc._outlier_qualifier_level("extremely") == 3
    assert desc._outlier_qualifier_level("quite") == 1


def test_line_split_empty_clause_branch():
    lines = desc._lines_from_categorical_phrase("often in forests, and .")
    assert lines and lines[0]["prefix"] == "Often in"


def test_remaining_helper_branch_matrix(monkeypatch, tmp_path):
    assert desc._ensure_climate_suffix("dry climate") == "dry climate"
    assert desc._resolve_entry_name(
        {"short_name": "alpha", "class_name": "class_1", "value": "class_1"},
        {"1": {"short_name": "legend-short", "name": "legend-name"}},
    ) == "alpha"
    assert desc._resolve_entry_name(
        {"short_name": "class_1", "class_name": "beta", "value": "class_1"},
        {"1": {"short_name": "legend-short", "name": "legend-name"}},
    ) == "beta"
    assert desc._resolve_entry_name(
        {"short_name": "class_1", "class_name": "class_1", "value": "class_1"},
        {"1": {"short_name": "legend-short", "name": "legend-name"}},
    ) == "legend-short"
    assert desc._resolve_entry_name(
        {"short_name": "class_1", "class_name": "class_1", "value": "class_1"},
        {"1": {"name": "legend-name"}},
    ) == "legend-name"
    assert desc._combine_parallel_labels("", "") == ""
    assert desc._combine_parallel_labels("", "x") == "x"
    assert desc._combine_parallel_labels("x", "") == "x"

    dims = [{"key": "k", "order": ["a", "b"], "combine": "join"}]
    assert (
        desc._combine_semantic_entries(
            {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "a"}}},
            {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": ""}}},
        )
        is None
    )
    assert (
        desc._combine_semantic_entries(
            {"_semantic": {"group": "g", "base_label": "", "dimensions": dims, "values": {"k": "a"}}},
            {"_semantic": {"group": "g", "base_label": "", "dimensions": dims, "values": {"k": "b"}}},
        )
        is None
    )

    candidate = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {}}}
    p1 = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "a"}}}
    p2 = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "a"}}}
    assert desc._is_semantically_subsumed_by_primary(candidate, [p1, p2]) is True
    assert desc._is_semantically_subsumed_by_primary(candidate, [{"_semantic": []}, p2]) is False
    assert desc._is_semantically_subsumed_by_primary({"_semantic": {"group": "", "base_label": "b", "dimensions": [], "values": {}}}, [p1, p2]) is False
    assert desc._is_semantically_subsumed_by_primary({"_semantic": {"group": "x", "base_label": "b", "dimensions": [], "values": {}}}, [p1, p2]) is False
    assert desc._is_semantically_subsumed_by_primary({"_semantic": {"group": "g", "base_label": "b", "dimensions": ["x"], "values": {}}}, [p1, p2]) is False

    assert desc._frequency_verb(0.81) == "almost always"
    assert desc._frequency_verb(0.01) is None
    assert desc._combine_entry_pair_or_single([{"name": ""}]) == ""
    assert desc._top_categorical_phrase_from_payload(variable_id="x", label="x", payload={}) is None

    monkeypatch.setattr(desc.summary_stats, "build_categorical_stats_for_location", lambda *_a, **_k: None)
    monkeypatch.setattr(desc.summary_stats, "load_categorical_distribution", lambda *_a, **_k: None)
    assert desc._top_categorical_phrase(tmp_path, variable_id="x", label="x", taxon_id=1, location_gid="USA") is None


def test_location_and_terrain_specific_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        desc,
        "load_config",
        lambda _n: SimpleNamespace(location_scope_by_level={0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}),
    )
    mapping = {
        "USA": SimpleNamespace(name="United States", parent_gid=None),
        "USA.1": SimpleNamespace(name="A", parent_gid="USA"),
    }
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], mapping))
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {})
    assert desc._build_location_text(1) == ""
    monkeypatch.setattr(
        gis_lookup,
        "location_counts_for_taxon",
        lambda _i: {("gadm_level0", "USA"): 10, ("gadm_level1", "USA.1"): 10},
    )
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level0", "USA"))
    assert desc._build_location_text(1, location_gid="USA") == "A in the United States"
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "other", "custom"))
    assert desc._build_location_text(1, location_gid="custom") == "custom"

    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda _v, _u, _s: (float("nan"), ""))
    assert desc._format_terrain_value(10) is None
    assert desc._slope_band_from_grade(12) == "gentle"
    assert desc._slope_band_from_grade(17) == "moderate"
    assert desc._slope_band_from_grade(31) == "very steep"
    assert desc._slope_range_phrase("mild", "flat") == "mild slopes to flat areas"

    monkeypatch.setattr(
        desc.summary_stats,
        "load_categorical_distribution",
        lambda *_a, **_k: {"distribution": [{"class_name": "x", "fraction": 1.0}], "totals": {"total_samples": "bad"}},
    )
    masses, total = desc._aspect_cardinal_masses(tmp_path)
    assert total == 0.0 and masses["north"] == 0.0
    assert desc._aspect_preference_text(tmp_path) is None


def test_outlier_pipeline_remaining_paths(monkeypatch, tmp_path):
    from util import indexing, taxa_navigation

    taxon = {"taxon_key": "1", "rank": "SPECIES", "path": str(tmp_path)}
    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "min", "count": 20, "percentile": 0.99, "ancestorTaxonId": "10", "label": ""},
            {"metric": "min", "count": 20, "percentile": 0.99, "ancestorTaxonId": "11", "label": ""},
        ],
    )
    monkeypatch.setattr(
        taxa_navigation,
        "get_parent_taxon",
        lambda t: {"taxon_key": "10", "rank": "GENUS"} if str(t.get("taxon_key")) == "1" else None,
    )
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(
        taxa_navigation,
        "get_taxon_by_id",
        lambda tid: {"taxon_key": str(tid), "rank": "ORDER", "scientific_name": "Anc", "path": str(tmp_path / str(tid))},
    )
    cand = desc._select_variable_outlier_candidate(
        variable_id="bio_1",
        taxon=taxon,
        taxon_dir=tmp_path,
        preferred_metrics=("mean", "min"),
        max_ancestor_rank="FAMILY",
    )
    assert cand is None

    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [{"metric": "min", "count": 20, "percentile": 0.99, "ancestorTaxonId": "10", "label": ""}],
    )
    monkeypatch.setattr(
        taxa_navigation,
        "get_taxon_by_id",
        lambda tid: {"taxon_key": str(tid), "rank": "FAMILY", "scientific_name": "Anc", "path": str(tmp_path / str(tid))},
    )
    cand2 = desc._select_variable_outlier_candidate(
        variable_id="bio_1",
        taxon=taxon,
        taxon_dir=tmp_path,
        preferred_metrics=("min",),
        max_ancestor_rank="FAMILY",
    )
    assert cand2 is not None and cand2["context"].startswith("family ")

    monkeypatch.setattr(desc.summary_stats, "_load_categorical_stats", lambda *_a, **_k: {"landcover": {}})
    assert desc._categorical_metric_fraction_for_aliases(tmp_path, variable_id="landcover", aliases=("a",), default=0.2) == 0.2
    monkeypatch.setattr(desc.summary_stats, "_load_categorical_stats", lambda *_a, **_k: {"landcover": {"a": "bad"}})
    assert desc._categorical_metric_fraction_for_aliases(tmp_path, variable_id="landcover", aliases=("a",), default=0.2) == 0.2
    assert desc._resolve_metric_name_for_variable(tmp_path, variable_id="landcover", candidates=()) == ""
    monkeypatch.setattr(desc.summary_stats, "_load_categorical_stats", lambda *_a, **_k: {"landcover": {}})
    assert desc._resolve_metric_name_for_variable(tmp_path, variable_id="landcover", candidates=("A",)) == "A"
    monkeypatch.setattr(desc.summary_stats, "_load_categorical_stats", lambda *_a, **_k: {"landcover": {"x": 1}})
    assert desc._resolve_metric_name_for_variable(tmp_path, variable_id="landcover", candidates=("A",)) == "A"
    assert desc._delta_adjusted_qualifier("bad", abs_delta=0.5) is None
    assert desc._delta_adjusted_qualifier("very", abs_delta=0.01) is None
    assert desc._pick_best_outlier_candidate([]) is None
    assert desc._join_outlier_labels([], "fallback") == "fallback"
    assert desc._entry_metric_aliases(tmp_path, variable_id="landcover", entry={"value": None, "class_name": ""}, legend={}) == ()


def test_remaining_categorical_and_status_edges(monkeypatch, tmp_path):
    from util import taxa_navigation

    # style/default branches
    assert desc._categorical_display_label(
        variable_id="x",
        style="other",
        class_name="Raw Name",
        class_id=None,
        group_value="",
        group_label="",
        legend_entry=None,
    ) == "Raw Name"
    assert desc._outlier_qualifier_level("nope") == 0

    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [{"value": "class_1", "fraction": 0.35}, {"value": "class_2", "fraction": 0.25}]})
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.1)
    monkeypatch.setattr(desc, "_location_label", lambda _g: "X")
    text = desc._location_delta_outlier_text(
        tmp_path,
        variable_id="landcover",
        top_metrics=[{"aliases": ("a",), "label": "A", "fraction": 0.35}, {"aliases": ("b",), "label": "B", "fraction": 0.25}],
        taxon_id=1,
        location_gid="USA",
    )
    assert text is not None and "less common" in text.lower()

    # fallback label + alias-empty skip + best-adjusted low branch
    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: [{"label": "", "metric": "m", "aliases": (), "fraction": 0.4}])
    assert desc._categorical_outlier_text({"taxon_key": "1"}, tmp_path, variable_id="unknown") is None

    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: [{"label": "", "metric": "m", "aliases": ("x",), "fraction": 0.4}])
    monkeypatch.setattr(desc, "_select_variable_outlier_candidate", lambda *_a, **_k: {"qualifier": "very", "context": "family X", "ancestor_taxon_id": "10", "polarity": "low", "depth": 0, "strength": 0.8})
    monkeypatch.setattr(desc, "_categorical_metric_fraction_for_aliases", lambda *_a, **_k: 0.5)
    monkeypatch.setattr(desc, "_delta_adjusted_qualifier", lambda *_a, **_k: "very")
    monkeypatch.setattr(desc, "_sentence_case", lambda t: t)
    monkeypatch.setattr(desc, "_join_outlier_labels", lambda labels, fb: fb if not labels else labels[0])
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"path": str(tmp_path / "anc")})
    out = desc._categorical_outlier_text({"taxon_key": "1"}, tmp_path, variable_id="unknown")
    assert out is not None and "less common" in out

    # numeric/scalar/location guards
    monkeypatch.setattr(desc.summary_stats, "gather_numeric_records", lambda *_a, **_k: [{"value": "x"}])
    assert desc._numeric_summary_for_context(taxon_id=1, taxon_dir=tmp_path, variable_id="bio_1", location_gid="USA") == {}
    assert desc._format_scalar_value(float("nan")) is None
    assert desc._format_scalar_value_for_system(float("nan")) is None
    # no-display-unit append branches in temp/precip
    taxon = {"taxon_key": "1", "rank": "SPECIES", "path": str(tmp_path)}
    monkeypatch.setattr(
        desc,
        "_numeric_summary_for_context",
        lambda *, variable_id, location_gid=None, **_k: {
            "bio_6": {"min": -2},
            "bio_5": {"max": 5},
            "bio_1": {"mean": 11 if location_gid else 10},
            "bio_12": {"min": 200, "max": 200, "mean": 300 if location_gid else 280},
        }.get(variable_id, {}),
    )
    monkeypatch.setattr(gis_lookup, "load_variable_metadata", lambda: ([], {}))
    monkeypatch.setattr(desc.units, "equivalent_unit", lambda u, _s: u)
    monkeypatch.setattr(desc.units, "display_unit", lambda _u: "")
    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda v, _u, _s: (v, ""))
    monkeypatch.setattr(desc, "_location_label", lambda _g: "X")
    weather = desc._weather_status_rows(taxon, tmp_path, taxon_id=1, location_gid="USA")[0]["detail"]
    assert weather is not None and "Can tolerate" in weather
    monkeypatch.setattr(desc, "_numeric_summary_for_context", lambda *, variable_id, **_k: {"bio_12": {"min": 200, "max": 500, "mean": None}}.get(variable_id, {}))
    weather2 = desc._weather_status_rows(taxon, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert weather2 is not None and "to" in weather2


def test_final_helper_edges(monkeypatch, tmp_path):
    assert desc._landcover_group_label("unknown_group", "Custom Label") == "custom label"
    assert desc._landcover_group_label("", "") == "landcover classes"
    assert desc._derive_koppen_thermal("BHK") == "cold"
    assert desc._derive_koppen_thermal("BHH") == "hot"
    assert desc._derive_koppen_thermal("X") == ""

    # semantic label fallback base + skipped invalid dim key
    monkeypatch.setitem(desc._CATEGORICAL_LAYER_RULES, "xvar", {"split_groups": {"g": {"base_label": "", "dimensions": [{"key": ""}, {"key": "k", "order": ["a"], "combine": "drop"}]}}})
    label, semantic = desc._semantic_label_from_group(variable_id="xvar", group="g", group_label="G", traits={"k": "z"})
    assert label == "G"
    assert semantic is not None and semantic["dimensions"]

    # combine semantic early mismatch branches + base-label return
    dims = [{"key": "k", "order": ["a", "b"], "combine": "drop"}]
    left = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "a"}}}
    right = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "b"}}}
    assert desc._combine_semantic_entries(left, right) == "b"
    assert desc._combine_semantic_entries({"_semantic": {"group": "a", "base_label": "b", "dimensions": [], "values": {}}}, {"_semantic": {"group": "a", "base_label": "c", "dimensions": [], "values": {}}}) is None
    assert desc._combine_semantic_entries({"_semantic": {"group": "a", "base_label": "b", "dimensions": []}}, {"_semantic": {"group": "a", "base_label": "b", "dimensions": [], "values": {}}}) is None

    # phrase payload koppen branch with code extraction + thermal derivation
    monkeypatch.setattr(
        gis_lookup,
        "load_layer_legend",
        lambda _v: {"1": {"name": "Desert (BWh)", "group": "desert", "group_label": "Desert", "traits": {}}},
    )
    text = desc._top_categorical_phrase_from_payload(
        variable_id="koppen_geiger",
        label="climate",
        payload={"distribution": [{"value": "class_1", "fraction": 1.0}]},
    )
    assert text is not None and "climates" in text

    # Empty ranking after sanitize.
    assert (
        desc._top_categorical_phrase_from_payload(
            variable_id="misc",
            label="x",
            payload={"distribution": [{"class_name": "none", "fraction": 1.0}]},
        )
        is None
    )

    # top_categorical_phrase no-location path falls through to None payload.
    monkeypatch.setattr(desc.summary_stats, "load_categorical_distribution", lambda *_a, **_k: None)
    assert desc._top_categorical_phrase(tmp_path, variable_id="x", label="x", taxon_id=None, location_gid=None) is None


def test_final_outlier_and_status_edges(monkeypatch, tmp_path):

    # location delta specific branches.
    assert desc._location_delta_outlier_text(tmp_path, variable_id="x", top_metrics=[], taxon_id=1, location_gid="USA") is None
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [1]})
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.34)
    out = desc._location_delta_outlier_text(
        tmp_path,
        variable_id="x",
        top_metrics=[{"aliases": ("a",), "label": "", "fraction": 0.35}],
        taxon_id=1,
        location_gid="USA",
    )
    assert out is None
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.05)
    out2 = desc._location_delta_outlier_text(
        tmp_path,
        variable_id="x",
        top_metrics=[{"aliases": ("a",), "label": "A", "fraction": 0.25}],
        taxon_id=1,
        location_gid="USA",
    )
    assert out2 is not None and "less common" in out2.lower()

    # categorical_outlier fallback label path + aliases-empty skip.
    monkeypatch.setattr(desc, "_top_categorical_class_metrics", lambda *_a, **_k: [{"label": "", "metric": "m", "aliases": (), "fraction": 0.4}])
    assert desc._categorical_outlier_text({"taxon_key": "1"}, tmp_path, variable_id="landcover") is None

    # numeric summary fallback and scalar parse failures.
    monkeypatch.setattr(desc.summary_stats, "load_numeric_summary", lambda *_a, **_k: {})
    assert desc._numeric_summary_for_context(taxon_id=None, taxon_dir=tmp_path, variable_id="x") == {}
    assert desc._format_scalar_value("bad") is None
    assert desc._format_scalar_value_for_system("bad") is None
    assert desc._location_label(None) == "this location"
    assert desc._location_label("   ") == "this location"
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "x", "target"))
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], {}))
    assert desc._location_label("X") == "target"

    # terrain/status minor lines.
    monkeypatch.setattr(
        desc,
        "_numeric_summary_for_context",
        lambda *, variable_id, **_k: {"elevation": {}, "slope": {"mean": 10, "10th percentile": 9}}.get(variable_id, {}),
    )
    monkeypatch.setattr(gis_lookup, "load_variable_metadata", lambda: ([], {"elevation": {"units": "m"}}))
    monkeypatch.setattr(desc.units, "equivalent_unit", lambda u, _s: u)
    monkeypatch.setattr(desc.units, "display_unit", lambda u: u or "")
    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda v, _u, _s: (v, ""))
    monkeypatch.setattr(desc, "_aspect_preference_text", lambda _p: None)
    terrain = desc._terrain_status_rows({"taxon_key": "1"}, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert terrain is not None and "slopes" in terrain

    # precipitation outlier insertion line path.
    monkeypatch.setattr(
        desc,
        "_numeric_summary_for_context",
        lambda *, variable_id, **_k: {"bio_12": {"min": 100, "max": 200, "mean": 150}}.get(variable_id, {}),
    )
    monkeypatch.setattr(gis_lookup, "load_variable_metadata", lambda: ([], {"bio_12": {"units": "mm"}}))
    weather = desc._weather_status_rows({"taxon_key": "1"}, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert weather is not None and "Can tolerate" in weather

    # lines parser prefix transform
    assert desc._lines_from_categorical_phrase("always in forests")[0]["prefix"] == "Always in"


def test_last_coverage_push_branches(monkeypatch, tmp_path):
    # name-based group inference branches
    assert desc._infer_landcover_group("shrubland zone", None)[0] == "shrubland"
    assert desc._infer_landcover_group("wetland zone", None)[0] == "wetlands"
    assert desc._infer_landcover_group("water body", None)[0] == "water"
    assert desc._infer_landcover_group("bare soil", None)[0] == "bare_areas"
    assert desc._infer_landcover_group("ice field", None)[0] == "ice_snow"
    assert desc._infer_landcover_group("lichen mat", None)[0] == "lichens_mosses"

    # semantic combine/internal branches
    dims = [{"key": "k", "order": ["a"], "combine": "join"}, {"key": "", "order": [], "combine": "join"}]
    left = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "a"}}}
    right = {"_semantic": {"group": "g", "base_label": "b", "dimensions": dims, "values": {"k": "a"}}}
    assert desc._combine_semantic_entries(left, right) is None
    assert desc._combine_semantic_entries({"_semantic": {"group": "g", "base_label": "b", "dimensions": [1], "values": {}}}, {"_semantic": {"group": "g", "base_label": "b", "dimensions": {}, "values": {}}}) is None
    assert desc._legend_for_entry({"slug": "sluggy"}, {"sluggy": {"name": "x"}}) == {"name": "x"}
    assert desc._is_semantically_subsumed_by_primary(
        {"_semantic": {"group": "g", "base_label": "b", "dimensions": [{"key": "k"}], "values": {"k": "a"}}},
        [{"_semantic": {"group": "g", "base_label": "b", "dimensions": [{"key": "k"}], "values": {"k": "a"}}}, {"_semantic": {"group": "g", "base_label": "b", "dimensions": [{"key": "x"}], "values": {"k": "a"}}}],
    ) is False

    # top phrase internals for koppen branch + specific ranking lines
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    txt = desc._top_categorical_phrase_from_payload(
        variable_id="koppen_geiger",
        label="climate",
        payload={"distribution": [{"class_name": "Cfb", "fraction": 0.4}, {"class_name": "Dfb", "fraction": 0.3}, {"class_name": "Cfb", "fraction": 0.2}]},
    )
    assert txt is not None

    # with-definite article + location internals
    assert desc._with_definite_article("the netherlands") == "the netherlands"
    monkeypatch.setattr(
        desc,
        "load_config",
        lambda _n: SimpleNamespace(location_scope_by_level={0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}),
    )
    mapping = {
        "USA": SimpleNamespace(name="United States", parent_gid=None),
        "USA.1": SimpleNamespace(name="A", parent_gid="USA"),
        "USA.2": SimpleNamespace(name="A", parent_gid="USA"),
        "USA.1.1": SimpleNamespace(name="Sub", parent_gid="USA.1"),
    }
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], mapping))
    monkeypatch.setattr(
        gis_lookup,
        "location_counts_for_taxon",
        lambda _i: {("gadm_level0", "USA"): 10, ("gadm_level1", "USA.1"): 6, ("gadm_level1", "USA.2"): 4, ("gadm_level2", "USA.1.1"): 6},
    )
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level1", "USA.1"))
    assert "Sub" in desc._build_location_text(1, location_gid="USA.1", min_fraction=0.7)
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level2", "USA.1.1"))
    assert desc._build_location_text(1, location_gid="USA.1.1") == "Sub in A"
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {("gadm_level0", "USA"): 10})
    assert desc._build_location_text(1).startswith("the United States")

    # terrain parse exceptions + slope branches
    assert desc._format_terrain_value("nope") is None
    assert desc._slope_grade_percent("nope") is None
    assert desc._slope_band_from_grade(16) == "moderate"

    monkeypatch.setattr(
        desc.summary_stats,
        "load_categorical_distribution",
        lambda *_a, **_k: {"distribution": [{"fraction": "bad"}, {"fraction": 0.5, "value": "class_2"}], "totals": {"total_samples": 200}},
    )
    masses, _ = desc._aspect_cardinal_masses(tmp_path)
    assert masses["north"] > 0 and masses["east"] > 0

    # outlier candidate fallback context build and strength tie-break
    from util import indexing, taxa_navigation

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "count": 20, "percentile": 0.95, "ancestorTaxonId": "10", "label": ""},
            {"metric": "mean", "count": 20, "percentile": 0.99, "ancestorTaxonId": "10", "label": ""},
        ],
    )
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda _t: None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"rank": "FAMILY", "scientific_name": "Anc", "path": str(tmp_path)})
    cand = desc._select_variable_outlier_candidate(
        variable_id="x",
        taxon={"taxon_key": "1", "rank": "SPECIES"},
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
        max_ancestor_rank=None,
    )
    assert cand is not None and "family Anc" in cand["context"]

    # alias skip and class-metric loops
    monkeypatch.setattr(desc.summary_stats, "_load_categorical_stats", lambda *_a, **_k: {"x": {"a": 1}})
    assert desc._categorical_metric_fraction_for_aliases(tmp_path, variable_id="x", aliases=("",), default=0.0) == 0.0

    # class metric selector edge lines
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [{"fraction": 0.0, "value": "x"}, {"fraction": 0.4, "value": "class_1", "class_name": "C"}]})
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    monkeypatch.setattr(desc, "_resolve_metric_name_for_variable", lambda *_a, **_k: "m")
    rows = desc._top_categorical_class_metrics(tmp_path, variable_id="landcover")
    assert rows and rows[0]["metric"] == "m"

    # level-2 phrase branch
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [1]})
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(desc, "_location_label", lambda _g: "X")
    t = desc._location_delta_outlier_text(tmp_path, variable_id="x", top_metrics=[{"aliases": ("a",), "label": "A", "fraction": 0.2}], taxon_id=1, location_gid="USA")
    assert t is not None and "less common" in t.lower()

    # precip with display units to exercise range_with_units with units.
    monkeypatch.setattr(
        desc,
        "_numeric_summary_for_context",
        lambda *, variable_id, location_gid=None, **_k: {"bio_12": {"min": 100, "max": 100, "median": 100 if location_gid else 100}}.get(variable_id, {}),
    )
    monkeypatch.setattr(gis_lookup, "load_variable_metadata", lambda: ([], {"bio_12": {"units": "mm"}}))
    monkeypatch.setattr(desc.units, "equivalent_unit", lambda u, _s: u)
    monkeypatch.setattr(desc.units, "display_unit", lambda u: u or "")
    monkeypatch.setattr(desc.units, "convert_value_for_system", lambda v, _u, _s: (v, ""))
    p = desc._weather_status_rows({"taxon_key": "1"}, tmp_path, taxon_id=1, location_gid=None)[0]["detail"]
    assert p == "Prefers xeric areas"


def test_final_remaining_lines_live_paths(monkeypatch, tmp_path):
    # Top phrase loop/primary/secondary lines.
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    text = desc._top_categorical_phrase_from_payload(
        variable_id="misc",
        label="x",
        payload={"distribution": [{"class_name": "alpha", "fraction": 0.6}, {"class_name": "beta", "fraction": 0.3}, {"class_name": "gamma", "fraction": 0.05}]},
    )
    assert text is not None

    # Build location text internals: dedupe, filtered-empty, total<=0, no-names path.
    monkeypatch.setattr(
        desc,
        "load_config",
        lambda _n: SimpleNamespace(location_scope_by_level={0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}),
    )
    mapping = {
        "USA": SimpleNamespace(name="United States", parent_gid=None),
        "USA.1": SimpleNamespace(name="Dup", parent_gid="USA"),
        "USA.2": SimpleNamespace(name="Dup", parent_gid="USA"),
        "USA.3": SimpleNamespace(name="Solo", parent_gid="USA"),
        "USA.1.1": SimpleNamespace(name="Sub", parent_gid="MISSING"),
    }
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], mapping))
    monkeypatch.setattr(
        gis_lookup,
        "location_counts_for_taxon",
        lambda _i: {
            ("gadm_level0", "USA"): 10,
            ("gadm_level1", "USA.1"): 6,
            ("gadm_level1", "USA.2"): 4,
            ("gadm_level1", "USA.3"): 0,
            ("gadm_level2", "USA.1.1"): 6,
        },
    )
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level1", "USA.1"))
    t1 = desc._build_location_text(1, location_gid="USA.1", min_fraction=0.99)  # filtered-empty branch
    assert t1 == "Dup"
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level2", "USA.1.1"))
    t2 = desc._build_location_text(1, location_gid="USA.1.1")  # missing parent fallback
    assert t2 == "Sub in MISSING"
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {("gadm_level0", "USA"): 10, ("gadm_level1", "USA.1"): 6, ("gadm_level1", "USA.2"): 4})
    t3 = desc._build_location_text(1, location_gid=None, limit=1)  # dedupe + has_more country text path
    assert "other countries" in t3 or "United States" in t3

    # Explicit error-handling lines.
    assert desc._format_terrain_value("bad") is None
    assert desc._slope_grade_percent("bad") is None
    assert desc._slope_band_from_grade(19.0) == "moderate"
    monkeypatch.setattr(
        desc.summary_stats,
        "load_categorical_distribution",
        lambda *_a, **_k: {"distribution": [{"fraction": "bad"}, {"fraction": 0.5, "value": "class_2"}], "totals": {"total_samples": 200}},
    )
    _masses, _total = desc._aspect_cardinal_masses(tmp_path)

    # Outlier candidate context fallback and tie-break by strength.
    from util import indexing, taxa_navigation

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "count": 20, "percentile": 0.95, "ancestorTaxonId": "10", "label": ""},
            {"metric": "mean", "count": 20, "percentile": 0.99, "ancestorTaxonId": "10", "label": ""},
        ],
    )
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda _t: None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"rank": "FAMILY", "scientific_name": None, "common_name": None, "taxon_key": "T10", "path": str(tmp_path)})
    candidate = desc._select_variable_outlier_candidate(
        variable_id="x",
        taxon={"taxon_key": "1", "rank": "SPECIES"},
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
    )
    assert candidate is not None and "t10" in candidate["context"].lower()

    # Forest phenology fallback line.
    label = desc._categorical_display_label(
        variable_id="landcover",
        style="group_map",
        class_name="Deciduous forest",
        class_id=1,
        group_value="forest",
        group_label="Forest",
        legend_entry={"traits": {}},
    )
    assert "forests" in label

    # top_categorical_class_metrics specific flow lines.
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [{"fraction": 0.2, "value": "class_1", "class_name": "C"}]})
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    monkeypatch.setattr(desc, "_resolve_metric_name_for_variable", lambda *_a, **_k: "metric_1")
    rows = desc._top_categorical_class_metrics(tmp_path, variable_id="landcover")
    assert rows and rows[0]["metric"] == "metric_1"

    # location delta level-2 text branch
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [1]})
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(desc, "_location_label", lambda _g: "X")
    txt = desc._location_delta_outlier_text(
        tmp_path,
        variable_id="x",
        top_metrics=[{"aliases": ("a",), "label": "A", "fraction": 0.2}],
        taxon_id=1,
        location_gid="USA",
    )
    assert txt is not None and "less common" in txt.lower()


def test_truly_last_branch_targets(monkeypatch, tmp_path):
    # _is_semantically_subsumed_by_primary values-not-dict branch
    cand = {"_semantic": {"group": "g", "base_label": "b", "dimensions": [{"key": "k"}], "values": {}}}
    p1 = {"_semantic": {"group": "g", "base_label": "b", "dimensions": [{"key": "k"}], "values": []}}
    p2 = {"_semantic": {"group": "g", "base_label": "b", "dimensions": [{"key": "k"}], "values": {}}}
    assert desc._is_semantically_subsumed_by_primary(cand, [p1, p2]) is False

    # top phrase specific branches: candidate skipped, no-primary return, combined==primary
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    txt_none = desc._top_categorical_phrase_from_payload(
        variable_id="misc",
        label="x",
        payload={"distribution": [{"class_name": "alpha", "fraction": 0.05}]},
    )
    assert txt_none is None
    txt_same = desc._top_categorical_phrase_from_payload(
        variable_id="misc",
        label="x",
        payload={"distribution": [{"class_name": "alpha", "fraction": 0.26}, {"class_name": "alpha", "fraction": 0.21}]},
    )
    assert txt_same is not None and txt_same.startswith("often in")

    # location text internal branches: total<=0, filtered-empty, dedupe continue, scope0/scope2 fallbacks, no-country
    monkeypatch.setattr(
        desc,
        "load_config",
        lambda _n: SimpleNamespace(location_scope_by_level={0: "gadm_level0", 1: "gadm_level1", 2: "gadm_level2"}),
    )
    mapping = {
        "USA": SimpleNamespace(name="United States", parent_gid=None),
        "USA.1": SimpleNamespace(name="Dup", parent_gid="USA"),
        "USA.2": SimpleNamespace(name="Dup", parent_gid="USA"),
        "USA.1.1": SimpleNamespace(name="Leaf", parent_gid=None),
    }
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: ([], mapping))
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {("gadm_level0", "USA"): -5})
    assert desc._build_location_text(1) == ""
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {("gadm_level0", "USA"): 10, ("gadm_level1", "USA.1"): 6, ("gadm_level1", "USA.2"): 4})
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level0", "USA"))
    assert desc._build_location_text(1, location_gid="USA", min_fraction=0.99) == "the United States"
    monkeypatch.setattr(gis_lookup, "location_lookup_for_gid", lambda _g: ("gid", "gadm_level2", "USA.1.1"))
    assert desc._build_location_text(1, location_gid="USA.1.1") == "Leaf"
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {})
    assert desc._build_location_text(1) == ""
    monkeypatch.setattr(gis_lookup, "location_counts_for_taxon", lambda _i: {("gadm_level0", "USA"): 10, ("gadm_level1", "USA.1"): 6, ("gadm_level1", "USA.2"): 4})
    s = desc._build_location_text(1)
    assert "Dup" in s

    # terrain/slope/aspect edge lines
    assert desc._format_terrain_value(float("nan")) is None
    assert desc._slope_grade_percent(float("nan")) is None
    assert desc._slope_band_from_grade(25.0) == "steep"
    monkeypatch.setattr(desc.summary_stats, "load_categorical_distribution", lambda *_a, **_k: {"distribution": [{"fraction": 0, "value": "1"}], "totals": {"total_samples": 200}})
    masses, _total = desc._aspect_cardinal_masses(tmp_path)
    assert masses["north"] == 0.0

    # outlier context-empty and strength tie branch
    from util import indexing, taxa_navigation

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "count": 20, "percentile": 0.95, "ancestorTaxonId": "10", "label": ""},
            {"metric": "mean", "count": 20, "percentile": 0.99, "ancestorTaxonId": "10", "label": ""},
        ],
    )
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda _t: None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"rank": "FAMILY", "scientific_name": "", "common_name": "", "taxon_key": "", "path": str(tmp_path)})
    assert (
        desc._select_variable_outlier_candidate(
            variable_id="x",
            taxon={"taxon_key": "1", "rank": "SPECIES"},
            taxon_dir=tmp_path,
            preferred_metrics=("mean",),
        )
        is None
    )
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"rank": "FAMILY", "scientific_name": "Anc", "path": str(tmp_path)})
    c = desc._select_variable_outlier_candidate(
        variable_id="x",
        taxon={"taxon_key": "1", "rank": "SPECIES"},
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
    )
    assert c is not None

    # top class metrics lines
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [{"fraction": 0.2, "value": None, "class_name": "", "short_name": ""}, {"fraction": 0.2, "value": "class_1", "class_name": "A"}]})
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    monkeypatch.setattr(desc, "_resolve_metric_name_for_variable", lambda *_a, **_k: "m")
    rows = desc._top_categorical_class_metrics(tmp_path, variable_id="landcover")
    assert rows

    # level-1 phrase branch in location delta
    monkeypatch.setattr(desc, "_load_categorical_payload_for_context", lambda *_a, **_k: {"distribution": [1]})
    monkeypatch.setattr(desc, "_fraction_for_aliases_from_payload", lambda *_a, **_k: 0.05)
    monkeypatch.setattr(desc, "_location_label", lambda _g: "X")
    t = desc._location_delta_outlier_text(tmp_path, variable_id="x", top_metrics=[{"aliases": ("a",), "label": "A", "fraction": 0.2}], taxon_id=1, location_gid="USA")
    assert "a bit less common" in t.lower()


def test_final_four_live_lines(monkeypatch, tmp_path):
    # line 541: empty-key continue in composed loop
    dims = [{"key": "k", "order": ["a", "b"], "combine": "join"}, {"key": "", "order": [], "combine": "join"}]
    out = desc._combine_semantic_entries(
        {"_semantic": {"group": "g", "base_label": "base", "dimensions": dims, "values": {"k": "a"}}},
        {"_semantic": {"group": "g", "base_label": "base", "dimensions": dims, "values": {"k": "b"}}},
    )
    assert out == "a and b base"

    # line 855: skip low-fraction candidate in paired-primary scan
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    text = desc._top_categorical_phrase_from_payload(
        variable_id="misc",
        label="x",
        payload={"distribution": [{"class_name": "alpha", "fraction": 0.6}, {"class_name": "beta", "fraction": 0.01}, {"class_name": "gamma", "fraction": 0.3}]},
    )
    assert text is not None

    # line 1527: tie-break by strength
    from util import indexing, taxa_navigation

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "count": 20, "percentile": 0.95, "ancestorTaxonId": "10", "label": "Anc"},
            {"metric": "mean", "count": 20, "percentile": 0.99, "ancestorTaxonId": "10", "label": "Anc"},
        ],
    )
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda _t: None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"rank": "FAMILY", "scientific_name": "Anc", "path": str(tmp_path)})
    cand = desc._select_variable_outlier_candidate(
        variable_id="x",
        taxon={"taxon_key": "1", "rank": "SPECIES"},
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
    )
    assert cand is not None and cand["strength"] == pytest.approx(0.99)

    # line 1868: inferred group label assignment when missing
    monkeypatch.setattr(
        desc,
        "_load_categorical_payload_for_context",
        lambda *_a, **_k: {"distribution": [{"fraction": 0.4, "value": "class_190", "class_name": "Urban"}]},
    )
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    monkeypatch.setattr(desc, "_resolve_metric_name_for_variable", lambda *_a, **_k: "m")
    rows = desc._top_categorical_class_metrics(tmp_path, variable_id="landcover")
    assert rows and rows[0]["label"] == "urban areas"


def test_absolute_last_two_lines(monkeypatch, tmp_path):
    # 1525: tie-break by greater strength at same level/depth
    from util import indexing, taxa_navigation

    monkeypatch.setattr(desc, "load_config", lambda _n: SimpleNamespace(skip_description_outliers=False))
    monkeypatch.setattr(
        indexing,
        "load_relative_ranks",
        lambda *_a, **_k: [
            {"metric": "mean", "count": 20, "percentile": 0.96, "ancestorTaxonId": "10", "label": "Anc"},
            {"metric": "mean", "count": 20, "percentile": 0.98, "ancestorTaxonId": "10", "label": "Anc"},
        ],
    )
    monkeypatch.setattr(taxa_navigation, "get_parent_taxon", lambda _t: None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _i: {"rank": "FAMILY", "scientific_name": "Anc", "path": str(tmp_path)})
    c = desc._select_variable_outlier_candidate(
        variable_id="x",
        taxon={"taxon_key": "1", "rank": "SPECIES"},
        taxon_dir=tmp_path,
        preferred_metrics=("mean",),
    )
    assert c is not None and c["strength"] == pytest.approx(0.98)

    # 1866: duplicate metric skip
    monkeypatch.setattr(
        desc,
        "_load_categorical_payload_for_context",
        lambda *_a, **_k: {"distribution": [{"fraction": 0.6, "value": "class_1", "class_name": "A"}, {"fraction": 0.4, "value": "class_2", "class_name": "B"}]},
    )
    monkeypatch.setattr(gis_lookup, "load_layer_legend", lambda _v: {})
    monkeypatch.setattr(desc, "_resolve_metric_name_for_variable", lambda *_a, **_k: "dup_metric")
    rows = desc._top_categorical_class_metrics(tmp_path, variable_id="landcover")
    assert len(rows) == 1 and rows[0]["metric"] == "dup_metric"

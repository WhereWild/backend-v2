"""Unit tests for util.gis_lookup helper behavior."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from util import gis_lookup


class _StubParquet:
    is_remote = False

    def __init__(self):
        self._exists: dict[Path, bool] = {}
        self._files: dict[Path, bytes] = {}
        self._read_table = None

    def current(self):
        return self

    def exists(self, path):
        return self._exists.get(Path(path), False)

    def open_input_file(self, path):
        payload = self._files[Path(path)]
        return io.BytesIO(payload)

    def read_table(self, path, **kwargs):
        if self._read_table is None:
            raise RuntimeError("read_table not configured")
        return self._read_table(path, **kwargs)


@pytest.fixture(autouse=True)
def _clear_caches():
    gis_lookup._load_gis_catalog.cache_clear()
    gis_lookup._get_layer.cache_clear()
    gis_lookup.load_layer_metadata.cache_clear()
    gis_lookup.load_temporal_registry.cache_clear()
    gis_lookup.load_variable_metadata.cache_clear()
    gis_lookup.load_layer_legend.cache_clear()
    gis_lookup.preload_layer_legends.cache_clear()
    gis_lookup.load_location_catalog.cache_clear()
    gis_lookup.location_taxa_membership.cache_clear()
    gis_lookup.location_taxa_for.cache_clear()
    gis_lookup.location_taxon_counts.cache_clear()
    gis_lookup._load_location_taxa_table.cache_clear()
    gis_lookup.get_layer_tile_info.cache_clear()
    yield


@pytest.fixture
def stub_env(monkeypatch, tmp_path):
    config = SimpleNamespace(
        gis_catalog_path=tmp_path / "catalog.json",
        gis_legends_root=tmp_path / "legends",
        location_hierarchy_path=tmp_path / "hierarchy.csv",
        gbif_region_set={"AFRICA", "EUROPE"},
        location_level_columns=["country", "state", "county"],
        location_scope_by_level=["country_scope", "state_scope", "county_scope"],
        location_catalog_path=tmp_path / "location_taxa.parquet",
        gis_root=tmp_path / "gis",
    )
    stub = _StubParquet()
    monkeypatch.setattr(gis_lookup, "CONFIG", config)
    monkeypatch.setattr(gis_lookup, "PARQUET", stub)
    return config, stub


def test_catalog_layer_and_variable_metadata(stub_env):
    config, stub = stub_env
    catalog = {
        "categories": [
            {
                "name": "physical",
                "display_name": "Physical",
                "layers": [{"id": "bio_1", "display_name": "Temp", "value_type": "numeric", "units": "C"}],
            },
            {
                "name": "temporal",
                "display_name": "Temporal",
                "windows": [24],
                "layers": [
                    {"id": "wind", "agg": "avg", "display_name": "Wind", "value_type": "numeric", "units": "m/s"},
                    {"id": "nowcast", "agg": "snapshot", "display_name": "Nowcast", "value_type": "categorical"},
                ],
            },
        ]
    }
    stub._files[config.gis_catalog_path] = json.dumps(catalog).encode("utf-8")

    assert gis_lookup._get_layer("bio_1")["id"] == "bio_1"
    assert gis_lookup._get_layer("missing") is None

    layers = gis_lookup.load_layer_metadata()
    assert "bio_1" in layers
    assert "wind_avg_24h" in layers
    assert "nowcast" in layers

    temporal = gis_lookup.load_temporal_registry()
    assert temporal["windows"] == [24]
    assert len(temporal["layers"]) == 2

    entries, mapping = gis_lookup.load_variable_metadata()
    assert [entry["id"] for entry in entries] == sorted(mapping.keys())
    assert mapping["bio_1"]["category"] == "Physical"
    assert mapping["wind_avg_24h"]["name"].startswith("Wind")


def test_metadata_skips_layers_without_id(stub_env):
    config, stub = stub_env
    catalog = {
        "categories": [
            {"name": "physical", "layers": [{"id": "bio_1"}, {"name": "skip-me"}]},
            {"name": "temporal", "windows": [6], "layers": [{"id": "wind", "agg": "avg"}, {"agg": "avg"}]},
        ]
    }
    stub._files[config.gis_catalog_path] = json.dumps(catalog).encode("utf-8")

    layer_ids = set(gis_lookup.load_layer_metadata().keys())
    assert layer_ids == {"bio_1", "wind_avg_6h"}

    _, by_id = gis_lookup.load_variable_metadata()
    assert set(by_id.keys()) == {"bio_1", "wind_avg_6h"}

    # Force defensive "missing id" branches for expanded temporal entries.
    gis_lookup.load_layer_metadata.cache_clear()
    gis_lookup.load_variable_metadata.cache_clear()
    original_expand = gis_lookup._expand_temporal_layers
    try:
        gis_lookup._expand_temporal_layers = lambda _category: [{"id": None}]  # type: ignore[assignment]
        assert gis_lookup.load_layer_metadata() == {"bio_1": {"id": "bio_1"}}
        assert gis_lookup.load_variable_metadata()[1] == {
            "bio_1": {
                "id": "bio_1",
                "name": "bio_1",
                "units": None,
                "description": None,
                "value_type": None,
                "category": "physical",
                "source_ids": [],
            }
        }
    finally:
        gis_lookup._expand_temporal_layers = original_expand  # type: ignore[assignment]


def test_temporal_registry_and_layer_ids_when_category_absent(stub_env):
    config, stub = stub_env
    stub._files[config.gis_catalog_path] = json.dumps(
        {"categories": [{"name": "other", "layers": [{"id": "x"}]}]}
    ).encode("utf-8")

    assert gis_lookup.load_temporal_registry() == {"windows": [], "layers": []}
    assert gis_lookup.list_layer_ids() == ["x"]


def test_list_layer_ids_with_temporal_category(stub_env):
    config, stub = stub_env
    catalog = {
        "categories": [
            {"name": "temporal", "windows": [6], "layers": [{"id": "wind", "agg": "avg"}, {"agg": "avg"}]},
            {"name": "physical", "layers": [{"id": "bio_1"}]},
        ]
    }
    stub._files[config.gis_catalog_path] = json.dumps(catalog).encode("utf-8")
    assert gis_lookup.list_layer_ids() == ["wind_avg_6h", "bio_1"]


def test_expand_temporal_layers_branches():
    category = {
        "windows": [6, 12],
        "layers": [
            {"id": "a", "agg": "avg", "display_name": "A", "code": "A1"},
            {"id": "b", "agg": "snapshot", "display_name": "B"},
            {"agg": "avg"},
        ],
    }
    expanded = gis_lookup._expand_temporal_layers(category)
    assert [item["id"] for item in expanded] == ["a_avg_6h", "a_avg_12h", "b"]


def test_temporal_layer_source_ids_override_category_source_ids(stub_env):
    config, stub = stub_env
    catalog = {
        "categories": [
            {
                "name": "temporal",
                "display_name": "Temporal",
                "source_ids": ["category_source"],
                "windows": [24],
                "layers": [
                    {
                        "id": "wind",
                        "agg": "avg",
                        "display_name": "Wind",
                        "source_ids": ["layer_source"],
                    },
                    {
                        "id": "nowcast",
                        "agg": "snapshot",
                        "display_name": "Nowcast",
                    },
                ],
            },
        ]
    }
    stub._files[config.gis_catalog_path] = json.dumps(catalog).encode("utf-8")

    expanded = gis_lookup._expand_temporal_layers(catalog["categories"][0])
    assert expanded[0]["source_ids"] == ["layer_source"]
    assert expanded[1]["source_ids"] is None

    _entries, mapping = gis_lookup.load_variable_metadata()
    assert mapping["wind_avg_24h"]["source_ids"] == ["layer_source"]
    assert mapping["nowcast"]["source_ids"] == ["category_source"]


def test_parse_temporal_layer_id_helpers():
    assert gis_lookup.parse_temporal_layer_id("temperature_2m_avg_24h") == (
        "temperature_2m",
        "avg",
        24,
    )
    assert gis_lookup.parse_temporal_layer_id("precipitation_sum_168h") == (
        "precipitation",
        "sum",
        168,
    )
    assert gis_lookup.parse_temporal_layer_id("weather_code_simple") is None
    assert gis_lookup.is_temporal_layer_id("temperature_2m_avg_24h")
    assert not gis_lookup.is_temporal_layer_id("weather_code_simple")


def test_temporal_feature_names_from_config_requires_explicit_aggregation():
    cfg = SimpleNamespace(
        temporal_window_hours_by_variable={"precipitation": (24,)},
        temporal_agg_by_variable={},
        temporal_window_hours_default=(24,),
    )

    with pytest.raises(ValueError, match="precipitation"):
        gis_lookup.temporal_feature_names_from_config(cfg)


def test_temporal_feature_names_from_config_uses_configured_windows():
    cfg = SimpleNamespace(
        temporal_window_hours_by_variable={
            "precipitation": (24,),
            "temperature_2m": (1, 8),
        },
        temporal_agg_by_variable={
            "precipitation": "sum",
            "temperature_2m": "avg",
        },
        temporal_window_hours_default=(24,),
    )

    assert gis_lookup.temporal_feature_names_from_config(cfg) == [
        "precipitation_sum_24h",
        "temperature_2m_avg_1h",
        "temperature_2m_avg_8h",
        "vapor_pressure_deficit_avg_1h",
        "vapor_pressure_deficit_avg_8h",
    ]


def test_load_layer_legend_and_preload(stub_env, monkeypatch):
    config, stub = stub_env
    config.gis_legends_root.mkdir(parents=True, exist_ok=True)
    legend_path = config.gis_legends_root / "landcover_legend.json"
    stub._files[legend_path] = json.dumps(
        {"classes": [{"id": 1, "name": "Warm Temperate"}, {"id": None, "name": "bad"}]}
    ).encode("utf-8")

    legend = gis_lookup.load_layer_legend("landcover")
    assert legend["1"]["name"] == "Warm Temperate"
    assert legend["warm temperate"]["id"] == 1

    bad_path = config.gis_legends_root / "bad_legend.json"
    stub._files[bad_path] = b"{not-json"
    assert gis_lookup.load_layer_legend("bad") == {}

    monkeypatch.setattr(
        gis_lookup,
        "load_layer_metadata",
        lambda: {"landcover": {"value_type": "categorical"}, "bio_1": {"value_type": "numeric"}},
    )
    assert gis_lookup.preload_layer_legends() == 1


def test_load_location_catalog_and_search_helpers(stub_env):
    config, stub = stub_env
    csv_payload = (
        "gid,name,level,parent_gid\n"
        "USA,United States,0,\n"
        "USA.UT,Utah,1,USA\n"
        "USA.UT.001,Salt Lake,2,USA.UT\n"
        "AFRICA.KE,Kenya,1,AFRICA\n"
        "BAD,Bad,nope,\n"
        ",Missing Gid,1,USA\n"
    )
    stub._exists[config.location_hierarchy_path] = True
    stub._files[config.location_hierarchy_path] = csv_payload.encode("utf-8")

    entries, by_gid = gis_lookup.load_location_catalog()
    assert "USA" in by_gid
    assert "USA.UT" in by_gid
    assert "AFRICA" in by_gid

    assert gis_lookup.strip_diacritics(" QuéBec ") == "quebec"
    assert gis_lookup.strip_diacritics("") == ""
    assert gis_lookup.search_locations("", 10) == []
    assert gis_lookup.search_locations("utah", 10)[0]["gid"] == "USA.UT"
    assert len(gis_lookup.search_locations("u", 1)) == 1

    direct_children = gis_lookup.list_children("USA", level=1, limit=10)
    assert [item["gid"] for item in direct_children] == ["USA.UT"]

    region_children = gis_lookup.list_children("africa", level=1, limit=1)
    assert [item["gid"] for item in region_children] == ["AFRICA.KE"]

    name_children = gis_lookup.list_children("united states", level=1, limit=1)
    assert [item["gid"] for item in name_children] == ["USA.UT"]

    super_fallback = gis_lookup.list_children("utah", level=2, limit=1)
    assert [item["gid"] for item in super_fallback] == ["USA.UT.001"]
    assert gis_lookup.list_children("", level=1, limit=10) == []


def test_location_gid_lookup_mask_and_context(stub_env):
    _config, _stub = stub_env
    assert not gis_lookup.is_valid_location_gid(None)
    assert not gis_lookup.is_valid_location_gid("  ")
    assert not gis_lookup.is_valid_location_gid("NaN")
    assert gis_lookup.is_valid_location_gid("USA")

    with pytest.raises(ValueError):
        gis_lookup.location_lookup_for_gid("null")
    assert gis_lookup.location_lookup_for_gid("EUROPE")[0] == "gbifRegion"
    assert gis_lookup.location_lookup_for_gid("USA")[1] == "country_scope"
    assert gis_lookup.location_lookup_for_gid("USA.UT")[1] == "state_scope"
    assert gis_lookup.location_lookup_for_gid("USA.UT.001")[1] == "county_scope"

    table = pa.table({"country": ["USA", "CAN"], "state": ["USA.UT", "CAN.AB"]})
    mask = gis_lookup.build_location_mask(table, "USA")
    assert mask is not None and mask.to_pylist() == [True, False]
    assert gis_lookup.build_location_mask(table, "null") is None
    assert gis_lookup.build_location_mask(pa.table({"x": [1]}), "USA") is None

    mapping = {
        "A": gis_lookup.LocationRecord(gid="A", name="Root", level=0, parent_gid=None),
        "A.B": gis_lookup.LocationRecord(gid="A.B", name="Child", level=1, parent_gid="A"),
        "A.B.C": gis_lookup.LocationRecord(gid="A.B.C", name="Leaf", level=2, parent_gid="A.B"),
        "A.X": gis_lookup.LocationRecord(gid="A.X", name="UnknownParent", level=1, parent_gid="missing"),
    }
    assert gis_lookup.resolve_location_context(mapping["A.B.C"], mapping) == ["Root", "Child"]
    assert gis_lookup.resolve_location_context(mapping["A.X"], mapping) == []


def test_location_taxa_membership_and_location_taxa_for(stub_env):
    config, stub = stub_env
    table = pa.table(
        {
            "scope": ["country_scope", "country_scope", "", "country_scope"],
            "gid": ["USA", "USA", "USA", "null"],
            "taxon_id": ["1", "x", "2", "3"],
            "count": [1, 1, 1, 1],
        }
    )
    monkeypatch_table = table
    # membership path uses _load_location_taxa_table helper
    gis_lookup._load_location_taxa_table.cache_clear()
    gis_lookup.location_taxa_membership.cache_clear()
    # direct monkeypatch keeps branch deterministic
    original = gis_lookup._load_location_taxa_table
    gis_lookup._load_location_taxa_table = lambda: monkeypatch_table  # type: ignore[assignment]
    try:
        membership = gis_lookup.location_taxa_membership()
        assert membership[("country_scope", "USA")] == frozenset({1})
    finally:
        gis_lookup._load_location_taxa_table = original  # type: ignore[assignment]
    gis_lookup.location_taxa_membership.cache_clear()
    gis_lookup._load_location_taxa_table = lambda: None  # type: ignore[assignment]
    try:
        assert gis_lookup.location_taxa_membership() == {}
    finally:
        gis_lookup._load_location_taxa_table = original  # type: ignore[assignment]

    gis_lookup.location_taxa_for.cache_clear()
    assert gis_lookup.location_taxa_for("country_scope", "USA") == frozenset()

    stub._exists[config.location_catalog_path] = True
    filtered = pa.table({"taxon_id": ["1", "2", "x"]})
    empty = pa.table({"taxon_id": []})
    stub._read_table = lambda _path, **kwargs: (
        filtered if kwargs.get("filters") else pa.table({"scope": ["country_scope"], "gid": ["USA"], "taxon_id": [5]})
    )
    gis_lookup.location_taxa_for.cache_clear()
    assert gis_lookup.location_taxa_for("country_scope", "USA") == frozenset({1, 2})
    assert gis_lookup.location_taxa_for("", "USA") == frozenset()
    assert gis_lookup.location_taxa_for("country_scope", "null") == frozenset()
    gis_lookup.location_taxa_for.cache_clear()
    stub._read_table = lambda _path, **kwargs: empty if kwargs.get("filters") else empty
    assert gis_lookup.location_taxa_for("country_scope", "USA") == frozenset()


def test_location_taxa_for_fallback_and_counts(stub_env):
    config, stub = stub_env
    stub._exists[config.location_catalog_path] = True

    calls = {"n": 0}

    def read_table_with_typeerror(_path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1 and "filters" in kwargs:
            raise TypeError("filters unsupported")
        if "columns" in kwargs and kwargs["columns"] == ["scope", "gid", "taxon_id"]:
            return pa.table({"scope": ["country_scope", "state_scope"], "gid": ["USA", "USA.UT"], "taxon_id": [10, 11]})
        return pa.table({"scope": ["country_scope"], "gid": ["USA"], "taxon_id": [10], "count": [2]})

    stub._read_table = read_table_with_typeerror
    assert gis_lookup.location_taxa_for("country_scope", "USA") == frozenset({10})

    gis_lookup._load_location_taxa_table.cache_clear()
    counts = gis_lookup.location_counts_for_taxon(10)
    assert counts[("country_scope", "USA")] == 2
    assert gis_lookup.location_counts_for_taxon("bad-id") == {}

    def read_table_fallback_fails(_path, **kwargs):
        if "filters" in kwargs:
            raise TypeError("filters unsupported")
        raise RuntimeError("fallback boom")

    stub._read_table = read_table_fallback_fails
    gis_lookup.location_taxa_for.cache_clear()
    assert gis_lookup.location_taxa_for("country_scope", "USA") == frozenset()

    stub._read_table = lambda _path, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    gis_lookup.location_taxa_for.cache_clear()
    assert gis_lookup.location_taxa_for("country_scope", "USA") == frozenset()


def test_load_location_taxa_table_branches(stub_env):
    config, stub = stub_env
    gis_lookup._load_location_taxa_table.cache_clear()
    assert gis_lookup._load_location_taxa_table() is None

    stub._exists[config.location_catalog_path] = True
    primary = pa.table({"scope": ["country_scope"], "gid": ["USA"], "taxon_id": [1], "count": [1]})
    fallback = pa.table({"scope": ["country_scope"], "gid": ["USA"], "taxon_id": [1]})
    calls = {"n": 0}

    def read_table_primary_then_fallback(_path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no count column")
        return fallback

    stub._read_table = lambda _path, **_kwargs: primary
    gis_lookup._load_location_taxa_table.cache_clear()
    assert gis_lookup._load_location_taxa_table().num_rows == 1

    stub._read_table = read_table_primary_then_fallback
    gis_lookup._load_location_taxa_table.cache_clear()
    assert gis_lookup._load_location_taxa_table().num_rows == 1

    stub._read_table = lambda _path, **_kwargs: (_ for _ in ()).throw(RuntimeError("all bad"))
    gis_lookup._load_location_taxa_table.cache_clear()
    assert gis_lookup._load_location_taxa_table() is None


def test_location_counts_without_count_column_and_invalid_rows(monkeypatch):
    table = pa.table(
        {
            "scope": ["country_scope", "country_scope", "", "country_scope"],
            "gid": ["USA", "USA", "USA", "null"],
            "taxon_id": [1, 1, 1, 1],
        }
    )
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: table)
    assert gis_lookup.location_counts_for_taxon(1) == {("country_scope", "USA"): 2}
    assert gis_lookup.location_counts_for_taxon(999) == {}
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: None)
    assert gis_lookup.location_counts_for_taxon(1) == {}

    broken = pa.table({"scope": ["x", "x"], "gid": ["Y", "Y"], "taxon_id": [1, 1], "count": ["bad", "0"]})
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: broken)
    assert gis_lookup.location_counts_for_taxon(1) == {("x", "Y"): 1}


def test_location_counts_filter_exception_returns_empty(monkeypatch):
    class _ExplodingTable:
        num_rows = 1

        def __getitem__(self, key):
            if key == "taxon_id":
                return pa.array([1])
            raise KeyError(key)

        def filter(self, _mask):
            raise RuntimeError("filter failed")

    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: _ExplodingTable())
    assert gis_lookup.location_counts_for_taxon(1) == {}


def test_list_children_super_fallback_and_no_matches(monkeypatch):
    records = [
        gis_lookup.LocationRecord(gid="A", name="Root", level=0, parent_gid=None),
        gis_lookup.LocationRecord(gid="A.B", name="Child", level=1, parent_gid="A"),
        gis_lookup.LocationRecord(gid="A.B.C", name="Leaf", level=2, parent_gid="A.B"),
    ]
    by_gid = {record.gid: record for record in records}
    monkeypatch.setattr(gis_lookup, "load_location_catalog", lambda: (records, by_gid))

    assert [item["gid"] for item in gis_lookup.list_children("root", level=2, limit=1)] == ["A.B.C"]
    assert gis_lookup.list_children("missing", level=2, limit=1) == []


def test_region_and_cog_path_helpers(stub_env, monkeypatch):
    config, _stub = stub_env
    assert gis_lookup._region_origin(-9.1) == -10
    assert gis_lookup.get_region_name(12.3, -25.7) == "lat10_lon-30"

    monkeypatch.setattr(gis_lookup, "_get_layer", lambda _layer_id: None)
    assert gis_lookup.get_cog_path("bio_1", 0, 0) is None

    monkeypatch.setattr(gis_lookup, "_get_layer", lambda _layer_id: {"filename_template": "{id}.tif"})
    with pytest.raises(KeyError):
        gis_lookup.get_cog_path("bio_1", 0, 0)

    monkeypatch.setattr(
        gis_lookup, "_get_layer", lambda _layer_id: {"region_root": "regions", "filename_template": "{id}.tif"}
    )
    expected = config.gis_root / "regions" / "lat0_lon0" / "bio_1.tif"
    assert gis_lookup.get_cog_path("bio_1", 0.4, 0.2) == expected


def test_location_taxon_counts_rollup_and_rollup_failure(monkeypatch):
    table = pa.table(
        {
            "scope": ["country_scope", "country_scope", "country_scope", "country_scope"],
            "gid": ["USA", "USA", "USA", "USA"],
            "taxon_id": ["20", "30", "bad", "40"],
            "count": [2, 3, 1, 0],
        }
    )
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: table)
    monkeypatch.setattr(gis_lookup, "is_valid_location_gid", lambda _gid: True)

    from util import taxa_navigation

    subspecies = {"taxon_key": "20", "rank": "SUBSPECIES"}
    genus = {"taxon_key": "11", "rank": "GENUS"}
    species = {"taxon_key": "10", "rank": "SPECIES"}
    species_30 = {"taxon_key": "30", "rank": "SPECIES"}
    by_key = {"20": subspecies, "30": species_30}
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda key: by_key.get(str(key)))
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(
        taxa_navigation,
        "get_parent_taxon",
        lambda taxon: genus if taxon is subspecies else (species if taxon is genus else None),
    )
    monkeypatch.setattr(taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))

    counts = gis_lookup.location_taxon_counts("country_scope", "USA", include_species_rollup=True)
    assert counts[20] == 2
    assert counts[30] == 3
    assert counts[10] == 2

    ancestor_counts = gis_lookup.location_taxon_counts(
        "country_scope",
        "USA",
        include_ancestor_rollup=True,
    )
    assert ancestor_counts[20] == 2
    assert ancestor_counts[10] == 2
    assert ancestor_counts[11] == 2

    gis_lookup.location_taxa_for.cache_clear()
    assert gis_lookup.location_taxa_for(
        "country_scope",
        "USA",
        include_ancestor_rollup=True,
    ) == frozenset({10, 11, 20, 30})

    # Rollup exceptions are swallowed and base mapping still returns.
    monkeypatch.setattr(
        taxa_navigation,
        "get_taxon_by_id",
        lambda _key: (_ for _ in ()).throw(RuntimeError("rollup failed")),
    )
    fallback = gis_lookup.location_taxon_counts("country_scope", "USA", include_species_rollup=True)
    assert fallback[20] == 2 and fallback[30] == 3
    ancestor_fallback = gis_lookup.location_taxon_counts("country_scope", "USA", include_ancestor_rollup=True)
    assert ancestor_fallback[20] == 2 and ancestor_fallback[30] == 3


def test_location_counts_for_taxon_species_rollup_and_fallback(monkeypatch):
    table = pa.table(
        {
            "scope": ["country_scope", "country_scope", "country_scope"],
            "gid": ["USA", "USA", "USA"],
            "taxon_id": [1, 2, 3],
            "count": [1, 1, 1],
        }
    )
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: table)
    monkeypatch.setattr(gis_lookup, "is_valid_location_gid", lambda _gid: True)

    from util import taxa_navigation

    species = {"taxon_key": "1", "rank": "SPECIES"}
    sub = {"taxon_key": "2", "rank": "SUBSPECIES"}
    genus = {"taxon_key": "3", "rank": "GENUS"}
    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda key: species if str(key) == "1" else None)
    monkeypatch.setattr(taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(taxa_navigation, "iter_descendants", lambda _taxon: [sub, genus])
    monkeypatch.setattr(taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    out = gis_lookup.location_counts_for_taxon(1)
    assert out[("country_scope", "USA")] == 2

    monkeypatch.setattr(
        taxa_navigation,
        "iter_descendants",
        lambda _taxon: (_ for _ in ()).throw(RuntimeError("taxonomy exploded")),
    )
    fallback = gis_lookup.location_counts_for_taxon(1)
    assert fallback[("country_scope", "USA")] == 1


def test_location_taxon_counts_remaining_branches(monkeypatch):
    gis_lookup.location_taxon_counts.cache_clear()
    assert gis_lookup.location_taxon_counts("", "USA") == {}

    monkeypatch.setattr(gis_lookup, "is_valid_location_gid", lambda _gid: True)
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: None)
    gis_lookup.location_taxon_counts.cache_clear()
    assert gis_lookup.location_taxon_counts("country_scope", "USA") == {}

    class _BadTable:
        num_rows = 1

        def __getitem__(self, _key):
            raise RuntimeError("boom")

    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: _BadTable())
    gis_lookup.location_taxon_counts.cache_clear()
    assert gis_lookup.location_taxon_counts("country_scope", "USA") == {}

    table = pa.table({"scope": ["country_scope"], "gid": ["USA"], "taxon_id": [1], "count": ["bad"]})
    monkeypatch.setattr(gis_lookup, "_load_location_taxa_table", lambda: table)
    gis_lookup.location_taxon_counts.cache_clear()
    assert gis_lookup.location_taxon_counts("country_scope", "USA") == {1: 1}

    from util import taxa_navigation

    monkeypatch.setattr(taxa_navigation, "get_taxon_by_id", lambda _k: None)
    gis_lookup.location_taxon_counts.cache_clear()
    assert gis_lookup.location_taxon_counts("country_scope", "USA", include_species_rollup=True) == {1: 1}

    monkeypatch.setattr(
        taxa_navigation,
        "get_taxon_by_id",
        lambda _k: (_ for _ in ()).throw(RuntimeError("rollup")),
    )
    gis_lookup.location_taxon_counts.cache_clear()
    assert gis_lookup.location_taxon_counts("country_scope", "USA", include_species_rollup=True) == {1: 1}


def test_get_layer_tile_info_success_and_errors(stub_env, monkeypatch, tmp_path):
    config, _stub = stub_env
    config.gis_root = tmp_path / "gis"
    regions = config.gis_root / "regions"
    regions.mkdir(parents=True, exist_ok=True)
    (regions / "lat0_lon0").mkdir(parents=True, exist_ok=True)
    sample_path = regions / "lat0_lon0" / "bio_1.tif"
    sample_path.write_bytes(b"x")

    monkeypatch.setattr(
        gis_lookup, "_get_layer", lambda _layer_id: {"region_root": "regions", "filename_template": "{id}.tif"}
    )

    class _Dataset:
        bounds = SimpleNamespace(top=10, bottom=0, right=20, left=0)
        transform = SimpleNamespace(e=-0.5, a=0.5)
        block_shapes = [(4, 8)]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    sys.modules["rasterio"] = SimpleNamespace(open=lambda _path: _Dataset())
    info = gis_lookup.get_layer_tile_info("bio_1")
    assert info["region_span_lat"] == 10
    assert info["pixel_size_lon"] == 0.5
    assert info["block_shape"] == (4, 8)

    gis_lookup.get_layer_tile_info.cache_clear()
    monkeypatch.setattr(gis_lookup, "_get_layer", lambda _layer_id: None)
    with pytest.raises(ValueError):
        gis_lookup.get_layer_tile_info("bio_1")

    gis_lookup.get_layer_tile_info.cache_clear()
    monkeypatch.setattr(
        gis_lookup, "_get_layer", lambda _layer_id: {"region_root": "empty", "filename_template": "{id}.tif"}
    )
    (config.gis_root / "empty").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        gis_lookup.get_layer_tile_info("bio_1")

    gis_lookup.get_layer_tile_info.cache_clear()
    monkeypatch.setattr(
        gis_lookup, "_get_layer", lambda _layer_id: {"region_root": "regions", "filename_template": "{id}.missing"}
    )
    with pytest.raises(FileNotFoundError):
        gis_lookup.get_layer_tile_info("bio_1")

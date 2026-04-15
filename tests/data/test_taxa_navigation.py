"""Unit tests for util.taxa_navigation."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from util import taxa_navigation as nav
from util.request_cancellation import RequestCancelledError


class _StubParquet:
    def __init__(self):
        self._exists = {}
        self._files = {}
        self._tables = {}
        self._metadata = {}
        self.is_remote = False

    def exists(self, path):
        return self._exists.get(Path(path), False)

    def open_input_file(self, path):
        return io.BytesIO(self._files[Path(path)])

    def read_metadata(self, path):
        return self._metadata[Path(path)]

    def read_table(self, path, **_kwargs):
        value = self._tables.get(Path(path))
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(path, **_kwargs)
        return value


@pytest.fixture(autouse=True)
def _clear_caches():
    nav.resolve_taxon_media.cache_clear()
    nav._load_payload.cache_clear()
    nav.load_catalog.cache_clear()
    nav.load_name_index.cache_clear()
    nav.load_search_names_by_taxon.cache_clear()
    nav.load_slug_index.cache_clear()
    nav._normalized_taxon_record.cache_clear()
    nav.load_taxon_media.cache_clear()
    nav.resolve_preferred_image_taxon_key.cache_clear()
    nav._child_index.cache_clear()
    yield


@pytest.fixture
def stub_env(monkeypatch, tmp_path):
    taxonomy_root = tmp_path / "taxonomy"
    taxonomy_root.mkdir()
    cfg = SimpleNamespace(
        taxonomy_root=taxonomy_root,
        taxon_catalog_path=tmp_path / "taxon_catalog.pkl",
        taxon_media_path=tmp_path / "taxon_media.pkl",
        occurrence_parquet_filename="occurrence.parquet",
        subspecies_equivalents={"SUBSPECIES", "VARIETY"},
        rank_synonyms={"SPECIES": {"SPECIES", "SP"}, "GENUS": {"GENUS"}},
        common_name_language="en",
    )
    stub = _StubParquet()
    monkeypatch.setattr(nav, "CONFIG", cfg)
    monkeypatch.setattr(nav, "PARQUET", stub)
    return cfg, stub


def _catalog_payload(cfg: SimpleNamespace):
    root = cfg.taxonomy_root
    g1 = root / "genus_10"
    s1 = g1 / "species_11"
    ss1 = s1 / "sub_12"
    return {
        "catalog": {
            "10": {
                "taxon_key": "10",
                "path": g1,
                "scientific_name": "Alpha_genus",
                "common_name": "alpha",
                "rank": "GENUS",
            },
            "11": {
                "taxon_key": "11",
                "path": s1,
                "scientific_name": "Alpha_species",
                "common_name": [{"name": "alpha bird", "language": "en", "lexicon": "english", "source": "inat"}],
                "rank": "SPECIES",
                "inat_preferred_image": "http://img/species.jpg",
            },
            "12": {
                "taxon_key": "12",
                "path": ss1,
                "scientific_name": "Alpha_sub",
                "common_name": [],
                "rank": "SUBSPECIES",
            },
        },
        "combined_name_index": {"alpha genus": ["10"]},
    }


def test_normalization_and_parent_helpers(stub_env, monkeypatch, tmp_path):
    cfg, _stub = stub_env
    abs_under_tax = cfg.taxonomy_root / "x" / "species_1"
    assert nav.normalize_taxon_path("x/species_1") == abs_under_tax
    assert nav.normalize_taxon_path(abs_under_tax) == abs_under_tax
    external = tmp_path / "outside" / "taxonomy" / "a" / "species_2"
    assert nav.normalize_taxon_path(external) == cfg.taxonomy_root / "a" / "species_2"

    assert nav.normalize_name("  Alpha__Bird  ") == "alpha bird"
    assert nav.normalize_name("") == ""

    called = {}
    monkeypatch.setattr(nav, "taxon_key_from_path", lambda p: called.setdefault("k", p.name) or "10")
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "10"})
    assert nav.get_parent_taxon({"path": cfg.taxonomy_root / "genus_1" / "species_2"}) == {"taxon_key": "10"}
    assert nav.get_parent_taxon({"path": ""}) is None
    assert nav.get_parent_taxon({"path": cfg.taxonomy_root}) is None
    assert nav.get_parent_taxon({"path": cfg.taxonomy_root / "genus_1"}) is None


def test_common_name_helpers_and_extract_language_fallback(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    assert nav._common_name_score("en", "english united states", "inat") > nav._common_name_score("fr", "", "")
    assert nav._matches_language("", "english", "en")
    assert not nav._matches_language("fr", "", "en")
    assert nav._format_common_name("USA oak") == "USA Oak"
    assert nav._format_common_name("o'connor tree") == "O'connor Tree"
    assert nav._format_common_name("") == ""

    t1 = {
        "common_name": [
            {"name": "chene", "language": "fr", "lexicon": "french", "source": "x"},
            {"name": "oak", "language": "en", "lexicon": "english", "source": "inat"},
        ],
        "inat_preferred_common_name": "red oak",
    }
    assert nav._extract_common_names(t1, "en")[0] == "Red Oak"
    assert "Oak" in nav._extract_common_names(t1, "en")
    t1b = {
        "common_name": [
            {"name": "oak", "language": "en", "lexicon": "english", "source": "inat"},
            123,
            {"name": "", "language": "en"},
        ],
        "inat_preferred_common_name": "oak",
    }
    assert nav._extract_common_names(t1b, "en")[0] == "Oak"

    t2 = {"common_name": ["alpha", " beta "], "inat_preferred_common_name": "alpha"}
    assert nav._extract_common_names(t2, "en")[0] == "Alpha"
    t2b = {"common_name": ["beta"], "inat_preferred_common_name": "alpha"}
    assert nav._extract_common_names(t2b, "en")[0] == "Alpha"
    t3 = {"common_name": "wolf, gray fox"}
    assert nav._extract_common_names(t3, None) == ["Wolf", "Gray Fox"]
    t3b = {"common_name": "wolf, gray fox", "inat_preferred_common_name": "alpha"}
    assert nav._extract_common_names(t3b, "en")[0] == "Alpha"
    assert nav._extract_common_names({"common_name": None}, None) == []

    parent = {"common_name": [{"name": "Parent Name", "language": "en", "lexicon": "english", "source": "inat"}]}
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: parent)
    names = nav.extract_common_names_for_language({"rank": "subspecies", "common_name": []}, "en")
    assert names == ["Parent Name"]
    assert nav.extract_common_names_for_language({"rank": "species", "common_name": "alpha"}, "en") == ["Alpha"]


def test_catalog_payload_and_lookup_paths(stub_env):
    cfg, stub = stub_env
    payload = _catalog_payload(cfg)
    stub._files[cfg.taxon_catalog_path] = pickle.dumps(payload)
    stub._exists[cfg.taxon_media_path] = True
    stub._files[cfg.taxon_media_path] = pickle.dumps({"11": {"url": "u"}})

    catalog = nav.load_catalog()
    assert "10" in catalog
    idx = nav.load_name_index()
    assert "alpha bird" in idx and "alpha species" in idx
    assert nav.get_taxon_by_id("10")["taxon_key"] == "10"
    assert nav.get_taxon_by_id("bad") is None
    assert nav.load_taxon_media()["11"]["url"] == "u"

    stub._exists[cfg.taxon_media_path] = False
    nav.load_taxon_media.cache_clear()
    assert nav.load_taxon_media() == {}

    # int-key fallback branch and empty-name skip in load_name_index.
    nav.load_catalog.cache_clear()
    nav._load_payload.cache_clear()
    nav._normalized_taxon_record.cache_clear()
    nav.load_name_index.cache_clear()
    nav.load_slug_index.cache_clear()
    payload2 = {
        "catalog": {
            20: {
                "taxon_key": "20",
                "path": cfg.taxonomy_root / "g_20",
                "scientific_name": " ",
                "common_name": ["", "x"],
                "rank": "SPECIES",
            },
        },
        "combined_name_index": {},
    }
    stub._files[cfg.taxon_catalog_path] = pickle.dumps(payload2)
    assert nav.get_taxon_by_id("20")["taxon_key"] == "20"
    assert nav.get_taxon_by_id(20)["taxon_key"] == "20"
    assert "x" in nav.load_name_index()


def test_resolve_taxon_reference_supports_slug_only(stub_env):
    cfg, stub = stub_env
    payload = _catalog_payload(cfg)
    stub._files[cfg.taxon_catalog_path] = pickle.dumps(payload)

    by_slug = nav.resolve_taxon_reference("alpha-species")
    by_name = nav.resolve_taxon_reference("alpha genus")

    assert by_slug is not None
    assert by_slug["taxon_key"] == "11"
    assert by_name is None


def test_resolve_taxon_reference_rejects_ambiguous_slug(stub_env):
    cfg, stub = stub_env
    payload = _catalog_payload(cfg)
    payload["catalog"]["21"] = {
        "taxon_key": "21",
        "path": cfg.taxonomy_root / "other_21",
        "scientific_name": "Alpha species",
        "common_name": "other alpha",
        "rank": "SPECIES",
    }
    stub._files[cfg.taxon_catalog_path] = pickle.dumps(payload)

    with pytest.raises(ValueError, match="Ambiguous taxon slug"):
        nav.resolve_taxon_reference("alpha-species")


def test_children_descendants_and_rank_iteration(stub_env, monkeypatch):
    cfg, _stub = stub_env
    root = {"taxon_key": "10", "path": cfg.taxonomy_root / "genus_10", "rank": "GENUS"}
    child = {"taxon_key": "11", "path": cfg.taxonomy_root / "genus_10" / "species_11", "rank": "SPECIES"}
    leaf = {"taxon_key": "12", "path": cfg.taxonomy_root / "genus_10" / "species_11" / "sub_12", "rank": "SUBSPECIES"}
    mapping = {"10": [child], "11": [leaf], "12": []}
    monkeypatch.setattr(nav, "get_children", lambda key: mapping.get(str(key), []))
    assert [t["taxon_key"] for t in nav.iter_descendants_dfs(root)] == ["11", "12"]
    assert [t["taxon_key"] for t in nav.iter_descendants(root, include_self=False)] == ["11", "12"]
    assert [t["taxon_key"] for t in nav.iter_descendants(root, include_self=True)] == ["10", "11", "12"]
    assert [t["taxon_key"] for t in nav.iter_descendants_by_rank(root, "species", include_self=True)] == ["11", "11"]
    assert nav.iter_descendants_by_rank(root, "", include_self=True) == []
    assert nav.canonical_rank("sp") == "SPECIES"
    assert nav.canonical_rank("other") == "OTHER"
    assert nav.canonical_rank(None) == ""
    assert nav.is_valid_descendant_rank("sp")
    assert nav.is_valid_descendant_rank("ORDER")
    assert not nav.is_valid_descendant_rank("other")


def test_search_taxa_by_name_and_limits(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    monkeypatch.setattr(
        nav, "load_name_index", lambda: {"alpha bird": ["11"], "alpha species": ["11"], "beta fox": ["20"]}
    )
    monkeypatch.setattr(
        nav.process,
        "extract",
        lambda *_a, **_k: [("alpha bird", 90, 0), ("alpha species", 80, 0), ("beta fox", 75, 0)],
    )
    monkeypatch.setattr(
        nav,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "rank": "SPECIES"} if str(key) in {"11", "20"} else None,
    )
    out = nav.search_taxa_by_name("alpha", limit=2)
    assert out and out[0][0]["taxon_key"] == "11"
    assert nav.search_taxa_by_name("  ", limit=2) == []

    monkeypatch.setattr(nav, "load_name_index", lambda: {})
    assert nav.search_taxa_by_name("alpha", limit=2) == []

    monkeypatch.setattr(nav, "load_name_index", lambda: {"alpha bird": ["11"], "alpha species": ["missing"]})
    monkeypatch.setattr(
        nav.process,
        "extract",
        lambda *_a, **_k: [("alpha bird", 65, 0), ("alpha species", 40, 0), ("other", 99, 0)],
    )
    monkeypatch.setattr(
        nav, "get_taxon_by_id", lambda key: {"taxon_key": "11", "rank": "SPECIES"} if str(key) == "11" else None
    )
    out2 = nav.search_taxa_by_name("alpha bird", limit=3)
    assert len(out2) == 1 and out2[0][0]["taxon_key"] == "11"

    # Single-token candidate that fails token-prefix check.
    monkeypatch.setattr(nav, "load_name_index", lambda: {"zebra fish": ["11"]})
    monkeypatch.setattr(nav.process, "extract", lambda *_a, **_k: [("zebra fish", 90, 0)])
    assert nav.search_taxa_by_name("alp", limit=3) == []

    # low-score and taxon-missing branches
    monkeypatch.setattr(nav, "load_name_index", lambda: {"alpha": ["404"]})
    monkeypatch.setattr(nav.process, "extract", lambda *_a, **_k: [("alpha", 10, 0), ("alpha", 80, 0)])
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: None)
    assert nav.search_taxa_by_name("alpha", limit=3) == []


def test_search_taxa_by_name_none_limit_uses_index_size(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    name_index = {
        "alpha bird": ["11"],
        "alpha species": ["12"],
        "beta fox": ["20"],
    }
    seen = {"used_extract_iter": False}

    def _extract_iter(*_args, **_kwargs):
        seen["used_extract_iter"] = True
        return iter([("alpha bird", 90, 0), ("alpha species", 80, 0), ("beta fox", 75, 0)])

    monkeypatch.setattr(nav, "load_name_index", lambda: name_index)
    monkeypatch.setattr(nav.process, "extract_iter", _extract_iter)
    monkeypatch.setattr(
        nav,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "rank": "SPECIES"} if str(key) in {"11", "12", "20"} else None,
    )

    out = nav.search_taxa_by_name("alpha", limit=None)

    assert seen["used_extract_iter"] is True
    assert [row[0]["taxon_key"] for row in out] == ["11", "12"]


def test_taxon_name_match_score_uses_aliases_from_combined_name_index(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    monkeypatch.setattr(nav, "load_name_index", lambda: {"hidden alias": ["11"]})
    nav.load_search_names_by_taxon.cache_clear()
    monkeypatch.setattr(nav.fuzz, "token_set_ratio", lambda *_a, **_k: 95.0)

    score = nav.taxon_name_match_score(
        {"taxon_key": "11", "scientific_name": "Alpha species", "common_name": "oak", "rank": "SPECIES"},
        "hidden",
    )

    assert score is not None
    assert score > 0


def test_search_taxa_by_name_honors_cancellation(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    monkeypatch.setattr(nav, "load_name_index", lambda: {"alpha bird": ["11"], "alpha species": ["12"]})
    monkeypatch.setattr(nav.process, "extract", lambda *_a, **_k: [("alpha bird", 95, 0), ("alpha species", 90, 1)])
    monkeypatch.setattr(
        nav,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "rank": "SPECIES"} if str(key) in {"11", "12"} else None,
    )

    calls = {"count": 0}

    def cancel_check():
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RequestCancelledError("Client disconnected")

    with pytest.raises(RequestCancelledError, match="Client disconnected"):
        nav.search_taxa_by_name("alpha", limit=10, cancel_check=cancel_check)


def test_search_taxa_by_name_matches_reordered_multiword_queries(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    monkeypatch.setattr(nav, "load_name_index", lambda: {"red oak": ["11"]})
    monkeypatch.setattr(nav.process, "extract", lambda *_a, **_k: [("red oak", 100, 0)])
    monkeypatch.setattr(
        nav,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "rank": "SPECIES"} if str(key) == "11" else None,
    )

    out = nav.search_taxa_by_name("oak red", limit=3)

    assert len(out) == 1
    assert out[0][0]["taxon_key"] == "11"


def test_search_taxa_by_name_prefers_exact_short_match(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    monkeypatch.setattr(nav, "load_name_index", lambda: {"oak": ["11"], "oak tree": ["12"]})
    monkeypatch.setattr(nav.process, "extract", lambda *_a, **_k: [("oak tree", 90, 0), ("oak", 80, 1)])
    monkeypatch.setattr(
        nav,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "rank": "SPECIES"} if str(key) in {"11", "12"} else None,
    )

    out = nav.search_taxa_by_name("oak", limit=2)

    assert len(out) == 2
    assert out[0][0]["taxon_key"] == "11"
    assert out[0][1] > out[1][1]


def test_resolve_taxon_media_and_preferred_image_payload(stub_env, monkeypatch):
    _cfg, _stub = stub_env
    root = {"taxon_key": "10", "rank": "GENUS"}
    species = {
        "taxon_key": "11",
        "rank": "SPECIES",
        "inat_preferred_image": "http://img/spec.jpg",
        "inat_preferred_image_creator": "Creator",
        "inat_preferred_image_license": "CC-BY",
        "inat_preferred_image_attribution": "Attrib",
        "inat_preferred_image_references": "Ref",
    }
    child = {"taxon_key": "12", "rank": "SUBSPECIES"}
    by_id = {"10": root, "11": species, "12": child}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda key: by_id.get(str(key)))
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"11": {"url": "media11"}, "12": {"url": "media12"}})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [child, species])
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: {"taxon_key": "10"})
    monkeypatch.setattr(nav, "get_children", lambda _key: [species, child])

    assert nav.resolve_taxon_media("10") == {"url": "media11"}
    assert nav.resolve_taxon_media("999") is None
    assert nav.preferred_image_url(species) == "http://img/spec.jpg"
    assert nav.preferred_image_url(None) is None
    assert nav.resolve_preferred_image_taxon_key("10") == "11"
    payload = nav.preferred_image_payload(root)
    assert payload["image_url"] == "http://img/spec.jpg"
    assert payload["image_creator"] == "Creator"
    assert payload["image_references"] == "Ref"
    assert nav.preferred_image_payload(None) == {}

    # payload fallback to attribution for creator/rights-holder
    species2 = {"taxon_key": "13", "inat_preferred_image": "u2", "inat_preferred_image_attribution": "Attrib Only"}
    original_resolve_image_key = nav.resolve_preferred_image_taxon_key
    monkeypatch.setattr(nav, "resolve_preferred_image_taxon_key", lambda _k: "13")
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: species2)
    p2 = nav.preferred_image_payload({"taxon_key": "10"})
    assert p2["image_creator"] == "Attrib Only"
    assert p2["image_rights_holder"] == "Attrib Only"

    monkeypatch.setattr(nav, "resolve_preferred_image_taxon_key", lambda _k: None)
    assert nav.preferred_image_payload({"taxon_key": "10"}) == {}
    monkeypatch.setattr(nav, "resolve_preferred_image_taxon_key", original_resolve_image_key)

    # no descendant records -> continue branch + first-descendant fallback
    sparse = {"taxon_key": "30", "rank": "GENUS"}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: sparse)
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"31": {"url": "d1"}})
    monkeypatch.setattr(
        nav,
        "iter_descendants_dfs",
        lambda _t: [{"taxon_key": "32", "rank": "GENUS"}, {"taxon_key": "31", "rank": "GENUS"}],
    )
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: None)
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("30") == {"url": "d1"}

    # sibling fallback path
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"40": {"url": "sib"}})
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "35", "rank": "GENUS"})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [])
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: {"taxon_key": "p"})
    monkeypatch.setattr(
        nav, "get_children", lambda _k: [{"taxon_key": "35", "rank": "GENUS"}, {"taxon_key": "40", "rank": "GENUS"}]
    )
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("35") == {"url": "sib"}

    # direct fallback branch when species/lower has direct media but no descendants
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "50", "rank": "SPECIES"})
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"50": {"url": "direct-spec"}})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [])
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: None)
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("50") == {"url": "direct-spec"}

    # sibling-descendant fallback branch
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "60", "rank": "GENUS"})
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"62": {"url": "from-desc"}})
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: {"taxon_key": "p"})
    monkeypatch.setattr(
        nav, "get_children", lambda _k: [{"taxon_key": "61", "rank": "GENUS"}, {"taxon_key": "60", "rank": "GENUS"}]
    )
    monkeypatch.setattr(
        nav,
        "iter_descendants_dfs",
        lambda t: [] if str(t.get("taxon_key")) == "60" else [{"taxon_key": "62", "rank": "SPECIES"}],
    )
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("60") == {"url": "from-desc"}

    # direct-media fallback for higher rank
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "70", "rank": "GENUS"})
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"70": {"url": "direct-genus"}})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [])
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: None)
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("70") == {"url": "direct-genus"}

    # preferred-image key helper branches
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: None)
    nav.resolve_preferred_image_taxon_key.cache_clear()
    assert nav.resolve_preferred_image_taxon_key("x") is None
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "41", "inat_preferred_image": "u"})
    nav.resolve_preferred_image_taxon_key.cache_clear()
    assert nav.resolve_preferred_image_taxon_key("41") == "41"
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "42"})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [])
    nav.resolve_preferred_image_taxon_key.cache_clear()
    assert nav.resolve_preferred_image_taxon_key("42") is None


def test_count_serialize_and_occurrence_filters(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env
    taxon = {
        "taxon_key": "11",
        "path": tmp_path / "genus_10/species_11",
        "scientific_name": "Alpha_species",
        "rank": "SPECIES",
    }
    occ_path = Path(taxon["path"]) / cfg.occurrence_parquet_filename
    stub._exists[occ_path] = True
    stub._metadata[occ_path] = SimpleNamespace(num_rows=7)
    assert nav.count_taxon_rows(taxon) == 7
    stub._exists[occ_path] = False
    assert nav.count_taxon_rows(taxon) is None

    monkeypatch.setattr(nav, "extract_common_names_for_language", lambda *_a, **_k: ["Alpha"])
    monkeypatch.setattr(nav, "preferred_image_payload", lambda _t: {"image_url": "u"})
    monkeypatch.setattr(nav, "resolve_taxon_media", lambda _k: None)
    serialized = nav.serialize_taxon(taxon)
    assert serialized["taxon_id"] == 11
    assert serialized["common_name"] == "Alpha"
    assert serialized["image_url"] == "u"
    assert nav.serialize_taxon({"taxon_key": "x"}) is None

    # media fallback when preferred image payload is empty
    monkeypatch.setattr(nav, "preferred_image_payload", lambda _t: {})
    monkeypatch.setattr(
        nav,
        "resolve_taxon_media",
        lambda _k: {
            "url": "mu",
            "license": "ml",
            "creator": "mc",
            "rightsHolder": "mr",
            "references": ["mref", "ignored"],
        },
    )
    serialized2 = nav.serialize_taxon(taxon)
    assert serialized2["image_url"] == "mu"
    assert serialized2["image_license"] == "ml"
    assert serialized2["image_references"] == "mref"

    table = pa.table(
        {
            "obscured": ["No", "No", "Yes"],
            "coordinateUncertaintyInMeters": [10, 900, 10],
            "decimalLatitude": [1.0, 2.0, None],
            "decimalLongitude": [1.0, 2.0, 3.0],
        }
    )
    mask = nav.base_observation_mask(table)
    assert mask.to_pylist() == [True, False, False]

    root = {"taxon_key": "10", "path": tmp_path / "taxa/10"}
    child = {"taxon_key": "11", "path": tmp_path / "taxa/11"}
    for p in (Path(root["path"]), Path(child["path"])):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        nav, "get_taxon_by_id", lambda key: root if str(key) == "10" else (child if str(key) == "11" else None)
    )
    monkeypatch.setattr(nav, "iter_descendants", lambda _t, include_self=True: [root, child])
    monkeypatch.setattr(nav.gis_lookup, "build_location_mask", lambda t, _g: pa.array([True] * t.num_rows))

    t_ok = pa.table(
        {
            "catalogNumber": ["a", "a", "b"],
            "decimalLatitude": [1.0, 1.0, 2.0],
            "decimalLongitude": [3.0, 3.0, 4.0],
            "obscured": ["No", "No", "No"],
            "coordinateUncertaintyInMeters": [10, 10, 10],
            "level0Gid": ["USA", "USA", "USA"],
            "level1Gid": ["USA.UT", "USA.UT", "USA.UT"],
            "level2Gid": ["USA.UT.001", "USA.UT.001", "USA.UT.001"],
            "gbifRegion": ["EUROPE", "EUROPE", "EUROPE"],
        }
    )
    stub._tables[Path(root["path"]) / cfg.occurrence_parquet_filename] = t_ok
    stub._exists[Path(root["path"]) / cfg.occurrence_parquet_filename] = True
    stub._tables[Path(child["path"]) / nav.combined_parquet_filename] = Exception("bad read")
    stub._exists[Path(child["path"]) / nav.combined_parquet_filename] = True

    chunks = list(nav.iter_filtered_occurrence_tables(10, extra_columns=("x",), location_gid="USA"))
    assert len(chunks) == 1 and chunks[0].num_rows == 3
    points = nav.load_occurrence_points(10, location_gid="USA")
    assert points == [
        {"catalogNumber": "a", "latitude": 1.0, "longitude": 3.0},
        {"catalogNumber": "b", "latitude": 2.0, "longitude": 4.0},
    ]

    # taxon not found branch
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: None)
    assert list(nav.iter_filtered_occurrence_tables(999)) == []


def test_misc_remaining_branches(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env
    # normalize absolute path without taxonomy segment
    outside = tmp_path / "plain" / "x"
    assert nav.normalize_taxon_path(outside) == outside

    assert nav._matches_language("fr", "", "") is True
    assert nav._format_common_name("'tis tree") == "'tis Tree"

    # preferred duplicate ordering branch in comma-string path
    names = nav._extract_common_names({"common_name": "alpha, beta", "inat_preferred_common_name": "alpha"}, "en")
    assert names[0] == "Alpha"

    # no fallback common names for empty subspecies data
    assert nav.extract_common_names_for_language({"rank": "SUBSPECIES", "common_name": []}, "en") == []

    # resolve_taxon_media fallback paths
    taxon = {"taxon_key": "20", "rank": "SPECIES"}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: taxon)
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {"20": {"url": "direct"}})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [])
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: None)
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("20") == {"url": "direct"}

    # iter_filtered location mask none and path-missing skip
    t = {"taxon_key": "1", "path": tmp_path / "missing"}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: t)
    monkeypatch.setattr(nav, "iter_descendants", lambda *_a, **_k: [t])
    monkeypatch.setattr(nav.gis_lookup, "build_location_mask", lambda *_a, **_k: None)
    assert list(nav.iter_filtered_occurrence_tables(1, location_gid="USA")) == []

    # hit loc_mask None continue with existing table.
    p2 = tmp_path / "hasdata2"
    p2.mkdir(parents=True, exist_ok=True)
    t_exist = {"taxon_key": "3", "path": p2}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: t_exist)
    monkeypatch.setattr(nav, "iter_descendants", lambda *_a, **_k: [t_exist])
    ok_table = pa.table(
        {
            "catalogNumber": ["c"],
            "decimalLatitude": [1.0],
            "decimalLongitude": [1.0],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [1],
            "level0Gid": ["A"],
            "level1Gid": ["A.B"],
            "level2Gid": ["A.B.C"],
            "gbifRegion": ["EUROPE"],
        }
    )
    occ2 = p2 / cfg.occurrence_parquet_filename
    stub._exists[occ2] = True
    stub._tables[occ2] = ok_table
    monkeypatch.setattr(nav.gis_lookup, "build_location_mask", lambda *_a, **_k: None)
    assert list(nav.iter_filtered_occurrence_tables(3, location_gid="USA")) == []

    # invalid lat/lon point skipped
    p = tmp_path / "hasdata"
    p.mkdir(parents=True, exist_ok=True)
    t2 = {"taxon_key": "2", "path": p}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: t2)
    monkeypatch.setattr(nav, "iter_descendants", lambda *_a, **_k: [t2])
    bad = pa.table(
        {
            "catalogNumber": ["x"],
            "decimalLatitude": ["bad"],
            "decimalLongitude": [1.0],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [1],
            "level0Gid": ["A"],
            "level1Gid": ["A.B"],
            "level2Gid": ["A.B.C"],
            "gbifRegion": ["EUROPE"],
        }
    )
    occ = p / cfg.occurrence_parquet_filename
    stub._exists[occ] = True
    stub._tables[occ] = bad
    assert nav.load_occurrence_points(2) == []

    # taxon_id_as_int falsey branch
    assert nav.taxon_id_as_int("") is None


def test_child_index_and_get_children_branches(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env
    root = cfg.taxonomy_root / "genus_1"
    child_dir = root / "species_2"
    root.mkdir(parents=True, exist_ok=True)
    child_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "catalog": {
            "1": {"taxon_key": "1", "path": root, "rank": "GENUS"},
            "2": {"taxon_key": "2", "path": child_dir, "rank": "SPECIES"},
            "3": {"taxon_key": "3", "path": "", "rank": "SPECIES"},
        },
        "combined_name_index": {},
    }
    stub._files[cfg.taxon_catalog_path] = pickle.dumps(payload)

    assert nav._child_index()["1"] == ["2"]
    assert nav.taxon_key_from_path(Path("a/b/species_2")) == "2"
    assert nav.taxon_key_from_path(Path("plain")) == "plain"

    kids = nav.get_children("1")
    assert [k["taxon_key"] for k in kids] == ["2"]
    assert nav.get_children("999") == []

    nav._child_index.cache_clear()
    monkeypatch.setattr(nav, "_child_index", lambda: {})
    stub.is_remote = True
    assert nav.get_children("1") == []
    stub.is_remote = False
    missing_parent = {"taxon_key": "100", "path": tmp_path / "none"}
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: missing_parent)
    assert nav.get_children("100") == []

    # uncached local filesystem iteration branch
    monkeypatch.setattr(nav, "_child_index", lambda: {})
    parent = {"taxon_key": "1", "path": root}
    monkeypatch.setattr(
        nav,
        "get_taxon_by_id",
        lambda key: parent if str(key) == "1" else (payload["catalog"]["2"] if str(key) == "2" else None),
    )
    assert [k["taxon_key"] for k in nav.get_children("1")] == ["2"]


def test_load_name_index_skips_empty_name(stub_env, monkeypatch):
    cfg, stub = stub_env
    payload = {
        "catalog": {
            "1": {
                "taxon_key": "1",
                "path": cfg.taxonomy_root / "a",
                "scientific_name": "Alpha",
                "common_name": ["", "Visible"],
            }
        },
        "combined_name_index": {},
    }
    stub._files[cfg.taxon_catalog_path] = pickle.dumps(payload)
    out = nav.load_name_index()
    assert "visible" in out


def test_resolve_taxon_media_returns_none_when_no_candidates(stub_env, monkeypatch):
    monkeypatch.setattr(nav, "get_taxon_by_id", lambda _k: {"taxon_key": "500", "rank": "GENUS"})
    monkeypatch.setattr(nav, "load_taxon_media", lambda: {})
    monkeypatch.setattr(nav, "iter_descendants_dfs", lambda _t: [])
    monkeypatch.setattr(nav, "get_parent_taxon", lambda _t: {"taxon_key": "p"})
    monkeypatch.setattr(
        nav, "get_children", lambda _k: [{"taxon_key": "500", "rank": "GENUS"}, {"taxon_key": "501", "rank": "GENUS"}]
    )
    nav.resolve_taxon_media.cache_clear()
    assert nav.resolve_taxon_media("500") is None

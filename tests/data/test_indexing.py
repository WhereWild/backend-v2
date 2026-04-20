"""Unit tests for util.indexing helper behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from util import indexing as idx
from util.request_cancellation import RequestCancelledError


class _StubParquet:
    def __init__(self):
        self._exists = {}
        self._tables = {}
        self._schemas = {}
        self.is_remote = False

    def exists(self, path):
        return self._exists.get(Path(path), False)

    def read_table(self, path, **_kwargs):
        value = self._tables.get(Path(path))
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(path, **_kwargs)
        return value

    def read_schema(self, path):
        value = self._schemas.get(Path(path))
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture(autouse=True)
def _clear_caches():
    idx._temporal_registry_config.cache_clear()
    idx._infer_sample_count_cached.cache_clear()
    idx._cached_metric_rows_for_taxon.cache_clear()
    yield


@pytest.fixture
def stub_env(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        taxonomy_root=tmp_path / "taxonomy",
        subspecies_equivalents={"SUBSPECIES", "VARIETY"},
        species_rank="SPECIES",
        common_name_language="en",
        occurrence_parquet_filename="occurrence.parquet",
    )
    cfg.taxonomy_root.mkdir(parents=True, exist_ok=True)
    stub = _StubParquet()
    monkeypatch.setattr(idx, "CONFIG", cfg)
    monkeypatch.setattr(idx, "PARQUET", stub)
    return cfg, stub


def test_temporal_helpers_and_targets(monkeypatch):
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_temporal_registry",
        lambda: {
            "windows": [6, "bad", 12],
            "layers": [{"id": "wind", "agg": "avg"}, {"id": "snap", "agg": "snapshot"}, {"id": ""}],
        },
    )
    expanded, base = idx._temporal_registry_config()
    assert "wind_avg_6h" in expanded and "snap" in expanded
    assert "wind" in base
    assert idx._extract_variable_from_metric_column("bio_1::mean") == "bio_1"
    assert idx._is_temporal_variable_id("wind_avg_12h")
    assert idx._is_temporal_metric_column("wind_avg_12h::mean")

    arrays = idx._harmonize_numeric_arrays([pa.array([1], type=pa.int32()), pa.array([2.0], type=pa.float64())])
    assert all(a.type == pa.float64() for a in arrays)

    targets = idx.index_targets_for_columns(
        {"bio_1", "wind_avg_6h", "wind_avg_12h"},
        layer_catalog={"bio_1": {"value_type": "numeric"}, "wind": {"value_type": "numeric", "agg": "avg"}},
    )
    assert ("bio_1", "numeric") in targets
    assert ("wind_avg_6h", "numeric") in targets


def test_temporal_and_global_loader_edge_branches(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_temporal_registry",
        lambda: {"windows": [0], "layers": [{"id": "wind", "agg": "avg", "windows": ["bad", -1]}]},
    )
    exp, _base = idx._temporal_registry_config()
    assert exp == frozenset()
    assert idx._extract_variable_from_metric_column("plain") == "plain"
    assert not idx._is_temporal_variable_id("")

    monkeypatch.setattr(idx.pds, "dataset", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    assert idx._load_global_relative_rows("1", "bio_1") is None

    class _BadDS:
        def to_table(self, **_kwargs):
            raise OSError("x")

    d = idx.global_relative_positions_dir()
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx.pds, "dataset", lambda *_a, **_k: _BadDS())
    assert idx._load_global_relative_rows("1", "bio_1") is None
    monkeypatch.setattr(
        idx.pds, "dataset", lambda *_a, **_k: type("DS", (), {"to_table": lambda self, **kw: pa.table({"x": []})})()
    )
    assert idx._load_global_relative_rows("1", "bio_1") is None

    out = idx._harmonize_numeric_arrays([pa.array(["a"], type=pa.string()), pa.array(["b"], type=pa.large_string())])
    assert all(a.type == pa.string() for a in out)


def test_global_rows_and_descendant_catalog_helpers(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env
    assert idx.global_relative_positions_dir() == cfg.taxonomy_root / idx.relative_rank_global_dirname
    stub.is_remote = True
    assert idx._load_global_relative_rows("1", "bio_1") is None
    stub.is_remote = False

    base = idx.global_relative_positions_dir()
    base.mkdir(parents=True, exist_ok=True)

    class _DS:
        def to_table(self, **_kwargs):
            return pa.table(
                {
                    "variable": ["bio_1"],
                    "metric": ["mean"],
                    "position": [0],
                    "count": [1],
                    "sampleCount": [1],
                    "contextTaxonId": ["1"],
                    "contextLabel": ["ctx"],
                }
            )

    monkeypatch.setattr(idx.pds, "dataset", lambda *_a, **_k: _DS())
    out = idx._load_global_relative_rows("1", "bio_1")
    assert out is not None
    assert out.num_rows == 1

    descendants = [{"taxon_key": "2"}, {"taxon_key": "1"}, {"taxon_key": "x"}, {"taxon_key": "1"}]
    ordered = idx._sorted_unique_descendants(descendants)
    assert [d["taxon_key"] for d in ordered] == ["1", "2", "x"]

    out_path = tmp_path / "species.parquet"
    monkeypatch.setattr(idx, "_infer_sample_count", lambda _t: 3)
    idx._write_descendant_catalog(out_path, [{"taxon_key": "1"}])
    assert out_path.exists()
    idx._write_descendant_catalog(out_path, [])
    assert not out_path.exists()


def test_infer_counts_and_rank_targets(monkeypatch):
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"count": "5"}})
    assert idx._infer_sample_count_cached("1", "/tmp/t") == 5
    idx._infer_sample_count_cached.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1"})
    monkeypatch.setattr(idx.taxa_navigation, "count_taxon_rows", lambda _t: 7)
    assert idx._infer_sample_count_cached("1", "/tmp/t") == 7
    idx._infer_sample_count_cached.cache_clear()
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    assert idx._infer_sample_count_cached("1", "/tmp/t") == 0

    assert "SPECIES" in idx._descendant_rank_targets("GENUS")
    assert idx._descendant_rank_targets("UNKNOWN")[0] == "KINGDOM"


def test_descendant_catalog_builders(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path / "g_1", "scientific_name": "Anc"}
    anc["path"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "iter_descendants_by_rank", lambda _a, r: [{"taxon_key": "2", "rank": r}])
    wrote = {}
    monkeypatch.setattr(idx, "_write_descendant_catalog", lambda p, d: wrote.setdefault("rows", (p, d)))
    idx.build_descendant_catalog_parquet("1", "species")
    assert wrote["rows"][0].name == "species.parquet"
    with pytest.raises(ValueError):
        idx.build_descendant_catalog_parquet("1", "")

    monkeypatch.setattr(idx, "_descendant_rank_targets", lambda _r: ["SPECIES"])
    monkeypatch.setattr(
        idx.taxa_navigation, "iter_descendants", lambda *_a, **_k: [{"taxon_key": "2", "rank": "SPECIES"}]
    )
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    called = {}
    monkeypatch.setattr(idx, "_write_descendant_catalog", lambda p, d: called.setdefault("x", (p, d)))
    idx.build_descendant_catalogs_for_ancestor("1")
    assert called["x"][0].name == "species.parquet"
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: None if str(k) == "999" else anc)
    with pytest.raises(ValueError):
        idx.build_descendant_catalogs_for_ancestor("999")


def test_metric_row_collection_and_rank_index_arrays(monkeypatch, tmp_path):
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 2.0, "count": 3}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"class_1": 0.3}})
    rows = idx._cached_metric_rows_for_taxon("1", "/tmp/x")
    assert any(r[0] == "bio_1::mean" for r in rows)
    assert idx._normalize_fallback_samples("x") == 0
    assert idx._normalize_sample_count(0) is None

    monkeypatch.setattr(
        idx,
        "_cached_metric_rows_for_taxon",
        lambda *_a, **_k: (("bio_1::mean", 1.0, 2), ("wind_avg_6h::mean", 3.0, 2), ("bio_1::std", 0.2, None)),
    )
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda c: c.startswith("wind_"))
    out = idx._collect_metric_entries_for_taxon({"taxon_key": "1", "path": tmp_path}, 5, exclude_columns={"bio_1::std"})
    assert list(out.keys()) == ["bio_1::mean"]

    arrays, lengths, metrics, max_len = idx._build_rank_index_arrays(out)
    assert "bio_1::mean" in arrays and lengths["bio_1::mean"] == 1 and "mean" in metrics and max_len == 1
    assert idx._build_rank_index_arrays({}) == ({}, {}, set(), 0)


def test_query_taxa_honors_cancellation_during_match_iteration(monkeypatch):
    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_a, **_k: [
            ({"taxon_key": "1", "rank": "SPECIES", "path": "/tmp/1"}, 91.0),
            ({"taxon_key": "2", "rank": "SPECIES", "path": "/tmp/2"}, 88.0),
        ],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda value: int(str(value)))

    calls = {"count": 0}

    def cancel_check():
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RequestCancelledError("Client disconnected")

    with pytest.raises(RequestCancelledError, match="Client disconnected"):
        idx.query_taxa(q="oak", cancel_check=cancel_check)


def test_write_rank_index_and_column_helpers(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    index_path = tmp_path / "species_index.parquet"
    entries = {"bio_1::mean": [{"taxon_key": "1", "value": 2.0, "sample_count": 3}]}
    idx._write_rank_index(index_path, entries, merge_existing=False)
    assert index_path.exists()

    schema = pq.read_schema(index_path)
    stub._schemas[index_path] = schema
    stub._tables[index_path] = pq.read_table(index_path)
    lengths = idx._load_column_lengths(index_path)
    assert lengths.get("bio_1::mean") == 1
    assert idx._resolve_column_name(index_path, "bio_1", "mean") == "bio_1::mean"
    col = idx._load_struct_column(index_path, "bio_1::mean", 1)
    assert len(col) == 1

    with pytest.raises(ValueError):
        idx._resolve_column_name(index_path, "missing", "x")


def test_relative_ranks_and_child_rankings(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            {"taxon_key": str(k), "rank": "SPECIES", "path": taxon_dir, "scientific_name": "S"}
            if str(k) == "1"
            else None
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(
        idx,
        "_load_global_relative_rows",
        lambda *_a, **_k: pa.table(
            {
                "variable": ["bio_1"],
                "metric": ["mean"],
                "position": [0],
                "count": [2],
                "sampleCount": [5],
                "contextTaxonId": ["10"],
                "contextLabel": ["G"],
            }
        ),
    )
    out = idx.load_relative_ranks(taxon_dir, "bio_1")
    assert out and out[0]["metric"] == "mean"

    index_path = taxon_dir / "species_index.parquet"
    arr = pa.StructArray.from_arrays(
        [pa.array(["1"], type=pa.string()), pa.array([1.5], type=pa.float64()), pa.array([5], type=pa.int32())],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    table = pa.table({"bio_1::mean": arr})
    pq.write_table(
        table.replace_schema_metadata({b"column_lengths": json.dumps({"bio_1::mean": 1}).encode("utf-8")}), index_path
    )
    stub._exists[index_path] = True
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["Name"])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    rankings, dist = idx.child_relative_rankings("1", "species", "bio_1", "mean", limit=10, order="asc")
    assert rankings and dist


def test_build_index_parquet_end_to_end_and_incremental(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    node = tmp_path / "species_1"
    child = node / "subspecies_2"
    child.mkdir(parents=True, exist_ok=True)
    node.mkdir(parents=True, exist_ok=True)

    parent_occ = node / "occurrence.parquet"
    child_occ = child / "occurrence.parquet"
    pq.write_table(
        pa.table(
            {
                "catalogNumber": ["b", "a"],
                "bio_1": [2.0, 1.0],
                "koppen": [1, 2],
                "obscured": ["No", "No"],
                "coordinateUncertaintyInMeters": [1, 1],
            }
        ),
        parent_occ,
    )
    pq.write_table(
        pa.table(
            {
                "catalogNumber": ["c"],
                "bio_1": [3.0],
                "koppen": [2],
                "obscured": ["No"],
                "coordinateUncertaintyInMeters": [1],
            }
        ),
        child_occ,
    )

    monkeypatch.setattr(
        idx.gis_lookup,
        "load_layer_metadata",
        lambda: {
            "bio_1": {"value_type": "numeric"},
            "koppen": {"value_type": "categorical"},
        },
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda p: Path(p).name.split("_")[-1])
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [{"taxon_key": "2", "path": child}])
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))

    idx.build_index_parquet(node)
    index_path = node / "occurrence_index.parquet"
    assert index_path.exists()
    schema = pq.read_schema(index_path)
    assert "bio_1" in schema.names and "koppen" in schema.names
    meta = schema.metadata or {}
    offsets = json.loads((meta.get(b"category_offsets") or b"{}").decode("utf-8"))
    assert "koppen" in offsets

    # incremental path: add a new layer and rebuild
    pq.write_table(
        pa.table(
            {
                "catalogNumber": ["a", "b"],
                "bio_1": [1.0, 2.0],
                "bio_2": [10.0, 11.0],
                "koppen": [2, 1],
                "obscured": ["No", "No"],
                "coordinateUncertaintyInMeters": [1, 1],
            }
        ),
        parent_occ,
    )
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_layer_metadata",
        lambda: {
            "bio_1": {"value_type": "numeric"},
            "bio_2": {"value_type": "numeric"},
            "koppen": {"value_type": "categorical"},
        },
    )
    idx.build_index_parquet(node)
    schema2 = pq.read_schema(index_path)
    assert "bio_2" in schema2.names


def test_query_taxa_text_results_use_full_match_set(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    def _search(_query, limit=10):
        rows = []
        for taxon_id in range(1, 121):
            rows.append(
                (
                    {
                        "taxon_key": str(taxon_id),
                        "path": tmp_path / f"species_{taxon_id}",
                        "rank": "SPECIES",
                        "scientific_name": f"Species {taxon_id}",
                    },
                    float(200 - taxon_id),
                )
            )
        if limit is None:
            return rows
        return rows[:limit]

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", _search)
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))

    payload = idx.query_taxa(q="oak", limit=5, offset=100)

    assert payload["total"] == 120
    assert [row["taxon_id"] for row in payload["results"]] == [101, 102, 103, 104, 105]


def test_query_taxa_text_results_bound_search_window_after_filtering(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    def _search(_query, limit=None, **_kwargs):
        rows = []
        for taxon_id in range(1, 1001):
            rows.append(
                (
                    {
                        "taxon_key": str(taxon_id),
                        "path": tmp_path / f"species_{taxon_id}",
                        "rank": "SPECIES",
                        "scientific_name": f"Species {taxon_id}",
                    },
                    float(2000 - taxon_id),
                )
            )
        if limit is None:
            return rows
        return rows[:limit]

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", _search)
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(
        idx,
        "_filter_matched_taxa",
        lambda matched_taxa, **_kwargs: [match for match in matched_taxa if match["taxon_id"] % 30 == 0],
    )

    payload = idx.query_taxa(q="oak", limit=5, offset=20)

    assert payload["matched_total"] == 625
    assert payload["eligible_total"] == 20
    assert payload["total"] == 20
    assert payload["results"] == []


def test_query_taxa_scoped_ranked_query_filters_inside_leaderboard_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    captured = {}

    def _ranked(*_args, **kwargs):
        captured["candidate_taxon_ids"] = kwargs.get("candidate_taxon_ids")
        captured["name_query"] = kwargs.get("name_query")
        return ([{"count": 400, "matched_count": 425, "taxon_id": 250, "match_score": 88.0}], None)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not prefilter scoped ranked queries")),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx, "child_relative_rankings", _ranked)

    payload = idx.query_taxa(
        q="oak",
        within_taxon_id="10",
        descendant_rank="SPECIES",
        sort_variable="bio_1",
        sort_metric="mean",
        limit=5,
    )

    assert payload["total"] == 400
    assert payload["matched_total"] == 425
    assert captured["candidate_taxon_ids"] is None
    assert captured["name_query"] == "oak"


def test_query_taxa_text_query_uses_bounded_search(monkeypatch):
    seen: dict[str, Any] = {}

    def _search(_query, limit=10):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", _search)

    payload = idx.query_taxa(q="oak", limit=5, offset=100)

    assert seen["limit"] == 2625
    assert payload["total"] == 0


def test_child_relative_rankings_keeps_location_matches_when_counts_missing_and_no_min_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    ancestor = {"taxon_key": "77", "rank": "GENUS", "path": tmp_path}
    target = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    target["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: ancestor if str(key) == "77" else (target if str(key) == "1" else None),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        idx,
        "_resolve_column_name",
        lambda *_args, **_kwargs: "bio_1::mean",
    )
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _path: True)
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _path: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_args, **_kwargs: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([7], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda *_args, **_kwargs: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_args, **_kwargs: {2: 4})
    monkeypatch.setattr(idx.gis_lookup, "location_counts_for_taxon", lambda _taxon_id: {})

    rows, _distribution = idx.child_relative_rankings(
        "77",
        "species",
        "bio_1",
        "mean",
        location_gid="USA",
        min_samples=0,
        return_distribution=False,
    )

    assert len(rows) == 1
    assert rows[0]["taxon_id"] == 1
    assert rows[0]["sample_count"] == 0


def test_query_taxa_rejects_unknown_descendant_rank():
    with pytest.raises(ValueError, match="Unknown descendant_rank: spcies"):
        idx.query_taxa(q="oak", descendant_rank="spcies")


def test_query_taxa_accepts_standard_higher_descendant_rank(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_a, **_k: [
            (
                {
                    "taxon_key": "10",
                    "path": tmp_path / "genus_10",
                    "rank": "GENUS",
                    "scientific_name": "Genus 10",
                },
                99.0,
            )
        ],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx, "_filter_matched_taxa", lambda matched_taxa, **_kwargs: matched_taxa)
    monkeypatch.setattr(idx, "child_relative_rankings", lambda *_a, **_k: ([{"count": 1, "taxon_id": 10}], None))

    payload = idx.query_taxa(
        q="genus",
        within_taxon_id="1",
        descendant_rank="GENUS",
        sort_variable="bio_1",
        sort_metric="mean",
    )

    assert payload["total"] == 1


def test_query_taxa_ranked_search_uses_full_match_set(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    ancestor = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path / "genus_10",
        "scientific_name": "Ancestor Genus",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not prefilter scoped ranked queries")),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_parent_taxon",
        lambda taxon: ancestor if str(taxon.get("taxon_key")) != "10" else None,
    )

    def _child_relative_rankings(*_args, candidate_taxon_ids=None, name_query=None, **_kwargs):
        seen["candidate_count"] = len(candidate_taxon_ids or [])
        seen["name_query"] = name_query
        return ([{"count": 300, "matched_count": 300, "taxon_id": 1, "match_score": 91.0}], None)

    monkeypatch.setattr(idx, "child_relative_rankings", _child_relative_rankings)

    payload = idx.query_taxa(
        q="species",
        within_taxon_id="10",
        descendant_rank="SPECIES",
        sort_variable="bio_1",
        sort_metric="mean",
        limit=5,
        offset=100,
    )

    assert seen["candidate_count"] == 0
    assert seen["name_query"] == "species"
    assert payload["matched_total"] == 300
    assert payload["total"] == 300


def test_query_taxa_direct_ranked_query_uses_bounded_text_search(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }
    for taxon in (species_one, species_two):
        taxon["path"].mkdir(parents=True, exist_ok=True)

    def _search_taxa_by_name(_query, limit=10, **_kwargs):
        seen["limit"] = limit
        return [
            (species_one, 80.0),
            (species_two, 95.0),
        ]

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", _search_taxa_by_name)
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(
        idx,
        "_load_summary_stats",
        lambda path: (
            {"bio_1": {"mean": 10.0, "count": 5}}
            if str(path).endswith("species_1")
            else {"bio_1": {"mean": 2.0, "count": 6}}
        ),
    )
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _path: {})

    payload = idx.query_taxa(
        q="species",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
        limit=5,
        offset=100,
    )

    assert seen["limit"] == 2625
    assert payload["total"] == 2
    assert payload["matched_total"] == 2
    assert payload["eligible_total"] == 2
    assert payload["results"] == []


def test_query_taxa_text_location_filter_skips_sample_count_when_min_samples_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    species = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", lambda _query, limit=250: [(species, 90.0)])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("level0Gid", "country_scope", "ETH"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _scope, _gid: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 3})
    monkeypatch.setattr(
        idx,
        "_matched_taxon_sample_count",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not resolve sample_count when min_samples=0")),
    )

    payload = idx.query_taxa(q="species", location_gid="ETH", min_samples=0)

    assert payload["total"] == 1
    assert payload["matched_total"] == 1
    assert payload["eligible_total"] == 1
    assert [row["taxon_id"] for row in payload["results"]] == [1]


def test_rank_candidate_taxa_keeps_missing_counts_when_min_samples_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    species = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species["path"].mkdir(parents=True, exist_ok=True)
    match_row = {"taxon": species, "taxon_id": 1, "match_score": 80.0}

    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx, "_taxon_metric_record", lambda *_args, **_kwargs: (5.0, None))

    rows = idx._rank_candidate_taxa(
        [match_row],
        "bio_1",
        "mean",
        order="asc",
        min_samples=0,
        include_species_like=False,
        within_taxon_id=None,
        descendant_rank=None,
        location_gid=None,
    )

    assert len(rows) == 1
    assert rows[0]["taxon_id"] == 1
    assert rows[0]["sample_count"] == 0


def test_rank_candidate_taxa_descending_uses_rank_position(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }
    for taxon in (species_one, species_two):
        taxon["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        idx,
        "_taxon_metric_record",
        lambda taxon, *_args, **_kwargs: (5.0, 4) if taxon["taxon_key"] == "1" else (10.0, 6),
    )

    rows = idx._rank_candidate_taxa(
        [
            {"taxon": species_one, "taxon_id": 1, "match_score": 80.0},
            {"taxon": species_two, "taxon_id": 2, "match_score": 70.0},
        ],
        "bio_1",
        "mean",
        order="desc",
        min_samples=0,
        include_species_like=False,
        within_taxon_id=None,
        descendant_rank=None,
        location_gid=None,
    )

    assert [row["taxon_id"] for row in rows] == [2, 1]
    assert [row["position"] for row in rows] == [2, 1]
    assert [row["count"] for row in rows] == [2, 2]
    assert [row["percentile"] for row in rows] == [1.0, 0.0]


def test_query_taxa_location_rollup_keeps_species_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    species = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", lambda _query, limit=None: [(species, 90.0)])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "iter_descendants",
        lambda _taxon, include_self=False: [] if include_self else [{"taxon_key": "11", "rank": "SUBSPECIES"}],
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("level0Gid", "country_scope", "ETH"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _scope, _gid: frozenset({11}))
    monkeypatch.setattr(
        idx.gis_lookup,
        "location_taxon_counts",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("should not roll up all location taxa when min_samples=0")
        ),
    )

    payload = idx.query_taxa(q="species", location_gid="ETH")

    assert payload["total"] == 1
    assert payload["matched_total"] == 1
    assert payload["eligible_total"] == 1
    assert [row["taxon_id"] for row in payload["results"]] == [1]


def test_query_taxa_location_filter_keeps_higher_rank_ranked_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    genus = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path / "genus_10",
        "scientific_name": "Genus Ten",
    }
    genus["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", lambda _query, limit=None: [(genus, 90.0)])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _path: {"bio_1": {"mean": 2.0, "count": 4}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _path: {})
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("level0Gid", "country_scope", "USA"))

    def _location_taxa_for(_scope, _gid):
        return frozenset({10})

    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", _location_taxa_for)
    monkeypatch.setattr(
        idx.gis_lookup,
        "location_taxon_counts",
        lambda _scope, _gid: {10: 4},
    )

    payload = idx.query_taxa(
        q="genus",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
        location_gid="USA",
    )

    assert payload["total"] == 1
    assert payload["matched_total"] == 1
    assert payload["eligible_total"] == 1
    assert [row["taxon_id"] for row in payload["results"]] == [10]
    assert payload["results"][0]["sample_count"] == 4


def test_query_taxa_text_only_filters_matches_by_location(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=None: [
            (species_one, 90.0),
            (species_two, 80.0),
        ],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("level0Gid", "country_scope", "ETH"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _scope, _gid: frozenset({2}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {2: 3})
    monkeypatch.setattr(idx.taxa_navigation, "iter_descendants", lambda _taxon, include_self=False: iter([]))

    payload = idx.query_taxa(q="species", location_gid="ETH")

    assert payload["total"] == 1
    assert payload["matched_total"] == 2
    assert payload["eligible_total"] == 1
    assert payload["empty_reason"] is None
    assert [row["taxon_id"] for row in payload["results"]] == [2]


def test_query_taxa_text_location_filter_keeps_higher_rank_matches_without_location_rollup(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    genus = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path / "genus_10",
        "scientific_name": "Genus Ten",
    }
    species = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", lambda _query, limit=250: [(genus, 90.0)])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "iter_descendants",
        lambda taxon, include_self=False: [species] if str(taxon.get("taxon_key")) == "10" and not include_self else [],
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("level0Gid", "country_scope", "CHN"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _scope, _gid: frozenset({1}))
    monkeypatch.setattr(
        idx.gis_lookup,
        "location_taxon_counts",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("should not roll up all location taxa when min_samples=0")
        ),
    )

    payload = idx.query_taxa(q="genus", location_gid="CHN", min_samples=0)

    assert payload["total"] == 1
    assert payload["matched_total"] == 1
    assert payload["eligible_total"] == 1
    assert [row["taxon_id"] for row in payload["results"]] == [10]


def test_query_taxa_text_only_filters_matches_by_min_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=None: [
            (species_one, 90.0),
            (species_two, 80.0),
        ],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx, "_infer_sample_count", lambda taxon: 5 if str(taxon.get("taxon_key")) == "1" else 12)

    payload = idx.query_taxa(q="species", min_samples=10)

    assert payload["total"] == 1
    assert payload["matched_total"] == 2
    assert payload["eligible_total"] == 1
    assert payload["empty_reason"] is None
    assert [row["taxon_id"] for row in payload["results"]] == [2]


def test_query_taxa_returns_no_query_metadata():
    payload = idx.query_taxa()

    assert payload["total"] == 0
    assert payload["matched_total"] == 0
    assert payload["eligible_total"] == 0
    assert payload["empty_reason"] == "no_query"


def test_query_taxa_reports_filtered_out_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )

    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=None: [
            (species_one, 90.0),
            (species_two, 80.0),
        ],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx, "_infer_sample_count", lambda _taxon: 0)

    payload = idx.query_taxa(q="species", min_samples=1)

    assert payload["matched_total"] == 2
    assert payload["eligible_total"] == 0
    assert payload["empty_reason"] == "filtered_out"
    assert payload["results"] == []


def test_child_relative_rankings_location_filter_uses_location_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES"},
            species_rank="SPECIES",
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: {
            "taxon_key": str(key),
            "rank": "SPECIES",
            "scientific_name": f"Species {key}",
            "path": tmp_path,
        },
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 3})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2", "3"], type=pa.string()),
                pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                pa.array([10, 10, 10], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1, 2}))
    monkeypatch.setattr(
        idx.gis_lookup,
        "location_taxon_counts",
        lambda _s, _t: {1: 1, 2: 3},
    )

    rows, distribution = idx.child_relative_rankings(
        "1",
        "species",
        "bio_1",
        "mean",
        min_samples=2,
        return_distribution=False,
        location_gid="USA",
    )

    assert distribution is None
    assert len(rows) == 1
    assert rows[0]["taxon_id"] == 2
    assert rows[0]["sampleCount"] == 3
    assert rows[0]["count"] == 1
    assert rows[0]["percentile"] == 0.0


def test_child_relative_rankings_descending_uses_filtered_rank_position(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES"},
            species_rank="SPECIES",
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: {
            "taxon_key": str(key),
            "rank": "SPECIES",
            "scientific_name": f"Species {key}",
            "path": tmp_path,
        },
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 3})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2", "3"], type=pa.string()),
                pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                pa.array([2, 5, 7], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1, 2, 3}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 2, 2: 5, 3: 7})

    rows, distribution = idx.child_relative_rankings(
        "1",
        "species",
        "bio_1",
        "mean",
        min_samples=2,
        order="desc",
        return_distribution=False,
        location_gid="USA",
    )

    assert distribution is None
    assert [row["taxon_id"] for row in rows] == [3, 2, 1]
    assert [row["count"] for row in rows] == [3, 3, 3]
    assert [row["sampleCount"] for row in rows] == [7, 5, 2]
    assert [row["position"] for row in rows] == [3, 2, 1]
    assert [row["percentile"] for row in rows] == [1.0, 0.5, 0.0]


def test_child_relative_rankings_location_rollup_keeps_species(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES"},
            species_rank="SPECIES",
        ),
    )
    taxa = {
        "1": {
            "taxon_key": "1",
            "rank": "SPECIES",
            "scientific_name": "Species 1",
            "path": tmp_path,
        },
        "11": {
            "taxon_key": "11",
            "rank": "SUBSPECIES",
            "scientific_name": "Species 1 subsp.",
            "path": tmp_path,
        },
    }
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda key: taxa.get(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1"], type=pa.string()),
                pa.array([2.0], type=pa.float64()),
                pa.array([1], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({11}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 4, 11: 4})

    rows, distribution = idx.child_relative_rankings(
        "1",
        "species",
        "bio_1",
        "mean",
        min_samples=2,
        return_distribution=False,
        location_gid="USA",
    )

    assert distribution is None
    assert [row["taxon_id"] for row in rows] == [1]
    assert rows[0]["sampleCount"] == 4


def test_ranked_query_location_filter_falls_back_to_per_taxon_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"}),
    )
    taxon = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "scientific_name": "Species 1",
        "path": tmp_path / "species_1",
    }

    monkeypatch.setattr(idx.taxa_navigation, "search_taxa_by_name", lambda *_a, **_k: [(taxon, 99.0)])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx, "_taxon_metric_record", lambda *_a, **_k: (2.5, 1))
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda *_a, **_k: frozenset({1}))
    monkeypatch.setattr(
        idx.gis_lookup,
        "location_taxon_counts",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_counts_for_taxon", lambda _taxon_id: {("scope", "target"): 7})

    payload = idx.query_taxa(
        q="species",
        sort_variable="bio_1",
        sort_metric="mean",
        min_samples=5,
        location_gid="USA",
    )

    assert payload["total"] == 1
    assert payload["results"][0]["sample_count"] == 7


def test_child_relative_rankings_falls_back_to_per_taxon_location_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES"},
            species_rank="SPECIES",
        ),
    )
    ancestor = {"taxon_key": "77", "rank": "GENUS", "path": tmp_path}
    taxon = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "scientific_name": "Species 1",
        "path": tmp_path,
    }
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: ancestor if str(key) == "77" else taxon,
    )
    monkeypatch.setattr(
        idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)) if str(key).isdigit() else None
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda *_a, **_k: frozenset({1}))
    monkeypatch.setattr(
        idx.gis_lookup,
        "location_taxon_counts",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_counts_for_taxon", lambda _taxon_id: {("scope", "target"): 7})

    rows, distribution = idx.child_relative_rankings(
        "77",
        "species",
        "bio_1",
        "mean",
        min_samples=5,
        return_distribution=False,
        location_gid="USA",
    )

    assert distribution is None
    assert len(rows) == 1
    assert rows[0]["sampleCount"] == 7


def test_build_index_parquet_error_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    bad_occ = node / "occurrence.parquet"
    pq.write_table(pa.table({"x": [1]}), bad_occ)
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    monkeypatch.setattr(idx.gis_lookup, "load_layer_metadata", lambda: {})
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [])
    with pytest.raises(ValueError):
        idx.build_index_parquet(node)


def test_child_rankings_intersect_candidate_taxa(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    ancestor_path = tmp_path / "genus_10"
    ancestor_path.mkdir(parents=True, exist_ok=True)
    index_path = ancestor_path / "species_index.parquet"

    column = pa.StructArray.from_arrays(
        [
            pa.array(["1", "2"], type=pa.string()),
            pa.array([1.0, 2.0], type=pa.float64()),
            pa.array([5, 6], type=pa.int32()),
        ],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    table = pa.table({"bio_1::mean": column})
    pq.write_table(
        table.replace_schema_metadata({b"column_lengths": json.dumps({"bio_1::mean": 2}).encode("utf-8")}),
        index_path,
    )
    stub._exists[index_path] = True
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)

    taxa = {
        "10": {
            "taxon_key": "10",
            "rank": "GENUS",
            "path": ancestor_path,
            "scientific_name": "Ancestor",
        },
        "1": {
            "taxon_key": "1",
            "rank": "SPECIES",
            "path": tmp_path / "species_1",
            "scientific_name": "Species One",
        },
        "2": {
            "taxon_key": "2",
            "rank": "SPECIES",
            "path": tmp_path / "species_2",
            "scientific_name": "Species Two",
        },
    }
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda key: taxa.get(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})

    ranked, distribution = idx.child_relative_rankings(
        "10",
        "species",
        "bio_1",
        "mean",
        candidate_taxon_ids=[2],
    )

    assert [row["taxonId"] for row in ranked] == [2]
    assert ranked[0]["count"] == 1
    assert ranked[0]["position"] == 1
    assert ranked[0]["percentile"] == 0.0
    assert distribution == [2.0]


def test_query_taxa_sorts_matched_taxa_without_leaderboard_scope(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }
    for taxon in (species_one, species_two):
        taxon["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [
            (species_one, 80.0),
            (species_two, 95.0),
        ][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(
        idx,
        "_load_summary_stats",
        lambda path: (
            {"bio_1": {"mean": 10.0, "count": 5}}
            if str(path).endswith("species_1")
            else {"bio_1": {"mean": 2.0, "count": 6}}
        ),
    )
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _path: {})

    out = idx.query_taxa(
        q="species",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
    )

    assert out["total"] == 2
    assert out["matched_total"] == 2
    assert out["eligible_total"] == 2
    assert out["empty_reason"] is None
    assert [row["taxon_id"] for row in out["results"]] == [2, 1]
    assert out["results"][0]["sort_value"] == 2.0
    assert out["results"][1]["sort_value"] == 10.0
    assert "taxonId" not in out["results"][0]
    assert "sampleCount" not in out["results"][0]
    assert "value" not in out["results"][0]


def test_query_taxa_reports_ranking_ineligible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_one["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [(species_one, 80.0)][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _path: {})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _path: {})

    out = idx.query_taxa(
        q="species",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
    )

    assert out["total"] == 0
    assert out["matched_total"] == 1
    assert out["eligible_total"] == 0
    assert out["empty_reason"] == "ranking_ineligible"
    assert out["results"] == []


def test_query_taxa_direct_ranked_eligible_total_matches_metric_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }
    for taxon in (species_one, species_two):
        taxon["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [
            (species_one, 80.0),
            (species_two, 95.0),
        ][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(
        idx,
        "_load_summary_stats",
        lambda path: {"bio_1": {"mean": 10.0, "count": 5}} if str(path).endswith("species_1") else {},
    )
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _path: {})

    out = idx.query_taxa(
        q="species",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
    )

    assert out["total"] == 1
    assert out["matched_total"] == 2
    assert out["eligible_total"] == 1
    assert [row["taxon_id"] for row in out["results"]] == [1]


def test_query_taxa_scope_excludes_ancestor_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    genus = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path / "genus_10",
        "scientific_name": "Genus Ten",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [(genus, 90.0)][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda key: genus if str(key) == "10" else None)
    monkeypatch.setattr(idx.taxa_navigation, "iter_descendants", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)

    out = idx.query_taxa(
        q="genus",
        within_taxon_id="10",
        descendant_rank="GENUS",
    )

    assert out["total"] == 0
    assert out["matched_total"] == 0
    assert out["eligible_total"] == 0
    assert out["empty_reason"] == "no_text_matches"
    assert out["results"] == []


def test_query_taxa_scoped_text_query_bypasses_global_text_prefilter(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    ancestor = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path / "genus_10",
        "scientific_name": "Ancestor Genus",
    }
    species = {
        "taxon_key": "11",
        "rank": "SPECIES",
        "path": tmp_path / "species_11",
        "scientific_name": "Scoped Species",
        "common_name": "American Something",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not use global prefilter")),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: ancestor if str(key) == "10" else None,
    )
    monkeypatch.setattr(idx.taxa_navigation, "iter_descendants", lambda *_args, **_kwargs: [species])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda taxon: ancestor if taxon is species else None)
    monkeypatch.setattr(
        idx.taxa_navigation,
        "taxon_name_match_score",
        lambda taxon, query: 92.0 if taxon is species and query == "american" else None,
    )
    monkeypatch.setattr(idx, "_filter_matched_taxa", lambda matched_taxa, **_kwargs: matched_taxa)

    out = idx.query_taxa(
        q="american",
        within_taxon_id="10",
    )

    assert out["total"] == 1
    assert out["matched_total"] == 1
    assert out["eligible_total"] == 1
    assert [row["taxon_id"] for row in out["results"]] == [11]


def test_query_taxa_scoped_text_query_uses_stable_tie_break_order(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    ancestor = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path / "genus_10",
        "scientific_name": "Ancestor Genus",
    }
    species_b = {
        "taxon_key": "12",
        "rank": "SPECIES",
        "path": tmp_path / "species_12",
        "scientific_name": "Scoped Species B",
        "common_name": "American Something Else",
    }
    species_a = {
        "taxon_key": "11",
        "rank": "SPECIES",
        "path": tmp_path / "species_11",
        "scientific_name": "Scoped Species A",
        "common_name": "American Something",
    }

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not use global prefilter")),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: ancestor if str(key) == "10" else None,
    )
    monkeypatch.setattr(idx.taxa_navigation, "iter_descendants", lambda *_args, **_kwargs: [species_b, species_a])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_parent_taxon",
        lambda taxon: ancestor if taxon is species_a or taxon is species_b else None,
    )
    monkeypatch.setattr(
        idx.taxa_navigation, "taxon_name_match_score", lambda _taxon, query: 92.0 if query == "american" else None
    )
    monkeypatch.setattr(idx, "_filter_matched_taxa", lambda matched_taxa, **_kwargs: matched_taxa)

    out = idx.query_taxa(
        q="american",
        within_taxon_id="10",
    )

    assert out["total"] == 2
    assert out["matched_total"] == 2
    assert out["eligible_total"] == 2
    assert [row["taxon_id"] for row in out["results"]] == [11, 12]


def test_query_taxa_scoped_ranked_query_bypasses_text_prefilter(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not prefilter scoped ranked queries")),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx,
        "_filter_matched_taxa",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not prefilter scoped ranked queries")),
    )
    monkeypatch.setattr(
        idx,
        "child_relative_rankings",
        lambda *_args, **_kwargs: ([], None),
    )
    monkeypatch.setattr(idx, "_count_scoped_query_matches", lambda *_args, **_kwargs: 1)

    out = idx.query_taxa(
        q="species",
        within_taxon_id="10",
        descendant_rank="SPECIES",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
        min_samples=1,
    )

    assert out["total"] == 0
    assert out["matched_total"] == 1
    assert out["eligible_total"] == 0
    assert out["empty_reason"] == "ranking_ineligible"
    assert out["results"] == []


def test_child_relative_rankings_scoped_text_query_matches_alias_only_names(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"}, species_rank="SPECIES"),
    )
    ancestor = {"taxon_key": "10", "rank": "GENUS", "path": tmp_path / "genus_10", "scientific_name": "Genus Ten"}
    species = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Visible Name",
        "common_name": "ordinary",
    }
    species["path"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: ancestor if str(key) == "10" else (species if str(key) == "1" else None),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)) if str(key).isdigit() else None
    )
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["Visible Name"])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(idx.taxa_navigation, "load_search_names_by_taxon", lambda: {"1": ("hidden alias",)})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1"], type=pa.string()),
                pa.array([2.0], type=pa.float64()),
                pa.array([5], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )

    rows, _distribution = idx.child_relative_rankings(
        "10",
        "species",
        "bio_1",
        "mean",
        name_query="hidden",
        return_distribution=False,
    )

    assert len(rows) == 1
    assert rows[0]["taxon_id"] == 1
    assert rows[0]["matched_count"] == 1


def test_query_taxa_scoped_ranked_query_counts_text_matches_without_metric(monkeypatch, tmp_path, stub_env):
    _cfg, stub = stub_env
    ancestor = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path,
        "scientific_name": "Genus Ten",
    }
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }

    index_path = tmp_path / "species_index.parquet"
    arr = pa.StructArray.from_arrays(
        [
            pa.array(["1", "2"], type=pa.string()),
            pa.array([1.0, None], type=pa.float64()),
            pa.array([5, 5], type=pa.int32()),
        ],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(
        pa.table({"bio_1::mean": arr}).replace_schema_metadata(
            {b"column_lengths": json.dumps({"bio_1::mean": 2}).encode("utf-8")}
        ),
        index_path,
    )
    stub._exists[index_path] = True
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)

    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES"},
            species_rank="SPECIES",
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: {
            "10": ancestor,
            "1": species_one,
            "2": species_two,
        }.get(str(key)),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        idx.taxa_navigation,
        "taxon_name_match_score",
        lambda taxon, query: 90.0 if query == "species" and "Species" in taxon["scientific_name"] else None,
    )

    out = idx.query_taxa(
        q="species",
        within_taxon_id="10",
        descendant_rank="SPECIES",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
    )

    assert out["total"] == 1
    assert out["matched_total"] == 2
    assert out["eligible_total"] == 1
    assert [row["taxon_id"] for row in out["results"]] == [1]


def test_query_taxa_scoped_ranked_query_fallback_counts_location_scoped_matches(monkeypatch, tmp_path, stub_env):
    _cfg, stub = stub_env
    ancestor = {
        "taxon_key": "10",
        "rank": "GENUS",
        "path": tmp_path,
        "scientific_name": "Genus Ten",
    }
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }

    index_path = tmp_path / "species_index.parquet"
    arr = pa.StructArray.from_arrays(
        [
            pa.array(["1", "2"], type=pa.string()),
            pa.array([None, None], type=pa.float64()),
            pa.array([5, 5], type=pa.int32()),
        ],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(
        pa.table({"bio_1::mean": arr}).replace_schema_metadata(
            {b"column_lengths": json.dumps({"bio_1::mean": 2}).encode("utf-8")}
        ),
        index_path,
    )
    stub._exists[index_path] = True
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)

    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES"},
            species_rank="SPECIES",
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: {
            "10": ancestor,
            "1": species_one,
            "2": species_two,
        }.get(str(key)),
    )
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: ancestor)
    monkeypatch.setattr(idx.taxa_navigation, "iter_descendants", lambda *_args, **_kwargs: [species_one, species_two])
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        idx.taxa_navigation,
        "taxon_name_match_score",
        lambda taxon, query: 90.0 if query == "species" and "Species" in taxon["scientific_name"] else None,
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _gid: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda *_args, **_kwargs: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_args, **_kwargs: {})

    out = idx.query_taxa(
        q="species",
        within_taxon_id="10",
        descendant_rank="SPECIES",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
        location_gid="USA",
    )

    assert out["total"] == 0
    assert out["matched_total"] == 1
    assert out["eligible_total"] == 0
    assert out["empty_reason"] == "ranking_ineligible"
    assert out["results"] == []


def test_query_taxa_direct_sort_ties_prefer_better_match_score(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }
    for taxon in (species_one, species_two):
        taxon["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [
            (species_one, 80.0),
            (species_two, 95.0),
        ][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _path: {"bio_1": {"mean": 5.0, "count": 5}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _path: {})

    out = idx.query_taxa(
        q="species",
        sort_variable="bio_1",
        sort_metric="mean",
        sort_order="asc",
    )

    assert [row["taxon_id"] for row in out["results"]] == [2, 1]


def test_query_taxa_rejects_invalid_sort_order_without_leaderboard_scope(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_one["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [(species_one, 80.0)][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))

    with pytest.raises(ValueError, match="order must be either 'asc' or 'desc'"):
        idx.query_taxa(
            q="species",
            sort_variable="bio_1",
            sort_metric="mean",
            sort_order="sideways",
        )


def test_rank_index_builders_more_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path, "scientific_name": "Anc"}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx, "_descendant_rank_targets", lambda _r: ["SPECIES"])

    catalog = tmp_path / "species.parquet"
    pq.write_table(pa.table({"taxon_key": ["2"], "sample_count": [3]}), catalog)
    stub._exists[catalog] = True
    stub._tables[catalog] = pq.read_table(catalog)
    called = {}
    monkeypatch.setattr(
        idx, "_build_rank_index_parquet", lambda a, r, **_k: called.setdefault("v", (a["taxon_key"], r))
    )
    idx.build_rank_indexes_for_ancestor("1")
    assert called["v"] == ("1", "SPECIES")

    # _build_rank_index_parquet early exits
    monkeypatch.setattr(stub, "exists", lambda _p: False)
    idx._build_rank_index_parquet(anc, "SPECIES")

    # child_relative_rankings additional branches
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: None if str(k) == "bad" else anc)
    with pytest.raises(ValueError):
        idx.child_relative_rankings("bad", "species", "bio_1", "mean")


def test_build_rank_index_parquet_modes(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    catalog_path = tmp_path / "species.parquet"
    pq.write_table(pa.table({"taxon_key": ["2", "3"], "sample_count": [3, 0]}), catalog_path)
    stub._exists[catalog_path] = True
    stub._tables[catalog_path] = pq.read_table(catalog_path)

    # existing index with non-temporal columns => incremental mode
    index_path = tmp_path / "species_index.parquet"
    arr = pa.StructArray.from_arrays(
        [pa.array(["2"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([3], type=pa.int32())],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(pa.table({"bio_1::mean": arr}), index_path)
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)

    by_id = {"2": {"taxon_key": "2", "path": tmp_path / "t2"}, "3": {"taxon_key": "3", "path": tmp_path / "t3"}}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: by_id.get(str(k)))
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda _c: False)
    monkeypatch.setattr(idx, "_normalize_fallback_samples", lambda v: int(v or 0))
    monkeypatch.setattr(
        idx,
        "_collect_metric_entries_for_taxon",
        lambda taxon, *_a, **_k: (
            {"bio_2::mean": [{"taxon_key": taxon["taxon_key"], "value": 2.0, "sample_count": 3}]}
            if taxon["taxon_key"] == "2"
            else {}
        ),
    )
    wrote = {}
    monkeypatch.setattr(idx, "_write_rank_index", lambda p, entries, **k: wrote.setdefault("v", (p, entries, k)))
    idx._build_rank_index_parquet(anc, "SPECIES")
    assert wrote["v"][2]["merge_existing"] is True

    # when no entries in incremental mode => up-to-date early return
    monkeypatch.setattr(idx, "_collect_metric_entries_for_taxon", lambda *_a, **_k: {})
    idx._build_rank_index_parquet(anc, "SPECIES")


def test_child_relative_rankings_fast_and_distribution_paths(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    t2 = {"taxon_key": "2", "rank": "SPECIES", "scientific_name": "Two"}
    t3 = {"taxon_key": "3", "rank": "SUBSPECIES", "scientific_name": "Three"}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: anc if str(k) == "1" else (t2 if str(k) == "2" else (t3 if str(k) == "3" else None)),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})

    index_path = tmp_path / "species_index.parquet"
    arr = pa.StructArray.from_arrays(
        [
            pa.array(["2", "3"], type=pa.string()),
            pa.array([1.0, 2.0], type=pa.float64()),
            pa.array([5, 5], type=pa.int32()),
        ],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(
        pa.table({"bio_1::mean": arr}).replace_schema_metadata(
            {b"column_lengths": json.dumps({"bio_1::mean": 2}).encode("utf-8")}
        ),
        index_path,
    )
    stub._exists[index_path] = True
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)

    # fast path: location filtered + no distribution
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({2}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {2: 5})
    out_fast, dist_fast = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", limit=10, order="asc", return_distribution=False, location_gid="USA"
    )
    assert len(out_fast) == 1 and dist_fast is None and out_fast[0]["taxonId"] == 2

    # standard path with distribution and desc order
    out_std, dist_std = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", limit=10, order="desc", include_species_like=True, return_distribution=True
    )
    assert len(out_std) == 2 and dist_std == [1.0, 2.0]

    with pytest.raises(ValueError):
        idx.child_relative_rankings("1", "species", "bio_1", "mean", order="bad")

    # no index path branch
    stub._exists[index_path] = False
    empty, dist = idx.child_relative_rankings("1", "species", "bio_1", "mean")
    assert empty == [] and dist is None


def test_write_rank_index_merge_and_cached_rows_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    index_path = tmp_path / "rank.parquet"
    existing_arr = pa.StructArray.from_arrays(
        [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([2], type=pa.int32())],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(
        pa.table({"bio_1::mean": existing_arr}).replace_schema_metadata(
            {
                b"column_lengths": json.dumps({"bio_1::mean": 1}).encode("utf-8"),
                b"metrics": json.dumps(["mean"]).encode("utf-8"),
            }
        ),
        index_path,
    )
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)
    entries = {"bio_2::mean": [{"taxon_key": "2", "value": 3.0, "sample_count": 4}]}
    idx._write_rank_index(index_path, entries, merge_existing=True)
    schema = pq.read_schema(index_path)
    assert "bio_1::mean" in schema.names and "bio_2::mean" in schema.names

    # merge existing fallback path on schema read errors
    monkeypatch.setattr(stub, "read_schema", lambda _p: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr(stub, "read_table", lambda _p, **_k: (_ for _ in ()).throw(OSError("x")))
    idx._write_rank_index(index_path, entries, merge_existing=True)

    # cached metric rows filters and empty result
    monkeypatch.setattr(
        idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": float("inf"), "count": -1, "min": None}}
    )
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"class_1": "bad"}})
    rows = idx._cached_metric_rows_for_taxon("1", "/tmp/a")
    assert rows == (("bio_1::count", -1.0, None),)


def test_descendant_catalog_rank_specific_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "ORDER", "path": tmp_path, "scientific_name": "Anc"}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx, "_descendant_rank_targets", lambda _r: ["SUBSPECIES", "SPECIES"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "iter_descendants",
        lambda *_a, **_k: [{"taxon_key": "s1", "rank": "SPECIES"}, {"taxon_key": "ss1", "rank": "SUBSPECIES"}],
    )
    called = []
    monkeypatch.setattr(idx, "_write_descendant_catalog", lambda p, d: called.append((p.name, len(d))))
    monkeypatch.setattr(stub, "exists", lambda p: p.name == "species.parquet")
    idx.build_descendant_catalogs_for_ancestor("1")
    # non-species ancestor skips subspecies; existing species file skips species write
    assert called == []


def test_load_relative_ranks_location_filtered_branch(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    target = {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir, "scientific_name": "S"}
    ancestor = {"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            target
            if str(k) == "1"
            else (
                ancestor
                if str(k) == "10"
                else {"taxon_key": str(k), "rank": "SPECIES", "path": taxon_dir, "scientific_name": "X"}
            )
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [ancestor])
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1, 2}))
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2"], type=pa.string()),
                pa.array([1.0, 2.0], type=pa.float64()),
                pa.array([5, 5], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    rows = idx.load_relative_ranks(taxon_dir, "bio_1", location_gid="USA")
    assert rows and rows[0]["metric"] == "mean"


def test_ancestor_contexts_and_relative_ranks_local_fallback(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env
    cfg.taxonomy_root = tmp_path / "taxonomy"
    tax_path = cfg.taxonomy_root / "genus_10" / "species_1"
    tax_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx, "CONFIG", cfg)

    # ancestor resolution with missing lookup fallback payload
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            None if str(k) == "10" else {"taxon_key": "1", "rank": "SPECIES", "path": tax_path, "scientific_name": "S"}
        ),
    )
    ctx = idx._ancestor_contexts(tax_path)
    assert ctx and ctx[0]["taxon_key"] == "10"

    # load_relative_ranks local fallback path (no global rows)
    taxon_dir = tax_path
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir, "scientific_name": "S"} if str(k) == "1" else None
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    pos = taxon_dir / "relative_ranks_positions.parquet"
    pos.touch()
    stub._exists[pos] = True
    stub._tables[pos] = Exception("force typeerror")
    stub.is_remote = True
    monkeypatch.setattr(stub, "read_table", lambda *_a, **_k: (_ for _ in ()).throw(TypeError("no filters")))
    monkeypatch.setattr(
        idx.pq,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": ["bio_1", "bio_1"],
                "metric": ["mean", "bad_metric"],
                "position": [0, 1],
                "count": [2, 2],
                "sampleCount": [5, 5],
                "contextTaxonId": ["10", "10"],
                "contextLabel": ["G", "G"],
            }
        ),
    )
    rows = idx.load_relative_ranks(taxon_dir, "bio_1")
    assert len(rows) == 1 and rows[0]["metric"] == "mean"
    stub.is_remote = False


def test_indexing_helper_branch_sweep(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    monkeypatch.setattr(idx, "_temporal_registry_config", lambda: (frozenset(), ("wind_avg",)))
    assert idx._is_temporal_variable_id("wind_avg_6h")
    assert not idx._is_temporal_variable_id("other_6h")

    monkeypatch.setattr(idx, "global_relative_positions_dir", lambda: tmp_path / "missing_global")
    assert idx._load_global_relative_rows("1", "bio_1") is None

    assert idx._harmonize_numeric_arrays([]) == []
    targets = idx.index_targets_for_columns(
        {"bio_1"},
        layer_catalog={
            "": {"value_type": "numeric"},
            "bio_1": {"value_type": "numeric"},
            "wind": {"value_type": "numeric", "agg": "avg"},
        },
    )
    assert targets == [("bio_1", "numeric")]

    assert idx._normalize_fallback_samples(None) == 0
    assert idx._normalize_sample_count("bad") is None
    assert idx._normalize_sample_count(3) == 3

    # _cached_metric_rows_for_taxon empty-source branches
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: None)
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: None)
    assert idx._cached_metric_rows_for_taxon("1", "/tmp/z") == ()

    # _load_column_lengths malformed metadata
    p = tmp_path / "x.parquet"
    schema = pa.schema([pa.field("x", pa.int64())]).with_metadata({b"column_lengths": b"not-json"})
    stub._schemas[p] = schema
    assert idx._load_column_lengths(p) == {}

    assert idx.build_density_curve([], point_count=8) is None


def test_rank_catalog_and_options_edge_paths(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path, "scientific_name": "Anc"}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx, "_infer_sample_count", lambda _t: 1)
    monkeypatch.setattr(
        idx.taxa_navigation,
        "iter_descendants_by_rank",
        lambda *_a, **_k: [{"taxon_key": "2"}, {"taxon_key": "2"}, {"taxon_key": ""}],
    )
    idx.build_descendant_catalog_parquet("1", "species")
    assert (tmp_path / "species.parquet").exists()

    anc_non_species = {"taxon_key": "9", "rank": "GENUS", "path": tmp_path, "scientific_name": "N"}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc_non_species)
    idx.build_descendant_catalog_parquet("9", "subspecies")
    assert not (tmp_path / "subspecies.parquet").exists()

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    with pytest.raises(ValueError):
        idx.build_descendant_catalog_parquet("missing", "species")
    with pytest.raises(ValueError):
        idx.list_rank_metric_options("missing", "species")

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    with pytest.raises(ValueError):
        idx.list_rank_metric_options("1", "")

    index_path = tmp_path / "species_index.parquet"
    stub._exists[index_path] = False
    assert idx.list_rank_metric_options("1", "species") == []

    stub._exists[index_path] = True
    stub._schemas[index_path] = OSError("boom")
    assert idx.list_rank_metric_options("1", "species") == []

    stub._schemas[index_path] = pa.schema(
        [pa.field("x", pa.int64()), pa.field("bio_1::mean", pa.int64()), pa.field("bio_1::max", pa.int64())]
    ).with_metadata({b"column_lengths": json.dumps({"bio_1::mean": 7, "bio_1::max": 0}).encode("utf-8")})
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_variable_metadata",
        lambda: ([{"id": "bio_12"}, {"id": "bio_1"}], {"bio_12": {"id": "bio_12"}, "bio_1": {"id": "bio_1"}}),
    )
    assert idx.list_rank_metric_options("1", "species") == [
        {"variable": "bio_1", "metric": "mean", "label": "Average", "column": "bio_1::mean", "count": 7}
    ]


def test_list_rank_metric_options_uses_variable_metadata_order(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    ancestor = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: ancestor)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [{"id": "bio_12"}, {"id": "bio_1"}],
            {"bio_12": {"id": "bio_12"}, "bio_1": {"id": "bio_1"}},
        ),
    )

    index_path = tmp_path / "species_index.parquet"
    stub._exists[index_path] = True
    stub._schemas[index_path] = pa.schema(
        [pa.field("bio_1::mean", pa.int64()), pa.field("bio_12::mean", pa.int64())]
    ).with_metadata(
        {b"column_lengths": json.dumps({"bio_1::mean": 3, "bio_12::mean": 4}).encode("utf-8")}
    )

    assert idx.list_rank_metric_options("1", "species") == [
        {"variable": "bio_12", "metric": "mean", "label": "Average", "column": "bio_12::mean", "count": 4},
        {"variable": "bio_1", "metric": "mean", "label": "Average", "column": "bio_1::mean", "count": 3},
    ]


def test_query_taxa_scales_named_categorical_fraction_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        idx, "CONFIG", SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES"})
    )
    species_one = {
        "taxon_key": "1",
        "rank": "SPECIES",
        "path": tmp_path / "species_1",
        "scientific_name": "Species One",
    }
    species_two = {
        "taxon_key": "2",
        "rank": "SPECIES",
        "path": tmp_path / "species_2",
        "scientific_name": "Species Two",
    }
    for taxon in (species_one, species_two):
        taxon["path"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        idx.taxa_navigation,
        "search_taxa_by_name",
        lambda _query, limit=100: [(species_one, 80.0), (species_two, 95.0)][:limit],
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(
        idx.taxa_navigation,
        "extract_common_names_for_language",
        lambda taxon, **_kwargs: [taxon["scientific_name"]],
    )
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(idx.taxa_navigation, "get_parent_taxon", lambda _taxon: None)
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_variable_metadata",
        lambda: ([], {"landcover": {"value_type": "categorical"}}),
    )
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _path: {})
    monkeypatch.setattr(
        idx,
        "_load_categorical_stats",
        lambda path: (
            {"landcover": {"bare_areas": 0.125, "count": 5}}
            if str(path).endswith("species_1")
            else {"landcover": {"bare_areas": 0.375, "count": 6}}
        ),
    )

    out = idx.query_taxa(
        q="species",
        sort_variable="landcover",
        sort_metric="bare_areas",
        sort_order="asc",
    )

    assert [row["taxon_id"] for row in out["results"]] == [1, 2]
    assert out["results"][0]["sort_value"] == 12.5
    assert out["results"][1]["sort_value"] == 37.5


def test_relative_and_child_ranking_edge_paths(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    ancestor = {"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "Anc", "common_name": None}
    target = {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir, "scientific_name": "Target", "common_name": None}

    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: target if str(k) == "1" else (ancestor if str(k) == "10" else None),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [ancestor])

    # local positions missing path
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    assert idx.load_relative_ranks(taxon_dir, "bio_1") == []

    # location filter rejects target taxon
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({2}))
    assert idx.load_relative_ranks(taxon_dir, "bio_1", location_gid="USA") == []

    # explicit contexts path with missing lookup taxon and missing index
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    assert idx.load_relative_ranks(taxon_dir, "bio_1", location_gid="USA") == []

    # child rankings invalid rank/order and error surfacing
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: target if str(k) == "1" else None)
    with pytest.raises(ValueError):
        idx.child_relative_rankings("1", "", "bio_1", "mean")
    with pytest.raises(ValueError):
        idx.child_relative_rankings("1", "species", "bio_1", "mean", order="weird")

    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    index_path = taxon_dir / "species_index.parquet"
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    stub._exists[index_path] = True
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad column")))
    with pytest.raises(ValueError):
        idx.child_relative_rankings("1", "species", "bio_1", "mean")

    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 0})
    assert idx.child_relative_rankings("1", "species", "bio_1", "mean") == ([], None)

    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([None], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset())
    assert idx.child_relative_rankings("1", "species", "bio_1", "mean", location_gid="USA") == ([], None)


def test_write_and_build_rank_index_remaining_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    p = tmp_path / "r.parquet"

    # _build_rank_index_arrays branches
    assert idx._build_rank_index_arrays({"bio_1::mean": []}) == ({}, {}, set(), 0)
    arrays, lengths, metrics, max_len = idx._build_rank_index_arrays(
        {
            "bio_1": [{"taxon_key": "1", "value": 1.0, "sample_count": 2}],
            "bio_2::mean": [
                {"taxon_key": "1", "value": 1.0, "sample_count": 2},
                {"taxon_key": "2", "value": 2.0, "sample_count": 3},
            ],
        }
    )
    assert lengths["bio_1"] == 1 and "bio_1" in metrics and max_len == 2 and len(arrays["bio_1"]) == 2

    # _write_rank_index empty unlink branch
    p.touch()
    idx._write_rank_index(p, {}, merge_existing=False)
    assert not p.exists()

    # merge branch with existing names lacking "::" + padding paths
    existing = pa.StructArray.from_arrays(
        [
            pa.array(["1", "2"], type=pa.string()),
            pa.array([1.0, 2.0], type=pa.float64()),
            pa.array([2, 2], type=pa.int32()),
        ],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(pa.table({"nocolon": existing}), p)
    stub._schemas[p] = pq.read_schema(p)
    stub._tables[p] = pq.read_table(p)
    idx._write_rank_index(
        p,
        {"bio_1::mean": [{"taxon_key": "1", "value": 1.0, "sample_count": 2}]},
        merge_existing=True,
    )
    assert p.exists()

    # atomic cleanup branch in _write_rank_index
    original_write = idx.pq.write_table
    monkeypatch.setattr(idx.pq, "write_table", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        idx._write_rank_index(
            p,
            {"bio_1::mean": [{"taxon_key": "1", "value": 1.0, "sample_count": 2}]},
            merge_existing=False,
        )
    monkeypatch.setattr(idx.pq, "write_table", original_write)

    anc = {"taxon_key": "1", "path": tmp_path}
    catalog = tmp_path / "species.parquet"
    index = tmp_path / "species_index.parquet"

    # _build_rank_index_parquet missing/invalid/empty catalog branches
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    idx._build_rank_index_parquet(anc, "SPECIES")
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    idx._build_rank_index_parquet(anc, "SPECIES")
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: pa.table({"taxon_key": [], "sample_count": []}))
    idx._build_rank_index_parquet(anc, "SPECIES")

    # index schema read failure and temporal-column rebuild path
    index.touch()
    monkeypatch.setattr(
        idx.PARQUET, "read_table", lambda *_a, **_k: pa.table({"taxon_key": ["x"], "sample_count": [1]})
    )
    monkeypatch.setattr(idx.PARQUET, "read_schema", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    idx._build_rank_index_parquet(anc, "SPECIES")
    monkeypatch.setattr(
        idx.PARQUET, "read_schema", lambda *_a, **_k: pa.schema([pa.field("wind_avg_6h::mean", pa.int64())])
    )
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda c: c.startswith("wind_"))
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "x", "path": tmp_path / "x"})
    monkeypatch.setattr(idx, "_collect_metric_entries_for_taxon", lambda *_a, **_k: {})
    idx._build_rank_index_parquet(anc, "SPECIES")

    # build_rank_indexes_for_ancestor missing ancestor / missing catalog cleanup
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    with pytest.raises(ValueError):
        idx.build_rank_indexes_for_ancestor("x")
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx, "_descendant_rank_targets", lambda _r: ["SPECIES"])
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    idx.build_rank_indexes_for_ancestor("1")
    assert not catalog.exists()


def test_indexing_density_ancestor_and_ranking_branches(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env
    curve = idx.build_density_curve([1.0, 2.0, 3.0], point_count=8)
    assert curve is not None
    assert curve["points"]

    # _ancestor_contexts branches
    cfg.taxonomy_root = tmp_path / "tax"
    cfg.taxonomy_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx, "CONFIG", cfg)
    assert idx._ancestor_contexts(Path("/")) == []
    assert idx._ancestor_contexts(tmp_path / "outside" / "species_1") == []

    p = cfg.taxonomy_root / "genus_10" / "species_1"
    p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            {"taxon_key": str(k), "rank": "GENUS", "path": p.parent, "scientific_name": "Genus"}
            if str(k) == "10"
            else None
        ),
    )
    ctx = idx._ancestor_contexts(p)
    assert ctx and ctx[0]["taxon_key"] == "10"

    # load_relative_ranks local fallback read path and filtering branches
    target = {"taxon_key": "1", "rank": "SPECIES", "path": p, "scientific_name": "Species", "common_name": None}
    ancestor = {"taxon_key": "10", "rank": "GENUS", "path": p.parent, "scientific_name": "Genus", "common_name": None}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: target if str(k) == "1" else (ancestor if str(k) == "10" else None),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [ancestor])
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    pos = p / "relative_ranks_positions.parquet"
    pos.touch()
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _path: True)
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: (_ for _ in ()).throw(TypeError("x")))
    stub.is_remote = True
    monkeypatch.setattr(
        idx.pq,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": ["bio_1"],
                "metric": ["mean"],
                "position": [0],
                "count": [1],
                "sampleCount": [1],
                "contextTaxonId": ["10"],
                "contextLabel": ["Genus"],
            }
        ),
    )
    out = idx.load_relative_ranks(p, "bio_1")
    assert out and out[0]["metric"] == "mean"
    stub.is_remote = False

    # child_relative_rankings fast and slow-path skip branches + media fallbacks
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: {
            "taxon_key": str(k),
            "rank": "SUBSPECIES" if str(k) == "2" else "SPECIES",
            "scientific_name": f"T{str(k)}",
            "path": p,
        },
    )
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "resolve_taxon_media",
        lambda _k: {"url": "u", "license": "l", "creator": "c", "rightsHolder": "r", "references": ["ref"]},
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda _t: {})
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["2", "3"], type=pa.string()),
                pa.array([None, 2.0], type=pa.float64()),
                pa.array([0, 2], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({2, 3}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {3: 2})
    fast, dist = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", location_gid="USA", return_distribution=False
    )
    assert fast and dist is None

    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda _t: {"image_url": "p"})
    slow, dist2 = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", include_species_like=True, return_distribution=True, min_samples=1
    )
    assert dist2 is not None and slow


def test_indexing_relative_and_child_additional_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    tdir = tmp_path / "species_1"
    tdir.mkdir(parents=True, exist_ok=True)
    taxon = {"taxon_key": "1", "rank": "SPECIES", "path": tdir, "scientific_name": "S", "common_name": None}
    ancestor = {"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G", "common_name": None}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: taxon if str(k) == "1" else ancestor)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [ancestor])

    # local fallback: non-remote PARQUET.read_table in TypeError fallback raises OSError
    pos = tdir / "relative_ranks_positions.parquet"
    pos.touch()
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    stub.is_remote = False

    calls = {"n": 0}

    def _read_table(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("x")
        raise OSError("x")

    monkeypatch.setattr(idx.PARQUET, "read_table", _read_table)
    assert idx.load_relative_ranks(tdir, "bio_1") == []

    # empty table and empty-filter branches
    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": [],
                "metric": [],
                "position": [],
                "count": [],
                "sampleCount": [],
                "contextTaxonId": [],
                "contextLabel": [],
            }
        ),
    )
    assert idx.load_relative_ranks(tdir, "bio_1") == []
    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": ["other"],
                "metric": ["mean"],
                "position": [0],
                "count": [1],
                "sampleCount": [1],
                "contextTaxonId": ["10"],
                "contextLabel": ["G"],
            }
        ),
    )
    assert idx.load_relative_ranks(tdir, "bio_1") == []

    # location-filtered branch with lookup_taxon missing and bad metric values
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(
        idx,
        "_resolve_column_name",
        lambda *_a, **_k: "bio_1::mean",
    )
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["2", "1"], type=pa.string()),
                pa.array([1.0, 1.0], type=pa.float64()),
                pa.array([1, 1], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": "bad"}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: None if str(k) == "1" else ancestor)
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []

    # child_relative_rankings fast path descending + skip branches
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            None
            if str(k) == "x"
            else {
                "taxon_key": str(k),
                "rank": "SUBSPECIES" if str(k) == "2" else "SPECIES",
                "scientific_name": f"T{str(k)}",
                "path": tdir,
            }
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "resolve_taxon_media",
        lambda _k: {"url": "u", "license": "l", "creator": "c", "rightsHolder": "r", "references": ["ref"]},
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda _t: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 4})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "x", "2", "3"], type=pa.string()),
                pa.array([1.0, 2.0, None, 4.0], type=pa.float64()),
                pa.array([1, 1, 1, None], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1, 2, 3}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 1, 3: 2})
    fast, _ = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", order="desc", return_distribution=False, location_gid="USA"
    )
    assert fast

    # child_relative_rankings slow path empty-eligible then media fallback record creation
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["2", "3"], type=pa.string()),
                pa.array([1.0, 2.0], type=pa.float64()),
                pa.array([None, None], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    assert idx.child_relative_rankings("1", "species", "bio_1", "mean", include_species_like=True, min_samples=1) == (
        [],
        None,
    )

    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["2"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([2], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    slow, _dist = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", include_species_like=True, return_distribution=False
    )
    assert slow and slow[0]["image_url"] == "u"


def test_indexing_helper_and_density_extra_branches(monkeypatch):
    mixed = idx._harmonize_numeric_arrays([pa.array([1], type=pa.int64()), pa.array(["x"], type=pa.string())])
    assert mixed[0].type == pa.int64() and mixed[1].type == pa.string()
    curve = idx.build_density_curve([2.0, 2.0], point_count=8)
    assert curve and curve["min"] < curve["max"]
    original_isfinite = idx.math.isfinite
    monkeypatch.setattr(idx.math, "isfinite", lambda _x: False)
    assert idx.build_density_curve([1.0, 2.0], point_count=8)
    monkeypatch.setattr(idx.math, "isfinite", original_isfinite)


def test_indexing_misc_helper_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env

    # _infer_sample_count / cached branches
    original_cached = idx._infer_sample_count_cached
    monkeypatch.setattr(idx, "_infer_sample_count_cached", lambda *_a, **_k: 4)
    assert idx._infer_sample_count({"taxon_key": "", "path": tmp_path}) == 4
    monkeypatch.setattr(idx, "_infer_sample_count_cached", original_cached)
    idx._infer_sample_count_cached.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"count": "bad"}})
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    assert idx._infer_sample_count_cached("1", str(tmp_path)) == 0
    idx._infer_sample_count_cached.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"count": None}})
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1"})
    monkeypatch.setattr(idx.taxa_navigation, "count_taxon_rows", lambda _t: None)
    assert idx._infer_sample_count_cached("1", str(tmp_path)) == 0
    idx.reset_rank_build_caches()
    assert idx._descendant_rank_targets("SUBSPECIES") == []
    assert idx._descendant_rank_targets("VARIETY") == []

    # _collect_metric_entries skip branches
    original_metric_rows = idx._cached_metric_rows_for_taxon
    monkeypatch.setattr(idx, "_cached_metric_rows_for_taxon", lambda *_a, **_k: ())
    assert idx._collect_metric_entries_for_taxon({"taxon_key": "1", "path": tmp_path}, 3) == {}
    monkeypatch.setattr(
        idx,
        "_cached_metric_rows_for_taxon",
        lambda *_a, **_k: (
            ("wind_avg_6h::mean", 1.0, 2),
            ("bio_1::mean", 1.0, 0),
        ),
    )
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda c: c.startswith("wind_"))
    assert idx._collect_metric_entries_for_taxon({"taxon_key": "1", "path": tmp_path}, 0) == {}

    # _cached_metric_rows source-empty branch
    monkeypatch.setattr(idx, "_cached_metric_rows_for_taxon", original_metric_rows)
    idx._cached_metric_rows_for_taxon.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})
    rows = idx._cached_metric_rows_for_taxon("1", str(tmp_path))
    assert rows and rows[0][0] == "bio_1::mean"

    # _write_rank_index merge existing metric parsing and padding branches
    p = tmp_path / "m.parquet"
    arr = pa.StructArray.from_arrays(
        [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(pa.table({"bio_1::mean": arr}), p)
    stub._schemas[p] = pq.read_schema(p)
    stub._tables[p] = pq.read_table(p)
    idx._write_rank_index(
        p,
        {
            "bio_2::mean": [
                {"taxon_key": "1", "value": 1.0, "sample_count": 1},
                {"taxon_key": "2", "value": 2.0, "sample_count": 1},
            ]
        },
        merge_existing=True,
    )
    assert p.exists()

    # _build_rank_index_parquet prints/skip cases for missing keys and lookups
    anc = {"taxon_key": "1", "path": tmp_path}
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(
        idx.PARQUET, "read_table", lambda *_a, **_k: pa.table({"taxon_key": [None, "x"], "sample_count": [1, 1]})
    )
    monkeypatch.setattr(
        idx.PARQUET, "read_schema", lambda *_a, **_k: pa.schema([pa.field("wind_avg_6h::mean", pa.int64())])
    )
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda c: c.startswith("wind_"))
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    monkeypatch.setattr(idx, "_collect_metric_entries_for_taxon", lambda *_a, **_k: {})
    idx._build_rank_index_parquet(anc, "SPECIES")

    # _load_column_lengths / _resolve_column_name / _load_struct_column edges
    bad_schema_path = tmp_path / "bad_schema.parquet"
    stub._schemas[bad_schema_path] = OSError("x")
    assert idx._load_column_lengths(bad_schema_path) == {}
    schema_path = tmp_path / "schema.parquet"
    stub._schemas[schema_path] = pa.schema([pa.field("bio_1::mean", pa.int64())]).with_metadata({})
    with pytest.raises(ValueError):
        idx._resolve_column_name(schema_path, "bio_1", "mean")
    stub._schemas[schema_path] = OSError("x")
    with pytest.raises(ValueError):
        idx._resolve_column_name(schema_path, "bio_1", "mean")
    stub._tables[schema_path] = pa.table({"bio_1::mean": arr})
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda p, **_k: stub._tables[Path(p)])
    col = idx._load_struct_column(schema_path, "bio_1::mean", 0)
    assert len(col) == 0

    # load_relative_ranks early-return branches
    assert idx.load_relative_ranks(tmp_path / "plain", "bio_1") == []
    assert idx.load_relative_ranks(tmp_path / "species_", "bio_1") == []


def test_indexing_descendant_and_relative_remaining_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    anc = {"taxon_key": "1", "rank": "SPECIES", "path": tmp_path, "scientific_name": "A"}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper() if r is not None else None)
    monkeypatch.setattr(idx, "_descendant_rank_targets", lambda _r: ["SUBSPECIES", "SPECIES", "GENUS"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "iter_descendants",
        lambda *_a, **_k: [
            {"taxon_key": "a", "rank": None},
            {"taxon_key": "b", "rank": "SUBSPECIES"},
            {"taxon_key": "c", "rank": "SPECIES"},
        ],
    )
    monkeypatch.setattr(stub, "exists", lambda p: p.name in {"subspecies.parquet", "species.parquet", "genus.parquet"})
    idx.build_descendant_catalogs_for_ancestor("1")

    # location-filtered load_relative_ranks continuation branches
    tdir = tmp_path / "species_1"
    tdir.mkdir(parents=True, exist_ok=True)
    target = {"taxon_key": "1", "rank": "SPECIES", "path": tdir, "scientific_name": "S", "common_name": None}
    ancestor = {"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G", "common_name": None}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: target if str(k) == "1" else ancestor)
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [ancestor])
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: None if str(k) == "1" else ancestor)
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda k: target if str(k) == "1" else ancestor)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")))
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []

    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {})
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []

    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": "bad"}})
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": float("inf")}})
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []


def test_list_rank_metric_options_uses_subspecies_storage_for_alias(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(common_name_language="en", subspecies_equivalents={"SUBSPECIES", "VARIETY", "FORM"}),
    )
    ancestor = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: ancestor)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")

    index_path = tmp_path / "subspecies_index.parquet"
    stub._exists[index_path] = True
    stub._schemas[index_path] = pa.schema([pa.field("bio_1::mean", pa.int64())]).with_metadata(
        {b"column_lengths": json.dumps({"bio_1::mean": 3}).encode("utf-8")}
    )

    assert idx.list_rank_metric_options("1", "variety") == [
        {"variable": "bio_1", "metric": "mean", "label": "Average", "column": "bio_1::mean", "count": 3}
    ]


def test_child_relative_rankings_uses_subspecies_storage_for_alias(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    monkeypatch.setattr(
        idx,
        "CONFIG",
        SimpleNamespace(
            common_name_language="en",
            subspecies_equivalents={"SUBSPECIES", "VARIETY", "FORM"},
            species_rank="SPECIES",
        ),
    )
    ancestor = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    variety = {"taxon_key": "2", "rank": "VARIETY", "scientific_name": "Variety Two", "path": tmp_path / "v2"}
    form = {"taxon_key": "3", "rank": "FORM", "scientific_name": "Form Three", "path": tmp_path / "f3"}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: (
            ancestor if str(key) == "1" else (variety if str(key) == "2" else (form if str(key) == "3" else None))
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})

    index_path = tmp_path / "subspecies_index.parquet"
    arr = pa.StructArray.from_arrays(
        [
            pa.array(["2", "3"], type=pa.string()),
            pa.array([1.0, 2.0], type=pa.float64()),
            pa.array([5, 5], type=pa.int32()),
        ],
        fields=[
            pa.field("taxonKey", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("sampleCount", pa.int32()),
        ],
    )
    pq.write_table(
        pa.table({"bio_1::mean": arr}).replace_schema_metadata(
            {b"column_lengths": json.dumps({"bio_1::mean": 2}).encode("utf-8")}
        ),
        index_path,
    )
    stub._exists[index_path] = True
    stub._schemas[index_path] = pq.read_schema(index_path)
    stub._tables[index_path] = pq.read_table(index_path)

    rows, distribution = idx.child_relative_rankings("1", "variety", "bio_1", "mean", limit=10, order="asc")

    assert distribution == [1.0]
    assert [row["taxon_id"] for row in rows] == [2]
    assert all(row["rank"] == "VARIETY" for row in rows)


def test_indexing_child_ranking_remaining_branches(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env
    anc = {"taxon_key": "1", "rank": "GENUS", "path": tmp_path}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            anc
            if str(k) == "1"
            else {
                "taxon_key": str(k),
                "rank": "SUBSPECIES" if str(k) == "2" else "SPECIES",
                "scientific_name": "T",
                "path": tmp_path,
            }
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "resolve_taxon_media",
        lambda _k: {"url": "u", "license": "l", "creator": "c", "rightsHolder": "r", "references": ["ref"]},
    )
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({2, 3}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {2: 2, 3: 2})

    # fast path parse-exception + preferred-image + media fallback
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["2", "3"], type=pa.string()),
                pa.array([None, 3.0], type=pa.float64()),
                pa.array([2, 2], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation, "preferred_image_payload", lambda t: {"image_url": "p"} if t["taxon_key"] == "3" else {}
    )
    fast, _ = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", return_distribution=False, location_gid="USA"
    )
    assert fast and fast[0].get("image_url")

    # slow path taxon-none, allowed skip, species-like skip
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["x", "2", "3"], type=pa.string()),
                pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                pa.array([1, 1, 1], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            None
            if str(k) == "x"
            else (
                anc
                if str(k) == "1"
                else {"taxon_key": str(k), "rank": "SUBSPECIES", "scientific_name": "T", "path": tmp_path}
            )
        ),
    )
    out, _ = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", include_species_like=False, return_distribution=True, location_gid="USA"
    )
    assert out == []


def test_build_index_parquet_rewrite_cleanup_branch(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    occ = node / "occurrence.parquet"
    pq.write_table(pa.table({"catalogNumber": ["b", "a"], "bio_1": [2.0, 1.0]}), occ)
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    monkeypatch.setattr(idx.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "numeric"}})
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [])
    original_write = idx.pq.write_table
    monkeypatch.setattr(idx.pq, "write_table", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        idx.build_index_parquet(node)
    monkeypatch.setattr(idx.pq, "write_table", original_write)


def test_build_index_parquet_origin_map_and_pending_skip(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    occ = node / "occurrence.parquet"
    pq.write_table(pa.table({"catalogNumber": ["a"], "bio_1": [1.0]}), occ)

    missing_child = node / "subspecies_missing"
    bad_child = node / "subspecies_bad"
    nocat_child = node / "subspecies_nocat"
    intcat_child = node / "subspecies_intcat"
    for d in (bad_child, nocat_child, intcat_child):
        d.mkdir(parents=True, exist_ok=True)
    bad_occ = bad_child / "occurrence.parquet"
    nocat_occ = nocat_child / "occurrence.parquet"
    intcat_occ = intcat_child / "occurrence.parquet"
    pq.write_table(pa.table({"catalogNumber": ["x"], "bio_1": [1.0]}), bad_occ)
    pq.write_table(pa.table({"x": [1]}), nocat_occ)
    pq.write_table(pa.table({"catalogNumber": [1], "bio_1": [2.0]}), intcat_occ)

    index_path = node / "occurrence_index.parquet"
    arr = pa.StructArray.from_arrays(
        [pa.array(["a"], type=pa.string()), pa.array([0], type=pa.int32()), pa.array([1.0], type=pa.float64())],
        fields=[
            pa.field("catalogNumber", pa.string()),
            pa.field("originId", pa.int32()),
            pa.field("value", pa.float64()),
        ],
    )
    meta = {
        b"origin_map": json.dumps(
            [
                {"id": 0, "relative_path": ".", "taxon_key": "1"},
                {"id": 1, "relative_path": "subspecies_missing", "taxon_key": "2"},
                {"id": 2, "relative_path": "subspecies_bad", "taxon_key": "3"},
                {"id": 3, "relative_path": "subspecies_nocat", "taxon_key": "4"},
                {"id": 4, "relative_path": "subspecies_intcat", "taxon_key": "5"},
            ]
        ).encode("utf-8"),
        b"column_lengths": json.dumps({"bio_1": 1}).encode("utf-8"),
        b"catalog_column": b"catalogNumber",
    }
    pq.write_table(pa.table({"bio_1": arr}).replace_schema_metadata(meta), index_path)

    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    monkeypatch.setattr(idx.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "numeric"}})
    monkeypatch.setattr(
        idx.taxa_navigation,
        "taxon_key_from_path",
        lambda p: Path(p).name.split("_")[-1] if "_" in Path(p).name else "1",
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_children",
        lambda _k: [{"taxon_key": "2", "path": missing_child}, {"taxon_key": "4", "path": nocat_child}],
    )
    real_read = idx.pq.read_table
    monkeypatch.setattr(
        idx.pq,
        "read_table",
        lambda p, **k: (_ for _ in ()).throw(RuntimeError("x")) if Path(p) == bad_occ else real_read(p, **k),
    )
    idx.build_index_parquet(node)


def test_build_index_parquet_categorical_cast_and_merge_paths(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    occ = node / "occurrence.parquet"
    pq.write_table(
        pa.table(
            {
                "catalogNumber": [1, 2, 3],
                "bio_cat": [1.1, 2.2, None],
                "bio_num": [5.0, None, None],
            }
        ),
        occ,
    )
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [])
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_layer_metadata",
        lambda: {"bio_cat": {"value_type": "categorical"}, "bio_num": {"value_type": "numeric"}},
    )
    idx.build_index_parquet(node)
    assert (node / "occurrence_index.parquet").exists()

    # merge-existing path with missing column_lengths metadata
    existing = pa.StructArray.from_arrays(
        [pa.array(["old"], type=pa.string()), pa.array([0], type=pa.int32()), pa.array([0.5], type=pa.float64())],
        fields=[
            pa.field("catalogNumber", pa.string()),
            pa.field("originId", pa.int32()),
            pa.field("value", pa.float64()),
        ],
    )
    idx_path = node / "occurrence_index.parquet"
    pq.write_table(pa.table({"old_col": existing}), idx_path)
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_layer_metadata",
        lambda: {"bio_num": {"value_type": "numeric"}},
    )
    idx.build_index_parquet(node)
    assert idx_path.exists()


def test_indexing_remaining_branch_targets(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env

    # build_index_parquet branch with missing parent file (line 244)
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx.gis_lookup, "load_layer_metadata", lambda: {})
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [])
    idx.build_index_parquet(node)

    # build_index_parquet child cast + empty filtered + combined-values skip
    occ = node / "occurrence.parquet"
    pq.write_table(pa.table({"catalogNumber": ["a", "b"], "bio_1": [None, None], "bio_2": [1.5, 2.5]}), occ)
    child = node / "subspecies_2"
    child.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"catalogNumber": [1], "bio_2": [3.5]}), child / "occurrence.parquet")
    monkeypatch.setattr(
        idx.gis_lookup,
        "load_layer_metadata",
        lambda: {"bio_1": {"value_type": "numeric"}, "bio_2": {"value_type": "categorical"}},
    )
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [{"taxon_key": "2", "path": child}])
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    idx.build_index_parquet(node)

    # build_index_parquet schema-read failure branch and final write cleanup
    idx_path = node / "occurrence_index.parquet"
    idx_path.touch(exist_ok=True)
    monkeypatch.setattr(idx.pq, "read_schema", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    original_write = idx.pq.write_table
    monkeypatch.setattr(idx.pq, "write_table", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        idx.build_index_parquet(node)
    monkeypatch.setattr(idx.pq, "write_table", original_write)

    # _write_descendant_catalog cleanup branch
    out = tmp_path / "species.parquet"
    monkeypatch.setattr(idx, "_infer_sample_count", lambda _t: 1)
    monkeypatch.setattr(idx.pq, "write_table", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        idx._write_descendant_catalog(out, [{"taxon_key": "1"}])
    monkeypatch.setattr(idx.pq, "write_table", original_write)

    # _cached_metric_rows 'not source' branch
    idx._cached_metric_rows_for_taxon.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    assert idx._cached_metric_rows_for_taxon("1", str(tmp_path))

    # _build_rank_index_parquet temporal rebuild + missing taxon_key(None) message path
    anc = {"taxon_key": "1", "path": tmp_path}
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(
        idx.PARQUET, "read_schema", lambda *_a, **_k: pa.schema([pa.field("wind_avg_6h::mean", pa.int64())])
    )
    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table({"taxon_key": pa.array([None], type=pa.string()), "sample_count": [1]}),
    )
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda c: c.startswith("wind_"))
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    monkeypatch.setattr(idx, "_collect_metric_entries_for_taxon", lambda *_a, **_k: {})
    idx._build_rank_index_parquet(anc, "SPECIES")

    # _ancestor_contexts parent==taxonomy_root break
    _cfg.taxonomy_root = tmp_path / "tx"
    (_cfg.taxonomy_root / "species_1").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx, "CONFIG", _cfg)
    assert idx._ancestor_contexts(_cfg.taxonomy_root / "species_1") == []

    # load_relative_ranks contexts-empty and empty-location branches
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1", "rank": "SPECIES", "path": tmp_path}
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [])
    assert idx.load_relative_ranks(tmp_path / "species_1", "bio_1") == []
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset())
    assert idx.load_relative_ranks(tmp_path / "species_1", "bio_1", location_gid="USA") == []

    # local ranks read error + cast fallback + invalid count branch
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    pos = tmp_path / "species_1" / "relative_ranks_positions.parquet"
    pos.parent.mkdir(parents=True, exist_ok=True)
    pos.touch()
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    assert idx.load_relative_ranks(tmp_path / "species_1", "bio_1") == []

    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": [1],
                "metric": ["mean"],
                "position": [0],
                "count": [0],
                "sampleCount": [1],
                "contextTaxonId": ["10"],
                "contextLabel": ["G"],
            }
        ),
    )
    orig_cast = idx.PC.cast
    orig_equal = idx.PC.equal
    monkeypatch.setattr(idx.PC, "cast", lambda *_a, **_k: (_ for _ in ()).throw(pa.ArrowInvalid("x")))
    monkeypatch.setattr(idx.PC, "equal", lambda *_a, **_k: pa.array([True]))
    assert idx.load_relative_ranks(tmp_path / "species_1", "bio_1") == []

    # location-filtered deeper continuation branches
    monkeypatch.setattr(idx.PC, "cast", orig_cast)
    monkeypatch.setattr(idx.PC, "equal", orig_equal)
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["2"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: {"taxon_key": "1", "rank": "SPECIES", "path": tmp_path} if str(k) == "1" else None,
    )
    assert idx.load_relative_ranks(tmp_path / "species_1", "bio_1", location_gid="USA") == []

    # child_relative_rankings fast parse fail + media fallback and slow allowed skip
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: {"taxon_key": str(k), "rank": "SPECIES", "scientific_name": "T", "path": tmp_path},
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "resolve_taxon_media",
        lambda _k: {"url": "u", "license": "l", "creator": "c", "rightsHolder": "r", "references": ["ref"]},
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda _t: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2"], type=pa.string()),
                pa.array([None, 2.0], type=pa.float64()),
                pa.array([2, 2], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1, 2}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 2, 2: 2})
    fast, _ = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", return_distribution=False, location_gid="USA"
    )
    assert fast and fast[0]["image_url"] == "u"

    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 2})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2"], type=pa.string()),
                pa.array([2.0, 3.0], type=pa.float64()),
                pa.array([2, 2], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    slow, _ = idx.child_relative_rankings("1", "species", "bio_1", "mean", return_distribution=True, location_gid="USA")
    assert slow


def test_indexing_last_missing_branches(stub_env, monkeypatch, tmp_path):
    cfg, stub = stub_env

    # 545-546 and 562-563 via categorical offsets + merge-existing max-len
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    occ = node / "occurrence.parquet"
    pq.write_table(pa.table({"catalogNumber": ["a", "b"], "koppen": [1, 2]}), occ)
    old = pa.StructArray.from_arrays(
        [
            pa.array(["x", "y"], type=pa.string()),
            pa.array([0, 0], type=pa.int32()),
            pa.array([1.0, 2.0], type=pa.float64()),
        ],
        fields=[
            pa.field("catalogNumber", pa.string()),
            pa.field("originId", pa.int32()),
            pa.field("value", pa.float64()),
        ],
    )
    pq.write_table(pa.table({"old_col": old}), node / "occurrence_index.parquet")
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    monkeypatch.setattr(idx.gis_lookup, "load_layer_metadata", lambda: {"koppen": {"value_type": "categorical"}})
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [])
    idx.build_index_parquet(node)

    # 833-835 and 857 in descendant catalogs
    anc = {"taxon_key": "1", "rank": "SPECIES", "path": tmp_path, "scientific_name": "A"}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: anc)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper() if r is not None else None)
    monkeypatch.setattr(idx, "_descendant_rank_targets", lambda _r: ["SUBSPECIES", "GENUS"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "iter_descendants",
        lambda *_a, **_k: [{"taxon_key": "s1", "rank": "SUBSPECIES"}, {"taxon_key": "g1", "rank": "GENUS"}],
    )
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    called = []
    monkeypatch.setattr(idx, "_write_descendant_catalog", lambda p, d: called.append((p.name, len(d))))
    idx.build_descendant_catalogs_for_ancestor("1")
    assert ("subspecies.parquet", 1) in called and ("genus.parquet", 1) in called

    # 942 in cached metric row source iteration
    idx._cached_metric_rows_for_taxon.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    assert idx._cached_metric_rows_for_taxon("1", str(tmp_path))

    # 1179 + 1201-1205 in rank-index builder
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(
        idx.PARQUET, "read_schema", lambda *_a, **_k: pa.schema([pa.field("wind_avg_6h::mean", pa.int64())])
    )

    class _T:
        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame({"taxon_key": [None], "sample_count": [1]}, dtype=object)

    (tmp_path / "species_index.parquet").touch()
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: _T())
    monkeypatch.setattr(idx, "_is_temporal_metric_column", lambda c: c.startswith("wind_"))
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    monkeypatch.setattr(idx, "_collect_metric_entries_for_taxon", lambda *_a, **_k: {})
    idx._build_rank_index_parquet({"taxon_key": "1", "path": tmp_path}, "SPECIES")

    # 1368-1369 in ancestor context traversal
    cfg.taxonomy_root = tmp_path / "tax_root"
    (cfg.taxonomy_root / "species_1").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx, "CONFIG", cfg)
    assert idx._ancestor_contexts(cfg.taxonomy_root / "species_1") == []

    # 1519 / 1563 / 1568 in location-filtered relative ranks
    tdir = cfg.taxonomy_root / "species_1"
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    original_load_column_lengths = idx._load_column_lengths
    original_resolve_column_name = idx._resolve_column_name
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1", "rank": "SPECIES", "path": tdir}
    )
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["2"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    assert idx.load_relative_ranks(tdir, "bio_1", location_gid="USA") == []

    # 1605-1606 and 1631-1632
    bad = tmp_path / "bad.parquet"
    monkeypatch.setattr(idx, "_load_column_lengths", original_load_column_lengths)
    monkeypatch.setattr(idx, "_resolve_column_name", original_resolve_column_name)
    monkeypatch.setattr(
        idx.PARQUET,
        "read_schema",
        lambda p: (_ for _ in ()).throw(ValueError("x")) if Path(p) == bad else stub._schemas[Path(p)],
    )
    assert idx._load_column_lengths(bad) == {}
    with pytest.raises(ValueError):
        idx._resolve_column_name(bad, "bio_1", "mean")

    # 1844 in child fast-path media fallback payload
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: {"taxon_key": str(k), "rank": "SPECIES", "scientific_name": "T", "path": tmp_path},
    )
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "resolve_taxon_media",
        lambda _k: {"url": "u", "license": "l", "creator": "c", "rightsHolder": "r", "references": ["ref"]},
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda _t: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([2.0], type=pa.float64()), pa.array([2], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 2})
    out, _ = idx.child_relative_rankings("1", "species", "bio_1", "mean", return_distribution=False, location_gid="USA")
    assert out and out[0]["image_license"] == "l"


def test_indexing_truly_last_lines(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env
    node = tmp_path / "species_1"
    node.mkdir(parents=True, exist_ok=True)
    occ = node / "occurrence.parquet"
    monkeypatch.setattr(stub, "read_table", lambda p, **k: pq.read_table(p, **k))
    monkeypatch.setattr(idx.taxa_navigation, "taxon_key_from_path", lambda _p: "1")
    monkeypatch.setattr(idx.taxa_navigation, "get_children", lambda _k: [])

    # 545-546: existing index + only-null new layer => no new layers
    old = pa.StructArray.from_arrays(
        [pa.array(["x"], type=pa.string()), pa.array([0], type=pa.int32()), pa.array([1.0], type=pa.float64())],
        fields=[
            pa.field("catalogNumber", pa.string()),
            pa.field("originId", pa.int32()),
            pa.field("value", pa.float64()),
        ],
    )
    pq.write_table(pa.table({"old_col": old}), node / "occurrence_index.parquet")
    pq.write_table(pa.table({"catalogNumber": ["a"], "bio_new": [None]}), occ)
    monkeypatch.setattr(idx.gis_lookup, "load_layer_metadata", lambda: {"bio_new": {"value_type": "numeric"}})
    idx.build_index_parquet(node)

    # 562-563: merge existing with shorter old array than new max_len
    pq.write_table(pa.table({"catalogNumber": ["a", "b"], "bio_new": [1.0, 2.0]}), occ)
    idx.build_index_parquet(node)

    # 942: empty metric bucket skip
    idx._cached_metric_rows_for_taxon.cache_clear()
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    assert idx._cached_metric_rows_for_taxon("1", str(tmp_path))

    # 1368-1369: parent segment without underscore under taxonomy root
    _cfg.taxonomy_root = tmp_path / "root"
    p = _cfg.taxonomy_root / "folder" / "species_1"
    p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(idx, "CONFIG", _cfg)
    assert idx._ancestor_contexts(p) == []

    # 1519 / 1563 / 1568
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({2}))
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})

    calls = {"n": 0}

    def _taxon_lookup(k):
        if str(k) != "1":
            return None
        calls["n"] += 1
        if calls["n"] == 1:
            return {"taxon_key": "1", "rank": "SPECIES", "path": p}
        return None

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", _taxon_lookup)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    assert idx.load_relative_ranks(p, "bio_1", location_gid="USA") == []

    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1", "rank": "SPECIES", "path": p}
    )
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    assert idx.load_relative_ranks(p, "bio_1", location_gid="USA") == []

    # 1844: fast path break on limit
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: {"taxon_key": str(k), "rank": "SPECIES", "scientific_name": "T", "path": p},
    )
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: ["N"])
    monkeypatch.setattr(
        idx.taxa_navigation,
        "resolve_taxon_media",
        lambda _k: {"url": "u", "license": "l", "creator": "c", "rightsHolder": "r", "references": ["ref"]},
    )
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda _t: {})
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 2})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2"], type=pa.string()),
                pa.array([2.0, 3.0], type=pa.float64()),
                pa.array([2, 2], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1, 2}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: 2, 2: 2})
    rows, _ = idx.child_relative_rankings(
        "1", "species", "bio_1", "mean", return_distribution=False, location_gid="USA", limit=1
    )
    assert len(rows) == 1


def test_indexing_targeted_uncovered_branches(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env

    # _load_global_relative_rows error + metric filter branch
    monkeypatch.setattr(idx, "global_relative_positions_dir", lambda: tmp_path)
    monkeypatch.setattr(idx.PARQUET, "is_remote", False)
    monkeypatch.setattr(idx.pds, "dataset", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")))
    assert idx._load_global_relative_rows("1", "bio_1") is None

    class _DS:
        def to_table(self, **_kwargs):
            return pa.table(
                {
                    "variable": ["bio_1"],
                    "metric": ["mean"],
                    "position": [0],
                    "count": [1],
                    "sampleCount": [1],
                    "contextTaxonId": ["10"],
                    "contextLabel": ["A"],
                }
            )

    monkeypatch.setattr(idx.pds, "dataset", lambda *_a, **_k: _DS())
    out = idx._load_global_relative_rows("1", "bio_1", metric_names=["mean"])
    assert out is not None and out.num_rows == 1

    # _context_column_rank + _descendant_catalog_sample_counts defensive paths.
    assert idx._context_column_rank("SUBSPECIES", "GENUS") == "SPECIES"
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    assert idx._descendant_catalog_sample_counts("missing", "species") == {}
    ancestor = {"taxon_key": "1", "path": tmp_path}
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: ancestor)
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: False)
    assert idx._descendant_catalog_sample_counts("no-file", "species") == {}
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    assert idx._descendant_catalog_sample_counts("read-fail", "species") == {}
    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table({"taxon_key": ["bad", "2", "3"], "sample_count": ["x", "0", "4"]}),
    )
    monkeypatch.setattr(
        idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)) if str(key).isdigit() else None
    )
    assert idx._descendant_catalog_sample_counts("ok", "species") == {3: 4}

    # _eligible_context_taxon_ids branches.
    monkeypatch.setattr(idx, "_descendant_catalog_sample_counts", lambda *_a, **_k: {1: 5, 2: 2, 3: 7})
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"rank": "GENUS", "taxon_key": "1", "path": tmp_path}
    )
    assert idx._eligible_context_taxon_ids(
        ancestor_taxon_id="1",
        target_rank="SPECIES",
        storage_rank="SPECIES",
        include_species_like=True,
        allowed_taxa=None,
        min_samples=0,
        location_counts=None,
    ) == {1, 2, 3}

    ranks = {"1": "SPECIES", "2": "SUBSPECIES", "3": "GENUS"}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: {"taxon_key": str(k), "rank": ranks.get(str(k), "SPECIES"), "path": tmp_path},
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    include_like = idx._eligible_context_taxon_ids(
        ancestor_taxon_id="1",
        target_rank="SPECIES",
        storage_rank="SPECIES",
        include_species_like=True,
        allowed_taxa=frozenset({1, 2, 3}),
        min_samples=2,
        location_counts={1: 3, 2: 2, 3: 2},
    )
    assert include_like == {1, 2}
    only_species = idx._eligible_context_taxon_ids(
        ancestor_taxon_id="1",
        target_rank="SPECIES",
        storage_rank="SPECIES",
        include_species_like=False,
        allowed_taxa=frozenset({1, 2, 3}),
        min_samples=0,
        location_counts=None,
    )
    assert only_species == {1}

    # location-filtered class metric branch in load_relative_ranks.
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    target = {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir, "scientific_name": "S"}
    ancestor_ctx = {"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            target
            if str(k) == "1"
            else (ancestor_ctx if str(k) == "10" else {"taxon_key": str(k), "rank": "SPECIES", "path": taxon_dir})
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: [ancestor_ctx])
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::class_1": 1})
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::class_1")
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {"bio_1": {"class_1": 0.4}})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([0.4], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx, "_eligible_context_taxon_ids", lambda **_k: {1, 2, 3})
    rows = idx.load_relative_ranks(taxon_dir, "bio_1", metric_names=["class_1"], location_gid="USA")
    assert rows and rows[0]["count"] == 3

    # child_relative_rankings location-counts fallback + class metric adjustment branch.
    anc = {"taxon_key": "99", "rank": "GENUS", "path": tmp_path}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            anc
            if str(k) == "99"
            else {"taxon_key": str(k), "rank": "SPECIES", "scientific_name": "T", "path": tmp_path}
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::class_1")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::class_1": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([2], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {1: None})
    monkeypatch.setattr(idx.gis_lookup, "location_counts_for_taxon", lambda _taxon_id: {("scope", "target"): 2})
    monkeypatch.setattr(idx, "_eligible_context_taxon_ids", lambda **_k: {1, 2, 3})
    ranked, dist = idx.child_relative_rankings(
        "99", "species", "bio_1", "class_1", return_distribution=True, location_gid="USA"
    )
    assert ranked and ranked[0]["count"] == 3 and dist == [1.0]


def test_indexing_remaining_defensive_edges(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env

    # _descendant_catalog_sample_counts: empty table and bad sample_count coercion.
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"path": tmp_path})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    idx._descendant_catalog_sample_counts.cache_clear()
    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: pa.table({"taxon_key": [], "sample_count": []}))
    assert idx._descendant_catalog_sample_counts("empty", "species") == {}

    idx._descendant_catalog_sample_counts.cache_clear()
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table({"taxon_key": ["1"], "sample_count": ["bad"]}),
    )
    assert idx._descendant_catalog_sample_counts("bad-count", "species") == {}

    # _eligible_context_taxon_ids guard branches.
    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", lambda _k: None)
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="missing",
            target_rank="SPECIES",
            storage_rank="SPECIES",
            include_species_like=False,
            allowed_taxa=None,
        )
        == set()
    )
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"rank": "GENUS", "path": tmp_path, "taxon_key": "1"}
    )
    monkeypatch.setattr(idx, "_descendant_catalog_sample_counts", lambda *_a, **_k: {})
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="1",
            target_rank="SPECIES",
            storage_rank="SPECIES",
            include_species_like=False,
            allowed_taxa=None,
        )
        == set()
    )

    monkeypatch.setattr(idx, "_descendant_catalog_sample_counts", lambda *_a, **_k: {1: 1, 2: 3})
    ranks = {"1": "SPECIES", "2": "GENUS"}
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: {"taxon_key": str(key), "rank": ranks.get(str(key), "SPECIES"), "path": tmp_path},
    )
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="1",
            target_rank="SPECIES",
            storage_rank="SPECIES",
            include_species_like=False,
            allowed_taxa=frozenset({2}),
        )
        == set()
    )
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="1",
            target_rank="SPECIES",
            storage_rank="SPECIES",
            include_species_like=False,
            allowed_taxa=None,
            min_samples=5,
        )
        == set()
    )
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: None if str(key) == "1" else {"taxon_key": "2", "rank": "GENUS", "path": tmp_path},
    )
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="1",
            target_rank="SPECIES",
            storage_rank="SPECIES",
            include_species_like=False,
            allowed_taxa=None,
        )
        == set()
    )

    # load_relative_ranks: filtered combine_chunks empty + requested metric mismatch + invalid counts.
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir}
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)))
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)

    class _Table:
        num_rows = 1

        def combine_chunks(self):
            return pa.table(
                {
                    "variable": [],
                    "metric": [],
                    "position": [],
                    "count": [],
                    "sampleCount": [],
                    "contextTaxonId": [],
                    "contextLabel": [],
                }
            )

    monkeypatch.setattr(idx.PARQUET, "read_table", lambda *_a, **_k: _Table())
    assert idx.load_relative_ranks(taxon_dir, "bio_1") == []

    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": ["bio_1", "bio_1", "bio_1"],
                "metric": ["median", "mean", "mean"],
                "position": [0, None, "bad"],
                "count": [1, 0, "x"],
                "sampleCount": [1, 1, 1],
                "contextTaxonId": ["10", "10", "10"],
                "contextLabel": ["G", "G", "G"],
            }
        ),
    )
    assert idx.load_relative_ranks(taxon_dir, "bio_1", metric_names=["mean"]) == []

    # child_relative_rankings: filtered-to-empty dict branch.
    anc = {"taxon_key": "77", "rank": "GENUS", "path": tmp_path}
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda k: (
            anc
            if str(k) == "77"
            else {"taxon_key": str(k), "rank": "SPECIES", "scientific_name": "T", "path": tmp_path}
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda k: int(str(k)) if str(k).isdigit() else None)
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda *_a, **_k: frozenset({1}))
    monkeypatch.setattr(idx.gis_lookup, "location_taxon_counts", lambda *_a, **_k: {2: 4})
    monkeypatch.setattr(idx.gis_lookup, "location_counts_for_taxon", lambda _taxon_id: {("scope", "target"): 1})
    out2, _dist2 = idx.child_relative_rankings("77", "species", "bio_1", "mean", location_gid="USA")
    assert out2


def test_child_relative_rankings_applies_candidate_filter_before_taxon_lookup(monkeypatch, tmp_path):
    ancestor = {"taxon_key": "77", "rank": "GENUS", "path": tmp_path}
    looked_up: list[str] = []

    def fake_get_taxon_by_id(key):
        normalized = str(key)
        looked_up.append(normalized)
        if normalized == "77":
            return ancestor
        return {
            "taxon_key": normalized,
            "rank": "SPECIES",
            "scientific_name": f"Taxon {normalized}",
            "path": tmp_path,
        }

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", fake_get_taxon_by_id)
    monkeypatch.setattr(
        idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)) if str(key).isdigit() else None
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper() if rank else "")
    monkeypatch.setattr(idx.taxa_navigation, "extract_common_names_for_language", lambda *_a, **_k: [])
    monkeypatch.setattr(idx.taxa_navigation, "resolve_taxon_media", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.taxa_navigation, "preferred_image_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 3})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [
                pa.array(["1", "2", "3"], type=pa.string()),
                pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                pa.array([5, 5, 5], type=pa.int32()),
            ],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )

    rows, _dist = idx.child_relative_rankings(
        "77",
        "species",
        "bio_1",
        "mean",
        candidate_taxon_ids=["2"],
        return_distribution=False,
    )

    assert [row["taxon_id"] for row in rows] == [2]
    assert looked_up == ["77", "2"]


def test_indexing_final_remaining_lines(stub_env, monkeypatch, tmp_path):
    _cfg, _stub = stub_env

    # _eligible_context_taxon_ids: taxon None and rank mismatch filters.
    monkeypatch.setattr(idx, "_descendant_catalog_sample_counts", lambda *_a, **_k: {1: 4, 2: 4})
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: (
            {"taxon_key": "root", "rank": "GENUS", "path": tmp_path}
            if str(key) == "root"
            else (None if str(key) == "1" else {"taxon_key": str(key), "rank": "GENUS", "path": tmp_path})
        ),
    )
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="root",
            target_rank="SPECIES",
            storage_rank="SPECIES",
            include_species_like=False,
            allowed_taxa=None,
        )
        == set()
    )

    # load_relative_ranks: requested_metrics mismatch and malformed count/position rows.
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        idx.taxa_navigation, "get_taxon_by_id", lambda _k: {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir}
    )
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(
        idx.PARQUET,
        "read_table",
        lambda *_a, **_k: pa.table(
            {
                "variable": ["bio_1", "bio_1", "bio_1"],
                "metric": ["median", "mean", "mean"],
                "position": [0, None, "oops"],
                "count": [1, 0, "oops"],
                "sampleCount": [1, 1, 1],
                "contextTaxonId": ["10", "10", "10"],
                "contextLabel": ["G", "G", "G"],
            }
        ),
    )
    assert idx.load_relative_ranks(taxon_dir, "bio_1", metric_names=["mean"]) == []


def test_indexing_final_relative_rank_continuation_lines(monkeypatch, tmp_path):
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    target = {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir, "scientific_name": "S", "common_name": None}
    ancestors = [
        {"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "A"},
        {"taxon_key": "11", "rank": "GENUS", "path": tmp_path, "scientific_name": "B"},
    ]
    calls = {"n": 0}

    def _get_taxon(_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return target
        if calls["n"] == 2:
            return None
        return target

    monkeypatch.setattr(idx.taxa_navigation, "get_taxon_by_id", _get_taxon)
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda r: str(r).upper())
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda _k: None)
    monkeypatch.setattr(idx, "_ancestor_contexts", lambda _p: ancestors)
    monkeypatch.setattr(idx.gis_lookup, "location_lookup_for_gid", lambda _g: ("x", "scope", "target"))
    monkeypatch.setattr(idx.gis_lookup, "location_taxa_for", lambda _s, _t: frozenset({1}))
    monkeypatch.setattr(idx.PARQUET, "exists", lambda _p: True)
    monkeypatch.setattr(idx, "_load_column_lengths", lambda _p: {"bio_1::mean": 1})
    monkeypatch.setattr(idx, "_resolve_column_name", lambda *_a, **_k: "bio_1::mean")
    monkeypatch.setattr(idx, "_load_summary_stats", lambda _p: {"bio_1": {"mean": 1.0}})
    monkeypatch.setattr(idx, "_load_categorical_stats", lambda _p: {})
    monkeypatch.setattr(
        idx,
        "_load_struct_column",
        lambda *_a, **_k: pa.StructArray.from_arrays(
            [pa.array(["1"], type=pa.string()), pa.array([1.0], type=pa.float64()), pa.array([1], type=pa.int32())],
            fields=[
                pa.field("taxonKey", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("sampleCount", pa.int32()),
            ],
        ),
    )
    assert idx.load_relative_ranks(taxon_dir, "bio_1", location_gid="USA") == []


def test_indexing_last_uncovered_defensive_paths(stub_env, monkeypatch, tmp_path):
    _cfg, stub = stub_env

    # _eligible_context_taxon_ids: non-species target with rank mismatch.
    monkeypatch.setattr(idx, "_descendant_catalog_sample_counts", lambda *_a, **_k: {2: 5})
    monkeypatch.setattr(idx.taxa_navigation, "canonical_rank", lambda rank: str(rank).upper())
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: (
            {"taxon_key": "root", "rank": "GENUS", "path": tmp_path}
            if str(key) == "root"
            else {"taxon_key": str(key), "rank": "SPECIES", "path": tmp_path}
        ),
    )
    assert (
        idx._eligible_context_taxon_ids(
            ancestor_taxon_id="root",
            target_rank="GENUS",
            storage_rank="GENUS",
            include_species_like=False,
            allowed_taxa=None,
        )
        == set()
    )

    # load_relative_ranks: requested metric mismatch, invalid row values, and class metric adjustment.
    taxon_dir = tmp_path / "species_1"
    taxon_dir.mkdir(parents=True, exist_ok=True)
    idx._eligible_context_taxon_count_cached.cache_clear()
    monkeypatch.setattr(
        idx.taxa_navigation,
        "get_taxon_by_id",
        lambda key: (
            {"taxon_key": "1", "rank": "SPECIES", "path": taxon_dir}
            if str(key) == "1"
            else (
                {"taxon_key": "10", "rank": "GENUS", "path": tmp_path}
                if str(key) == "10"
                else {"taxon_key": str(key), "rank": "SPECIES", "path": tmp_path}
            )
        ),
    )
    monkeypatch.setattr(idx.taxa_navigation, "taxon_id_as_int", lambda key: int(str(key)))
    monkeypatch.setattr(
        idx,
        "_descendant_catalog_sample_counts",
        lambda *_a, **_k: {101: 5, 102: 5, 103: 5},
    )
    monkeypatch.setattr(
        idx,
        "_ancestor_contexts",
        lambda _p: [{"taxon_key": "10", "rank": "GENUS", "path": tmp_path, "scientific_name": "G"}],
    )
    monkeypatch.setattr(idx, "_load_global_relative_rows", lambda *_a, **_k: None)
    positions_path = taxon_dir / "relative_ranks_positions.parquet"
    stub._exists[positions_path] = True
    stub._tables[positions_path] = pa.table(
        {
            "variable": ["bio_1", "bio_1", "bio_1", "bio_1"],
            "metric": ["mean", "class_1", "class_1", "class_1"],
            "position": [0.0, 1.0, float("nan"), 0.0],
            "count": [1, 0, 1, 1],
            "sampleCount": [1, 1, 1, 1],
            "contextTaxonId": ["10", "10", "10", "10"],
            "contextLabel": ["G", "G", "G", "G"],
        }
    )
    rows = idx.load_relative_ranks(taxon_dir, "bio_1", metric_names=["class_1"])
    assert len(rows) == 1
    assert rows[0]["metric"] == "class_1"
    assert rows[0]["count"] == 3
    assert rows[0]["position"] == 3

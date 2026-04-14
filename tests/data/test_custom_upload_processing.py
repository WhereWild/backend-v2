from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest
from fastapi import HTTPException

from util import custom_upload_processing as cup


def test_normalize_coordinate_columns_observation_name_uses_explicit_alias():
    frame = pd.DataFrame(
        {
            "scientificName": ["Quercus robur"],
            "name": ["English oak"],
            "latitude": [51.5],
            "longitude": [-0.1],
        }
    )

    normalized = cup._normalize_coordinate_columns(frame)

    assert normalized["observationName"].tolist() == ["English oak"]
    assert normalized["scientificName"].tolist() == ["Quercus robur"]


def test_normalize_coordinate_columns_does_not_promote_scientific_name():
    frame = pd.DataFrame(
        {
            "scientificName": ["Quercus robur"],
            "decimalLatitude": [51.5],
            "decimalLongitude": [-0.1],
        }
    )

    normalized = cup._normalize_coordinate_columns(frame)

    assert "observationName" not in normalized.columns


def test_ensure_catalog_numbers_uses_explicit_aliases():
    frame = pd.DataFrame(
        {
            "datasetId": ["dataset-1"],
            "occurrenceID": ["occ-123"],
        }
    )

    normalized = cup._ensure_catalog_numbers(frame)

    assert normalized["catalogNumber"].tolist() == ["occ-123"]
    assert "datasetId" in normalized.columns


def test_ensure_catalog_numbers_does_not_promote_plain_id():
    frame = pd.DataFrame(
        {
            "id": ["row-1", "row-2"],
        }
    )

    normalized = cup._ensure_catalog_numbers(frame)

    assert normalized["catalogNumber"].tolist() == ["Observation #1", "Observation #2"]
    assert "id" in normalized.columns


def test_ensure_observation_names_fills_missing_values():
    frame = pd.DataFrame(
        {
            "catalogNumber": ["a", "b"],
            "observationName": ["", None],
        }
    )

    normalized = cup._ensure_observation_names(frame)

    assert normalized["observationName"].tolist() == ["Observation #1", "Observation #2"]


def test_build_internal_upload_dataframe_preserves_user_columns():
    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1"],
            "observationName": ["English oak"],
            "decimalLatitude": [51.5],
            "decimalLongitude": [-0.1],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [25],
            "year": [2024],
            "datasetId": ["dataset-1"],
        }
    )

    internal = cup._build_internal_upload_dataframe(frame)

    assert list(internal.columns) == list(frame.columns)


def test_build_internal_upload_dataframe_rejects_reserved_enrichment_columns(monkeypatch):
    monkeypatch.setattr(cup.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "numeric"}})
    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1"],
            "observationName": ["English oak"],
            "decimalLatitude": [51.5],
            "decimalLongitude": [-0.1],
            "bio_1": [999.0],
        }
    )

    with pytest.raises(HTTPException, match="reserved for derived enrichment data"):
        cup._build_internal_upload_dataframe(frame)


def test_build_index_archive_uses_full_dataframe_for_artifacts_and_trimmed_export(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _capture_summary(frame, _directory):
        captured["summary_columns"] = list(frame.columns)

    def _capture_index(frame, _directory):
        captured["index_columns"] = list(frame.columns)

    monkeypatch.setattr(cup, "CONFIG", SimpleNamespace(occurrence_parquet_filename="occurrence.parquet"))
    monkeypatch.setattr(cup, "_write_summary_artifacts_from_dataframe", _capture_summary)
    monkeypatch.setattr(cup, "_write_occurrence_index_from_dataframe", _capture_index)
    monkeypatch.setattr(cup.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "numeric"}})
    monkeypatch.setattr(
        cup.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [
                {
                    "id": "bio_1",
                    "name": "Annual Mean Temperature",
                    "category": "Terrain",
                    "units": "C",
                    "value_type": "numeric",
                }
            ],
            {
                "bio_1": {
                    "id": "bio_1",
                    "name": "Annual Mean Temperature",
                    "category": "Terrain",
                    "units": "C",
                    "value_type": "numeric",
                }
            },
        ),
    )
    monkeypatch.setattr(cup.summary_stats, "_layer_value_type", lambda column: "numeric" if column == "bio_1" else None)

    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1"],
            "observationName": ["English oak"],
            "decimalLatitude": [51.5],
            "decimalLongitude": [-0.1],
            "tileId": ["n50w010"],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [25],
            "bio_1": [10.0],
            "year": [2024],
            "datasetId": ["dataset-1"],
        }
    )

    archive_path, _archive_name, work_dir = cup._build_index_archive(frame)
    try:
        exported = pd.read_parquet(work_dir / "occurrence.parquet")
        metadata = pd.read_parquet(work_dir / "variable_metadata.parquet")
        assert archive_path.exists()
        assert "obscured" in captured["summary_columns"]
        assert "coordinateUncertaintyInMeters" in captured["summary_columns"]
        assert "year" not in captured["summary_columns"]
        assert "datasetId" not in captured["summary_columns"]
        assert "obscured" in captured["index_columns"]
        assert "coordinateUncertaintyInMeters" in captured["index_columns"]
        assert "year" not in captured["index_columns"]
        assert "datasetId" not in captured["index_columns"]
        assert list(exported.columns) == [
            "catalogNumber",
            "observationName",
            "decimalLatitude",
            "decimalLongitude",
            "tileId",
            "obscured",
            "coordinateUncertaintyInMeters",
            "year",
            "datasetId",
            "Annual Mean Temperature",
        ]
        assert metadata["exported_name"].tolist() == ["Annual Mean Temperature"]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_index_archive_falls_back_consistently_when_readable_step_fails(monkeypatch):
    original_write_metadata = cup._write_variable_metadata_manifest

    def _capture_summary(_frame, _directory):
        return None

    def _capture_index(_frame, _directory):
        return None

    def _raise_for_staged_metadata(directory: Path, *, rename_map=None, output_path=None):
        if output_path is not None:
            raise RuntimeError("staged metadata write failed")
        return original_write_metadata(directory, rename_map=rename_map, output_path=output_path)

    monkeypatch.setattr(cup, "CONFIG", SimpleNamespace(occurrence_parquet_filename="occurrence.parquet"))
    monkeypatch.setattr(cup, "_write_summary_artifacts_from_dataframe", _capture_summary)
    monkeypatch.setattr(cup, "_write_occurrence_index_from_dataframe", _capture_index)
    monkeypatch.setattr(cup, "_write_variable_metadata_manifest", _raise_for_staged_metadata)
    monkeypatch.setattr(cup.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "numeric"}})
    monkeypatch.setattr(
        cup.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [
                {
                    "id": "bio_1",
                    "name": "Annual Mean Temperature",
                    "category": "Terrain",
                    "units": "C",
                    "value_type": "numeric",
                }
            ],
            {
                "bio_1": {
                    "id": "bio_1",
                    "name": "Annual Mean Temperature",
                    "category": "Terrain",
                    "units": "C",
                    "value_type": "numeric",
                }
            },
        ),
    )
    monkeypatch.setattr(cup.summary_stats, "_layer_value_type", lambda column: "numeric" if column == "bio_1" else None)

    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1"],
            "observationName": ["English oak"],
            "decimalLatitude": [51.5],
            "decimalLongitude": [-0.1],
            "tileId": ["n50w010"],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [25],
            "bio_1": [10.0],
        }
    )

    _archive_path, _archive_name, work_dir = cup._build_index_archive(frame)
    try:
        exported = pd.read_parquet(work_dir / "occurrence.parquet")
        metadata = pd.read_parquet(work_dir / "variable_metadata.parquet")
        assert list(exported.columns) == [
            "catalogNumber",
            "observationName",
            "decimalLatitude",
            "decimalLongitude",
            "tileId",
            "obscured",
            "coordinateUncertaintyInMeters",
            "bio_1",
        ]
        assert metadata["exported_name"].tolist() == ["bio_1"]
        assert list(metadata.columns) == ["id", "name", "exported_name", "category", "units", "value_type", "source_ids"]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_index_archive_keeps_internal_artifacts_id_based(monkeypatch):
    monkeypatch.setattr(cup, "CONFIG", SimpleNamespace(occurrence_parquet_filename="occurrence.parquet"))
    monkeypatch.setattr(cup.gis_lookup, "load_layer_metadata", lambda: {"bio_1": {"value_type": "numeric"}})
    monkeypatch.setattr(
        cup.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [
                {
                    "id": "bio_1",
                    "name": "Annual Mean Temperature",
                    "category": "Terrain",
                    "units": "C",
                    "value_type": "numeric",
                }
            ],
            {
                "bio_1": {
                    "id": "bio_1",
                    "name": "Annual Mean Temperature",
                    "category": "Terrain",
                    "units": "C",
                    "value_type": "numeric",
                }
            },
        ),
    )
    monkeypatch.setattr(cup.summary_stats, "_layer_value_type", lambda column: "numeric" if column == "bio_1" else None)

    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1", "occ-2"],
            "observationName": ["English oak", "English oak 2"],
            "decimalLatitude": [51.5, 52.0],
            "decimalLongitude": [-0.1, -0.2],
            "tileId": ["n50w010", "n50w010"],
            "obscured": ["No", "No"],
            "coordinateUncertaintyInMeters": [25, 30],
            "bio_1": [10.0, 12.0],
        }
    )

    _archive_path, _archive_name, work_dir = cup._build_index_archive(frame)
    try:
        summary = pd.read_parquet(work_dir / "summary_stats.parquet")
        density = pd.read_parquet(work_dir / cup.summary_stats.density_graph_filename)
        index_table = pq.read_table(work_dir / "occurrence_index.parquet")
        exported = pd.read_parquet(work_dir / "occurrence.parquet")

        assert summary["variable"].tolist() == ["bio_1"]
        assert summary["variableName"].tolist() == ["Annual Mean Temperature"]
        assert summary["variableCategory"].tolist() == ["Terrain"]

        assert density["variable"].tolist() == ["bio_1"]
        assert density["variableName"].tolist() == ["Annual Mean Temperature"]
        assert density["variableCategory"].tolist() == ["Terrain"]

        assert index_table.schema.names == ["bio_1"]
        assert json.loads(index_table.schema.metadata[b"origin_map"].decode("utf-8")) == [
            {"id": 0, "relative_path": ".", "taxon_key": "uploaded"}
        ]
        assert json.loads(index_table.schema.metadata[b"column_lengths"].decode("utf-8")) == {"bio_1": 2}
        assert index_table.schema.metadata[b"catalog_column"] == b"catalogNumber"
        assert list(exported.columns) == [
            "catalogNumber",
            "observationName",
            "decimalLatitude",
            "decimalLongitude",
            "tileId",
            "obscured",
            "coordinateUncertaintyInMeters",
            "Annual Mean Temperature",
        ]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_index_archive_uses_circular_summary_for_aspect_deg(monkeypatch):
    monkeypatch.setattr(cup, "CONFIG", SimpleNamespace(occurrence_parquet_filename="occurrence.parquet"))
    monkeypatch.setattr(cup.gis_lookup, "load_layer_metadata", lambda: {"aspect_deg": {"value_type": "circular"}})
    monkeypatch.setattr(
        cup.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [
                {
                    "id": "aspect_deg",
                    "name": "Aspect",
                    "category": "Terrain",
                    "units": "degrees",
                    "value_type": "circular",
                }
            ],
            {
                "aspect_deg": {
                    "id": "aspect_deg",
                    "name": "Aspect",
                    "category": "Terrain",
                    "units": "degrees",
                    "value_type": "circular",
                }
            },
        ),
    )
    monkeypatch.setattr(
        cup.summary_stats, "_layer_value_type", lambda column: "circular" if column == "aspect_deg" else None
    )

    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1", "occ-2"],
            "observationName": ["North wrap A", "North wrap B"],
            "decimalLatitude": [51.5, 52.0],
            "decimalLongitude": [-0.1, -0.2],
            "tileId": ["n50w010", "n50w010"],
            "obscured": ["No", "No"],
            "coordinateUncertaintyInMeters": [25, 30],
            "aspect_deg": [359.0, 1.0],
        }
    )

    _archive_path, _archive_name, work_dir = cup._build_index_archive(frame)
    try:
        summary = pd.read_parquet(work_dir / "summary_stats.parquet")
        row = summary[summary["variable"] == "aspect_deg"].iloc[0]
        assert row["mean"] == pytest.approx(0.0)
        assert row["median"] == pytest.approx(0.0)
        assert row["range"] == pytest.approx(2.0)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_build_index_archive_writes_categorical_value_lookup(monkeypatch):
    monkeypatch.setattr(cup, "CONFIG", SimpleNamespace(occurrence_parquet_filename="occurrence.parquet"))
    monkeypatch.setattr(cup.gis_lookup, "load_layer_metadata", lambda: {"landcover": {"value_type": "categorical"}})
    monkeypatch.setattr(
        cup.gis_lookup,
        "load_variable_metadata",
        lambda: (
            [
                {
                    "id": "landcover",
                    "name": "Land Cover",
                    "category": "Surface",
                    "units": None,
                    "value_type": "categorical",
                }
            ],
            {
                "landcover": {
                    "id": "landcover",
                    "name": "Land Cover",
                    "category": "Surface",
                    "units": None,
                    "value_type": "categorical",
                }
            },
        ),
    )
    monkeypatch.setattr(
        cup.gis_lookup,
        "load_layer_legend",
        lambda variable_id: (
            {
                "52": {"id": 52, "name": "Impervious surfaces", "description": "Built"},
                "impervious surfaces": {"id": 52, "name": "Impervious surfaces", "description": "Built"},
            }
            if variable_id == "landcover"
            else {}
        ),
    )
    monkeypatch.setattr(
        cup.summary_stats, "_layer_value_type", lambda column: "categorical" if column == "landcover" else None
    )

    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1", "occ-2"],
            "observationName": ["Urban patch", "Urban patch 2"],
            "decimalLatitude": [51.5, 52.0],
            "decimalLongitude": [-0.1, -0.2],
            "tileId": ["n50w010", "n50w010"],
            "obscured": ["No", "No"],
            "coordinateUncertaintyInMeters": [25, 30],
            "landcover": [52, 52],
        }
    )

    _archive_path, _archive_name, work_dir = cup._build_index_archive(frame)
    try:
        categorical_stats = pd.read_parquet(work_dir / "categorical_stats.parquet")
        categorical_lookup = pd.read_parquet(work_dir / "categorical_value_lookup.parquet")
        index_table = pq.read_table(work_dir / "occurrence_index.parquet")
        index_value = index_table["landcover"].to_pylist()[0]["value"]
        lookup_row = categorical_lookup[
            (categorical_lookup["variable"] == "landcover") & (categorical_lookup["code"] == index_value)
        ].iloc[0]
        stats_row = categorical_stats[
            (categorical_stats["variable"] == "landcover") & (categorical_stats["metric"] == "class_52")
        ].iloc[0]

        assert lookup_row["metric"] == "class_52"
        assert lookup_row["metric"] in categorical_stats["metric"].tolist()
        assert lookup_row["label"] == "Impervious surfaces"
        assert stats_row["metricLabel"] == "Impervious surfaces"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_add_gis_columns_preserves_non_conflicting_user_columns(monkeypatch):
    captured = {}

    def _fake_process_tiles(worklist):
        captured["worklist"] = worklist.to_pandas()

    monkeypatch.setattr(cup.enrich_tree, "_load_layer_ids", lambda: ["bio_1"])
    monkeypatch.setattr(cup.enrich_tree, "_process_tiles", _fake_process_tiles)

    frame = pd.DataFrame(
        {
            "catalogNumber": ["occ-1"],
            "observationName": ["English oak"],
            "decimalLatitude": [51.5],
            "decimalLongitude": [-0.1],
            "tileId": ["n50w010"],
            "obscured": ["No"],
            "coordinateUncertaintyInMeters": [25],
            "year": [2024],
        }
    )

    enriched = cup._add_gis_columns(frame)

    assert "bio_1" not in enriched.columns
    assert enriched["year"].tolist() == [2024]
    assert captured["worklist"]["missingLayers"].tolist() == [["bio_1"]]

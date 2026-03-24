"""Unit tests for util.config coverage and contract behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from util import config as config_module


def test_resolve_env_path_uses_env_value(monkeypatch, tmp_path):
    target = tmp_path / "custom-root"
    monkeypatch.setenv("WW_TEST_PATH", str(target))
    resolved = config_module._resolve_env_path("WW_TEST_PATH", tmp_path / "default")
    assert resolved == target.resolve()


def test_gis_catalog_and_legends_env_and_fallback_paths(monkeypatch, tmp_path):
    cfg = config_module.GlobalConfig(project_root=tmp_path)

    env_catalog = tmp_path / "env-catalog.json"
    env_legends = tmp_path / "env-legends"
    monkeypatch.setenv("GIS_CATALOG_PATH", str(env_catalog))
    monkeypatch.setenv("GIS_LEGENDS_ROOT", str(env_legends))
    assert cfg.gis_catalog_path == env_catalog.resolve()
    assert cfg.gis_legends_root == env_legends.resolve()

    monkeypatch.delenv("GIS_CATALOG_PATH", raising=False)
    monkeypatch.delenv("GIS_LEGENDS_ROOT", raising=False)

    config_gis_root = tmp_path / "config" / "gis"
    config_gis_root.mkdir(parents=True, exist_ok=True)
    catalog_file = config_gis_root / cfg.catalog_json_filename
    legends_dir = config_gis_root / "legends"
    catalog_file.write_text("{}", encoding="utf-8")
    legends_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.gis_catalog_path == catalog_file.resolve()
    assert cfg.gis_legends_root == legends_dir.resolve()

    catalog_file.unlink()
    legends_dir.rmdir()
    assert cfg.gis_catalog_path == (cfg.gis_root / cfg.catalog_json_filename).resolve()
    assert cfg.gis_legends_root == (cfg.gis_root / "legends").resolve()


def test_global_config_derived_paths_and_collections(monkeypatch, tmp_path):
    data_root = tmp_path / "ww-data"
    temporal_cache_root = tmp_path / "tmp-cache"
    monkeypatch.setenv("WHEREWILD_DATA_ROOT", str(data_root))
    monkeypatch.setenv("TEMPORAL_CACHE_ROOT", str(temporal_cache_root))
    cfg = config_module.GlobalConfig(project_root=tmp_path)

    assert cfg.gis_regions_root == cfg.gis_root / "regions"
    assert cfg.gis_landcover_root == cfg.gis_root / "landcover"
    assert cfg.parquet_root == cfg.data_root / "parquet"

    assert cfg.taxa_csv_path == cfg.species_dir / cfg.taxa_csv_filename
    assert cfg.vernacular_tsv_path == cfg.data_root / cfg.vernacular_filename
    assert cfg.occurrence_path == cfg.species_dir / cfg.occurrence_filename
    assert cfg.gbif_occurrence_path == cfg.species_dir / cfg.gbif_occurrence_txt
    assert cfg.gbif_multimedia_path == cfg.species_dir / cfg.gbif_multimedia_txt
    assert cfg.gbif_taxon_lookup_path == cfg.taxonomy_root / cfg.gbif_taxon_lookup_filename
    assert cfg.inat_mapping_offline_path == cfg.taxonomy_root / cfg.inat_mapping_offline_filename
    assert cfg.inat_mapping_api_path == cfg.taxonomy_root / cfg.inat_mapping_api_filename
    assert cfg.inat_mapping_obs_path == cfg.taxonomy_root / cfg.inat_mapping_obs_filename
    assert cfg.gbif_regions_path == cfg.gis_locations_root / cfg.gbif_regions_filename
    assert cfg.gadm_gpkg_path == cfg.gis_root / cfg.gadm_gpkg_filename
    assert cfg.bioclim_root == cfg.data_root / "bioclim"
    assert cfg.temporal_cache_root == temporal_cache_root.resolve()
    assert cfg.taxon_catalog_path == (cfg.taxonomy_root / cfg.taxon_catalog_filename).resolve()
    assert cfg.taxon_media_path == cfg.taxonomy_root / cfg.taxon_media_filename
    assert cfg.location_hierarchy_path == cfg.gis_locations_root / cfg.location_hierarchy_filename
    assert cfg.location_catalog_path == cfg.gis_locations_root / cfg.location_catalog_filename

    assert cfg.leaf_rank_set == frozenset(cfg.leaf_ranks)
    assert cfg.gbif_region_set == frozenset(cfg.gbif_regions)
    synonyms = cfg.rank_synonyms
    assert "SPECIES" in synonyms and "SP" in synonyms["SPECIES"]
    assert cfg.location_columns == (
        ("level0Gid", "gadm_level0"),
        ("level1Gid", "gadm_level1"),
        ("level2Gid", "gadm_level2"),
    )
    assert cfg.occurrence_all_columns == cfg.occurrence_base_columns + cfg.annotation_columns
    indices = cfg.occurrence_list_column_indices
    for col in cfg.occurrence_list_columns:
        assert cfg.occurrence_all_columns[indices[col]] == col


def test_load_config_unknown_name_raises_and_global_is_cached(monkeypatch):
    monkeypatch.setattr(config_module, "_CONFIG_CACHE", {})
    first = config_module.load_config("global")
    second = config_module.load_config("global")
    assert first is second
    with pytest.raises(KeyError):
        config_module.load_config("not-a-config")

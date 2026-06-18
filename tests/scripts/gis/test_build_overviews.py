# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import math
from unittest.mock import MagicMock, patch

import pytest

import scripts.gis.build_overviews as bo

CATALOG = {
    "categories": [
        {
            "id": "bioclimate",
            "display_name": "Bioclimatic",
            "layers": [
                {"id": "bio1", "filename": "bio1.tif", "value_type": "interval", "source": "chelsa_v2_1"},
                {"id": "koppen", "filename": "koppen.tif", "value_type": "nominal", "source": "chelsa_v2_1"},
            ],
        }
    ]
}


@pytest.fixture(autouse=True)
def patch_catalog(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(CATALOG))
    monkeypatch.setattr(bo, "CATALOG_PATH", catalog_path)


# --- _load_layer_meta ---

def test_load_layer_meta_keys(tmp_path):
    meta = bo._load_layer_meta()
    assert set(meta) == {"bio1.tif", "koppen.tif"}


# --- _is_class_based ---

def test_is_class_based_interval():
    assert bo._is_class_based({"value_type": "interval"}) is False


def test_is_class_based_nominal():
    assert bo._is_class_based({"value_type": "nominal"}) is True


def test_is_class_based_ordinal():
    assert bo._is_class_based({"value_type": "ordinal"}) is True


def test_is_class_based_none():
    assert bo._is_class_based(None) is False


def test_is_class_based_missing_key():
    assert bo._is_class_based({}) is False


# --- _next_power_of_two ---

def test_next_power_of_two_exact():
    assert bo._next_power_of_two(8.0) == 8


def test_next_power_of_two_rounds_up():
    assert bo._next_power_of_two(9.0) == 16


def test_next_power_of_two_lte_one():
    assert bo._next_power_of_two(1.0) == 1
    assert bo._next_power_of_two(0.5) == 1


# --- _overview_factor_close ---

def test_overview_factor_close_exact():
    assert bo._overview_factor_close(8, 8) is True


def test_overview_factor_close_within_tolerance():
    assert bo._overview_factor_close(9, 8) is True  # diff=1, tolerance=max(2, ...)


def test_overview_factor_close_outside_tolerance():
    assert bo._overview_factor_close(100, 8) is False


# --- _has_required_overviews ---

def test_has_required_overviews_empty_desired():
    assert bo._has_required_overviews([2, 4], []) is True


def test_has_required_overviews_empty_existing():
    assert bo._has_required_overviews([], [2, 4]) is False


def test_has_required_overviews_all_present():
    assert bo._has_required_overviews([2, 4, 8], [2, 4, 8]) is True


def test_has_required_overviews_missing_one():
    assert bo._has_required_overviews([2, 4], [2, 4, 8]) is False


# --- _overview_factors_for_dataset ---

def _make_ds(res=0.008333, width=43200, height=20880):
    ds = MagicMock()
    ds.transform.a = res
    ds.transform.e = -res
    ds.width = width
    ds.height = height
    return ds


def test_overview_factors_returns_list():
    factors = bo._overview_factors_for_dataset(_make_ds())
    assert isinstance(factors, list)
    assert len(factors) > 0
    assert all(f > 0 for f in factors)


def test_overview_factors_are_powers_of_two():
    factors = bo._overview_factors_for_dataset(_make_ds())
    assert all(math.log2(f) == int(math.log2(f)) for f in factors)


def test_overview_factors_zero_res_returns_empty():
    ds = _make_ds(res=0)
    assert bo._overview_factors_for_dataset(ds) == []


def test_overview_factors_capped_at_max():
    factors = bo._overview_factors_for_dataset(_make_ds())
    assert all(f <= bo.MAX_OVERVIEW_FACTOR for f in factors)


def test_overview_factors_very_coarse_res_returns_empty():
    # res > target_dst_res → desired <= 1 → empty
    ds = _make_ds(res=100.0)
    assert bo._overview_factors_for_dataset(ds) == []


# --- _build_cog ---

def test_build_cog_interval_uses_average(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    src.touch()

    with patch("scripts.gis.build_overviews.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        bo._build_cog(src, dst, nominal=False, overview_factors=[2, 4])

    calls_str = " ".join(str(c) for c in mock_run.call_args_list)
    assert "average" in calls_str.lower()
    assert "nearest" not in calls_str.lower()


def test_build_cog_nominal_uses_mode(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    src.touch()

    with patch("scripts.gis.build_overviews.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        bo._build_cog(src, dst, nominal=True, overview_factors=[2, 4])

    calls_str = " ".join(str(c) for c in mock_run.call_args_list)
    assert "mode" in calls_str.lower()
    assert "nearest" not in calls_str.lower()


def test_build_cog_cleans_up_base_tif_on_error(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    src.touch()

    with patch("scripts.gis.build_overviews.subprocess.run", side_effect=RuntimeError("gdal fail")):
        with pytest.raises(RuntimeError):
            bo._build_cog(src, dst, nominal=False, overview_factors=[2])

    assert not dst.with_suffix(".base.tif").exists()


def test_build_cog_skips_gdaladdo_when_no_factors(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    src.touch()

    with patch("scripts.gis.build_overviews.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        bo._build_cog(src, dst, nominal=False, overview_factors=[])

    commands = [c.args[0][0] for c in mock_run.call_args_list]
    assert "gdaladdo" not in commands


# --- main ---

def test_main_raises_if_layers_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bo, "LAYERS_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        bo.main()


def test_main_skips_files_with_sufficient_overviews(tmp_path, monkeypatch):
    monkeypatch.setattr(bo, "LAYERS_DIR", tmp_path)
    (tmp_path / "bio1.tif").touch()

    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.overviews.return_value = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    mock_ds.transform.a = 0.008333
    mock_ds.transform.e = -0.008333
    mock_ds.width = 43200
    mock_ds.height = 20880

    with patch("scripts.gis.build_overviews.rasterio.open", return_value=mock_ds), \
         patch("scripts.gis.build_overviews._build_cog") as mock_build:
        bo.main()

    mock_build.assert_not_called()


def test_main_upgrades_file_with_insufficient_overviews(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bo, "LAYERS_DIR", tmp_path)
    (tmp_path / "bio1.tif").touch()

    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.overviews.return_value = [2]  # has some but not enough
    mock_ds.transform.a = 0.008333
    mock_ds.transform.e = -0.008333
    mock_ds.width = 43200
    mock_ds.height = 20880

    with patch("scripts.gis.build_overviews.rasterio.open", return_value=mock_ds), \
         patch("scripts.gis.build_overviews._build_cog") as mock_build, \
         patch("scripts.gis.build_overviews.os.replace"):
        bo.main()

    mock_build.assert_called_once()
    assert "upgrading" in capsys.readouterr().out


def test_main_builds_cog_for_file_without_overviews(tmp_path, monkeypatch):
    monkeypatch.setattr(bo, "LAYERS_DIR", tmp_path)
    tif = tmp_path / "bio1.tif"
    tif.touch()

    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.overviews.return_value = []
    mock_ds.transform.a = 0.008333
    mock_ds.transform.e = -0.008333
    mock_ds.width = 43200
    mock_ds.height = 20880

    with patch("scripts.gis.build_overviews.rasterio.open", return_value=mock_ds), \
         patch("scripts.gis.build_overviews._build_cog") as mock_build, \
         patch("scripts.gis.build_overviews.os.replace"):
        bo.main()

    mock_build.assert_called_once()


def test_main_continues_after_failed_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bo, "LAYERS_DIR", tmp_path)
    (tmp_path / "bad.tif").touch()
    (tmp_path / "bio1.tif").touch()

    call_count = 0

    def fake_open(path):
        nonlocal call_count
        call_count += 1
        if "bad" in str(path):
            raise RuntimeError("corrupt file")
        mock_ds = MagicMock()
        mock_ds.__enter__ = lambda s: s
        mock_ds.__exit__ = MagicMock(return_value=False)
        mock_ds.overviews.return_value = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        mock_ds.transform.a = 0.008333
        mock_ds.transform.e = -0.008333
        mock_ds.width = 43200
        mock_ds.height = 20880
        return mock_ds

    with patch("scripts.gis.build_overviews.rasterio.open", side_effect=fake_open):
        bo.main()

    out = capsys.readouterr().out
    assert "failed" in out
    assert call_count == 2

# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from unittest.mock import MagicMock, patch

import pytest

import scripts.gis.prop_metadata as pm

CATALOG = {
    "categories": [
        {
            "id": "test",
            "display_name": "Test",
            "layers": [
                {
                    "id": "bio1",
                    "filename": "bio1.tif",
                    "value_type": "interval",
                    "display_name": "Annual Mean Temperature",
                },
                {"id": "koppen", "filename": "koppen.tif", "value_type": "nominal"},
                {"id": "salinity", "filename": "salinity.tif", "value_type": "ordinal"},
                {"id": "no_legend", "filename": "no_legend.tif", "value_type": "nominal"},
                {"id": "weird_type", "filename": "weird_type.tif", "value_type": "bogus"},
                # Derived/on-the-fly layers with no static file at all --
                # both share the same missing filename, which is exactly
                # why _load_layer_meta() filters these out rather than
                # letting them collide on one `None` dict key.
                {"id": "slope", "filename": None, "value_type": "ratio"},
                {"id": "aspect", "filename": None, "value_type": "circular"},
            ],
        }
    ]
}

KOPPEN_LEGEND = {
    "layer_id": "koppen",
    "classes": [
        {"id": 1, "name": "Tropical", "traits": {"color": "#ff0000"}},
        {"id": 2, "name": "Arid", "traits": {"color": "#ffff00"}},
    ],
}

SALINITY_LEGEND = {
    "layer_id": "salinity",
    "classes": [
        {"id": 0, "name": "Non saline"},
        {"id": 1, "name": "Slightly saline"},
    ],
}

CB_COLORS = {
    "salinity": {
        "viridis": {"0": "#440154", "1": "#3b528b"},
        "plasma": {"0": "#0d0887", "1": "#7e03a8"},
    },
}


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(CATALOG))
    monkeypatch.setattr(pm, "CATALOG_PATH", catalog_path)

    legends_dir = tmp_path / "legends"
    legends_dir.mkdir()
    (legends_dir / "koppen_legend.json").write_text(json.dumps(KOPPEN_LEGEND))
    (legends_dir / "salinity_legend.json").write_text(json.dumps(SALINITY_LEGEND))
    monkeypatch.setattr(pm, "LEGENDS_DIR", legends_dir)

    cb_colors_path = tmp_path / "cb_colors.json"
    cb_colors_path.write_text(json.dumps(CB_COLORS))
    monkeypatch.setattr(pm, "CB_COLORS_PATH", cb_colors_path)


# --- _load_layer_meta -------------------------------------------------------


def test_load_layer_meta_skips_layers_with_no_filename():
    meta = pm._load_layer_meta()
    assert set(meta) == {
        "bio1.tif", "koppen.tif", "salinity.tif", "no_legend.tif", "weird_type.tif",
    }


# --- _classes_for_layer ------------------------------------------------------


def test_classes_for_nominal_layer_uses_traits_color():
    cb_colors = pm._load_cb_colors()
    classes = pm._classes_for_layer("koppen", "nominal", cb_colors)
    assert classes == [
        {"id": 1, "name": "Tropical", "color": "#ff0000"},
        {"id": 2, "name": "Arid", "color": "#ffff00"},
    ]


def test_classes_for_ordinal_layer_uses_default_colormap():
    cb_colors = pm._load_cb_colors()
    classes = pm._classes_for_layer("salinity", "ordinal", cb_colors)
    assert classes == [
        {"id": 0, "name": "Non saline", "color": "#440154"},
        {"id": 1, "name": "Slightly saline", "color": "#3b528b"},
    ]


def test_classes_for_layer_with_no_legend_file_returns_empty():
    cb_colors = pm._load_cb_colors()
    assert pm._classes_for_layer("no_legend", "nominal", cb_colors) == []


# --- _embed_metadata ---------------------------------------------------------


def _mock_dataset(existing_tags: dict) -> MagicMock:
    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: s
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.tags.return_value = dict(existing_tags)
    return mock_ds


def test_embed_metadata_writes_new_tags_and_reports_changed(tmp_path):
    mock_ds = _mock_dataset({})
    with patch("scripts.gis.prop_metadata.rasterio.open", return_value=mock_ds) as mock_open:
        changed = pm._embed_metadata(
            tmp_path / "koppen.tif",
            "nominal",
            [{"id": 1, "name": "Tropical", "color": "#ff0000"}],
        )
    assert changed is True
    mock_open.assert_called_once_with(
        tmp_path / "koppen.tif", "r+", IGNORE_COG_LAYOUT_BREAK="YES"
    )
    mock_ds.update_tags.assert_called_once_with(
        WHEREWILD_VALUE_TYPE="nominal",
        WHEREWILD_LEGEND='[{"id":1,"name":"Tropical","color":"#ff0000"}]',
    )


def test_embed_metadata_no_op_when_already_up_to_date(tmp_path):
    legend_json = '[{"id":1,"name":"Tropical","color":"#ff0000"}]'
    mock_ds = _mock_dataset(
        {"WHEREWILD_VALUE_TYPE": "nominal", "WHEREWILD_LEGEND": legend_json}
    )
    with patch("scripts.gis.prop_metadata.rasterio.open", return_value=mock_ds):
        changed = pm._embed_metadata(
            tmp_path / "koppen.tif",
            "nominal",
            [{"id": 1, "name": "Tropical", "color": "#ff0000"}],
        )
    assert changed is False
    mock_ds.update_tags.assert_not_called()


def test_embed_metadata_continuous_layer_writes_no_legend_tag(tmp_path):
    mock_ds = _mock_dataset({})
    with patch("scripts.gis.prop_metadata.rasterio.open", return_value=mock_ds):
        changed = pm._embed_metadata(tmp_path / "bio1.tif", "interval", [])
    assert changed is True
    mock_ds.update_tags.assert_called_once_with(WHEREWILD_VALUE_TYPE="interval")


def test_embed_metadata_writes_the_display_name_when_given(tmp_path):
    mock_ds = _mock_dataset({})
    with patch("scripts.gis.prop_metadata.rasterio.open", return_value=mock_ds):
        changed = pm._embed_metadata(
            tmp_path / "bio1.tif", "interval", [], "Annual Mean Temperature"
        )
    assert changed is True
    mock_ds.update_tags.assert_called_once_with(
        WHEREWILD_VALUE_TYPE="interval",
        WHEREWILD_NAME="Annual Mean Temperature",
    )


def test_embed_metadata_rewrites_when_only_the_display_name_changed(tmp_path):
    mock_ds = _mock_dataset(
        {"WHEREWILD_VALUE_TYPE": "interval", "WHEREWILD_NAME": "Old name"}
    )
    with patch("scripts.gis.prop_metadata.rasterio.open", return_value=mock_ds):
        changed = pm._embed_metadata(tmp_path / "bio1.tif", "interval", [], "New name")
    assert changed is True
    mock_ds.update_tags.assert_called_once()


def test_embed_metadata_no_op_when_the_display_name_is_already_current(tmp_path):
    mock_ds = _mock_dataset(
        {"WHEREWILD_VALUE_TYPE": "interval", "WHEREWILD_NAME": "Annual Mean Temperature"}
    )
    with patch("scripts.gis.prop_metadata.rasterio.open", return_value=mock_ds):
        changed = pm._embed_metadata(
            tmp_path / "bio1.tif", "interval", [], "Annual Mean Temperature"
        )
    assert changed is False
    mock_ds.update_tags.assert_not_called()


# --- main --------------------------------------------------------------------


def test_main_raises_without_layers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "LAYERS_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        pm.main()


def test_main_embeds_metadata_for_each_recognized_layer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pm, "LAYERS_DIR", tmp_path)
    for name in ["bio1.tif", "koppen.tif", "salinity.tif", "unrelated.tif"]:
        (tmp_path / name).touch()

    calls: dict[str, dict] = {}

    def fake_open(path, mode, **kwargs):
        mock_ds = _mock_dataset({})
        mock_ds.update_tags.side_effect = lambda **tags: calls.setdefault(
            path.name, tags
        )
        return mock_ds

    with patch("scripts.gis.prop_metadata.rasterio.open", side_effect=fake_open):
        pm.main()

    assert calls["bio1.tif"] == {
        "WHEREWILD_VALUE_TYPE": "interval",
        "WHEREWILD_NAME": "Annual Mean Temperature",
    }
    # A catalog entry with no display_name gets no name tag at all.
    assert "WHEREWILD_NAME" not in calls["koppen.tif"]
    assert calls["koppen.tif"]["WHEREWILD_VALUE_TYPE"] == "nominal"
    assert json.loads(calls["koppen.tif"]["WHEREWILD_LEGEND"]) == [
        {"id": 1, "name": "Tropical", "color": "#ff0000"},
        {"id": 2, "name": "Arid", "color": "#ffff00"},
    ]
    assert calls["salinity.tif"]["WHEREWILD_VALUE_TYPE"] == "ordinal"
    # unrelated.tif has no catalog entry at all -- must be left completely
    # untouched, not just given empty tags.
    assert "unrelated.tif" not in calls

    out = capsys.readouterr().out
    assert "total=4" in out
    assert "updated=3" in out
    assert "skipped=1" in out


def test_main_skips_layer_with_unrecognized_value_type(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "LAYERS_DIR", tmp_path)
    (tmp_path / "weird_type.tif").touch()

    with patch("scripts.gis.prop_metadata.rasterio.open") as mock_open:
        pm.main()

    mock_open.assert_not_called()


def test_main_continues_past_a_failed_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pm, "LAYERS_DIR", tmp_path)
    (tmp_path / "bio1.tif").touch()
    (tmp_path / "koppen.tif").touch()

    def fake_open(path, mode, **kwargs):
        if path.name == "bio1.tif":
            raise RuntimeError("corrupt file")
        return _mock_dataset({})

    with patch("scripts.gis.prop_metadata.rasterio.open", side_effect=fake_open):
        pm.main()

    out = capsys.readouterr().out
    assert "failed bio1.tif" in out
    assert "updated=1" in out

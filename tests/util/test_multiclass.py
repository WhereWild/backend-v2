# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import Counter

import numpy as np
import pandas as pd
import pytest

import util.stats as st
from util.gis import vector_sampling_field
from util.multiclass import (
    OBS_TOTAL_KEY,
    class_ids_to_mask,
    mask_counts,
    mask_to_class_ids,
    split_obs_total,
)

_MULTI_LAYER = {
    "id": "lith_multi", "value_type": "nominal", "multi_valued": True,
    "vector_field": "LITH_PRIMARY", "vector_set_field": "LITH_MASK",
}


def test_mask_round_trip():
    assert mask_to_class_ids(class_ids_to_mask([0, 3, 51])) == [0, 3, 51]
    assert class_ids_to_mask([]) == 0


def test_mask_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match='outside'):
        class_ids_to_mask([52])


def test_mask_counts_ignores_nan_and_zero():
    masks = [class_ids_to_mask([1, 2]), class_ids_to_mask([2]), 0, np.nan]
    counts = mask_counts(masks)
    assert counts[1] == 1
    assert counts[2] == 2
    assert counts[OBS_TOTAL_KEY] == 2


def test_split_obs_total_plain_counter_unchanged():
    counts, total = split_obs_total(Counter({1: 3}))
    assert counts == Counter({1: 3})
    assert total is None


def test_multi_valued_fractions_use_observation_total():
    counts = mask_counts([class_ids_to_mask([1, 2])] * 3 + [class_ids_to_mask([2])])
    summary, distribution = st._nominal_stats(counts, 4)
    assert summary["total_samples"] == 4
    fractions = {d["class_id"]: d["fraction"] for d in distribution}
    assert fractions == {1: 0.75, 2: 1.0}  # sums past 1
    assert summary["unique_classes"] == 2
    assert 0.0 < summary["entropy"] <= np.log(2) + 1e-9


def test_multi_valued_counts_merge_by_addition():
    a = mask_counts([class_ids_to_mask([1, 2])])
    b = mask_counts([class_ids_to_mask([2]), class_ids_to_mask([2, 3])])
    a.update(b)
    summary, _ = st._nominal_stats(a, 3)
    assert summary["total_samples"] == 3


def test_multi_valued_cat_entries_have_no_reserved_class_row():
    counts = mask_counts([class_ids_to_mask([1, 2])] * 2)
    summary, _ = st._nominal_stats(counts, 2)
    entries = st._nominal_cat_entries("lith_multi", _MULTI_LAYER, counts, summary)
    metrics = {e["metric"]: e["value"] for e in entries}
    assert "class_-1" not in metrics
    assert metrics["class_1"] == 1.0
    assert metrics["class_2"] == 1.0
    assert metrics["total_samples"] == 2.0


def test_nominal_counts_dispatches_on_flag():
    series = pd.Series([class_ids_to_mask([4, 5]), class_ids_to_mask([5])], dtype=float)
    multi = st._nominal_counts(series, "lith_multi", _MULTI_LAYER)
    assert multi[4] == 1
    assert multi[5] == 2
    assert multi[OBS_TOTAL_KEY] == 2
    plain = st._nominal_counts(pd.Series([1.0, 1.0, 2.0]), "kg2", {"value_type": "nominal"})
    assert plain == Counter({1: 2, 2: 1})


def test_vector_sampling_field_prefers_set_field_only_when_multi_valued():
    assert vector_sampling_field(_MULTI_LAYER) == "LITH_MASK"
    assert vector_sampling_field({"vector_field": "ECO_ID"}) == "ECO_ID"
    assert vector_sampling_field({"vector_field": "ECO_ID", "vector_set_field": "X"}) == "ECO_ID"


def test_sample_vector_batch_returns_exact_masks(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    import util.gis as gis

    mask = class_ids_to_mask([2, 40])
    gdf = gpd.GeoDataFrame(
        {"LITH_PRIMARY": [2], "LITH_MASK": [mask]},
        geometry=[box(-1_000_000, -1_000_000, 1_000_000, 1_000_000)],
        crs=3857,
    )
    path = tmp_path / "multi_vector.parquet"
    gdf.to_parquet(path)
    gis._load_vector_layer.cache_clear() if hasattr(gis._load_vector_layer, "cache_clear") else None
    out = gis.sample_vector_batch(
        path, vector_sampling_field(_MULTI_LAYER), np.array([0.0, 60.0]), np.array([0.0, 0.0])
    )
    assert int(out[0]) == mask
    assert mask_to_class_ids(out[0]) == [2, 40]
    assert np.isnan(out[1])

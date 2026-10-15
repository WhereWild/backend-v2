# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Multi-valued nominal layers: one observation can belong to several classes
at once (e.g. a rock unit that is both carbonate and siliciclastic).

A catalog layer opts in with ``"multi_valued": true`` (value_type stays
"nominal"). The per-observation set of class ids is stored in the layer's
ordinary scalar column as a bitmask — bit ``i`` set means class id ``i`` is
present — so every pipeline stage that moves a float64 column around
(enrichment, carry-forward, parquet) keeps working unchanged. Only the
stages that interpret the value decode it.

Class ids must therefore be 0..MAX_MULTI_CLASSES-1; 52 is the most a float64
column can hold exactly. Not supported for custom (uploaded) layers.

Fractions for these layers are "share of observations that include the
class", so they can sum to more than 1. To keep that denominator through
the Counter-based stats accumulators (which merge by addition), the number
of contributing observations is carried inside the class Counter under
OBS_TOTAL_KEY; see util/stats.py.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

MAX_MULTI_CLASSES = 52
OBS_TOTAL_KEY = -1


def class_ids_to_mask(ids) -> int:
    mask = 0
    for i in ids:
        i = int(i)
        if not 0 <= i < MAX_MULTI_CLASSES:
            raise ValueError(
                f"multi-valued class id {i} outside 0..{MAX_MULTI_CLASSES - 1}"
            )
        mask |= 1 << i
    return mask


def mask_to_class_ids(mask) -> list[int]:
    m = int(mask)
    return [i for i in range(MAX_MULTI_CLASSES) if m >> i & 1]


def mask_counts(values) -> Counter:
    """Per-class observation counts from an array of bitmasks.

    Observations whose mask is NaN or 0 (no class) are ignored. The number of
    counted observations is stored under OBS_TOTAL_KEY.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    masks = arr.astype(np.int64)
    masks = masks[masks > 0]
    counts: Counter = Counter()
    if masks.size == 0:
        return counts
    for bit in range(MAX_MULTI_CLASSES):
        n = int(((masks >> bit) & 1).sum())
        if n:
            counts[bit] = n
    counts[OBS_TOTAL_KEY] = int(masks.size)
    return counts


def split_obs_total(counts: Counter) -> tuple[Counter, int | None]:
    """Return (class counts without the reserved key, observation total or None)."""
    if OBS_TOTAL_KEY not in counts:
        return counts, None
    rest = Counter({k: v for k, v in counts.items() if k != OBS_TOTAL_KEY})
    return rest, int(counts[OBS_TOTAL_KEY])

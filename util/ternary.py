# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generic machinery for 3-part ("ternary") compositional variables — data whose
components sum to a constant (e.g. sand/silt/clay percentages) and so can't be
treated as independent 1D distributions without lying about their joint shape.

This module is deliberately specific to 3-part compositions: the ILR/CLR
transform generalizes to any number of parts, but rendering the result as a
2D triangle does not — a genuinely N-ary (N > 3) compositional variable would
need its own projection/visualization, not just a bigger version of this.

Three independent pieces:
  - `composition_group_members` — catalog-driven lookup of which 3 columns
    make up a composition and their triangle display order (a *display*
    convention, independent of any classifier's own argument order).
  - `build_ternary_density_grid` — a KDE fit over real occurrence data,
    genuinely per-taxon (the actual observed distribution).
  - `build_ternary_classification_overlay` — given any classifier function,
    the class id and exact class-boundary lines over the same grid. This is
    static per classifier (identical for every taxon), so it's cached rather
    than recomputed — see `functools.lru_cache` below.

Soil texture (sand/silt/clay, classified by `util.gis.derive_soil_texture_array`)
is the only current caller; see `util.gis.COMPOSITION_CLASSIFIERS` for the
registry a future compositional variable would add itself to.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy.stats import gaussian_kde

DEFAULT_TERNARY_RESOLUTION = 24  # grid points per triangle edge, for the density KDE grid
# Classification is pure arithmetic (no fitting), so it's cheap to use a much
# finer grid than the density KDE — this is what actually makes boundary
# lines look like straight lines instead of a coarse, kinked approximation.
DEFAULT_CLASSIFICATION_RESOLUTION = 300
_MIN_SAMPLES = 2
_SAMPLE_CAP = 200  # capped raw-composition sample for a scatter overlay

_AXIS_ORDER = {"top": 0, "bottom_left": 1, "bottom_right": 2}


def composition_group_members(layer_meta: dict[str, dict]) -> dict[str, list[str]]:
    """{group_id: [col_top, col_bottom_left, col_bottom_right]}, derived from
    catalog `composition_group`/`composition_axis` fields (see
    config/gis/catalog.json) — only groups with all 3 axes present are
    returned. A future compositional variable is picked up automatically by
    tagging its 3 member columns in the catalog; nothing here is soil-specific.
    This axis order is a *display* convention (which corner a column renders
    at) — it is independent of, and must not be confused with, whatever
    positional argument order a classifier function expects (see `_classify`)."""
    members: dict[str, list[tuple[int, str]]] = {}
    for col, layer in layer_meta.items():
        group = layer.get("composition_group")
        axis = layer.get("composition_axis")
        if not group or axis not in _AXIS_ORDER:
            continue
        members.setdefault(group, []).append((_AXIS_ORDER[axis], col))
    return {
        group: [col for _, col in sorted(cols)]
        for group, cols in members.items()
        if len(cols) == 3
    }

# Helmert sub-basis: an orthonormal basis for the zero-sum hyperplane CLR
# coordinates live in. This is what makes ILR an *isometric* log-ratio — an
# ordinary Euclidean KDE on these 2 coordinates corresponds to a well-defined
# density on the simplex, unlike three independent 1D KDEs (which would
# ignore that the three parts sum to a constant and lie about the joint shape).
_ILR_HELMERT = np.array([
    [1 / np.sqrt(2), 1 / np.sqrt(6)],
    [-1 / np.sqrt(2), 1 / np.sqrt(6)],
    [0.0, -2 / np.sqrt(6)],
])


def _clr(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, None)
    logx = np.log(x)
    return logx - logx.mean(axis=-1, keepdims=True)


def _ilr(x: np.ndarray) -> np.ndarray:
    return _clr(x) @ _ILR_HELMERT


def _grid_points(resolution: int) -> np.ndarray:
    return np.array([
        (i / resolution, j / resolution, (resolution - i - j) / resolution)
        for i in range(resolution + 1)
        for j in range(resolution + 1 - i)
    ])


def build_ternary_density_grid(
    triples: np.ndarray, resolution: int = DEFAULT_TERNARY_RESOLUTION,
) -> dict | None:
    """Build a ternary KDE density grid from (a, b, c) percent triples.

    `triples` is an (n, 3) array of any 3-part composition (percentages
    summing to ~100). Returns a dict with "resolution" (grid points per
    triangle edge — the (a, b, c) coordinate of grid index (i, j) is implied
    as (i/N, j/N, (N-i-j)/N), never stored), "density" (flat list, one value
    per grid vertex in row-major (i, j) order), and a capped raw-sample of
    the input compositions for a scatter overlay. Returns None if there
    isn't enough data or the fit fails.
    """
    try:
        arr = np.asarray(triples, dtype=np.float64)
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.shape[0] < _MIN_SAMPLES:
            return None
        row_sums = arr.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            arr = arr[row_sums[:, 0] > 0]
            row_sums = arr.sum(axis=1, keepdims=True)
        if arr.shape[0] < _MIN_SAMPLES:
            return None
        comp = arr / row_sums  # renormalize off ~100 +/- rounding noise

        coords = _ilr(comp)
        kde = gaussian_kde(coords.T)

        grid_points = _grid_points(resolution)
        grid_ilr = _ilr(grid_points)
        density = kde(grid_ilr.T)
        max_density = float(density.max())
        if max_density <= 0 or not math.isfinite(max_density):
            return None
        density_norm = density / max_density

        rng = np.random.default_rng(0)
        n_sample = min(_SAMPLE_CAP, comp.shape[0])
        sample_idx = rng.choice(comp.shape[0], size=n_sample, replace=False)
        sample = comp[sample_idx]

        return {
            "resolution": resolution,
            "density": density_norm.tolist(),
            "sample_a": sample[:, 0].tolist(),
            "sample_b": sample[:, 1].tolist(),
            "sample_c": sample[:, 2].tolist(),
        }
    except Exception:
        return None


_BOUNDARY_BISECT_ITERS = 22  # ~2^-22 of an edge's length — visually exact


def _classify(classify_fn: Callable, axis_columns: tuple[str, str, str], a, b, c):
    """Calls `classify_fn` with (a, b, c) values passed as keyword arguments
    named by `axis_columns` — the catalog composition_group's member column
    ids, in the same [top, bottom_left, bottom_right] order the density grid
    uses for a/b/c. This is the one place that reconciles two independent
    orderings that would otherwise silently rotate the result: axis_columns
    is a *display* ordering (which corner a column renders at), while a
    classifier function's positional arguments carry their own fixed
    semantic order (e.g. `derive_soil_texture_array(sand, silt, clay)`) that
    has nothing to do with triangle geometry. Requiring classify_fn's
    parameter names to match its composition's catalog column ids — true for
    `derive_soil_texture_array(sand, silt, clay)` against catalog ids
    "sand"/"silt"/"clay" — is the registration contract for any future
    classifier added to `util.gis.COMPOSITION_CLASSIFIERS`."""
    return classify_fn(**{axis_columns[0]: a, axis_columns[1]: b, axis_columns[2]: c})


def _bisect_class_boundaries_vectorized(
    classify_fn: Callable, axis_columns: tuple[str, str, str],
    p0: np.ndarray, class0: np.ndarray, p1: np.ndarray,
) -> np.ndarray:
    """Binary-search along many edges at once for the precise (a, b, c)
    composition where each one's class changes. `p0`/`p1` are (n, 3) arrays
    of edge endpoints and `class0` is the (n,) array of `p0`'s class ids.
    Classifying all n edges' midpoints in a single vectorized call per
    iteration (instead of one Python-level call per edge) is what makes a
    fine classification grid cheap — the classifier is already vectorized
    (`np.select` under the hood for the USDA rules), so there's no reason to
    call it one point at a time."""
    lo, hi = p0.copy(), p1.copy()
    for _ in range(_BOUNDARY_BISECT_ITERS):
        mid = (lo + hi) / 2.0
        mid_pct = mid * 100.0
        mid_class = _classify(classify_fn, axis_columns, mid_pct[:, 0], mid_pct[:, 1], mid_pct[:, 2])
        matches = mid_class == class0
        lo = np.where(matches[:, None], mid, lo)
        hi = np.where(matches[:, None], hi, mid)
    return (lo + hi) / 2.0


def _classification_boundary_segments(
    classify_fn: Callable, axis_columns: tuple[str, str, str],
    grid_points: np.ndarray, class_ids: np.ndarray, resolution: int,
) -> tuple[list[float], list[float]]:
    """Classification boundary line segments, one pair of endpoints per mesh
    face edge that crosses a class change (bisected to the exact composition
    where the class changes). Deliberately simple — connect each face's
    crossing points in edge order, no special-casing — because the real fix
    for a jagged/kinked line at low resolution is a fine enough grid, not
    extra logic to paper over a coarse one. `resolution` here is the
    classification grid's own resolution, independent of (and typically much
    finer than) the density KDE's — classification is cheap arithmetic with
    no fitting involved, so a much finer grid costs almost nothing, as long
    as the bisection is vectorized across all crossing edges at once (see
    `_bisect_class_boundaries_vectorized`) rather than looped one at a time.
    Returns two flat lists of (a, b) values — consecutive pairs of entries
    are one segment's two endpoints (c is always derivable as 1 - a - b, so
    it isn't transmitted)."""
    index_of: dict[tuple[int, int], int] = {}
    idx = 0
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            index_of[(i, j)] = idx
            idx += 1

    # Enumerate every mesh edge as (i0, i1) index pairs, grouped by which
    # face(s) they belong to, without calling the classifier yet.
    face_edges: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for i in range(resolution):
        for j in range(resolution - i):
            a = index_of[(i, j)]
            b = index_of[(i + 1, j)]
            c = index_of[(i, j + 1)]
            face_edges.append(((a, b), (b, c), (c, a)))
            d = index_of.get((i + 1, j + 1))
            if d is not None:
                face_edges.append(((b, d), (d, c), (c, b)))

    # Collect the distinct crossing edges across the whole grid, then bisect
    # all of them in one vectorized pass.
    crossing_edges: dict[tuple[int, int], int] = {}  # (i0, i1) -> position in the batch arrays
    p0_list: list[np.ndarray] = []
    p1_list: list[np.ndarray] = []
    class0_list: list[int] = []
    for edges in face_edges:
        for i0, i1 in edges:
            if class_ids[i0] == class_ids[i1]:
                continue
            key = (i0, i1)
            if key in crossing_edges or (i1, i0) in crossing_edges:
                continue
            crossing_edges[key] = len(p0_list)
            p0_list.append(grid_points[i0])
            p1_list.append(grid_points[i1])
            class0_list.append(int(class_ids[i0]))

    if not p0_list:
        return [], []

    crossings = _bisect_class_boundaries_vectorized(
        classify_fn, axis_columns,
        np.array(p0_list), np.array(class0_list), np.array(p1_list),
    )

    def _crossing_point(i0: int, i1: int) -> np.ndarray | None:
        pos = crossing_edges.get((i0, i1))
        if pos is None:
            pos = crossing_edges.get((i1, i0))
        return None if pos is None else crossings[pos]

    boundary_a: list[float] = []
    boundary_b: list[float] = []
    for edges in face_edges:
        points = []
        for i0, i1 in edges:
            crossing = _crossing_point(i0, i1)
            if crossing is not None:
                points.append(crossing)
        for k in range(1, len(points)):
            boundary_a.extend([float(points[k - 1][0]), float(points[k][0])])
            boundary_b.extend([float(points[k - 1][1]), float(points[k][1])])

    return boundary_a, boundary_b


@lru_cache(maxsize=32)
def build_ternary_classification_overlay(
    resolution: int,
    classify_fn: Callable,
    axis_columns: tuple[str, str, str],
    line_resolution: int = DEFAULT_CLASSIFICATION_RESOLUTION,
) -> dict:
    """Class id per grid vertex, plus classification boundary line segments,
    for any classifier function over a 3-part composition. Identical for
    every taxon that shares the same classifier (classification doesn't
    depend on occurrence data at all), so this is cached rather than
    recomputed per request — the cache key is (resolution, classify_fn,
    axis_columns, line_resolution), all hashable. `axis_columns` is the
    composition_group's member column ids in [top, bottom_left,
    bottom_right] order — see `_classify`'s docstring for why this can't
    just be passed positionally.

    `resolution` and `line_resolution` are deliberately independent:
    `class_ids` must match `resolution` (the density grid's own resolution,
    e.g. 24) so the frontend can map density-mesh faces to a class for the
    optional shading overlay — but the boundary *lines* are just raw (a, b)
    coordinates, not tied to any mesh, so they're computed on a much finer
    `line_resolution` instead. Classification is pure arithmetic (no
    fitting), so the finer grid costs almost nothing, and it's what actually
    makes a straight threshold (e.g. "silt >= 50%") render as a straight
    line instead of a coarse, kinked approximation.
    """
    grid_points = _grid_points(resolution)
    grid_pct = grid_points * 100.0
    class_ids = _classify(classify_fn, axis_columns, grid_pct[:, 0], grid_pct[:, 1], grid_pct[:, 2])

    line_grid_points = _grid_points(line_resolution)
    line_grid_pct = line_grid_points * 100.0
    line_class_ids = _classify(
        classify_fn, axis_columns, line_grid_pct[:, 0], line_grid_pct[:, 1], line_grid_pct[:, 2],
    )
    boundary_a, boundary_b = _classification_boundary_segments(
        classify_fn, axis_columns, line_grid_points, line_class_ids, line_resolution,
    )
    return {
        "class_ids": class_ids.astype(int).tolist(),
        "boundary_a": boundary_a,
        "boundary_b": boundary_b,
    }

#!/usr/bin/env python3
"""
gen_colors.py

Reads the legend JSON files from config/gis/legends/ and generates
CB-safe color mappings for two modes:
  cb   — colorblind-friendly (works for deu/pro/tri simultaneously)
  ach  — achromatopsia (monochromacy; pure lightness variation)

Output: <FRONTEND_DIR>/components/sections/speciesOccurrenceMap/cbColors.ts

FRONTEND_DIR defaults to ../frontend (relative to the backend root).
Override with the FRONTEND_DIR environment variable.

Algorithm
---------
For each variable the script:
  1. Selects the smallest expert palette that has enough slots for all groups:
       ≤7 groups  → Okabe-Ito (7 colors)
       8-10 groups → Paul Tol Muted (10 colors)
       11-12 groups → Krzywinski 12-color
  2. Assigns groups to palette slots via the Hungarian algorithm (LSAP),
     minimising total Lab ΔE76 distance between each group's source-color
     centroid and its assigned palette slot.
  3. Within a group, members vary slightly in L around the slot's native L,
     ordered darkest source first.
  Achromatopsia: groups are distributed across L [12, 88]; members vary
  slightly within their group's allocated range.

Usage
-----
  python scripts/gen_colors.py
  FRONTEND_DIR=/path/to/frontend python scripts/gen_colors.py
"""

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# LCH ↔ sRGB (no external deps required)
# ---------------------------------------------------------------------------

_D65 = (0.95047, 1.0, 1.08883)
_EPS_CUBE = 0.008856
_EPS_LIN = 0.04045
_EPS_DELIN = 0.0031308
_CBRT_EPS = 0.2069  # ≈ (6/29)

def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip('#')
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0

def _rgb_to_hex(r: float, g: float, b: float) -> str:
    def ch(v: float) -> int:
        return max(0, min(255, round(v * 255)))
    return f'#{ch(r):02x}{ch(g):02x}{ch(b):02x}'

def _lin(c: float) -> float:
    return c / 12.92 if c <= _EPS_LIN else ((c + 0.055) / 1.055) ** 2.4

def _delin(c: float) -> float:
    return c * 12.92 if c <= _EPS_DELIN else 1.055 * c ** (1.0 / 2.4) - 0.055

def _rgb_lin_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z

def _xyz_to_rgb_lin(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r =  x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g =  x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b =  x *  0.0556434 + y * -0.2040259 + z *  1.0572252
    return r, g, b

def _lab_f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > _EPS_CUBE else 7.787 * t + 16.0 / 116.0

def _lab_f_inv(t: float) -> float:
    return t ** 3.0 if t > _CBRT_EPS else (t - 16.0 / 116.0) / 7.787

def hex_to_lch(h: str) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb(h)
    rl, gl, bl = _lin(r), _lin(g), _lin(b)
    x, y, z = _rgb_lin_to_xyz(rl, gl, bl)
    xn, yn, zn = x / _D65[0], y / _D65[1], z / _D65[2]
    L = 116.0 * _lab_f(yn) - 16.0
    a = 500.0 * (_lab_f(xn) - _lab_f(yn))
    b2 = 200.0 * (_lab_f(yn) - _lab_f(zn))
    C = math.sqrt(a * a + b2 * b2)
    H = math.degrees(math.atan2(b2, a)) % 360.0
    return L, C, H

def lch_to_hex(L: float, C: float, H: float) -> str:
    a = C * math.cos(math.radians(H))
    b2 = C * math.sin(math.radians(H))
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b2 / 200.0
    x = _lab_f_inv(fx) * _D65[0]
    y = _lab_f_inv(fy) * _D65[1]
    z = _lab_f_inv(fz) * _D65[2]
    rl, gl, bl = _xyz_to_rgb_lin(x, y, z)
    r = max(0.0, min(1.0, _delin(rl)))
    g = max(0.0, min(1.0, _delin(gl)))
    b = max(0.0, min(1.0, _delin(bl)))
    return _rgb_to_hex(r, g, b)

# ---------------------------------------------------------------------------
# Expert CB-safe palettes
#
# Okabe-Ito (7):  ≤7 groups — the standard CVD-safe palette
# Paul Tol Muted (10): 8-10 groups — widely used in scientific publishing
# Krzywinski 12:  11-12 groups — mathematically derived, deu/pro/tri safe
#
# All three palettes are designed to be distinguishable for deuteranopia,
# protanopia, AND tritanopia simultaneously, so one palette serves all modes.
# ---------------------------------------------------------------------------

# fmt: off
_PALETTE_OKABE_ITO: List[str] = [
    '#D55E00',  # vermillion
    '#0072B2',  # blue
    '#009E73',  # bluish green
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#CC79A7',  # reddish purple
]

_PALETTE_TOL_MUTED: List[str] = [
    '#332288',  # indigo
    '#44AA99',  # teal
    '#AA4499',  # purple
    '#CC6677',  # rose
    '#117733',  # green
    '#882255',  # wine
    '#88CCEE',  # cyan
    '#999933',  # olive
    '#DDCC77',  # sand
    '#DDDDDD',  # pale gray
]

_PALETTE_KRZYWINSKI_12: List[str] = [
    '#006655',  # blue-green
    '#009988',  # teal
    '#44BB99',  # light teal
    '#BBCC33',  # yellow-green
    '#AAAA00',  # olive-green
    '#EEDD88',  # sand
    '#EE8866',  # orange
    '#FFAABB',  # pink/rose
    '#992288',  # purple
    '#661100',  # dark magenta
    '#77AADD',  # light blue
    '#99DDFF',  # cyan-blue
]
# fmt: on

# Groups that are data artifacts, not real categories — always rendered as
# neutral gray regardless of palette; excluded from slot assignment.
_NEUTRAL_GROUPS = {'filled', 'undefined'}
_NEUTRAL_HEX = '#aaaaaa'

_SHAPES = [
    'circle', 'square', 'triangle', 'diamond', 'ring',
    'triangle-down', 'cross', 'plus', 'star', 'hexagon',
    'pentagon', 'arrow',
]

def _select_palette(n_groups: int) -> List[str]:
    if n_groups <= len(_PALETTE_OKABE_ITO):
        return _PALETTE_OKABE_ITO
    elif n_groups <= len(_PALETTE_TOL_MUTED):
        return _PALETTE_TOL_MUTED
    else:
        return _PALETTE_KRZYWINSKI_12

# ---------------------------------------------------------------------------
# Palette slot assignment
# ---------------------------------------------------------------------------

def _lab(hex_color: str) -> Tuple[float, float, float]:
    L, C, H = hex_to_lch(hex_color)
    return L, C * math.cos(math.radians(H)), C * math.sin(math.radians(H))

def _lab_dist(a: str, b: str) -> float:
    L1, a1, b1 = _lab(a)
    L2, a2, b2 = _lab(b)
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def _lab_dist_t(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _lab_centroid(hex_colors: List[str]) -> Tuple[float, float, float]:
    labs = [_lab(h) for h in hex_colors]
    n = len(labs)
    return (sum(l[0] for l in labs) / n, sum(l[1] for l in labs) / n, sum(l[2] for l in labs) / n)

def _farthest_point_slots(palette: List[str], n: int) -> List[int]:
    """Pick n palette indices that maximise minimum pairwise Lab distance."""
    gray = '#808080'
    chosen = [max(range(len(palette)), key=lambda i: _lab_dist(palette[i], gray))]
    remaining = [i for i in range(len(palette)) if i != chosen[0]]
    while len(chosen) < n:
        best = max(remaining, key=lambda i: min(_lab_dist(palette[i], palette[j]) for j in chosen))
        chosen.append(best)
        remaining.remove(best)
    return chosen

def _linear_sum_assignment(cost: List[List[float]]) -> List[int]:
    """Hungarian algorithm (O(n³)). Returns assignment[i] = j for row i → col j."""
    n = len(cost)
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = row assigned to column j (1-indexed)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minval = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minval[j]:
                        minval[j] = cur
                        way[j] = j0
                    if minval[j] < delta:
                        delta, j1 = minval[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minval[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    result = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result

def _assign_palette_slots(
    groups: Dict[str, List[dict]],
    palette: List[str],
) -> Dict[str, str]:
    """
    Solve the Linear Sum Assignment Problem (Hungarian algorithm) to find the
    bijection from groups → palette slots that minimises total Lab (ΔE76)
    distance between each group's source-color centroid and its assigned slot.

    When n_groups < palette_size, farthest-point sampling first selects the
    n_groups most perceptually distinct candidate slots so the assignment has
    maximum spread before minimisation.
    """
    n_groups = len(groups)

    if n_groups == len(palette):
        candidate_indices = list(range(len(palette)))
    else:
        candidate_indices = _farthest_point_slots(palette, n_groups)

    candidates = [palette[i] for i in candidate_indices]
    group_names = list(groups.keys())
    group_labs = [
        _lab_centroid([c['traits']['color'] for c in groups[g]])
        for g in group_names
    ]
    palette_labs = [_lab(c) for c in candidates]

    cost = [
        [_lab_dist_t(group_labs[i], palette_labs[j]) for j in range(n_groups)]
        for i in range(n_groups)
    ]
    assignment = _linear_sum_assignment(cost)

    return {group_names[i]: candidates[assignment[i]] for i in range(n_groups)}

# ---------------------------------------------------------------------------
# Achromatopsia L assignment
# ---------------------------------------------------------------------------

def _ach_l_values(all_count: int, group_rank: int, group_size: int) -> List[float]:
    """
    Spread ALL classes across L [12, 88] by group rank, then vary slightly
    within the group. Step scales with available space to avoid collisions.
    """
    inter = (88.0 - 12.0) / max(all_count - 1, 1)
    base = 12.0 + inter * group_rank
    step = min(inter / max(group_size, 1), 4.0)
    half = (group_size - 1) * step / 2.0
    return [base - half + i * step for i in range(group_size)]

# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_variable(classes: List[dict]) -> Dict[str, Dict]:
    """Return {'cb', 'ach', 'shapes'} → {class_id → value} dicts."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for cls in classes:
        groups[cls['group']].append(cls)

    # Track first-appearance order for stable shape assignment
    seen_groups: List[str] = []
    for cls in classes:
        if cls['group'] not in seen_groups:
            seen_groups.append(cls['group'])

    # Sort members within each group by source L (darkest first)
    for members in groups.values():
        members.sort(key=lambda c: hex_to_lch(c['traits']['color'])[0])

    result: Dict[str, Dict] = {'cb': {}, 'ach': {}, 'shapes': {}}

    # Shapes: sequential from _SHAPES by first-appearance group order
    for gi, g in enumerate(seen_groups):
        shape = _SHAPES[gi % len(_SHAPES)]
        for cls in groups[g]:
            result['shapes'][cls['id']] = shape

    # Achromatopsia: pure lightness spread, palette irrelevant
    group_order = list(groups.keys())
    for gi, g in enumerate(group_order):
        members = groups[g]
        L_vals = _ach_l_values(len(group_order), gi, len(members))
        for idx, cls in enumerate(members):
            result['ach'][cls['id']] = lch_to_hex(max(8.0, min(92.0, L_vals[idx])), 0.0, 0.0)

    # Colorblind-friendly: palette slot per group, small L variation within.
    # Neutral groups (filled, undefined) get a fixed gray and are excluded
    # from palette slot assignment so they don't consume a real slot.
    real_groups = {g: m for g, m in groups.items() if g not in _NEUTRAL_GROUPS}
    palette = _select_palette(len(real_groups))
    slot_assignment = _assign_palette_slots(real_groups, palette)
    for g in groups:
        if g in _NEUTRAL_GROUPS:
            slot_assignment[g] = _NEUTRAL_HEX

    for g, members in groups.items():
        base_hex = slot_assignment[g]
        base_L, base_C, base_H = hex_to_lch(base_hex)
        n = len(members)
        step = min(4.0, 16.0 / max(n - 1, 1)) if n > 1 else 0.0
        half = (n - 1) * step / 2.0
        for idx, cls in enumerate(members):
            L = max(8.0, min(92.0, base_L - half + idx * step))
            result['cb'][cls['id']] = lch_to_hex(L, base_C, base_H)

    return result

# ---------------------------------------------------------------------------
# TypeScript output
# ---------------------------------------------------------------------------

_TS_HEADER = """\
// AUTO-GENERATED by scripts/gen_colors.py — do not edit by hand.
// Re-run the script to regenerate after updating legend JSON files.

export type CbMode = 'colorblind' | 'achromatopsia';
export type ShapeKey = 'circle' | 'square' | 'triangle' | 'diamond' | 'ring' | 'triangle-down' | 'cross' | 'plus' | 'star' | 'hexagon' | 'pentagon' | 'arrow';

/** Maps variable layer_id → CB mode → class ID → CB-safe hex color. */
export const CB_CLASS_COLORS: Record<string, Partial<Record<CbMode, Record<number, string>>>> = {
"""

_MODE_TS_KEY: Dict[str, str] = {
    'cb':  'colorblind',
    'ach': 'achromatopsia',
}

def _format_variable(layer_id: str, data: Dict[str, Dict]) -> str:
    lines = [f'  {layer_id}: {{']
    for mode_key, ts_key in _MODE_TS_KEY.items():
        entries = data[mode_key]
        inner = ', '.join(f'{cid}: \'{color}\'' for cid, color in sorted(entries.items()))
        lines.append(f'    {ts_key}: {{ {inner} }},')
    lines.append('  },')
    return '\n'.join(lines)

def _format_shapes_variable(layer_id: str, shapes: Dict[int, str]) -> str:
    inner = ', '.join(f"{cid}: '{s}'" for cid, s in sorted(shapes.items()))
    return f'  {layer_id}: {{ {inner} }},'

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)

    frontend_dir = os.environ.get(
        'FRONTEND_DIR',
        os.path.join(backend_dir, '..', 'frontend'),
    )
    frontend_dir = os.path.abspath(frontend_dir)

    legends_dir = os.path.join(backend_dir, 'config', 'gis', 'legends')
    out_path = os.path.join(
        frontend_dir,
        'components', 'sections', 'speciesOccurrenceMap', 'cbColors.ts',
    )

    if not os.path.isdir(legends_dir):
        print(f'Error: legends directory not found: {legends_dir}', file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(out_path)
    if not os.path.isdir(out_dir):
        print(f'Error: frontend output directory not found: {out_dir}', file=sys.stderr)
        print(f'  (FRONTEND_DIR={frontend_dir})', file=sys.stderr)
        sys.exit(1)

    variables = {}
    for filename in sorted(os.listdir(legends_dir)):
        if not filename.endswith('_legend.json'):
            continue
        path = os.path.join(legends_dir, filename)
        with open(path) as f:
            data = json.load(f)
        layer_id = data['layer_id']
        classes = data['classes']
        all_groups = sorted(set(c['group'] for c in classes))
        real_group_count = sum(1 for g in all_groups if g not in _NEUTRAL_GROUPS)
        palette_name = (
            'Okabe-Ito' if real_group_count <= 7
            else 'Tol Muted' if real_group_count <= 10
            else 'Krzywinski-12'
        )
        neutral = [g for g in all_groups if g in _NEUTRAL_GROUPS]
        neutral_note = f' ({", ".join(neutral)} → gray)' if neutral else ''
        print(f'  {layer_id}: {len(classes)} classes, {real_group_count} real groups → {palette_name}{neutral_note}')
        variables[layer_id] = generate_variable(classes)

    ts = _TS_HEADER
    ts += '\n'.join(_format_variable(lid, data) for lid, data in variables.items())
    ts += '\n};\n\n'
    ts += '/** Look up a CB-safe color, falling back to the source color if not covered. */\n'
    ts += 'export function getCbColor(\n'
    ts += '  variableId: string,\n'
    ts += '  classId: number,\n'
    ts += '  cbMode: CbMode | null | undefined,\n'
    ts += '  fallback: string,\n'
    ts += '): string {\n'
    ts += '  if (!cbMode) return fallback;\n'
    ts += '  // Strip temporal suffix (e.g. weather_code_simple_mode_168h → weather_code_simple)\n'
    ts += '  const baseId = variableId.replace(/_(avg|sum|mode|snapshot)_\\d+h$/i, \'\');\n'
    ts += '  return CB_CLASS_COLORS[baseId]?.[cbMode]?.[classId] ?? fallback;\n'
    ts += '}\n'
    ts += '\n'
    ts += '/** Maps variable layer_id → class ID → shape key used in achromatopsia mode. */\n'
    ts += 'export const CB_CLASS_SHAPES: Record<string, Record<number, ShapeKey>> = {\n'
    ts += '\n'.join(_format_shapes_variable(lid, data['shapes']) for lid, data in variables.items())
    ts += '\n};\n\n'
    ts += '/** Look up the shape for a class in achromatopsia mode. */\n'
    ts += 'export function getCbShape(variableId: string, classId: number): ShapeKey {\n'
    ts += '  const baseId = variableId.replace(/_(avg|sum|mode|snapshot)_\\d+h$/i, \'\');\n'
    ts += "  return CB_CLASS_SHAPES[baseId]?.[classId] ?? 'circle';\n"
    ts += '}\n'

    with open(out_path, 'w') as f:
        f.write(ts)
    print(f'Written → {out_path}')

if __name__ == '__main__':
    main()

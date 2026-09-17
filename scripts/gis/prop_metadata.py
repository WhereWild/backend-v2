# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Embeds this project's own WHEREWILD_VALUE_TYPE/WHEREWILD_LEGEND GDAL
metadata into every GeoTIFF in data/gis/layers/, from the exact same
catalog/legend data the map/API already serve from -- so any of these
production layers, dropped straight into frontend's /gis-editor for local
inspection, opens already configured with the correct data type and full
legend instead of falling back to /gis-editor's own (deliberately
conservative, sample-based) auto-detection.

WHAT GETS WRITTEN, AND WHY THOSE TWO ITEMS SPECIFICALLY
    Two dataset-level (no `sample` attribute -- these apply to the whole
    file, not one band) GDAL_METADATA items, exactly matching what
    frontend/components/gisEditor/tiffMetadataWriter.ts writes when a user
    manually saves a file from /gis-editor, and what
    rasterMetadata.ts's readWherewildConfig() reads back:

      WHEREWILD_VALUE_TYPE  "nominal" | "ordinal" | "interval" | "ratio" |
                             "circular" -- taken directly from this layer's
                             catalog.json entry, never guessed.
      WHEREWILD_LEGEND      For nominal/ordinal layers only: a JSON array
                             of {id, name, color}, built from this layer's
                             own config/gis/legends/<id>_legend.json (id,
                             name) plus its real display color --
                             classes[].traits.color for nominal (the same
                             color the map itself renders that class
                             with), or config/gis/cb_colors.json's own
                             DEFAULT_COLORMAP entry for ordinal (matching
                             frontend/components/gisEditor/paletteColors.ts's
                             own defaultOrdinalColor(), which also defaults
                             to viridis -- ordinal's real per-pixel color
                             always comes from whichever colormap the user
                             has picked at render time regardless; this is
                             just a sane starting legend swatch, same as
                             it is in the frontend's own auto-detected
                             default).

WHY EMBEDDING THIS MATTERS BEYOND JUST SAVING A DETECTION PASS
    Some of this can't be recovered from pixel values at all, no matter
    how thorough a scan gets: "ratio" vs "interval" depends on whether
    zero is truly meaningful for a variable, which isn't something pixel
    values alone can prove (see frontend's dataTypeDetection.ts's own
    comment on this) -- a temperature layer and a rainfall layer can have
    indistinguishable-looking value distributions but very different
    correct answers. Embedding the catalog's own authoritative value_type
    makes that ambiguity moot for these specific, already-classified
    layers.

REAL GDAL DOUBLE-ESCAPES XML METADATA -- CONFIRMED, HANDLED ON THE READ SIDE
    rasterio's update_tags() (-> GDAL's SetMetadataItem) has a confirmed,
    longstanding quirk where it XML-escapes an Item's text TWICE, not
    once (its own reader silently undoes both passes, so this is
    invisible to anything reading the file back through GDAL/rasterio
    itself -- e.g. a literal '&' round-trips fine through GDAL, but is
    stored on disk as "&amp;amp;", not "&amp;"). frontend's geotiff.js
    does zero unescaping on read, so readWherewildConfig() had to be
    updated (see its own unescapeXmlEntities()) to undo however many
    passes were actually applied. Nothing to do here -- just documented
    so the next person touching either side of this isn't caught out by
    it again.

RUN AFTER build_overviews.py IN THE PIPELINE
    Reads/writes each COG's tags in place via a lightweight rasterio r+
    open -- doesn't rebuild pixel data, overviews, or COG layout at all,
    so ordering relative to enrich_tree/etc. downstream doesn't matter.
    Running right after build_overviews.py specifically just means it's
    acting on the final, fully-(re)built COG for this run rather than one
    that's stale/about to be rebuilt anyway.

VECTOR LAYERS (ecoregions/biome) ARE NOT HANDLED HERE
    Both share data/gis/layers/ecoregions_vector.parquet -- a GeoParquet
    file, not a GeoJSON, and /gis-editor's vector path only reads .geojson
    today. There's no COG/GDAL_METADATA equivalent to embed this into.
    Left as a follow-up; see the PR/commit this shipped in for the
    options considered.

Usage:
    uv run python -m scripts.gis.prop_metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import rasterio

CATALOG_PATH = Path("config/gis/catalog.json")
LEGENDS_DIR = Path("config/gis/legends")
CB_COLORS_PATH = Path("config/gis/cb_colors.json")
LAYERS_DIR = Path("data/gis/layers")

# Matches frontend/components/gisEditor/dataTypeDetection.ts's own
# ValueTypeGuess union exactly.
VALID_VALUE_TYPES = {"nominal", "ordinal", "interval", "ratio", "circular"}

# Matches frontend/components/sections/speciesOccurrenceMap/variableColors.ts's
# DEFAULT_COLORMAP -- the colormap an ordinal layer's legend swatches are
# sampled from before the user ever touches the colormap picker.
DEFAULT_COLORMAP = "viridis"


def _load_layer_meta() -> dict[str, dict]:
    """{filename: layer_entry} for every catalog layer that actually has a
    static file -- same shape/source as build_overviews.py's own helper of
    the same name, just skipping the (many) derived/on-the-fly catalog
    entries with no filename at all, which would otherwise all collide on
    a single `None` dict key."""
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    return {
        layer["filename"]: layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("filename")
    }


def _load_cb_colors() -> dict:
    if not CB_COLORS_PATH.exists():
        return {}
    with open(CB_COLORS_PATH) as f:
        return json.load(f)


def _classes_for_layer(layer_id: str, value_type: str, cb_colors: dict) -> list[dict]:
    """Builds the {id, name, color} legend list for one nominal/ordinal
    layer, from its own legend file. Returns [] if there's no legend file
    for this layer -- a nominal/ordinal catalog entry with no
    <id>_legend.json is either a derived layer with no static COG to embed
    into anyway, or genuinely missing legend data this script has no
    business inventing."""
    legend_path = LEGENDS_DIR / f"{layer_id}_legend.json"
    if not legend_path.exists():
        return []
    with open(legend_path) as f:
        legend = json.load(f)
    ordinal_colors = cb_colors.get(layer_id, {}).get(DEFAULT_COLORMAP, {})
    classes: list[dict] = []
    for cls in legend.get("classes", []):
        cid = cls.get("id")
        if cid is None:
            continue
        if value_type == "ordinal":
            color = ordinal_colors.get(str(cid))
        else:
            color = (cls.get("traits") or {}).get("color")
        classes.append({"id": cid, "name": cls.get("name") or str(cid), "color": color})
    return classes


def _embed_metadata(path: Path, value_type: str, classes: list[dict]) -> bool:
    """Writes WHEREWILD_VALUE_TYPE/WHEREWILD_LEGEND, returning True if
    anything actually changed (so main() can report real work done vs. an
    already-up-to-date no-op)."""
    legend_json = json.dumps(classes, separators=(",", ":")) if classes else None
    # IGNORE_COG_LAYOUT_BREAK: same reasoning as build_overviews.py's own
    # _fill_nodata_with_zero -- a tag-only edit via r+ against an
    # already-COG-laid-out file otherwise gets refused outright by GDAL's
    # COG driver, even though nothing about the file's actual tile/
    # overview layout is being touched here at all.
    with rasterio.open(path, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as ds:
        tags = ds.tags()
        up_to_date = (
            tags.get("WHEREWILD_VALUE_TYPE") == value_type
            and tags.get("WHEREWILD_LEGEND") == legend_json
        )
        if up_to_date:
            return False
        new_tags = {"WHEREWILD_VALUE_TYPE": value_type}
        if legend_json is not None:
            new_tags["WHEREWILD_LEGEND"] = legend_json
        # update_tags() merges into the existing domain rather than
        # replacing it -- if a layer's catalog value_type ever changes
        # away from nominal/ordinal (legend_json now None), this won't
        # clear a stale WHEREWILD_LEGEND left over from when it wasn't.
        # Not worth a full SetMetadata reset for a case that shouldn't
        # come up in practice (a layer's measurement level changing at
        # all is rare); worth knowing if it ever does.
        ds.update_tags(**new_tags)
        return True


def main() -> None:
    if not LAYERS_DIR.exists():
        raise FileNotFoundError(f"Layers directory not found: {LAYERS_DIR}")

    layer_meta = _load_layer_meta()
    cb_colors = _load_cb_colors()
    total = updated = skipped = 0

    for path in sorted(LAYERS_DIR.glob("*.tif")):
        total += 1
        layer = layer_meta.get(path.name)
        if not layer:
            skipped += 1
            continue
        value_type = str(layer.get("value_type") or "").lower()
        if value_type not in VALID_VALUE_TYPES:
            skipped += 1
            continue
        layer_id = layer.get("id")
        classes = (
            _classes_for_layer(layer_id, value_type, cb_colors)
            if value_type in ("nominal", "ordinal") and layer_id
            else []
        )
        try:
            if _embed_metadata(path, value_type, classes):
                print(
                    f"[prop-metadata] embedded {value_type} "
                    f"({len(classes)} classes) -> {path.name}"
                )
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"[prop-metadata] failed {path.name}: {exc}")

    print(f"[prop-metadata] done  total={total}  updated={updated}  skipped={skipped}")


if __name__ == "__main__":  # pragma: no cover
    main()

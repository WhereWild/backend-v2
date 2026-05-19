"""
Download CHELSA layers defined in config/gis/catalog.json into data/gis/layers/.

Only processes layers whose source is "chelsa_v2_1". Re-running skips files
that already exist. After each download, reads scale/offset from the file and
patches them into the catalog if they are currently null, or warns if they
differ from what the catalog already records.
"""

import json
import subprocess
from pathlib import Path

import rasterio

CATALOG_PATH = Path("config/gis/catalog.json")
LAYERS_DIR   = Path("data/gis/layers")
SOURCE_ID    = "chelsa_v2_1"


def _load_catalog() -> dict:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _chelsa_layers(catalog: dict) -> list[dict]:
    return [
        layer
        for category in catalog["categories"]
        for layer in category["layers"]
        if layer.get("source") == SOURCE_ID
    ]


def _download(url: str, dest: Path) -> None:
    subprocess.run(
        [
            "aria2c",
            "--split=8",
            "--max-connection-per-server=8",
            "--continue=true",
            "--max-tries=12",
            "--retry-wait=15",
            "--connect-timeout=60",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            url,
        ],
        check=True,
    )


def _read_file_scaling(path: Path) -> tuple[float | None, float | None]:
    with rasterio.open(path) as ds:
        scale  = ds.scales[0]  if ds.scales  else None
        offset = ds.offsets[0] if ds.offsets else None
        # rasterio returns 1.0/0.0 when nothing is embedded — treat as absent
        if scale == 1.0 and offset == 0.0:
            return None, None
        return scale, offset


def _inspect(path: Path) -> None:
    with rasterio.open(path) as ds:
        scale, offset = _read_file_scaling(path)
        print(f"  CRS        : {ds.crs}")
        print(f"  Shape      : {ds.height} x {ds.width}")
        print(f"  Dtype      : {ds.dtypes[0]}")
        print(f"  Nodata     : {ds.nodata}")
        print(f"  Bounds     : {ds.bounds}")
        print(f"  Scale      : {scale}")
        print(f"  Offset     : {offset}")


def _sync_scaling(layer: dict, path: Path) -> bool:
    """Patch null scale/offset in the layer dict from file metadata. Returns True if changed."""
    file_scale, file_offset = _read_file_scaling(path)
    changed = False

    for key, file_val in [("scale_factor", file_scale), ("add_offset", file_offset)]:
        catalog_val = layer.get(key)
        if catalog_val is None and file_val is not None:
            print(f"  {key}: null → {file_val} (read from file)")
            layer[key] = file_val
            changed = True
        elif catalog_val is not None and file_val is not None and catalog_val != file_val:
            print(f"  WARNING: catalog {key}={catalog_val} differs from file {file_val} — keeping catalog value")

    return changed


def main() -> None:
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()
    layers = _chelsa_layers(catalog)
    catalog_dirty = False

    for layer in layers:
        dest = LAYERS_DIR / layer["filename"]
        if dest.exists():
            print(f"[skip] {layer['id']} — already at {dest}")
        else:
            print(f"[download] {layer['id']} ({layer['display_name']})")
            _download(layer["download_url"], dest)
            print(f"  Saved to {dest}")

        print(f"[inspect] {layer['id']}")
        _inspect(dest)

        if _sync_scaling(layer, dest):
            catalog_dirty = True
        print()

    if catalog_dirty:
        with open(CATALOG_PATH, "w") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Catalog updated: {CATALOG_PATH}")


if __name__ == "__main__":
    main()

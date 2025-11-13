"""Smoke-test script that calls GDAL via Docker from Python.

Run from the repo root:

    python examples/python/reproject_point.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path, label: str) -> None:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        cwd=cwd,
    )
    print(f"{label}\n{result.stdout.strip()}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    _run(
        ["docker", "compose", "run", "--rm", "gdal", "gdalinfo", "--version"],
        repo_root,
        "GDAL responded from inside Docker:",
    )
    _run(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "gdal",
            "python",
            "-c",
            (
                "import rasterio; import geopandas as gpd; import pyproj; "
                "print(f'rasterio {rasterio.__version__}'); "
                "print(f'geopandas {gpd.__version__}'); "
                "print(f'pyproj {pyproj.__version__}')"
            ),
        ],
        repo_root,
        "Python libraries available inside Docker:",
    )


if __name__ == "__main__":
    main()

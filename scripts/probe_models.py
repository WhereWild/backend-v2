"""Probe all data_spatial models on Open-Meteo S3 for variable availability."""
from __future__ import annotations

import json
from datetime import datetime

import fsspec

WANT = {"precipitation", "cloud_cover", "snowfall_water_equivalent", "dew_point_2m"}

fs = fsspec.filesystem("s3", anon=True)

print("Listing models in s3://openmeteo/data_spatial/ ...")
models = [p.split("/")[-1] for p in fs.ls("s3://openmeteo/data_spatial/", detail=False)]
print(f"Found {len(models)} models: {models}\n")

for model in sorted(models):
    try:
        with fs.open(f"s3://openmeteo/data_spatial/{model}/latest.json") as f:
            meta = json.load(f)

        ref = datetime.fromisoformat(meta["reference_time"].replace("Z", "+00:00"))
        valid = datetime.fromisoformat(meta["valid_times"][0].replace("Z", "+00:00"))
        run_dir = f"{ref.year:04d}/{ref.month:02d}/{ref.day:02d}/{ref.hour:02d}{ref.minute:02d}Z"
        fname = valid.strftime("%Y-%m-%dT%H%M") + ".om"
        path = f"s3://openmeteo/data_spatial/{model}/{run_dir}/{fname}"

        import fsspec as _fs2
        from omfiles import OmFileReader
        backend = _fs2.open(path, mode="rb", s3={"anon": True})
        root = OmFileReader(backend)
        available = sorted(root.get_child_by_index(i).name for i in range(root.num_children))

        hits = sorted(WANT & set(available))
        marker = "  *** MATCH ***" if hits else ""
        print(f"{model:30s}  ref={meta['reference_time']}  hits={hits or 'none'}{marker}")
        if hits:
            print(f"  all vars: {available}")
    except Exception as e:
        print(f"{model:30s}  ERROR: {e}")

print("\nDone.")

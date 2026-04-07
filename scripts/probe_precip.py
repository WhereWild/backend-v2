"""Check global models for any precipitation-like variables (broad name match)."""
from __future__ import annotations
import json
import fsspec
from datetime import datetime
from omfiles import OmFileReader

GLOBAL_MODELS = [
    "ecmwf_ifs", "ecmwf_ifs025", "ecmwf_aifs025_single",
    "meteofrance_arpege_world025", "jma_gsm", "ncep_aigfs025",
    "kma_gdps", "ukmo_global_deterministic_10km",
    "ncep_gfs013", "ncep_gfs025", "ncep_gfs_graphcast025",
    "cma_grapes_global", "dwd_icon",
]

PRECIP_KEYWORDS = {"precip", "rain", "shower", "snowfall", "snow_water", "liquid", "runoff", "convective_precip"}

fs = fsspec.filesystem("s3", anon=True)
for model in GLOBAL_MODELS:
    try:
        with fs.open(f"s3://openmeteo/data_spatial/{model}/latest.json") as f:
            meta = json.load(f)
        ref = datetime.fromisoformat(meta["reference_time"].replace("Z", "+00:00"))
        valid = datetime.fromisoformat(meta["valid_times"][0].replace("Z", "+00:00"))
        run_dir = f"{ref.year:04d}/{ref.month:02d}/{ref.day:02d}/{ref.hour:02d}{ref.minute:02d}Z"
        fname = valid.strftime("%Y-%m-%dT%H%M") + ".om"
        path = f"s3://openmeteo/data_spatial/{model}/{run_dir}/{fname}"
        backend = fsspec.open(path, mode="rb", s3={"anon": True})
        root = OmFileReader(backend)
        available = sorted(root.get_child_by_index(i).name for i in range(root.num_children))
        hits = [v for v in available if any(kw in v.lower() for kw in PRECIP_KEYWORDS)]
        print(f"\n{model} (ref={meta['reference_time']}):")
        print(f"  precip-like: {hits if hits else 'NONE'}")
    except Exception as e:
        print(f"\n{model}: ERROR {e}")

print("\nDone.")

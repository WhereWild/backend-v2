"""Temporal enrichment utilities.

Enriches occurrence parquets with time-windowed weather statistics from
Open-Meteo ERA5 data (s3://openmeteo/data/, public/anonymous).

Processing model: chunks are processed sequentially in ascending time order.
Each chunk is downloaded on-demand, processed, then deleted. A tail buffer
(last max_window_steps timesteps per active grid cell) is kept in memory
across chunk boundaries so 2160h windows spanning two chunks are handled
correctly without re-downloading.

# TODO: elevation correction
# Requires target elevation column in occurrence parquets (from DEM pipeline,
# not yet built). Apply lapse rate: (model_elev - obs_elev) * 0.0065 °C/m.
# Model elevation raster: s3://openmeteo/data/{model}/static/HSURF.om
# Applicable variables: temperature_2m, dew_point_2m, soil_temperature_0_to_7cm
"""
from __future__ import annotations

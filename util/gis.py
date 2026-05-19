# Hilbert curve order for spatial indexing.
# Order 13 → 2^13 × 2^13 grid → ~4.9km cells at equator.
# Smaller than a 256-pixel tile at 30m resolution (~7.68km), so observations
# in the same COG internal tile get consecutive indices. Trivially holds for
# all coarser rasters. Index fits in int32 (max value 2^26 - 1 ≈ 67M).
_HILBERT_ORDER = 13


def hilbert_index(latitude: float, longitude: float) -> int:
    """Return a Hilbert curve index for a coordinate (order 13, ~4.9km cells).

    Sort observations by this value before COG raster sampling to maximise
    spatial cache locality across all raster resolutions ≥ 30m.
    """
    n = 1 << _HILBERT_ORDER
    x = min(max(int((longitude + 180.0) / 360.0 * n), 0), n - 1)
    y = min(max(int((latitude + 90.0) / 180.0 * n), 0), n - 1)

    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d

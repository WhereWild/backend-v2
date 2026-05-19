from util.gis import _HILBERT_ORDER, hilbert_index


def test_hilbert_index_fits_in_int32():
    h = hilbert_index(40.0, -105.0)
    assert 0 <= h < (1 << (2 * _HILBERT_ORDER))


def test_hilbert_nearby_points_same_or_close_index():
    # Two points ~150m apart should map to the same cell
    a = hilbert_index(40.0, -105.0)
    b = hilbert_index(40.001, -105.001)
    assert a == b


def test_hilbert_distant_points_differ():
    denver = hilbert_index(40.0, -105.0)
    sydney = hilbert_index(-33.0, 151.0)
    assert abs(denver - sydney) > 1_000_000


def test_hilbert_poles_and_antimeridian():
    # Edge coordinates should not raise and stay in range
    max_val = (1 << (2 * _HILBERT_ORDER)) - 1
    for lat, lon in [(-90, -180), (90, 180), (0, 0), (-90, 180), (90, -180)]:
        h = hilbert_index(lat, lon)
        assert 0 <= h <= max_val


def test_hilbert_deterministic():
    assert hilbert_index(51.5, -0.1) == hilbert_index(51.5, -0.1)

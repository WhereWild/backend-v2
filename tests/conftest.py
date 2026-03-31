"""Shared pytest fixtures for the WhereWild test suite."""
from __future__ import annotations

import resource
import time

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI test client. Startup events fire once."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Known-good test data (confirmed present with all precomputed files)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def known_taxon_id() -> int:
    """Quercus robur — has occurrence, summary_stats, density_graph, occurrence_index."""
    return 2878688


@pytest.fixture(scope="session")
def known_genus_taxon_id() -> int:
    """Quercus genus — used for relative-rankings tests."""
    return 2877951


@pytest.fixture(scope="session")
def known_numeric_var() -> str:
    return "bio_1"


@pytest.fixture(scope="session")
def known_categorical_var() -> str:
    return "koppen_geiger"


@pytest.fixture(scope="session")
def known_location_gid() -> str:
    """United States country-level GID."""
    return "USA"


@pytest.fixture(scope="session")
def known_species_location_gid() -> str:
    """Great Britain — Q. robur is native to Europe and has GBIF records here."""
    return "GBR"


@pytest.fixture(scope="session")
def known_categorical_class_value() -> str:
    """A Koppen-Geiger class value common in European oak habitat."""
    return "Cfb"


@pytest.fixture(scope="session")
def data_root():
    from util.config import load_config
    return load_config("global").data_root


@pytest.fixture(scope="session")
def parquet_storage():
    from util.config import load_config
    from util.storage import get_parquet_storage
    cfg = load_config("global")
    return get_parquet_storage(cfg.data_root, cfg.project_root)


def pytest_addoption(parser):
    parser.addoption(
        "--mem-report",
        action="store_true",
        default=False,
        help="Print per-test RSS memory diagnostics (before/after/delta + peak RSS).",
    )
    parser.addoption(
        "--mem-top",
        action="store",
        type=int,
        default=10,
        help="How many tests to include in memory summary when --mem-report is enabled.",
    )


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = float(parts[1])
                        return kb / 1024.0
    except OSError:
        pass
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return float(peak_mb)


@pytest.fixture(autouse=True)
def _per_test_memory_report(request):
    if not request.config.getoption("--mem-report"):
        yield
        return
    before = _current_rss_mb()
    start = time.perf_counter()
    yield
    after = _current_rss_mb()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    duration = time.perf_counter() - start
    stats = getattr(request.config, "_mem_report_stats", None)
    if stats is None:
        stats = []
        setattr(request.config, "_mem_report_stats", stats)
    stats.append(
        {
            "nodeid": request.node.nodeid,
            "before": before,
            "after": after,
            "delta": after - before,
            "peak": peak_mb,
            "duration": duration,
        }
    )
    terminal = request.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(
            (
                f"[mem] {request.node.nodeid} "
                f"rss_before={before:.1f}MB rss_after={after:.1f}MB "
                f"delta={after - before:+.1f}MB peak={peak_mb:.1f}MB "
                f"duration={duration:.2f}s"
            )
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--mem-report"):
        return
    stats = getattr(config, "_mem_report_stats", [])
    if not stats:
        terminalreporter.write_line("[mem-summary] no per-test memory data captured")
        return
    top_n = max(1, int(config.getoption("--mem-top") or 10))
    by_delta = sorted(stats, key=lambda item: item["delta"], reverse=True)
    terminalreporter.write_sep("-", f"Memory Summary (top {min(top_n, len(by_delta))} by RSS delta)")
    for item in by_delta[:top_n]:
        terminalreporter.write_line(
            (
                f"[mem-summary] {item['nodeid']} "
                f"delta={item['delta']:+.1f}MB "
                f"before={item['before']:.1f}MB after={item['after']:.1f}MB "
                f"peak={item['peak']:.1f}MB duration={item['duration']:.2f}s"
            )
        )
    global_peak = max(item["peak"] for item in stats)
    terminalreporter.write_line(f"[mem-summary] process_peak_rss={global_peak:.1f}MB")

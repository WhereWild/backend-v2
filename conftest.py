def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-fixtures",
        action="store_true",
        default=False,
        help="Force re-fetch all Open-Meteo fixture data even if files already exist (~20 API requests)",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live S3 tests (test_live.py). Use via: pt --temporal",
    )
    parser.addoption(
        "--regenerate-live",
        action="store_true",
        default=False,
        help="Re-fetch API ground truth for live tests and rewrite live_fixtures.json",
    )

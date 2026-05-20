"""
Enrich per-taxon occurrence parquets with time-windowed ERA5 weather statistics.

Reads temporal layers from config/gis/catalog.json (category id="temporal").
Respects VARS_TO_ENRICH env var — same semantics as enrich_tree: if set,
only enriches temporal variables whose id appears in the comma-separated list.
Non-temporal ids in VARS_TO_ENRICH are silently ignored here (enrich_tree
handles them; temporal ids are ignored there).

Usage:
    python -m scripts.enrich_temporal
    VARS_TO_ENRICH=precipitation,temperature_2m python -m scripts.enrich_temporal
"""
from __future__ import annotations

import os

_raw_vars = os.environ.get("VARS_TO_ENRICH", "")
VARS_TO_ENRICH: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None


def main() -> None:
    # TODO: implement in Phase 4 (after util/temporal.py core is complete)
    raise NotImplementedError("enrich_temporal not yet implemented — see PLAN.md")


if __name__ == "__main__":  # pragma: no cover
    main()

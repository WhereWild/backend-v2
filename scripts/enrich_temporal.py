"""
Enrich per-taxon occurrence parquets with time-windowed ERA5 weather statistics.

Reads temporal layers from config/gis/catalog.json (category id="temporal").
Respects VARS_TO_ENRICH env var — same semantics as enrich_tree: if set,
only enriches temporal variables whose id appears in the comma-separated list.
Non-temporal ids in VARS_TO_ENRICH are silently ignored here (enrich_tree
handles them; temporal ids are ignored there).
If VARS_TO_ENRICH is set but contains no temporal variable ids, all temporal
variables are enriched (the assumption is the list was meant for enrich_tree).

Usage:
    python -m scripts.enrich_temporal
    VARS_TO_ENRICH=precipitation,temperature_2m python -m scripts.enrich_temporal
"""
from __future__ import annotations

import os
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.compute as pc

from config.config import load_config
from util.temporal import (
    _PREFETCH_WORKERS,
    TailBuffer,
    TemporalLayer,
    _download_layer_chunk,
    build_chunk_index,
    build_occ_index,
    derive_vpd,
    load_temporal_layers,
    map_to_worklist,
    process_chunk,
    process_chunk_mode,
    window_steps,
    write_back,
)

CATALOG_PATH = Path("config/gis/catalog.json")

_raw_vars = os.environ.get("VARS_TO_ENRICH", "")
VARS_TO_ENRICH: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _cleanup_cache(cache_dir: str) -> None:
    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return
    for path in cache_root.rglob("*"):
        try:
            if path.is_file():
                path.unlink()
        except Exception as exc:
            print(f"[cleanup] failed to remove {path}: {exc}")


def _filter_layers(all_layers: list[TemporalLayer], vars_to_enrich: list[str] | None) -> list[TemporalLayer]:
    """Return layers to process, applying VARS_TO_ENRICH filter.

    If vars_to_enrich contains at least one temporal id, restrict to those.
    If it contains no temporal ids (all spatial), do all temporal layers.
    """
    if vars_to_enrich is None:
        return all_layers
    temporal_ids = {layer.id for layer in all_layers}
    requested = [v for v in vars_to_enrich if v in temporal_ids]
    if not requested:
        return all_layers
    requested_set = set(requested)
    return [layer for layer in all_layers if layer.id in requested_set]


# ---------------------------------------------------------------------------
# Per-layer processing
# ---------------------------------------------------------------------------

def _run_layer(
    layer: TemporalLayer,
    occ_table,
    cfg,
    stop: threading.Event,
) -> None:
    print(
        f"[layer] id={layer.id} model={layer.model} agg={layer.agg} "
        f"windows={layer.windows} grid_mode={layer.grid_mode}"
    )

    chunk_var = layer.sources[0] if layer.sources else layer.id
    try:
        chunk_index = build_chunk_index(
            layer.model, chunk_var, min_year=cfg.temporal_min_year
        )
    except Exception as exc:
        print(f"[skip] {layer.id}: could not build chunk index — {exc}")
        return

    print(
        f"[chunks] {layer.id}: {len(chunk_index.ranges)} chunks, "
        f"resolution={chunk_index.resolution:.0f}s"
    )

    worklist = map_to_worklist(occ_table, chunk_index, layer.grid_mode, layer.grid_step)
    if worklist.num_rows == 0:
        print(f"[skip] {layer.id}: no observations mapped to any chunk")
        return

    steps = window_steps(chunk_index.resolution, tuple(layer.windows))
    tail_buffer: TailBuffer = {}

    chunks_with_obs = [
        entry
        for entry in chunk_index.ranges
        if worklist.filter(pc.equal(worklist["chunk_num"], entry.chunk_num)).num_rows > 0
    ]
    total_chunks = len(chunks_with_obs)

    prefetch_vars = layer.sources if layer.sources else [layer.id]
    total_rows = worklist.num_rows
    rows_done = 0
    t_start = time.monotonic()

    # Submit all downloads upfront; process each chunk as its download finishes.
    # Pool limits to _PREFETCH_WORKERS concurrent downloads so chunks N+1..N+7
    # are fetched while chunk N is being processed.
    with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as pool:
        dl_futures = [
            pool.submit(_download_layer_chunk, entry, layer.model, prefetch_vars, cfg.temporal_cache_dir)
            for entry in chunks_with_obs
        ]

        for chunk_idx, (chunk_entry, dl_fut) in enumerate(zip(chunks_with_obs, dl_futures), 1):
            if stop.is_set():
                print(f"[stop] {layer.id}: interrupted before chunk {chunk_entry.chunk_num}")
                return

            dl_fut.result()  # wait for this chunk's files to be ready

            chunk_worklist = worklist.filter(
                pc.equal(worklist["chunk_num"], chunk_entry.chunk_num)
            )

            try:
                if layer.sources:
                    updates, tail_buffer = process_chunk_mode(
                        chunk_entry,
                        chunk_worklist,
                        tail_buffer,
                        layer.model,
                        layer.sources,
                        layer.id,
                        steps,
                        chunk_index.resolution,
                        cfg.temporal_cache_dir,
                    )
                else:
                    updates, tail_buffer = process_chunk(
                        chunk_entry,
                        chunk_worklist,
                        tail_buffer,
                        layer.model,
                        layer.id,
                        steps,
                        layer.agg,
                        cfg.temporal_cache_dir,
                    )
                write_back(updates)
            except Exception:
                print(f"[error] {layer.id} chunk={chunk_entry.chunk_num}")
                traceback.print_exc()
                raise

            rows_done += chunk_worklist.num_rows
            elapsed = time.monotonic() - t_start
            rate = rows_done / elapsed if elapsed > 0 else 0.0
            remaining = total_rows - rows_done
            eta_s = remaining / rate if rate > 0 else float("inf")
            rss = _rss_mb()
            rss_str = f" rss={rss:.0f}MB" if rss is not None else ""
            print(
                f"[progress] {layer.id} chunk {chunk_idx}/{total_chunks} "
                f"rows={rows_done}/{total_rows} "
                f"rate={rate:.0f}/s eta={eta_s:.0f}s{rss_str}"
            )

    print(f"[done] {layer.id} rows={rows_done} elapsed={time.monotonic() - t_start:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config("global")
    stop = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"[signal] received {signum}, stopping after current chunk...")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception as exc:
            print(f"[warn] could not register handler for signal {sig}: {exc}")

    Path(cfg.temporal_cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        all_layers = load_temporal_layers(CATALOG_PATH)
        active_layers = _filter_layers(all_layers, VARS_TO_ENRICH)
        active_ids = {layer.id for layer in active_layers}
        print(f"[init] active layers: {[layer.id for layer in active_layers]}")

        print(f"[occ_index] scanning root={str(cfg.plantae_key)} min_year={cfg.temporal_min_year}")
        occ_table = build_occ_index(
            str(cfg.plantae_key),
            cfg.data_root,
            cfg.occurrence_parquet_filename,
            cfg.temporal_min_year,
        )
        print(f"[occ_index] {occ_table.num_rows} observations")

        if occ_table.num_rows == 0:
            print("[done] no observations to enrich")
            return

        # Non-derived layers first
        for layer in active_layers:
            if layer.derived or stop.is_set():
                continue
            _run_layer(layer, occ_table, cfg, stop)

        # Derived passes (only if their deps were processed or already present)
        if "vapor_pressure_deficit" in active_ids and not stop.is_set():
            vpd_layer = next(layer for layer in active_layers if layer.id == "vapor_pressure_deficit")
            print(f"[derive] vapor_pressure_deficit windows={vpd_layer.windows}")
            try:
                derive_vpd(
                    str(cfg.plantae_key),
                    cfg.data_root,
                    cfg.occurrence_parquet_filename,
                    vpd_layer.windows,
                )
            except Exception:
                print("[error] derive_vpd failed")
                traceback.print_exc()

    finally:
        # Cache clearing disabled for debugging — re-enable when done
        # _cleanup_cache(cfg.temporal_cache_dir)
        print("[cleanup] cache preserved (debug mode)")


if __name__ == "__main__":  # pragma: no cover
    main()

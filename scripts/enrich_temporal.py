# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    CLEAR_CACHE=0 python -m scripts.enrich_temporal   # keep cache for quick re-runs
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
    iter_occ_index_batches,
    load_temporal_layers,
    map_to_worklist,
    process_chunk,
    process_chunk_mode,
    process_chunk_vpd,
    window_steps,
    write_back,
)

# Chunks with this many observations or fewer are read via HTTP range requests
# instead of a full download. Pre-2010 chunks and carry-forward tails are
# typically sparse, so this avoids downloading GB-sized files for a handful of points.
_RANGE_REQUEST_THRESHOLD = int(os.environ.get("TEMPORAL_RANGE_REQUEST_THRESHOLD", "1000"))

CATALOG_PATH = Path("config/gis/catalog.json")

_raw_vars = os.environ.get("VARS_TO_ENRICH", "")
VARS_TO_ENRICH: list[str] | None = [v.strip() for v in _raw_vars.split(",") if v.strip()] or None

CLEAR_CACHE: bool = os.environ.get("CLEAR_CACHE", "1") != "0"

# Number of occurrence rows processed per batch. Keeps peak RSS bounded
# regardless of total observation count.
_BATCH_ROWS = int(os.environ.get("TEMPORAL_BATCH_ROWS", "5000000"))


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
    occ_index_path: Path,
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
            layer.model, chunk_var, min_date=cfg.temporal_min_date
        )
    except Exception as exc:
        print(f"[skip] {layer.id}: could not build chunk index — {exc}")
        return

    print(
        f"[chunks] {layer.id}: {len(chunk_index.ranges)} chunks, "
        f"resolution={chunk_index.resolution:.0f}s"
    )

    primary_var = layer.sources[0] if layer.sources else layer.id
    steps = window_steps(chunk_index.resolution, tuple(layer.windows))

    chunks_eligible = list(chunk_index.ranges)

    # Build secondary source chunk indices for time-range lookup inside process_chunk_*.
    secondary_indices: dict[str, object] = {}
    for src_var in layer.sources[1:]:
        try:
            secondary_indices[src_var] = build_chunk_index(
                layer.model, src_var, min_date=cfg.temporal_min_date
            )
        except Exception as exc:
            print(f"[warn] {layer.id}: could not build chunk index for {src_var} — {exc}")

    t_start = time.monotonic()
    total_rows_done = 0
    batch_num = 0
    stopped = False

    for occ_batch in iter_occ_index_batches(occ_index_path, _BATCH_ROWS):
        if stop.is_set():
            print(f"[stop] {layer.id}: interrupted")
            stopped = True
            break
        batch_num += 1

        worklist = map_to_worklist(occ_batch, chunk_index, layer.grid_mode, layer.grid_step)
        if worklist.num_rows == 0:
            continue

        batch_chunk_worklists: dict[int, object] = {}
        for entry in chunks_eligible:
            sl = worklist.filter(pc.equal(worklist["chunk_num"], entry.chunk_num))
            if sl.num_rows > 0:
                batch_chunk_worklists[entry.chunk_num] = sl

        if not batch_chunk_worklists:
            continue

        chunks_this_batch = [e for e in chunks_eligible if e.chunk_num in batch_chunk_worklists]

        # Split chunks by observation density. Sparse chunks use HTTP range
        # requests to read only the needed grid cells; dense chunks are downloaded
        # in full first since the per-cell overhead would dominate otherwise.
        sparse_set = {
            e.chunk_num for e in chunks_this_batch
            if batch_chunk_worklists[e.chunk_num].num_rows <= _RANGE_REQUEST_THRESHOLD
        }
        dense_chunks = [e for e in chunks_this_batch if e.chunk_num not in sparse_set]

        print(
            f"[prefetch] {layer.id} batch={batch_num}: "
            f"{len(dense_chunks)} download + {len(sparse_set)} range-req "
            f"({sum(batch_chunk_worklists[e.chunk_num].num_rows for e in dense_chunks)} dense obs, "
            f"{sum(batch_chunk_worklists[e.chunk_num].num_rows for e in chunks_this_batch if e.chunk_num in sparse_set)} sparse obs)"
        )

        # Prefetch primary source for dense chunks only.
        if dense_chunks:
            t_dl = time.monotonic()
            with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as dl_pool:
                for fut in [dl_pool.submit(_download_layer_chunk, e, layer.model, [primary_var], cfg.temporal_cache_dir) for e in dense_chunks]:
                    fut.result()
            print(f"[prefetch] {layer.id} batch={batch_num}: downloads done ({time.monotonic() - t_dl:.1f}s)")

        batch_rows_done = 0
        tail_buffer: TailBuffer = {}
        batch_updates: dict[str, dict[str, list]] = {}

        for chunk_entry in chunks_this_batch:
            if stop.is_set():
                print(f"[stop] {layer.id}: interrupted before chunk {chunk_entry.chunk_num}")
                stopped = True
                break
            chunk_worklist = batch_chunk_worklists[chunk_entry.chunk_num]
            use_range = chunk_entry.chunk_num in sparse_set
            print(
                f"[chunk] {layer.id} chunk={chunk_entry.chunk_num} "
                f"obs={chunk_worklist.num_rows} mode={'range-req' if use_range else 'download'}"
            )
            try:
                if layer.id == "vapor_pressure_deficit":
                    updates, tail_buffer = process_chunk_vpd(
                        chunk_entry, chunk_worklist, tail_buffer,
                        layer.model, layer.sources, layer.id,
                        steps, chunk_index.resolution, cfg.temporal_cache_dir,
                        secondary_indices=secondary_indices,
                        range_request=use_range,
                    )
                elif layer.sources:
                    updates, tail_buffer = process_chunk_mode(
                        chunk_entry, chunk_worklist, tail_buffer,
                        layer.model, layer.sources, layer.id,
                        steps, chunk_index.resolution, cfg.temporal_cache_dir,
                        secondary_indices=secondary_indices,
                        range_request=use_range,
                    )
                else:
                    updates, tail_buffer = process_chunk(
                        chunk_entry, chunk_worklist, tail_buffer,
                        layer.model, layer.id, steps, layer.agg, cfg.temporal_cache_dir,
                        range_request=use_range,
                    )
                for tpath, colmap in updates.items():
                    batch_updates.setdefault(tpath, {})
                    for col, pairs in colmap.items():
                        batch_updates[tpath].setdefault(col, []).extend(pairs)
            except Exception:
                print(f"[error] {layer.id} batch={batch_num} chunk={chunk_entry.chunk_num}")
                traceback.print_exc()
                raise

            batch_rows_done += chunk_worklist.num_rows

        total_rows_done += batch_rows_done
        rss = _rss_mb()
        rss_str = f" rss={rss:.0f}MB" if rss is not None else ""
        print(
            f"[batch] {layer.id} batch={batch_num} "
            f"rows={total_rows_done}{rss_str} elapsed={time.monotonic() - t_start:.0f}s"
        )

        if batch_updates and not stop.is_set():
            write_back(batch_updates)

        if stopped:
            break

    if not stopped:
        if total_rows_done == 0:
            print(f"[skip] {layer.id}: no observations mapped to any chunk")
        else:
            print(f"[done] {layer.id} rows={total_rows_done} elapsed={time.monotonic() - t_start:.1f}s")


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
    occ_index_path = Path(cfg.temporal_cache_dir) / "occ_index.parquet"

    try:
        all_layers = load_temporal_layers(CATALOG_PATH)
        active_layers = _filter_layers(all_layers, VARS_TO_ENRICH)
        print(f"[init] active layers: {[layer.id for layer in active_layers]}")

        # One group per non-derived layer: a row is re-queued if ANY layer is incomplete.
        skip_col_groups: list[list[str]] = [
            [f"{layer.id}_{layer.agg}_{w}h" for w in layer.windows]
            for layer in active_layers
            if not layer.derived
        ]

        print(f"[occ_index] scanning root={str(cfg.plantae_key)}")
        n_obs = build_occ_index(
            str(cfg.plantae_key),
            cfg.data_root,
            cfg.occurrence_parquet_filename,
            occ_index_path,
            min_date=cfg.temporal_min_date,
            skip_if_cols=skip_col_groups if skip_col_groups else None,
        )
        print(f"[occ_index] {n_obs} observations")

        if n_obs == 0:
            print("[done] no observations to enrich")
            return

        for layer in active_layers:
            if layer.derived or stop.is_set():
                continue
            _run_layer(layer, occ_index_path, cfg, stop)

    finally:
        if occ_index_path.exists():
            occ_index_path.unlink()
        if CLEAR_CACHE:
            print(f"[cleanup] clearing cache {cfg.temporal_cache_dir}")
            _cleanup_cache(cfg.temporal_cache_dir)
        else:
            print(f"[cleanup] cache preserved (CLEAR_CACHE=0): {cfg.temporal_cache_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

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

The .om chunk cache under cfg.temporal_cache_dir is cleared only after a run
finishes cleanly. If the run is interrupted (Ctrl-C, SIGTERM, crash) the cache
is left in place as a warm cache for the rerun.
"""
from __future__ import annotations

import gc
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from config.config import load_config
from util.temporal import (
    _PREFETCH_WORKERS,
    TailBuffer,
    TemporalLayer,
    _chunk_entry_for_time,
    _download_layer_chunk,
    build_chunk_index,
    build_per_layer_occ_indices,
    iter_occ_index_batches,
    load_temporal_layers,
    map_to_worklist,
    process_chunk,
    process_chunk_mode,
    process_chunk_vpd,
    process_chunk_wind,
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


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


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

    for occ_batch in iter_occ_index_batches(occ_index_path, _BATCH_ROWS):
        batch_num += 1

        worklist = map_to_worklist(occ_batch, chunk_index, layer.grid_mode, layer.grid_step, accumulated=layer.accumulated)
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

        # Prefetch primary + secondary sources for dense chunks in parallel.
        # Secondary vars (e.g. cloud_cover for weather_code_simple) have different
        # chunk boundaries, so we compute their correct chunk entries here using the
        # same _chunk_entry_for_time logic process_chunk_mode uses — not the primary
        # chunk entry, which would download the wrong file.
        if dense_chunks:
            t_dl = time.monotonic()
            prefetch_tasks: list[tuple[object, str]] = []
            for e in dense_chunks:
                prefetch_tasks.append((e, primary_var))
                if secondary_indices:
                    pfs_ts = e.start - e.file_offset * chunk_index.resolution
                    for var in layer.sources[1:]:
                        if var in secondary_indices:
                            sec_entry, _ = _chunk_entry_for_time(secondary_indices[var], pfs_ts)
                            if sec_entry is not None:
                                prefetch_tasks.append((sec_entry, var))
            seen: set[tuple[int, str]] = set()
            unique_tasks = []
            for e, v in prefetch_tasks:
                key = (e.chunk_num, v)
                if key not in seen:
                    seen.add(key)
                    unique_tasks.append((e, v))
            with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as dl_pool:
                futs = [dl_pool.submit(_download_layer_chunk, e, layer.model, [v], cfg.temporal_cache_dir) for e, v in unique_tasks]
                for fut in futs:
                    fut.result()
            print(f"[prefetch] {layer.id} batch={batch_num}: downloads done ({time.monotonic() - t_dl:.1f}s)")

        batch_rows_done = 0
        tail_buffer: TailBuffer = {}
        batch_updates: dict[str, dict[str, list]] = {}

        for chunk_entry in chunks_this_batch:
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
                elif layer.id in ("wind_speed_10m", "wind_direction_10m"):
                    updates, tail_buffer = process_chunk_wind(
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
                        source_accumulated=layer.source_accumulated,
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

        if batch_updates:
            write_back(batch_updates)

        gc.collect()
        pa.default_memory_pool().release_unused()

    if total_rows_done == 0:
        print(f"[skip] {layer.id}: no observations mapped to any chunk")
    else:
        print(f"[done] {layer.id} rows={total_rows_done} elapsed={time.monotonic() - t_start:.1f}s")


# ---------------------------------------------------------------------------
# Background prefetch
# ---------------------------------------------------------------------------
#
# Each layer is processed as a single pass: every dense chunk is downloaded in
# full up front, then processing runs. That download phase is pure dead time on
# the critical path (~20-45% of each layer's wall time in practice).
#
# The prefetcher below runs on its own thread from the moment the occurrence
# indices are built. It walks the layers in processing order, works out which
# chunks each one will pull in full (mirroring the per-batch logic in
# _run_layer), and downloads them ahead of time. Downloads land atomically via
# _download_chunk's .tmp+rename, and per-file locks stop the main thread and the
# prefetcher racing on the same file, so by the time _run_layer asks for a chunk
# it is usually already on disk and its own prefetch call returns immediately.
# Sparse chunks (HTTP range requests) never touch the cache dir and are skipped
# here.


def _plan_layer_downloads(layer: TemporalLayer, occ_index_path: Path, cfg):
    """Yield (chunk_entry, model, [var]) for every dense chunk `layer` will need.

    Scans the whole occurrence index for the layer up front. Mirrors the dense
    vs. sparse split and secondary-source chunk resolution in _run_layer.
    """
    chunk_var = layer.sources[0] if layer.sources else layer.id
    chunk_index = build_chunk_index(layer.model, chunk_var, min_date=cfg.temporal_min_date)
    chunks_eligible = list(chunk_index.ranges)

    secondary_indices: dict[str, object] = {}
    for src_var in layer.sources[1:]:
        try:
            secondary_indices[src_var] = build_chunk_index(
                layer.model, src_var, min_date=cfg.temporal_min_date
            )
        except Exception:
            pass

    seen: set[tuple[int, str]] = set()
    for occ_batch in iter_occ_index_batches(occ_index_path, _BATCH_ROWS):
        worklist = map_to_worklist(
            occ_batch, chunk_index, layer.grid_mode, layer.grid_step,
            accumulated=layer.accumulated,
        )
        if worklist.num_rows == 0:
            continue
        for entry in chunks_eligible:
            sl = worklist.filter(pc.equal(worklist["chunk_num"], entry.chunk_num))
            if sl.num_rows <= _RANGE_REQUEST_THRESHOLD:
                continue  # sparse (or absent) — served by range requests
            key = (entry.chunk_num, chunk_var)
            if key not in seen:
                seen.add(key)
                yield entry, layer.model, [chunk_var]
            if secondary_indices:
                pfs_ts = entry.start - entry.file_offset * chunk_index.resolution
                for var in layer.sources[1:]:
                    sec_idx = secondary_indices.get(var)
                    if sec_idx is None:
                        continue
                    sec_entry, _ = _chunk_entry_for_time(sec_idx, pfs_ts)
                    if sec_entry is None:
                        continue
                    sec_key = (sec_entry.chunk_num, var)
                    if sec_key not in seen:
                        seen.add(sec_key)
                        yield sec_entry, layer.model, [var]


def _start_prefetcher(
    active_layers: list[TemporalLayer],
    layer_index_paths: dict[str, Path],
    counts: dict[str, int],
    cfg,
    prefetch_stop: threading.Event,
) -> threading.Thread:
    def _run() -> None:
        submitted = 0
        failures = 0
        try:
            with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as pool:
                futs = []
                for layer in active_layers:
                    if layer.derived or prefetch_stop.is_set():
                        continue
                    if counts.get(layer.id, 0) == 0:
                        continue
                    idx_path = layer_index_paths.get(layer.id)
                    if idx_path is None or not idx_path.exists():
                        continue
                    try:
                        for entry, model, dl_vars in _plan_layer_downloads(layer, idx_path, cfg):
                            if prefetch_stop.is_set():
                                break
                            futs.append(pool.submit(
                                _download_layer_chunk, entry, model, dl_vars,
                                cfg.temporal_cache_dir,
                            ))
                            submitted += 1
                    except Exception as exc:
                        print(f"[prefetch-bg] {layer.id}: planning failed — {exc}")
                for fut in futs:
                    if prefetch_stop.is_set():
                        break
                    try:
                        fut.result()
                    except Exception as exc:
                        failures += 1
                        print(f"[prefetch-bg] download failed — {exc}")
                if prefetch_stop.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
            print(f"[prefetch-bg] done: {submitted} chunk files, {failures} failures")
        except Exception:
            traceback.print_exc()

    thread = threading.Thread(target=_run, name="temporal-prefetch", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # No custom signal handling: Ctrl-C / SIGTERM propagate as usual and the
    # process exits immediately. Carry-forward means almost every value is
    # already enriched on a rerun, so there's nothing worth stopping cleanly for.
    cfg = load_config("global")

    Path(cfg.temporal_cache_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.temporal_cache_dir)

    all_layers = load_temporal_layers(CATALOG_PATH)
    active_layers = _filter_layers(all_layers, VARS_TO_ENRICH)
    print(f"[init] active layers: {[layer.id for layer in active_layers]}")

    non_derived = [layer for layer in active_layers if not layer.derived]
    layer_index_paths: dict[str, Path] = {
        layer.id: cache_dir / f"occ_index_{layer.id}.parquet"
        for layer in non_derived
    }

    prefetch_stop = threading.Event()
    prefetch_thread: threading.Thread | None = None
    try:
        print(f"[occ_index] scanning roots={list(cfg.taxonomy_roots)}")
        counts = build_per_layer_occ_indices(
            list(cfg.taxonomy_roots),
            cfg.data_root,
            cfg.occurrence_parquet_filename,
            layers=non_derived,
            index_paths=layer_index_paths,
            min_date=cfg.temporal_min_date,
        )

        if all(n == 0 for n in counts.values()):
            print("[done] no observations to enrich")
        else:
            prefetch_thread = _start_prefetcher(
                active_layers, layer_index_paths, counts, cfg, prefetch_stop,
            )
            for layer in active_layers:
                if layer.derived:
                    continue
                n = counts.get(layer.id, 0)
                if n == 0:
                    print(f"[skip] {layer.id}: no observations to enrich")
                    continue
                print(f"[occ_index] {layer.id}: {n} observations")
                _run_layer(layer, layer_index_paths[layer.id], cfg)
    finally:
        # Always tell the prefetcher to stop; it's a daemon thread so the wait
        # is short. Nothing else runs here — cache cleanup below is reached only
        # on a clean, complete run, never on Ctrl-C / SIGTERM / a crash.
        prefetch_stop.set()
        if prefetch_thread is not None:
            prefetch_thread.join(timeout=10)

    for idx_path in layer_index_paths.values():
        try:
            if idx_path.exists():
                idx_path.unlink()
        except Exception:
            pass
    if CLEAR_CACHE:
        print(f"[cleanup] clearing cache {cfg.temporal_cache_dir}")
        _cleanup_cache(cfg.temporal_cache_dir)
    else:
        print(f"[cleanup] cache preserved (CLEAR_CACHE=0): {cfg.temporal_cache_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

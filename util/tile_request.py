from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Callable

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from util import tiles

LOGGER = logging.getLogger("uvicorn.error")


async def run_tile_render_with_cancellation(
    request: Request,
    render_fn: Callable[..., Any],
    /,
    **kwargs: Any,
) -> Any:
    cancel_event = threading.Event()

    async def watch_disconnect() -> None:
        while not cancel_event.is_set():
            if await request.is_disconnected():
                cancel_event.set()
                return
            await asyncio.sleep(0.1)

    def cancel_check() -> None:
        if cancel_event.is_set():
            raise tiles.TileRenderCancelled()

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await run_in_threadpool(render_fn, cancel_check=cancel_check, **kwargs)
    finally:
        cancel_event.set()
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


def log_tile_cancellation(route: str, reason: str, **fields: object) -> None:
    details = " ".join(f"{key}={value!r}" for key, value in fields.items() if value is not None)
    if details:
        LOGGER.warning("[tile-cancelled] route=%s reason=%s %s", route, reason, details)
    else:
        LOGGER.warning("[tile-cancelled] route=%s reason=%s", route, reason)


@asynccontextmanager
async def acquire_tile_render_slot(
    semaphore: asyncio.Semaphore,
    request: Request,
    *,
    route: str,
    **fields: object,
) -> AsyncIterator[None]:
    while True:
        if await request.is_disconnected():
            log_tile_cancellation(route, "queued_disconnect", **fields)
            raise tiles.TileRenderCancelled()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except TimeoutError:
            continue
    try:
        yield
    finally:
        semaphore.release()


async def render_species_deep_zoom_tile(
    request: Request,
    semaphore: asyncio.Semaphore,
    *,
    taxon_id: int,
    z: int,
    x: int,
    y: int,
    tile_size: int,
    max_native_zoom: int,
    parent_tile_max_size: int,
    model_id: str | None,
    reproject: bool,
    forecast_hours: int,
    apply_phenology: bool,
    phenology_only: bool,
) -> bytes | None:
    zoom_diff = z - max_native_zoom
    scale = 2**zoom_diff
    parent_x = x // scale
    parent_y = y // scale
    subtile_x = x % scale
    subtile_y = y % scale
    parent_tile_size = min(tile_size * scale, parent_tile_max_size)
    fields = {
        "taxon_id": taxon_id,
        "z": z,
        "x": x,
        "y": y,
        "model_id": model_id,
    }

    async with acquire_tile_render_slot(semaphore, request, route="species", **fields):
        if await request.is_disconnected():
            log_tile_cancellation("species", "pre_render_disconnect", **fields)
            return None
        try:
            parent_payload = await run_tile_render_with_cancellation(
                request,
                tiles.render_model_tile_bytes,
                taxon_id=taxon_id,
                z=max_native_zoom,
                x=parent_x,
                y=parent_y,
                model_id=model_id,
                tile_size=parent_tile_size,
                reproject=reproject,
                forecast_hours=forecast_hours,
                apply_phenology=apply_phenology,
                phenology_only=phenology_only,
            )
        except tiles.TileRenderCancelled:
            log_tile_cancellation("species", "during_render", **fields)
            return None

        if await request.is_disconnected():
            log_tile_cancellation("species", "post_render_disconnect", **fields)
            return None

        return tiles.crop_subtile_png(
            parent_payload,
            parent_tile_size=parent_tile_size,
            scale=scale,
            subtile_x=subtile_x,
            subtile_y=subtile_y,
            tile_size=tile_size,
        )

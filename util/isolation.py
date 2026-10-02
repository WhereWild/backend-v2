# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import multiprocessing
from collections.abc import Callable


def run_isolated(target: Callable[..., object], *args: object) -> None:
    """Run target(*args) in a fresh spawned process and wait for it.

    For pipeline phases whose memory must be gone before the next phase
    starts. Python frees large arrays, rasterio/GDAL buffers and caches, but
    glibc's allocator mostly doesn't return that memory to the OS, so a
    process that ran a heavy pass keeps tens of GB of RSS for the rest of its
    life. A phase run here gives all of it back when its process exits.

    Spawning the next phase isn't enough on its own: the parent then sits in
    join() still holding everything. Both the heavy pass and whatever follows
    it need to run this way, so the parent stays small.

    spawn (not fork) gives each child a genuinely fresh interpreter and heap.
    It also re-imports the target's module, so any runtime override of a
    module global (tests patch them) is lost in the child: pass paths and
    settings explicitly as args.

    Raises RuntimeError if the child exits non-zero, including when it's
    OOM-killed (exit code -9).
    """
    proc = multiprocessing.get_context("spawn").Process(target=target, args=args)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"{target.__name__} subprocess failed (exit code {proc.exitcode})")

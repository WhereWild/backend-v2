from __future__ import annotations

from typing import Callable, TypeAlias


CancelCheck: TypeAlias = Callable[[], None]


class RequestCancelledError(RuntimeError):
    """Raised when request work should stop because the client disconnected."""


def noop_cancel_check() -> None:
    """Default cancellation callback used when no request-scoped checker is available."""

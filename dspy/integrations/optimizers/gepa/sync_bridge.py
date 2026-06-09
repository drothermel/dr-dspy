"""Sync bridge for the external GEPA adapter protocol."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

T = TypeVar("T")


def run_gepa_sync(coro: Coroutine[object, object, T]) -> T:
    """Run an async GEPA adapter coroutine from sync entrypoints.

    Raises ``RuntimeError`` when called from a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "GEPA adapter sync methods cannot be called from a running event loop. Use the async methods directly instead."
    )

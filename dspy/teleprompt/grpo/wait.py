from __future__ import annotations

import asyncio
from collections.abc import Callable  # noqa: TC003 — predicate typing at runtime


async def wait_until(predicate: Callable[[], bool], poll_interval: float = 1.0) -> None:
    while not predicate():  # noqa: ASYNC110 — polls external finetune job status
        await asyncio.sleep(poll_interval)

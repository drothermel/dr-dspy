import asyncio

import pytest

from dspy.integrations.optimizers.gepa.sync_bridge import run_gepa_sync


async def _returns_value() -> str:
    return "ok"


def test_run_gepa_sync_runs_coroutine_when_no_loop():
    assert run_gepa_sync(_returns_value()) == "ok"


def test_run_gepa_sync_raises_inside_running_loop():
    async def _inside_loop() -> None:
        coro = _returns_value()
        try:
            with pytest.raises(RuntimeError, match="running event loop"):
                run_gepa_sync(coro)
        finally:
            coro.close()

    asyncio.run(_inside_loop())

from __future__ import annotations

from typing import Any


def _run_sub_lm_async(coro):
    import asyncio
    import contextvars

    ctx = contextvars.copy_context()

    def _run_in_context() -> Any:
        return ctx.run(asyncio.run, coro)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_in_context()
    raise RuntimeError("RLM sub-LM queries cannot run inside an active asyncio loop from sync REPL tools.")

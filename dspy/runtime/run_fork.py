from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext


def fork_worker_run(run: RunContext, **overrides: Any) -> RunContext:
    """Return an isolated run for concurrent or batch execution."""
    return run.fork(optimization_trace=[], call_log=[], **overrides)

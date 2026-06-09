from __future__ import annotations

from typing import Any

from dspy.primitives import Example
from dspy.primitives.batch_result import BatchFailure, BatchResult
from dspy.runtime.async_parallel import BoundedRunStats, resolve_max_concurrency, resolve_max_errors, run_bounded
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.runtime.run_fork import fork_worker_run


class Parallel:
    def __init__(
        self,
        run: RunContext | None = None,
        max_concurrency: int | None = None,
        max_errors: int | None = None,
        access_examples: bool = True,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
        timeout: int = 120,
    ) -> None:
        self.run = run
        exec_cfg = run.execution if run is not None else None
        self.max_concurrency = max_concurrency or (exec_cfg.max_concurrency if exec_cfg is not None else None)
        if max_errors is None and exec_cfg is not None:
            self.max_errors = exec_cfg.max_errors
        else:
            self.max_errors = max_errors
        self.access_examples = access_examples
        if provide_traceback is None and exec_cfg is not None:
            self.provide_traceback = exec_cfg.provide_traceback
        else:
            self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar
        self.timeout = timeout
        self._last_stats = BoundedRunStats()
        self._active_run: RunContext | None = None

    async def _run_pair(self, pair: tuple[Any, Any]) -> Any:
        module, example = pair
        run = self._active_run
        assert run is not None
        item_run = fork_worker_run(run)
        if isinstance(example, Example):
            if self.access_examples:
                return await module(**example.as_inputs(), run=item_run)
            return await module(example, run=item_run)
        if isinstance(example, dict):
            return await module(**example, run=item_run)
        if isinstance(example, list) and module.__class__.__name__ == "Parallel":
            return await module(example, run=item_run)
        if isinstance(example, tuple):
            return await module(*example, run=item_run)
        raise ValueError(
            f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
        )

    async def __call__(
        self,
        exec_pairs: list[tuple[Any, Any]],
        run: RunContext | None = None,
        max_concurrency: int | None = None,
    ) -> BatchResult:
        run = resolve_run(run=run, bound_run=self.run)
        self._active_run = run
        concurrency = resolve_max_concurrency(
            explicit=max_concurrency,
            configured=self.max_concurrency,
            run=run,
        )
        max_errors = resolve_max_errors(self.max_errors, run)
        provide_traceback = (
            self.provide_traceback if self.provide_traceback is not None else run.execution.provide_traceback
        )
        try:
            results, stats = await run_bounded(
                items=exec_pairs,
                fn=self._run_pair,
                max_concurrency=concurrency,
                max_errors=max_errors,
                provide_traceback=provide_traceback,
                disable_progress_bar=self.disable_progress_bar,
                timeout=float(self.timeout) if self.timeout > 0 else None,
            )
        finally:
            self._active_run = None
        self._last_stats = stats
        failures: list[BatchFailure] = []
        for failed_idx in stats.failed_indices:
            if failed_idx < len(exec_pairs):
                _, original_example = exec_pairs[failed_idx]
                exception = stats.exceptions_map.get(failed_idx)
                if exception is not None:
                    failures.append(BatchFailure(input=original_example, exception=exception))
        return BatchResult(results=tuple(results), failures=tuple(failures))

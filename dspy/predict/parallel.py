from __future__ import annotations

from typing import Any

from dspy.primitives.example import Example
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.utils.async_parallel import BoundedRunStats, run_bounded


class Parallel:
    def __init__(
        self,
        run: RunContext | None = None,
        num_threads: int | None = None,
        max_concurrency: int | None = None,
        max_errors: int | None = None,
        access_examples: bool = True,
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
        timeout: int = 120,
        straggler_limit: int = 3,
    ) -> None:
        self.run = run
        exec_cfg = run.execution if run is not None else None
        concurrency = max_concurrency if max_concurrency is not None else num_threads
        self.max_concurrency = concurrency or (exec_cfg.num_threads if exec_cfg is not None else None)
        self.num_threads = self.max_concurrency
        if max_errors is None and exec_cfg is not None:
            self.max_errors = exec_cfg.max_errors
        else:
            self.max_errors = max_errors
        self.access_examples = access_examples
        self.return_failed_examples = return_failed_examples
        if provide_traceback is None and exec_cfg is not None:
            self.provide_traceback = exec_cfg.provide_traceback
        else:
            self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar
        self.timeout = timeout
        self.straggler_limit = straggler_limit
        self.failed_examples: list[Any] = []
        self.exceptions: list[BaseException] = []
        self._last_stats = BoundedRunStats()
        self._active_run: RunContext | None = None

    async def _run_pair(self, pair: tuple[Any, Any]) -> Any:
        module, example = pair
        run = self._active_run
        assert run is not None
        if isinstance(example, Example):
            if self.access_examples:
                return await module(**example.inputs(), run=run)
            return await module(example, run=run)
        if isinstance(example, dict):
            return await module(**example, run=run)
        if isinstance(example, list) and module.__class__.__name__ == "Parallel":
            return await module(example, run=run)
        if isinstance(example, tuple):
            return await module(*example, run=run)
        raise ValueError(
            f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
        )

    async def __call__(
        self,
        exec_pairs: list[tuple[Any, Any]],
        run: RunContext | None = None,
        num_threads: int | None = None,
        max_concurrency: int | None = None,
    ) -> list[Any] | tuple[list[Any], list[Any], list[BaseException]]:
        run = resolve_run(run=run, bound_run=self.run)
        self._active_run = run
        concurrency = max_concurrency if max_concurrency is not None else num_threads
        concurrency = concurrency or self.max_concurrency or run.execution.num_threads
        max_errors = self.max_errors if self.max_errors is not None else run.execution.max_errors
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
            )
        finally:
            self._active_run = None
        self._last_stats = stats
        if self.return_failed_examples:
            self.failed_examples = []
            self.exceptions = []
            for failed_idx in stats.failed_indices:
                if failed_idx < len(exec_pairs):
                    _, original_example = exec_pairs[failed_idx]
                    self.failed_examples.append(original_example)
                    if exception := stats.exceptions_map.get(failed_idx):
                        self.exceptions.append(exception)
            return (results, self.failed_examples, self.exceptions)
        return results

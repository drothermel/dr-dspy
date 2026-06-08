from __future__ import annotations

from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.primitives.example import Example
from dspy.utils.async_parallel import BoundedRunStats, run_bounded


class Parallel:
    def __init__(
        self,
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
        concurrency = max_concurrency if max_concurrency is not None else num_threads
        self.max_concurrency = concurrency or settings.num_threads
        self.num_threads = self.max_concurrency
        self.max_errors = settings.max_errors if max_errors is None else max_errors
        self.access_examples = access_examples
        self.return_failed_examples = return_failed_examples
        self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar
        self.timeout = timeout
        self.straggler_limit = straggler_limit
        self.failed_examples: list[Any] = []
        self.exceptions: list[BaseException] = []
        self._last_stats = BoundedRunStats()

    async def _run_pair(self, pair: tuple[Any, Any]) -> Any:
        module, example = pair
        if isinstance(example, Example):
            if self.access_examples:
                return await module(**example.inputs())
            return await module(example)
        if isinstance(example, dict):
            return await module(**example)
        if isinstance(example, list) and module.__class__.__name__ == "Parallel":
            return await module(example)
        if isinstance(example, tuple):
            return await module(*example)
        raise ValueError(
            f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
        )

    async def __call__(
        self, exec_pairs: list[tuple[Any, Any]], num_threads: int | None = None, max_concurrency: int | None = None
    ) -> list[Any] | tuple[list[Any], list[Any], list[BaseException]]:
        concurrency = max_concurrency if max_concurrency is not None else num_threads
        concurrency = concurrency or self.max_concurrency
        results, stats = await run_bounded(
            items=exec_pairs,
            fn=self._run_pair,
            max_concurrency=concurrency,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            disable_progress_bar=self.disable_progress_bar,
        )
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

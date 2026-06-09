from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, TypeVar

import tqdm

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
logger = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")


class BoundedRunStats:
    def __init__(self) -> None:
        self.failed_indices: list[int] = []
        self.exceptions_map: dict[int, BaseException] = {}


async def run_bounded(
    *,
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    max_concurrency: int,
    max_errors: int | None = None,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
    compare_results: bool = False,
) -> tuple[list[R | None], BoundedRunStats]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")
    stats = BoundedRunStats()
    results: list[R | None] = [None] * len(items)
    error_count = 0
    cancel = asyncio.Event()
    lock = asyncio.Lock()
    pbar = tqdm.tqdm(total=len(items), dynamic_ncols=True, disable=disable_progress_bar or len(items) == 0)

    async def run_indexed(index: int, item: T) -> None:
        nonlocal error_count
        if cancel.is_set():
            return
        try:
            outcome = await fn(item)
        except Exception as exc:
            if provide_traceback:
                logger.exception("Error for %r: %s", item, exc)
            else:
                logger.exception("Error for %r: %s. Set `provide_traceback=True` for traceback.", item, exc)
            async with lock:
                stats.failed_indices.append(index)
                stats.exceptions_map[index] = exc
                error_count += 1
                if max_errors is not None and error_count >= max_errors:
                    cancel.set()
            return
        results[index] = outcome
        if compare_results:
            completed = [r for r in results if r is not None]
            total_score = sum(r[-1] for r in completed if isinstance(r, tuple))
            pct = round(100 * total_score / len(items), 1) if items else 0
            pbar.set_description(f"Average Metric: {total_score:.2f} / {len(items)} ({pct}%)")
        else:
            completed = len([r for r in results if r is not None])
            pbar.set_description(f"Processed {completed} / {len(items)} examples")
        pbar.update()

    try:
        sem = asyncio.Semaphore(max_concurrency)

        async def run_one(index: int, item: T) -> None:
            if cancel.is_set():
                return
            async with sem:
                if cancel.is_set():
                    return
                await run_indexed(index=index, item=item)

        await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
    finally:
        pbar.close()
    if cancel.is_set():
        raise RuntimeError("Execution cancelled due to errors or interruption.")
    return (results, stats)

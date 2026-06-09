import asyncio
import logging

import pytest

from dspy.runtime.async_parallel import BoundedRunAbortedError, run_bounded


@pytest.mark.asyncio
async def test_run_bounded_omits_traceback_when_disabled(caplog):
    async def fail(_item: int) -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR), pytest.raises(BoundedRunAbortedError, match="Failed indices"):
        await run_bounded(
            items=[1],
            fn=fail,
            max_concurrency=1,
            max_errors=1,
            provide_traceback=False,
        )

    assert "boom" in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_run_bounded_includes_traceback_when_enabled(caplog):
    async def fail(_item: int) -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR), pytest.raises(BoundedRunAbortedError, match="Failed indices"):
        await run_bounded(
            items=[1],
            fn=fail,
            max_concurrency=1,
            max_errors=1,
            provide_traceback=True,
        )

    assert "Traceback" in caplog.text


@pytest.mark.asyncio
async def test_run_bounded_aborted_error_includes_stats():
    async def fail(_item: int) -> None:
        raise ValueError("boom")

    with pytest.raises(BoundedRunAbortedError) as exc_info:
        await run_bounded(
            items=[1],
            fn=fail,
            max_concurrency=1,
            max_errors=1,
            provide_traceback=False,
        )

    assert exc_info.value.stats.failed_indices == [0]
    assert isinstance(exc_info.value.stats.exceptions_map[0], ValueError)


@pytest.mark.asyncio
async def test_run_bounded_progress_hook():
    async def succeed(item: int) -> tuple[int, bool]:
        return (item, item > 1)

    def metric_progress(results, total):
        completed = [r for r in results if r is not None]
        total_score = sum(r[-1] for r in completed if isinstance(r, tuple))
        return f"score={total_score}/{total}"

    results, _stats = await run_bounded(
        items=[1, 2, 3],
        fn=succeed,
        max_concurrency=1,
        progress_hook=metric_progress,
        disable_progress_bar=True,
    )
    assert results == [(1, False), (2, True), (3, True)]


@pytest.mark.asyncio
async def test_run_bounded_timeout_records_failure():
    async def slow(_item: int) -> int:
        await asyncio.sleep(0.2)
        return _item

    results, stats = await run_bounded(
        items=[1],
        fn=slow,
        max_concurrency=1,
        timeout=0.05,
        disable_progress_bar=True,
    )
    assert results == [None]
    assert stats.failed_indices == [0]
    assert isinstance(stats.exceptions_map[0], TimeoutError)

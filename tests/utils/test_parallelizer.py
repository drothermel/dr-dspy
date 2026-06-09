import asyncio

import pytest

from dspy.runtime.async_parallel import BoundedRunAbortedError, run_bounded


async def _run_bounded(items, fn, **kwargs):
    return await run_bounded(items=items, fn=fn, **kwargs)


def test_worker_independence(make_run):

    async def task(item):
        return item * 2

    data = [1, 2, 3, 4, 5]
    results, _stats = asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=3))
    assert results == [2, 4, 6, 8, 10]


def test_parallel_execution_speed(make_run):
    import time

    async def task(item):
        await asyncio.sleep(0.1)
        return item

    data = [1, 2, 3, 4, 5]
    start_time = time.time()
    asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=5))
    end_time = time.time()
    assert end_time - start_time < len(data)


def test_max_errors_handling(make_run):

    async def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    with pytest.raises(BoundedRunAbortedError, match="Failed indices"):
        asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=3, max_errors=1))


def test_max_errors_not_met(make_run):

    async def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    results, _stats = asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=3, max_errors=2))
    assert results == [1, 2, None, 4, 5]


def test_run_bounded_tracks_failed_indices_and_exceptions(make_run):

    async def task(item):
        if item == 3:
            raise ValueError("test error for 3")
        if item == 5:
            raise RuntimeError("test error for 5")
        return item

    data = [1, 2, 3, 4, 5]
    results, stats = asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=3, max_errors=3))
    assert results == [1, 2, None, 4, None]
    assert sorted(stats.failed_indices) == [2, 4]
    assert len(stats.exceptions_map) == 2
    assert isinstance(stats.exceptions_map[2], ValueError)
    assert str(stats.exceptions_map[2]) == "test error for 3"
    assert isinstance(stats.exceptions_map[4], RuntimeError)
    assert str(stats.exceptions_map[4]) == "test error for 5"


def test_sequential_execution(make_run):

    async def task(item):
        return item * 2

    data = [1, 2, 3, 4, 5]
    results, _stats = asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=1))
    assert results == [2, 4, 6, 8, 10]


def test_sequential_max_errors_not_met(make_run):

    async def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    results, _stats = asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=1, max_errors=2))
    assert results == [1, 2, None, 4, 5]


def test_sequential_max_errors_exceeded(make_run):

    async def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    with pytest.raises(BoundedRunAbortedError, match="Failed indices"):
        asyncio.run(_run_bounded(items=data, fn=task, max_concurrency=1, max_errors=1))


def test_compare_results():

    async def task(item):
        return (item, item > 2)

    data = [1, 2, 3, 4, 5]
    results, _stats = asyncio.run(
        _run_bounded(items=data, fn=task, max_concurrency=1, compare_results=True, disable_progress_bar=True)
    )
    assert results == [(1, False), (2, False), (3, True), (4, True), (5, True)]

import logging

import pytest

from dspy.utils.async_parallel import run_bounded


@pytest.mark.asyncio
async def test_run_bounded_omits_traceback_when_disabled(caplog):
    async def fail(_item: int) -> None:
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="cancelled"):
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

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="cancelled"):
        await run_bounded(
            items=[1],
            fn=fail,
            max_concurrency=1,
            max_errors=1,
            provide_traceback=True,
        )

    assert "Traceback" in caplog.text

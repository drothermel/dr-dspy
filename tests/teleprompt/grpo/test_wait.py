import asyncio
import sys

from dspy.teleprompt.grpo.wait import wait_until


def test_wait_until_completes_without_recursion():
    state = {"count": 0}

    def predicate() -> bool:
        return state["count"] >= 1

    async def _run() -> None:
        async def _tick() -> None:
            await asyncio.sleep(0.01)
            state["count"] += 1

        tick_task = asyncio.create_task(_tick())
        await wait_until(predicate, poll_interval=0.005)
        await tick_task

    before_depth = sys.getrecursionlimit()
    asyncio.run(_run())
    assert state["count"] == 1
    assert sys.getrecursionlimit() == before_depth

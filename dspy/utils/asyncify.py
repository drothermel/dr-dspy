import functools
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import anyio
from anyio import CapacityLimiter

from dspy.dsp.utils.settings import settings
from dspy.dsp.utils.utils import dotdict

if TYPE_CHECKING:
    from dspy.primitives.module import Module

_limiter = None


def get_async_max_workers() -> int:

    return settings.async_max_workers


def get_limiter() -> CapacityLimiter:
    async_max_workers = get_async_max_workers()

    global _limiter
    if _limiter is None:
        _limiter = CapacityLimiter(async_max_workers)
    elif _limiter.total_tokens != async_max_workers:
        _limiter.total_tokens = async_max_workers

    return _limiter


def asyncify(program: "Module") -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
    program in parallel with another task (e.g., another DSPy program).

    This implementation propagates the current thread's configuration context to the worker thread.

    Args:
        program: The DSPy program to be wrapped for asynchronous execution.

    Returns:
        An async function: An async function that, when awaited, runs the program in a worker thread.
            The current thread's configuration context is inherited for each call.
    """

    async def async_program(*args: Any, **kwargs: Any) -> Any:
        # Capture the current overrides at call-time.
        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.get().copy()

        def wrapped_program(*a: Any, **kw: Any) -> Any:
            from dspy.dsp.utils.settings import thread_local_overrides

            original_overrides = thread_local_overrides.get()
            token = thread_local_overrides.set(dotdict({**original_overrides, **parent_overrides.copy()}))
            try:
                return program(*a, **kw)
            finally:
                thread_local_overrides.reset(token)

        partial_f = functools.partial(wrapped_program, *args, **kwargs)
        return await anyio.to_thread.run_sync(partial_f, abandon_on_cancel=True, limiter=get_limiter())  # ty:ignore[unresolved-attribute]

    return async_program

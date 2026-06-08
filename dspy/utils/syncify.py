import asyncio
from collections.abc import Awaitable
from types import MethodType
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from dspy.primitives.module import Module

T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In notebooks/Jupyter, patch nested event loops so run_until_complete can drive the coroutine from sync code.
        import nest_asyncio  # ty:ignore[unresolved-import]

        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)
    return asyncio.run(coro)  # ty:ignore[invalid-argument-type]


def syncify(program: "Module", in_place: bool = True) -> "Module":
    """Convert an async DSPy module to a sync program.

    There are two modes of this function:

    - `in_place=True` (recommended): Modify the module in place. But this may not work if you already have a `forward`
        method which does different things from `aforward`.
    - `in_place=False`: Return a wrapper module. This changes the module's architecture, but it's more robust.

    Args:
        program: The async program to convert, must have an `aforward` method implemented.
        in_place: If True, modify the module in place. Otherwise, return a wrapper module.

    Returns:
        The sync program, which has a `forward` method that can be called from a synchronous context.
    """
    if in_place:

        def forward(self: "Module", *args: object, **kwargs: object) -> object:
            return run_async(self.aforward(*args, **kwargs))

        # Create the `forward` method in place.
        program.forward = MethodType(forward, program)  # ty:ignore[unresolved-attribute]
        return program
    from dspy.primitives.module import Module

    class SyncWrapper(Module):
        def __init__(self, program: "Module") -> None:
            self.program = program

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return run_async(self.program.aforward(*args, **kwargs))

    return SyncWrapper(program)

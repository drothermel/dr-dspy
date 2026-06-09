from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

F = TypeVar("F", type, Callable[..., Any])


def _mark_experimental(api: object, version: str | None) -> None:
    marked = cast("Any", api)
    marked.__dspy_experimental__ = True
    marked.__dspy_experimental_version__ = version


@overload
def experimental(f: F, version: str | None = None) -> F: ...


@overload
def experimental(f: None = None, version: str | None = None) -> Callable[[F], F]: ...


def experimental(f: F | None = None, version: str | None = None) -> F | Callable[[F], F]:
    if f is not None:
        _mark_experimental(f, version)
        return f

    def decorator(fn: F) -> F:
        _mark_experimental(fn, version)
        return fn

    return decorator


def is_experimental(api: object) -> bool:
    return bool(getattr(api, "__dspy_experimental__", False))

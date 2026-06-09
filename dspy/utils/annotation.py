from typing import Any, Callable, ParamSpec, TypeVar, cast, overload

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
_Decorated = type[T] | Callable[P, R]


@overload
def experimental(f: _Decorated, version: str | None = None) -> _Decorated: ...


@overload
def experimental(f: None = None, version: str | None = None) -> Callable[[_Decorated], _Decorated]: ...


def experimental(f: type[T] | Callable[P, R] | None = None, version: str | None = None) -> Any:
    if f:
        return _experimental(f, version)

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        return _experimental(f, version)

    return decorator


def _experimental(api: Callable[P, R], version: str | None = None) -> Callable[P, R]:
    api_any = cast("Any", api)
    api_any.__dspy_experimental__ = True
    api_any.__dspy_experimental_version__ = version
    return api


def is_experimental(api: object) -> bool:
    return bool(getattr(api, "__dspy_experimental__", False))

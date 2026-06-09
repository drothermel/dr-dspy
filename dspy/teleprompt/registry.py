from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

TeleprompterT = TypeVar("TeleprompterT")

_TELEPROMPTER_PARAMS: dict[type, type[BaseModel]] = {}


def register_teleprompter(*, params: type[BaseModel]) -> Callable[[TeleprompterT], TeleprompterT]:
    def decorator(cls: TeleprompterT) -> TeleprompterT:
        _TELEPROMPTER_PARAMS[cast("type", cls)] = params
        return cls

    return decorator


def compile_params_type(optimizer: type | object) -> type[BaseModel]:
    cls = optimizer if isinstance(optimizer, type) else optimizer.__class__
    if cls not in _TELEPROMPTER_PARAMS:
        raise TypeError(
            f"Optimizer {cls.__name__} is not registered. Apply @register_teleprompter(params=...) to the class."
        )
    return _TELEPROMPTER_PARAMS[cls]


def validate_compile_params(optimizer: object, params: BaseModel) -> None:
    expected = compile_params_type(optimizer)
    if not isinstance(params, expected):
        raise TypeError(
            f"Expected compile params of type {expected.__name__} for {optimizer.__class__.__name__}, "
            f"got {type(params).__name__}"
        )


def registered_teleprompters() -> dict[type, type[BaseModel]]:
    return dict(_TELEPROMPTER_PARAMS)

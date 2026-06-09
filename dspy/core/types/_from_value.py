from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

_MISSING = object()


def config_data(value: Any, *, str_field: str | None = None, bool_field: str | None = None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    if str_field is not None and isinstance(value, str):
        return {str_field: value}
    if bool_field is not None and isinstance(value, bool):
        return {bool_field: value}
    raise TypeError(f"Cannot convert {type(value)!r} to a config object.")

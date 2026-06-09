"""JSON-safe serialization helpers.

Precedence in ``to_jsonable``:
1. ``BaseModel`` instances via ``model_dump(mode="json")``
2. objects with a ``to_dict()`` method
3. other objects with ``model_dump(exclude_none=True)``
4. containers, then ``repr`` for unknown types
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_CIRCULAR_SENTINEL = "<circular>"


def to_jsonable(value: Any, *, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return _CIRCULAR_SENTINEL
    seen.add(value_id)
    try:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            return {str(key): to_jsonable(item, _seen=seen) for key, item in value.items()}
        if isinstance(value, tuple):
            return [to_jsonable(item, _seen=seen) for item in value]
        if isinstance(value, list):
            return [to_jsonable(item, _seen=seen) for item in value]
        return repr(value)
    finally:
        seen.discard(value_id)

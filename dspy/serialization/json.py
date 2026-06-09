"""JSON-safe serialization helpers.

Precedence in ``to_jsonable``:
1. JSON primitives (``None``, ``bool``, ``int``, ``float``, ``str``)
2. Circular references → ``"<circular>"``
3. ``enum.Enum`` members via ``.value``
4. ``BaseModel`` instances via recursive ``model_dump(mode="json")``
5. Objects with ``to_dict()`` (when not a ``BaseModel``) via recursive conversion
6. ``dict`` / ``list`` / ``tuple`` / ``set`` containers (recurse; sets become lists)
7. ``TypeAdapter(type(value)).dump_python(value, mode="json")`` when supported
8. Fallback: ``str(value)``; with ``strict=True``, raise ``TypeError`` instead

Human-readable formatted strings (for example REPL history ``format()`` output) are
intentional exceptions and should not pass through this helper.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, TypeAdapter

_CIRCULAR_SENTINEL = "<circular>"


def to_jsonable(value: Any, *, strict: bool = False, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return _CIRCULAR_SENTINEL
    seen.add(value_id)
    try:
        if isinstance(value, enum.Enum):
            return to_jsonable(value.value, strict=strict, _seen=seen)
        if isinstance(value, BaseModel):
            return to_jsonable(value.model_dump(mode="json"), strict=strict, _seen=seen)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return to_jsonable(value.to_dict(), strict=strict, _seen=seen)
        if isinstance(value, dict):
            return {str(key): to_jsonable(item, strict=strict, _seen=seen) for key, item in value.items()}
        if isinstance(value, tuple):
            return [to_jsonable(item, strict=strict, _seen=seen) for item in value]
        if isinstance(value, list):
            return [to_jsonable(item, strict=strict, _seen=seen) for item in value]
        if isinstance(value, set):
            sorted_items = _sorted_set_items(value)
            return [to_jsonable(item, strict=strict, _seen=seen) for item in sorted_items]
        if hasattr(value, "model_dump"):
            return to_jsonable(value.model_dump(exclude_none=True), strict=strict, _seen=seen)
        try:
            return TypeAdapter(type(value)).dump_python(value, mode="json")
        except Exception:
            pass
        if strict:
            raise TypeError(f"Value of type {type(value).__name__} is not JSON-serializable.")
        return str(value)
    finally:
        seen.discard(value_id)


def _sorted_set_items(value: set[Any]) -> list[Any]:
    try:
        return sorted(value)
    except TypeError:
        return list(value)

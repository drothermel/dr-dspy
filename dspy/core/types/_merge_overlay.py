from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


def _merge_model_overlay(
    left: TModel | None,
    right: TModel | None,
    *,
    model: type[TModel],
    nested_fields: frozenset[str],
    extensions_field: str = "extensions",
) -> TModel | None:
    """Merge two Pydantic models with right overriding left.

    Semantics:
    - Base state comes from ``left`` with ``exclude_none=True``.
    - Only fields present in ``right.model_fields_set`` are applied.
    - Scalars: right wins, including explicit ``None`` (clears the field).
    - Nested configs in ``nested_fields``: shallow dict merge when both sides are
      non-``None``; explicit ``None`` on right clears.
    - ``extensions_field``: union-merge; ``None`` on right clears all extensions;
      an empty mapping on right is a no-op that preserves left keys; colliding keys
      use right.
    """
    if left is None:
        return right
    if right is None:
        return left
    data = left.model_dump(exclude_none=True)
    has_extensions = extensions_field in model.model_fields
    extensions = {**getattr(left, extensions_field)} if has_extensions else {}
    for key in right.model_fields_set:
        value = getattr(right, key)
        if has_extensions and key == extensions_field:
            if value is None:
                extensions = {}
            elif isinstance(value, Mapping):
                extensions.update(value)
            else:
                extensions = dict(value)
            continue
        if key in nested_fields:
            if value is None:
                data[key] = None
            else:
                left_value = data.get(key)
                right_value = value.model_dump(exclude_none=True)
                if isinstance(left_value, dict) and right_value:
                    data[key] = {**left_value, **right_value}
                else:
                    data[key] = right_value
            continue
        if isinstance(value, BaseModel):
            data[key] = value.model_dump(exclude_none=True)
        else:
            data[key] = value
    if has_extensions:
        data[extensions_field] = extensions
    return model(**data)

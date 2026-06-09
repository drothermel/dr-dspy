from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dspy.primitives.record_store import RecordStore


def _is_public_api_name(cls: type, key: str) -> bool:
    for base in cls.__mro__:
        if base is object:
            break
        descriptor = base.__dict__.get(key)
        if descriptor is None:
            continue
        if isinstance(descriptor, (types.FunctionType, types.MethodType, classmethod, staticmethod, property)):
            return True
    return False


class RecordBacked:
    """Mixin: attribute access for string keys in a backing mapping store."""

    _RECORD_ATTR: ClassVar[str]
    _RECORD_RESERVED: ClassVar[frozenset[str]]

    @override
    def __getattribute__(self, key: str) -> Any:
        if key.startswith("_"):
            return object.__getattribute__(self, key)

        cls = object.__getattribute__(self, "__class__")
        reserved = type.__getattribute__(cls, "_RECORD_RESERVED")
        if key in reserved:
            return object.__getattribute__(self, key)

        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass

        store_attr = type.__getattribute__(cls, "_RECORD_ATTR")
        store = object.__getattribute__(self, store_attr)
        if key in store:
            return store[key]
        raise AttributeError(f"{cls.__name__!r} object has no attribute {key!r}")

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return

        cls = object.__getattribute__(self, "__class__")
        reserved = type.__getattribute__(cls, "_RECORD_RESERVED")
        if key in reserved:
            object.__setattr__(self, key, value)
            return

        if _is_public_api_name(cls, key):
            raise AttributeError(
                f"Cannot set attribute {key!r} on {cls.__name__}; use bracket notation "
                f"({cls.__name__}[{key!r}] = value) for dynamic fields that collide with API names."
            )

        store_attr = type.__getattribute__(cls, "_RECORD_ATTR")
        store = object.__getattribute__(self, store_attr)
        store[key] = value


class RecordStoreFacade(RecordBacked):
    """Mixin: mapping protocol delegation to a ``RecordStore`` backing field."""

    def _backing_store(self) -> RecordStore:
        return object.__getattribute__(self, type(self)._RECORD_ATTR)

    def __getitem__(self, key: str) -> Any:
        return self._backing_store()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._backing_store()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._backing_store()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._backing_store()

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def keys(self, include_dspy: bool = False) -> list[str]:
        return self._backing_store().keys(include_dspy=include_dspy)

    def values(self, include_dspy: bool = False) -> list[Any]:
        return self._backing_store().values(include_dspy=include_dspy)

    def items(self, include_dspy: bool = False) -> list[tuple[str, Any]]:
        return self._backing_store().items(include_dspy=include_dspy)

    def get(self, key: str, default: Any = None) -> Any:
        return self._backing_store().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._backing_store().to_dict()

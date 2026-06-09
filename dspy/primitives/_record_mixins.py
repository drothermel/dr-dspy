from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dspy.primitives.record_store import RecordStore


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
        store_attr = type.__getattribute__(cls, "_RECORD_ATTR")
        store = object.__getattribute__(self, store_attr)
        if key in store:
            return store[key]
        return object.__getattribute__(self, key)

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return

        cls = object.__getattribute__(self, "__class__")
        reserved = type.__getattribute__(cls, "_RECORD_RESERVED")
        if key in reserved:
            object.__setattr__(self, key, value)
        else:
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

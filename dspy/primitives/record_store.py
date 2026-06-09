from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from typing_extensions import override

from dspy.serialization.json import to_jsonable


class _RecordBacked:
    """Internal mixin: attribute access for string keys in a backing mapping store."""

    _RECORD_ATTR: ClassVar[str]
    _RECORD_RESERVED: ClassVar[frozenset[str]]

    @override
    def __getattribute__(self, key: str) -> Any:
        cls = type(self)
        reserved = object.__getattribute__(cls, "_RECORD_RESERVED")
        store_attr = object.__getattribute__(cls, "_RECORD_ATTR")
        if key in reserved or key.startswith("_"):
            return object.__getattribute__(self, key)
        store = object.__getattribute__(self, store_attr)
        if key in store:
            return store[key]
        return object.__getattribute__(self, key)

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        cls = type(self)
        reserved = object.__getattribute__(cls, "_RECORD_RESERVED")
        store_attr = object.__getattribute__(cls, "_RECORD_ATTR")
        if key in reserved or key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            store = object.__getattribute__(self, store_attr)
            store[key] = value


class _RecordStoreFacade(_RecordBacked):
    """Mixin: mapping protocol delegation to a RecordStore backing field."""

    def _backing_store(self) -> "RecordStore":
        return object.__getattribute__(self, type(self)._RECORD_ATTR)

    def __getitem__(self, key: str) -> Any:
        return self._backing_store()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._backing_store()[key] = value

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


class RecordStore(_RecordBacked):
    """String-keyed record bag with consistent attribute and mapping access."""

    _RECORD_ATTR = "_data"
    _RECORD_RESERVED = frozenset({"_data"})

    __hash__ = None

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", dict(data) if data else {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordStore):
            return NotImplemented
        return self._data == other._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self, include_dspy: bool = False) -> list[str]:
        return [k for k in self._data if not k.startswith("dspy_") or include_dspy]

    def values(self, include_dspy: bool = False) -> list[Any]:
        return [v for k, v in self._data.items() if not k.startswith("dspy_") or include_dspy]

    def items(self, include_dspy: bool = False) -> list[tuple[str, Any]]:
        return [(k, v) for k, v in self._data.items() if not k.startswith("dspy_") or include_dspy]

    def copy(self) -> "RecordStore":
        return RecordStore(self._data)

    def to_dict(self) -> dict[str, Any]:
        return {key: to_jsonable(value) for key, value in self._data.items()}

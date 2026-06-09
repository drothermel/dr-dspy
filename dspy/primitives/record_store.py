from collections.abc import Iterator, Mapping
from typing import Any

from typing_extensions import override

from dspy.serialization.json import to_jsonable

_RESERVED_INSTANCE_ATTRS = frozenset({"_data"})


class RecordStore:
    """String-keyed record bag with consistent attribute and mapping access."""

    __hash__ = None

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", dict(data) if data else {})

    @override
    def __getattribute__(self, key: str) -> Any:
        if key in _RESERVED_INSTANCE_ATTRS or key.startswith("_"):
            return super().__getattribute__(key)
        data = super().__getattribute__("_data")
        if key in data:
            return data[key]
        return super().__getattribute__(key)

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        if key in _RESERVED_INSTANCE_ATTRS or key.startswith("_"):
            super().__setattr__(key, value)
        else:
            super().__getattribute__("_data")[key] = value

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

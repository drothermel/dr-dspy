from collections.abc import Mapping
from typing import Any

from typing_extensions import override

from dspy.primitives.record_store import RecordStore

_EXAMPLE_RESERVED = frozenset({"_store", "_demos", "_input_keys"})


class Example:
    __hash__ = None

    def __init__(
        self,
        *,
        _store: RecordStore | Mapping[str, Any] | None = None,
        _input_keys: frozenset[str] | None = None,
    ) -> None:
        if _store is None:
            store = RecordStore()
        elif isinstance(_store, RecordStore):
            store = _store
        else:
            store = RecordStore(_store)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_demos", [])
        object.__setattr__(self, "_input_keys", _input_keys)

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, input_keys: tuple[str, ...] = ()) -> "Example":
        return cls(_store=RecordStore(record), _input_keys=frozenset(input_keys))

    @property
    def input_keys(self) -> frozenset[str]:
        if self._input_keys is None:
            return frozenset()
        return self._input_keys

    @override
    def __getattribute__(self, key: str) -> Any:
        if key in _EXAMPLE_RESERVED or key.startswith("_"):
            return super().__getattribute__(key)
        store = super().__getattribute__("_store")
        if key in store:
            return store[key]
        return super().__getattribute__(key)

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        if key in _EXAMPLE_RESERVED or key.startswith("_"):
            super().__setattr__(key, value)
        else:
            super().__getattribute__("_store")[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len([k for k in self._store if not k.startswith("dspy_")])

    @override
    def __repr__(self) -> str:
        d = {k: v for k, v in self._store.items() if not k.startswith("dspy_")}
        return f"Example({d})" + f" (input_keys={sorted(self.input_keys)})"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Example) and self._store == other._store and self.input_keys == other.input_keys

    def keys(self, include_dspy=False):
        return self._store.keys(include_dspy=include_dspy)

    def values(self, include_dspy=False):
        return self._store.values(include_dspy=include_dspy)

    def items(self, include_dspy=False):
        return self._store.items(include_dspy=include_dspy)

    def get(self, key, default=None):
        return self._store.get(key, default)

    def with_input_keys(self, *keys: str) -> "Example":
        return self.fork(_input_keys=frozenset(keys))

    def as_inputs(self) -> dict[str, Any]:
        if not self._input_keys:
            raise ValueError(
                "Input keys have not been set for this example. Use Example.from_record(..., input_keys=(...))."
            )
        return {key: self._store[key] for key in self._store if key in self._input_keys}

    def as_labels(self) -> dict[str, Any]:
        input_keys = self.input_keys
        return {key: self._store[key] for key in self._store if key not in input_keys and not key.startswith("dspy_")}

    def __iter__(self):
        return iter(dict(self._store))

    def fork(self, **updates: Any) -> "Example":
        store = self._store.copy()
        input_keys = self._input_keys
        for key, value in updates.items():
            if key == "_input_keys":
                input_keys = value
            elif key == "_store":
                store = value.copy() if isinstance(value, RecordStore) else RecordStore(value).copy()
            else:
                store[key] = value
        return Example(_store=store, _input_keys=input_keys)

    def without(self, *keys):
        copied = self.fork()
        for key in keys:
            del copied[key]
        return copied

    def to_dict(self) -> dict[str, Any]:
        return self._store.to_dict()

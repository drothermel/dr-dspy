from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from typing_extensions import override


class Example:
    def __init__(self, *, _store: dict[str, Any] | None = None, _input_keys: frozenset[str] | None = None) -> None:
        self._store = _store or {}
        self._demos: list[Any] = []
        self._input_keys = _input_keys

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, input_keys: tuple[str, ...] = ()) -> "Example":
        return cls(_store=dict(record), _input_keys=frozenset(input_keys))

    @property
    def input_keys(self) -> frozenset[str]:
        if self._input_keys is None:
            return frozenset()
        return self._input_keys

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError
        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    @override
    def __setattr__(self, key, value) -> None:
        if key.startswith("_") or key in dir(self.__class__):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value) -> None:
        self._store[key] = value

    def __delitem__(self, key) -> None:
        del self._store[key]

    def __contains__(self, key) -> bool:
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
    def __eq__(self, other):
        return isinstance(other, Example) and self._store == other._store and self.input_keys == other.input_keys

    @override
    def __hash__(self):
        return hash((tuple(sorted(self._store.items())), self.input_keys))

    def keys(self, include_dspy=False):
        return [k for k in self._store if not k.startswith("dspy_") or include_dspy]

    def values(self, include_dspy=False):
        return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def items(self, include_dspy=False):
        return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

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
                store = value.copy()
            else:
                store[key] = value
        return Example(_store=store, _input_keys=input_keys)

    def without(self, *keys):
        copied = self.fork()
        for key in keys:
            del copied[key]
        return copied

    def to_dict(self) -> dict[str, Any]:
        def convert_to_serializable(value):
            if hasattr(value, "to_dict") and callable(value.to_dict):
                return value.to_dict()
            if isinstance(value, BaseModel):
                return value.model_dump()
            if isinstance(value, list):
                return [convert_to_serializable(item) for item in value]
            if isinstance(value, dict):
                return {k: convert_to_serializable(v) for k, v in value.items()}
            return value

        serializable_store = {}
        for k, v in self._store.items():
            serializable_store[k] = convert_to_serializable(v)
        return serializable_store

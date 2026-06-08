from typing import Any

from pydantic import BaseModel
from typing_extensions import override


class Example:
    def __init__(self, base=None, **kwargs) -> None:
        self._store = {}
        self._demos = []
        self._input_keys = None
        if base and isinstance(base, type(self)):
            self._store = base._store.copy()
        elif base and isinstance(base, dict):
            self._store = base.copy()
        self._store.update(kwargs)

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
        return f"Example({d})" + f" (input_keys={self._input_keys})"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    @override
    def __eq__(self, other):
        return isinstance(other, Example) and self._store == other._store

    @override
    def __hash__(self):
        return hash(tuple(self._store.items()))

    def keys(self, include_dspy=False):
        return [k for k in self._store if not k.startswith("dspy_") or include_dspy]

    def values(self, include_dspy=False):
        return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def items(self, include_dspy=False):
        return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def get(self, key, default=None):
        return self._store.get(key, default)

    def with_inputs(self, *keys):
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self):
        if self._input_keys is None:
            raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")
        d = {key: self._store[key] for key in self._store if key in self._input_keys}
        new_instance = type(self)(base=d)
        new_instance._input_keys = self._input_keys
        return new_instance

    def labels(self):
        input_keys = self.inputs().keys()
        d = {key: self._store[key] for key in self._store if key not in input_keys}
        return type(self)(d)

    def __iter__(self):
        return iter(dict(self._store))

    def copy(self, **kwargs):
        return type(self)(base=self, **kwargs)

    def without(self, *keys):
        copied = self.copy()
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

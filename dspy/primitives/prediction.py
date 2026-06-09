from collections.abc import Mapping
from typing import Any

from typing_extensions import override

from dspy.primitives.record_store import RecordStore, _RecordBacked


class Prediction(_RecordBacked):
    _RECORD_ATTR = "_store"
    _RECORD_RESERVED = frozenset({"_store", "_completions", "_lm_usage"})

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_store", RecordStore(kwargs))
        object.__setattr__(self, "_completions", None)
        object.__setattr__(self, "_lm_usage", None)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Prediction":
        return cls(**dict(record))

    def get_lm_usage(self):
        return self._lm_usage

    def set_lm_usage(self, value) -> None:
        self._lm_usage = value

    @classmethod
    def from_completions(cls, list_or_dict, task_spec=None):
        obj = cls()
        obj._completions = Completions(list_or_dict, task_spec=task_spec)
        object.__setattr__(obj, "_store", RecordStore({k: v[0] for k, v in obj._completions.items()}))
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __iter__(self):
        return iter(self._store)

    def keys(self, include_dspy=False):
        return self._store.keys(include_dspy=include_dspy)

    def items(self, include_dspy=False):
        return self._store.items(include_dspy=include_dspy)

    def get(self, key, default=None):
        return self._store.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._store.to_dict()

    @override
    def __repr__(self) -> str:
        store_repr = ",\n    ".join((f"{k}={v!r}" for k, v in self._store.items()))
        if self._completions is None or len(self._completions) == 1:
            return f"Prediction(\n    {store_repr}\n)"
        num_completions = len(self._completions)
        return f"Prediction(\n    {store_repr},\n    completions=Completions(...)\n) ({num_completions - 1} completions omitted)"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    def __float__(self) -> float:
        if "score" not in self._store:
            raise ValueError("Prediction object does not have a 'score' field to convert to float.")
        return float(self._store["score"])

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() + other
        if isinstance(other, Prediction):
            return self.__float__() + float(other)
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return other + self.__float__()
        if isinstance(other, Prediction):
            return float(other) + self.__float__()
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() / other
        if isinstance(other, Prediction):
            return self.__float__() / float(other)
        raise TypeError(f"Unsupported type for division: {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return other / self.__float__()
        if isinstance(other, Prediction):
            return float(other) / self.__float__()
        raise TypeError(f"Unsupported type for division: {type(other)}")

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() < other
        if isinstance(other, Prediction):
            return self.__float__() < float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __le__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() <= other
        if isinstance(other, Prediction):
            return self.__float__() <= float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() > other
        if isinstance(other, Prediction):
            return self.__float__() > float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() >= other
        if isinstance(other, Prediction):
            return self.__float__() >= float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    @property
    def completions(self):
        return self._completions


class Completions(_RecordBacked):
    _RECORD_ATTR = "_completions"
    _RECORD_RESERVED = frozenset({"_completions", "task_spec"})

    def __init__(self, list_or_dict, task_spec=None) -> None:
        self.task_spec = task_spec
        if isinstance(list_or_dict, list):
            kwargs = {}
            for arg in list_or_dict:
                for k, v in arg.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict
        self._validate_completion_lists(kwargs)
        object.__setattr__(self, "_completions", kwargs)

    @staticmethod
    def _validate_completion_lists(kwargs: dict[str, object]) -> None:
        if not all(isinstance(v, list) for v in kwargs.values()):
            raise ValueError("All Completions values must be lists")
        if not kwargs:
            return
        lists = [v for v in kwargs.values() if isinstance(v, list)]
        length = len(lists[0])
        if not all(len(v) == length for v in lists):
            raise ValueError("All Completions lists must have the same length")

    def items(self):
        return self._completions.items()

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")
            return Prediction.from_record({k: v[key] for k, v in self._completions.items()})
        return self._completions[key]

    def __len__(self) -> int:
        if not self._completions:
            return 0
        return len(next(iter(self._completions.values())))

    def __contains__(self, key) -> bool:
        return key in self._completions

    @override
    def __repr__(self) -> str:
        items_repr = ",\n    ".join((f"{k}={v!r}" for k, v in self._completions.items()))
        return f"Completions(\n    {items_repr}\n)"

    @override
    def __str__(self) -> str:
        return self.__repr__()

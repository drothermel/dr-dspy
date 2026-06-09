from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dspy.primitives.record_store import RecordStore, _RecordBacked

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from dspy.task_spec import TaskSpec


class Prediction(_RecordBacked):
    """Model output container backed by a field store.

    Equality compares store fields and attached completions, not numeric scores.

    Rich comparisons and arithmetic (``+``, ``/``, ``<``, etc.) coerce operands
    through ``float(self["score"])``. A missing ``score`` field raises
    ``ValueError``. Supported operand types are ``int``, ``float``, and
    ``Prediction``.
    """

    _RECORD_ATTR = "_store"
    _RECORD_RESERVED = frozenset({"_store", "_completions", "_lm_usage"})

    __hash__ = None

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_store", RecordStore(kwargs))
        object.__setattr__(self, "_completions", None)
        object.__setattr__(self, "_lm_usage", None)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Prediction:
        return cls(**dict(record))

    def get_lm_usage(self) -> dict[str, dict[str, Any]] | None:
        return self._lm_usage

    def set_lm_usage(self, value: dict[str, dict[str, Any]] | None) -> None:
        self._lm_usage = value

    @classmethod
    def from_completions(
        cls,
        list_or_dict: list[dict[str, Any]] | dict[str, list[Any]],
        task_spec: TaskSpec | None = None,
    ) -> Prediction:
        obj = cls()
        completions = Completions(list_or_dict, task_spec=task_spec)
        object.__setattr__(obj, "_completions", completions)
        object.__setattr__(obj, "_store", RecordStore({k: v[0] for k, v in completions.items()}))
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def keys(self, include_dspy: bool = False) -> list[str]:
        return self._store.keys(include_dspy=include_dspy)

    def values(self, include_dspy: bool = False) -> list[Any]:
        return self._store.values(include_dspy=include_dspy)

    def items(self, include_dspy: bool = False) -> list[tuple[str, Any]]:
        return self._store.items(include_dspy=include_dspy)

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prediction):
            return NotImplemented
        return self._store == other._store and self._completions == other._completions

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

    def __add__(self, other: float | int | Prediction) -> float:
        return self.__float__() + _coerce_score_operand(other, operation="addition")

    def __radd__(self, other: float | int | Prediction) -> float:
        return _coerce_score_operand(other, operation="addition") + self.__float__()

    def __truediv__(self, other: float | int | Prediction) -> float:
        return self.__float__() / _coerce_score_operand(other, operation="division")

    def __rtruediv__(self, other: float | int | Prediction) -> float:
        return _coerce_score_operand(other, operation="division") / self.__float__()

    def __lt__(self, other: float | int | Prediction) -> bool:
        return self.__float__() < _coerce_score_operand(other, operation="comparison")

    def __le__(self, other: float | int | Prediction) -> bool:
        return self.__float__() <= _coerce_score_operand(other, operation="comparison")

    def __gt__(self, other: float | int | Prediction) -> bool:
        return self.__float__() > _coerce_score_operand(other, operation="comparison")

    def __ge__(self, other: float | int | Prediction) -> bool:
        return self.__float__() >= _coerce_score_operand(other, operation="comparison")

    @property
    def completions(self) -> Completions | None:
        return self._completions


def _coerce_score_operand(other: object, *, operation: str) -> float:
    if isinstance(other, (float, int)):
        return float(other)
    if isinstance(other, Prediction):
        return float(other)
    raise TypeError(f"Unsupported type for {operation}: {type(other)}")


class Completions(_RecordBacked):
    _RECORD_ATTR = "_completions"
    _RECORD_RESERVED = frozenset({"_completions", "task_spec"})

    def __init__(
        self,
        list_or_dict: list[dict[str, Any]] | dict[str, list[Any]],
        task_spec: TaskSpec | None = None,
    ) -> None:
        self.task_spec = task_spec
        if isinstance(list_or_dict, list):
            kwargs: dict[str, list[Any]] = {}
            for arg in list_or_dict:
                for k, v in arg.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict
        self._validate_completion_lists(kwargs)
        object.__setattr__(self, "_completions", kwargs)

    @staticmethod
    def _validate_completion_lists(kwargs: dict[str, list[Any]]) -> None:
        if not all(isinstance(v, list) for v in kwargs.values()):
            raise ValueError("All Completions values must be lists")
        if not kwargs:
            return
        lists = list(kwargs.values())
        length = len(lists[0])
        if not all(len(v) == length for v in lists):
            raise ValueError("All Completions lists must have the same length")

    def items(self) -> list[tuple[str, list[Any]]]:
        return list(self._completions.items())

    def __getitem__(self, key: int | str) -> Prediction | list[Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")
            return Prediction.from_record({k: v[key] for k, v in self._completions.items()})
        return self._completions[key]

    def __len__(self) -> int:
        if not self._completions:
            return 0
        return len(next(iter(self._completions.values())))

    def __contains__(self, key: object) -> bool:
        return key in self._completions

    @override
    def __repr__(self) -> str:
        items_repr = ",\n    ".join((f"{k}={v!r}" for k, v in self._completions.items()))
        return f"Completions(\n    {items_repr}\n)"

    @override
    def __str__(self) -> str:
        return self.__repr__()

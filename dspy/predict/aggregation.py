from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.evaluate.metrics import normalize_text
from dspy.primitives import Completions, Prediction

if TYPE_CHECKING:
    from collections.abc import Callable

    from dspy.task_spec import TaskSpec

_MAJORITY_INPUT_TYPES = (Prediction, Completions, list)


def default_normalize(s: str) -> str | None:
    return normalize_text(s) or None


def _iter_completion_records(completions: Completions | list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(completions, Completions):
        for index in range(len(completions)):
            record = completions[index]
            if isinstance(record, Prediction):
                records.append(dict(record))
            elif isinstance(record, dict):
                records.append(record)
            else:
                raise TypeError(f"Expected Prediction or dict completion record, got {type(record).__name__}.")
        return records
    for completion in completions:
        if isinstance(completion, Prediction):
            records.append(dict(completion))
        else:
            records.append(completion)
    return records


def majority(
    prediction_or_completions: Prediction | Completions | list[dict[str, Any]],
    normalize: Callable[[str], str | None] | None = default_normalize,
    field: str | None = None,
) -> Prediction:
    if not isinstance(prediction_or_completions, _MAJORITY_INPUT_TYPES):
        accepted = ", ".join(type_.__name__ for type_ in _MAJORITY_INPUT_TYPES)
        raise TypeError(f"majority expected one of ({accepted}), got {type(prediction_or_completions).__name__}.")

    if isinstance(prediction_or_completions, Prediction):
        completions = prediction_or_completions.completions
        if completions is None:
            raise TypeError("majority expected Prediction with completions, got Prediction.completions=None.")
    else:
        completions = prediction_or_completions

    task_spec: TaskSpec | None = getattr(completions, "task_spec", None)
    completion_records = _iter_completion_records(completions)
    if not completion_records:
        raise ValueError("majority requires at least one completion.")

    if not field:
        if task_spec is not None:
            field = list(task_spec.output_fields.keys())[-1]
        else:
            field = list(completion_records[0].keys())[-1]

    normalize_fn = normalize if normalize is not None else lambda value: value
    normalized_values = [normalize_fn(str(completion[field])) for completion in completion_records]
    normalized_values_ = [value for value in normalized_values if value is not None]
    value_counts: dict[str | None, int] = {}
    for value in normalized_values_ or normalized_values:
        value_counts[value] = value_counts.get(value, 0) + 1
    majority_value = max(value_counts, key=lambda value: value_counts[value])
    completion = completion_records[0]
    for candidate in completion_records:
        if normalize_fn(str(candidate[field])) == majority_value:
            completion = candidate
            break
    return Prediction.from_completions([completion], task_spec=task_spec)

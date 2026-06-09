from __future__ import annotations

from typing import Any

from dspy.errors import DSPyError
from dspy.primitives import Example


class UnknownPredictorInTraceError(DSPyError):
    """Trace step references a predictor id not present in predictor2name."""


def trace_to_demos(trace: list[Any], predictor2name: dict[int, str]) -> dict[str, list[Example]]:
    name2traces: dict[str, list[Example]] = {}
    for step in trace:
        predictor, inputs, outputs = step
        demo = Example.from_record({"augmented": True, **inputs, **outputs})
        try:
            predictor_name = predictor2name[id(predictor)]
        except KeyError as exc:
            raise UnknownPredictorInTraceError(
                f"No predictor mapping for id={id(predictor)!r}; known ids: {sorted(predictor2name)}"
            ) from exc
        name2traces[predictor_name] = name2traces.get(predictor_name, [])
        name2traces[predictor_name].append(demo)
    return name2traces

from __future__ import annotations

from typing import Any

from dspy.predict.call_options import PredictOptions, ensure_predict_options_built

RESERVED_PREDICT_INPUT_KEYS = frozenset(
    {"run", "options", "lm", "config", "demos", "task_spec", "_trace", "prediction"}
)


def reject_reserved_predict_inputs(inputs: dict[str, Any]) -> None:
    reserved = sorted(key for key in inputs if key in RESERVED_PREDICT_INPUT_KEYS)
    if reserved:
        raise ValueError(
            f"Reserved keyword(s) {reserved} must not be passed as task inputs. "
            "Use run= for RunContext and options=PredictOptions(...) for lm, config, demos, task_spec, trace, and prediction."
        )


def resolve_predict_options(options: PredictOptions | None) -> PredictOptions:
    ensure_predict_options_built()
    return options or PredictOptions()

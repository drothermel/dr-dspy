from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, SkipValidation

from dspy.core.types.config import LMConfig  # noqa: TC001 — Pydantic field type
from dspy.core.types.lm import LMForward
from dspy.runtime.call_options import ModuleCallOptions
from dspy.task_spec.task_spec import TaskSpec


class PredictOptions(ModuleCallOptions):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    lm: SkipValidation[LMForward | None] = None
    config: LMConfig | None = None
    demos: list[dict[str, Any]] | None = None
    task_spec: SkipValidation[TaskSpec | None] = None
    trace: bool = True
    prediction: dict[str, Any] | None = None


_predict_options_built = False


def ensure_predict_options_built() -> None:
    global _predict_options_built
    if _predict_options_built:
        return

    PredictOptions.model_rebuild(_types_namespace={"LMForward": LMForward, "TaskSpec": TaskSpec})
    _predict_options_built = True


ensure_predict_options_built()

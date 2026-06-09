from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, SkipValidation

from dspy.core.types.config import LMConfig

if TYPE_CHECKING:
    from dspy.core.types.lm import LMForward
    from dspy.task_spec.task_spec import TaskSpec


class ModuleCallOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PredictOptions(ModuleCallOptions):
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
    from dspy.core.types.lm import LMForward
    from dspy.task_spec.task_spec import TaskSpec

    PredictOptions.model_rebuild(_types_namespace={"LMForward": LMForward, "TaskSpec": TaskSpec})
    _predict_options_built = True


ensure_predict_options_built()

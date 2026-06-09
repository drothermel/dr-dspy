from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from dspy.core.types.config import LMConfig


class ModuleCallOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PredictOptions(ModuleCallOptions):
    lm: Any | None = None
    config: LMConfig | None = None
    demos: list[dict[str, Any]] | None = None
    task_spec: Any | None = None
    trace: bool = True
    prediction: dict[str, Any] | None = None

from __future__ import annotations

from typing import Any

from dr_llm.llm import EffortSpec, SamplingControls
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec, parse_reasoning_spec
from pydantic import BaseModel, ConfigDict, field_validator

DR_LLM_EXTENSION_KEY = "dr_llm"


class DrLlmProviderControls(BaseModel):
    """Provider-native dr-llm controls carried by the DSPy dr-llm bridge."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    reasoning: ReasoningSpec | None = None
    effort: EffortSpec | None = None
    sampling: SamplingControls | None = None

    @field_validator("reasoning", mode="before")
    @classmethod
    def _parse_reasoning(cls, value: Any) -> ReasoningSpec | None:
        if value is None:
            return None
        return parse_reasoning_spec(value)

    @field_validator("effort", mode="before")
    @classmethod
    def _parse_effort(cls, value: Any) -> EffortSpec | None:
        if value is None or isinstance(value, EffortSpec):
            return value
        return EffortSpec(value)

    @field_validator("sampling", mode="before")
    @classmethod
    def _parse_sampling(cls, value: Any) -> SamplingControls | None:
        if value is None or isinstance(value, SamplingControls):
            return value
        return SamplingControls(**value)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def parse_dr_llm_controls(value: object | None) -> DrLlmProviderControls:
    if value is None:
        return DrLlmProviderControls()
    if isinstance(value, DrLlmProviderControls):
        return value
    return DrLlmProviderControls.model_validate(value)

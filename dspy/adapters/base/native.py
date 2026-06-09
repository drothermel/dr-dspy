from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from dspy.adapters.types.citation import Citations
from dspy.adapters.types.field_type import NativeResponseFieldType, implements_parse_lm_output
from dspy.adapters.types.reasoning import Reasoning
from dspy.core.types.config import LMConfig, LMReasoningConfig, NativeAdaptationMode
from dspy.task_spec import TaskSpec

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM

_DEFAULT_NATIVE_RESPONSE_TYPES: list[type[NativeResponseFieldType]] = [Citations, Reasoning]


class AdapterNativeMixin:
    @staticmethod
    def _ensure_native_response_type_parses_output(native_type: type[NativeResponseFieldType]) -> None:
        if not implements_parse_lm_output(native_type):
            raise TypeError(
                f"{native_type.__name__} is listed in native_response_types but does not implement parse_lm_output(). Native response fields must parse typed LMOutput values."
            )

    def _adapt_reasoning_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM, config: LMConfig) -> TaskSpec:
        if "reasoning" in config.model_fields_set and config.reasoning is None:
            return task_spec
        if config.reasoning is not None and config.reasoning.effort is not None:
            reasoning_effort = config.reasoning.effort
        elif isinstance(lm.kwargs.get("reasoning"), Mapping):
            reasoning_effort = lm.kwargs["reasoning"].get("effort")
        else:
            reasoning_effort = None
        if reasoning_effort is None or not lm.supports_reasoning:
            return task_spec
        if lm.reasoning_adaptation_mode is NativeAdaptationMode.SKIP:
            return task_spec
        config.reasoning = LMReasoningConfig(effort=reasoning_effort)
        return task_spec.delete(field_name)

    def _adapt_citations_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM) -> TaskSpec:
        if lm.citations_adaptation_mode is NativeAdaptationMode.SKIP:
            return task_spec.delete(field_name)
        return task_spec

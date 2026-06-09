from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dspy.adapters.types.base_type import Type
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.reasoning import Reasoning
from dspy.core.types.config import LMConfig, LMReasoningConfig
from dspy.task_spec import TaskSpec

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMMessage
    from dspy.utils.callback import BaseCallback
_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]


class AdapterMixinBase:
    callbacks: list[BaseCallback]
    use_native_function_calling: bool
    parallel_tool_calls: bool | None
    native_response_types: list[type[Type]]

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        raise NotImplementedError

    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        raise NotImplementedError

    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | list[dict[str, Any]]:
        raise NotImplementedError

    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        raise NotImplementedError

    def _get_turn_log_field_name(self, task_spec: TaskSpec) -> str | None:
        raise NotImplementedError

    def _get_tool_call_input_field_name(self, task_spec: TaskSpec) -> str | None:
        raise NotImplementedError

    def _get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None:
        raise NotImplementedError

    def format_conversation_history(
        self, task_spec: TaskSpec, turn_log_field_name: str, inputs: dict[str, Any]
    ) -> list[LMMessage]:
        raise NotImplementedError


class AdapterNativeMixin(AdapterMixinBase):
    @staticmethod
    def _ensure_native_response_type_parses_output(native_type: type[Type]) -> None:
        if native_type.parse_lm_output.__func__ is Type.parse_lm_output.__func__:
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
        if "gpt-5" in lm.model and lm.model_type == "chat":
            return task_spec
        config.reasoning = LMReasoningConfig(effort=reasoning_effort)
        return task_spec.delete(field_name)

    def _adapt_citations_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM) -> TaskSpec:
        if lm.model.startswith("anthropic/"):
            return task_spec.delete(field_name)
        return task_spec

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from dspy.adapters.types.base_type import Type
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMMessage
    from dspy.core.types.config import LMConfig
    from dspy.runtime.callback import Callback
    from dspy.task_spec import TaskSpec

ComposedAdapterT = TypeVar("ComposedAdapterT", bound="ComposedAdapter")


class FormattableAdapter(Protocol):
    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]: ...

    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | list[dict[str, Any]]: ...

    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str: ...

    def _get_turn_log_field_name(self, task_spec: TaskSpec) -> str | None: ...

    def _get_tool_call_input_field_name(self, task_spec: TaskSpec) -> str | None: ...

    def _get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None: ...

    def format_conversation_history(
        self, task_spec: TaskSpec, turn_log_field_name: str, inputs: dict[str, Any]
    ) -> list[LMMessage]: ...


class ChatFormattableAdapter(FormattableAdapter, Protocol):
    """Adapter surface required by ChatFormatMixin.format_finetune_data."""


class NativeAdaptableAdapter(Protocol):
    native_response_types: list[type[Type]]
    use_native_function_calling: bool
    parallel_tool_calls: bool | None

    @staticmethod
    def _ensure_native_response_type_parses_output(native_type: type[Type]) -> None: ...

    def _adapt_reasoning_native(
        self, task_spec: TaskSpec, field_name: str, lm: BaseLM, config: LMConfig
    ) -> TaskSpec: ...

    def _adapt_citations_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM) -> TaskSpec: ...


class ConversationFormattingAdapter(FormattableAdapter, NativeAdaptableAdapter, Protocol):
    """Formattable adapter surface required for turn-log conversation expansion."""


class ComposedAdapter(FormattableAdapter, NativeAdaptableAdapter, Protocol):
    callbacks: list[Callback]

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]: ...

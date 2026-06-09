from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, TypeVar

from dspy.adapters.call.mode import AdapterCallMode

if TYPE_CHECKING:
    from dspy.adapters.base.adapter import Adapter
    from dspy.adapters.call.preprocessors.chain import CallPreprocessorChain
    from dspy.adapters.format.message_assembler import MessageAssembler
    from dspy.adapters.types.field_type import NativeResponseFieldType
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMConfig, LMMessage, UserMessageContent
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
    ) -> UserMessageContent: ...

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
    """Adapter surface required for header-style finetune data formatting."""


class DirectParseAdapter(Protocol):
    """Adapter that supports standalone ``parse`` for LM completion text."""

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]: ...


class PipelineOnlyAdapter(Protocol):
    """Adapter whose main call path does not use standalone ``parse``."""

    call_mode: AdapterCallMode


class NativeAdaptableAdapter(Protocol):
    native_response_types: list[type[NativeResponseFieldType]]
    use_native_function_calling: bool
    parallel_tool_calls: bool | None

    @staticmethod
    def _ensure_native_response_type_parses_output(native_type: type[NativeResponseFieldType]) -> None: ...

    def _adapt_reasoning_native(
        self, task_spec: TaskSpec, field_name: str, lm: BaseLM, config: LMConfig
    ) -> TaskSpec: ...

    def _adapt_citations_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM) -> TaskSpec: ...


class ConversationFormattingAdapter(FormattableAdapter, NativeAdaptableAdapter, Protocol):
    """Formattable adapter surface required for turn-log conversation expansion."""


class MessageAssemblerHost(ConversationFormattingAdapter, Protocol):
    message_assembler: MessageAssembler

    def format_system_message(self, task_spec: TaskSpec) -> str: ...

    def format_demos(self, task_spec: TaskSpec, demos: list[dict[str, Any]]) -> list[LMMessage]: ...


class ComposedAdapter(FormattableAdapter, NativeAdaptableAdapter, DirectParseAdapter, Protocol):
    callbacks: list[Callback]
    preprocessor_chain: CallPreprocessorChain
    call_mode: AdapterCallMode | None

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]: ...


def is_pipeline_only_adapter(adapter: Adapter | ComposedAdapter) -> TypeGuard[PipelineOnlyAdapter]:
    return adapter.call_mode == AdapterCallMode.TWO_STEP


def is_direct_parse_adapter(adapter: Adapter | ComposedAdapter) -> TypeGuard[DirectParseAdapter]:
    return not is_pipeline_only_adapter(adapter)

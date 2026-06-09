from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.base.native_adaptation import (
    DEFAULT_NATIVE_RESPONSE_TYPES,
    adapt_citations_native,
    adapt_reasoning_native,
    ensure_native_response_type_parses_output,
)
from dspy.adapters.base.protocols import MessageAssemblerHost, is_pipeline_only_adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.mode import AdapterCallMode
from dspy.adapters.call.pipeline import AdapterCallPipeline
from dspy.adapters.call.postprocess import enrich_parsed_value_from_lm_output
from dspy.adapters.call.preprocessors import default_preprocessor_chain
from dspy.adapters.format.message_assembler import MessageAssembler
from dspy.adapters.types.field_type import NativeResponseFieldType
from dspy.core.types import (
    LMConfig,
    LMMessage,
    LMRequest,
    LMResponse,
    LMToolSpec,
    UserMessageContent,
    merge_lm_request_config,
)
from dspy.errors import AdapterOperationError, AdapterParseError
from dspy.runtime.callback import with_callbacks
from dspy.runtime.run_context import RunContext
from dspy.task_spec import FieldBinding, TaskSpec

if TYPE_CHECKING:
    from dspy.adapters.call.policies.parse_fallback import ParseFallbackPolicy
    from dspy.adapters.call.policies.response_format import ResponseFormatPolicy
    from dspy.adapters.format.field_formatter import FieldFormatter
    from dspy.clients.base_lm import BaseLM
    from dspy.runtime.callback import Callback
    from dspy.runtime.config import CallSite


class Adapter:
    """Base adapter for formatting task specs into LM requests and parsing responses.

    Native function-calling defaults by adapter subclass:

    | Adapter | ``use_native_function_calling`` default | Rationale |
    |---------|----------------------------------------|-----------|
    | ChatAdapter / XMLAdapter | ``False`` | Text marker parsing; FC optional |
    | JSONAdapter / BAMLAdapter | ``True`` | Structured JSON + tool calls via provider FC |
    | TwoStepAdapter | inherits kwargs | Main call text; extraction uses inner adapter |
    """

    response_format_policy: ResponseFormatPolicy | None = None
    parse_fallback_policy: ParseFallbackPolicy | None = None
    call_mode: AdapterCallMode | None = None
    capabilities: AdapterCapabilities = AdapterCapabilities()

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[NativeResponseFieldType]] | None = None,
        parallel_tool_calls: bool | None = None,
        allow_json_repair: bool = False,
    ) -> None:
        """Configure adapter behavior.

        ``parallel_tool_calls`` merges into ``LMToolChoice.parallel`` only when
        ``tool_choice`` is ``None`` or ``tool_choice.parallel`` is ``None``.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.parallel_tool_calls = parallel_tool_calls
        self.allow_json_repair = allow_json_repair
        self.native_response_types = native_response_types or DEFAULT_NATIVE_RESPONSE_TYPES
        self.field_formatter: FieldFormatter | None = None
        self.message_assembler = MessageAssembler(cast("MessageAssemblerHost", self))
        self.preprocessor_chain = default_preprocessor_chain()

    def _require_field_formatter(self) -> FieldFormatter:
        if self.field_formatter is None:
            raise NotImplementedError(f"{type(self).__name__} does not configure field_formatter.")
        return self.field_formatter

    def format_field_with_value(
        self,
        fields_with_values: dict[FieldBinding, Any],
        *,
        role_label: str | None = None,
        **kwargs: Any,
    ) -> str:
        if self.field_formatter is None:
            raise NotImplementedError(f"{type(self).__name__} does not configure field_formatter.")
        effective_role = role_label if role_label is not None else kwargs.get("role")
        return self.field_formatter.format_field_with_value(fields_with_values, role_label=effective_role)

    @staticmethod
    def _ensure_native_response_type_parses_output(native_type: type[NativeResponseFieldType]) -> None:
        ensure_native_response_type_parses_output(native_type)

    def _adapt_reasoning_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM, config: LMConfig) -> TaskSpec:
        return adapt_reasoning_native(task_spec=task_spec, field_name=field_name, lm=lm, config=config)

    def _adapt_citations_native(self, task_spec: TaskSpec, field_name: str, lm: BaseLM) -> TaskSpec:
        return adapt_citations_native(task_spec=task_spec, field_name=field_name, lm=lm)

    def _get_turn_log_field_name(self, task_spec: TaskSpec) -> str | None:
        return self.message_assembler.get_turn_log_field_name(task_spec)

    def _get_tool_call_input_field_name(self, task_spec: TaskSpec) -> str | None:
        return self.message_assembler.get_tool_call_input_field_name(task_spec)

    def _get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None:
        return self.message_assembler.get_tool_call_output_field_name(task_spec)

    def format_conversation_history(
        self,
        task_spec: TaskSpec,
        turn_log_field_name: str,
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
        return self.message_assembler.format_conversation_history(
            task_spec=task_spec, turn_log_field_name=turn_log_field_name, inputs=inputs
        )

    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        return self.message_assembler.format(task_spec=task_spec, demos=demos, inputs=inputs)

    def format_system_message(self, task_spec: TaskSpec) -> str:
        return f"{self.format_field_description(task_spec)}\n{self.format_field_structure(task_spec)}\n{self.format_task_description(task_spec)}"

    def format_field_description(self, task_spec: TaskSpec) -> str:
        raise NotImplementedError

    def format_field_structure(self, task_spec: TaskSpec) -> str:
        raise NotImplementedError

    def format_task_description(self, task_spec: TaskSpec) -> str:
        raise NotImplementedError

    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> UserMessageContent:
        raise NotImplementedError

    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        raise NotImplementedError

    def format_demos(self, task_spec: TaskSpec, demos: list[dict[str, Any]]) -> list[LMMessage]:
        return self.message_assembler.format_demos(task_spec=task_spec, demos=demos)

    def _call_preprocess(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any],
        task_spec: TaskSpec,
        inputs: dict[str, Any],
    ) -> tuple[TaskSpec, list[LMToolSpec], LMConfig]:
        return self.preprocessor_chain.run(
            self,
            lm=lm,
            config=config,
            task_spec=task_spec,
            inputs=inputs,
        )

    def _call_postprocess(
        self,
        processed_task_spec: TaskSpec,
        original_task_spec: TaskSpec,
        response: LMResponse,
    ) -> list[dict[str, Any]]:
        if is_pipeline_only_adapter(self):
            raise AdapterOperationError(
                f"{type(self).__name__} does not support _call_postprocess parse. "
                "Use AdapterCallPipeline.execute via the adapter call path."
            )
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(original_task_spec)
        for output in response.outputs:
            tool_calls = output.tool_calls
            text = output.text
            if text is not None and (not (tool_calls and tool_call_output_field_name)):
                value = self.parse(task_spec=processed_task_spec, completion=text)
            elif tool_calls and tool_call_output_field_name:
                value = (
                    self.parse(task_spec=processed_task_spec, completion=text)
                    if text and processed_task_spec.output_fields
                    else {}
                )
            elif text is None and (not processed_task_spec.output_fields):
                value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    task_spec=original_task_spec,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )
            value = enrich_parsed_value_from_lm_output(
                self,
                value=value,
                output=output,
                original_task_spec=original_task_spec,
            )
            values.append(value)
        return values

    def _render_request(
        self, lm: BaseLM, config: LMConfig, tools: list[LMToolSpec], messages: Sequence[LMMessage]
    ) -> LMRequest:
        return LMRequest(
            model=lm.model, messages=list(messages), tools=tools, config=merge_lm_request_config(lm=lm, config=config)
        )

    async def _call_lm(self, lm: BaseLM, request: LMRequest, *, run: RunContext, compiled=None) -> LMResponse:
        return await lm(request, run=run, compiled=compiled)

    async def __call__(
        self,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        run: RunContext,
        call_site: CallSite | None = None,
    ) -> list[dict[str, Any]]:
        return await AdapterCallPipeline.execute(
            self,
            lm=lm,
            config=config,
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            run=run,
            call_site=call_site,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.format = with_callbacks(kind="adapter")(cls.format)
        cls.parse = with_callbacks(kind="adapter")(cls.parse)

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        """Parse LM text into output fields.

        Contract: return a dict whose keys match ``task_spec.output_fields`` exactly.
        Missing or extra keys raise ``AdapterParseError`` via ``validate_parsed_fields``.
        """
        raise NotImplementedError

    def format_finetune_data(
        self,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support finetune data formatting. "
            "Use an adapter with capabilities.supports_finetune=True."
        )

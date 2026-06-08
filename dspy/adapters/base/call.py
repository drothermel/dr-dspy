from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.base.native import AdapterNativeMixin
from dspy.adapters.base.tool_calls import _provider_tool_call_to_tool_call_dict
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import ToolCalls
from dspy.core.types import (
    LMConfig,
    LMMessage,
    LMRequest,
    LMResponse,
    LMToolChoice,
    LMToolSpec,
    _coerce_tool_spec,
    coerce_lm_config,
    merge_lm_request_config,
)
from dspy.task_spec import TaskSpec
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM


class AdapterCallMixin(AdapterNativeMixin):
    def _call_preprocess(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any],
        task_spec: TaskSpec,
        inputs: dict[str, Any],
    ) -> tuple[TaskSpec, list[LMToolSpec], LMConfig]:
        if not isinstance(config, LMConfig):
            config = coerce_lm_config(config)
        tools: list[LMToolSpec] = []
        if not self.use_native_function_calling:
            if config.tool_choice is not None:
                config = config.model_copy(update={"tool_choice": None})
        else:
            tool_call_input_field_name = self._get_tool_call_input_field_name(task_spec)
            tool_call_output_field_name = self._get_tool_call_output_field_name(task_spec)

            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.adapters.types.tool.Tool]`."
                )

            if tool_call_output_field_name and lm.supports_function_calling:
                if tool_call_input_field_name is None:
                    raise ValueError("Tool call input field is required when native function calling is enabled.")
                input_tools = inputs[tool_call_input_field_name]
                input_tools = input_tools if isinstance(input_tools, list) else [input_tools]

                tools = [_coerce_tool_spec(tool) for tool in input_tools]
                if self.parallel_tool_calls is not None:
                    if config.tool_choice is None:
                        config = config.model_copy(
                            update={"tool_choice": LMToolChoice(mode="auto", parallel=self.parallel_tool_calls)}
                        )
                    elif config.tool_choice.parallel is None:
                        config = config.model_copy(
                            update={
                                "tool_choice": config.tool_choice.model_copy(
                                    update={"parallel": self.parallel_tool_calls}
                                )
                            }
                        )

                task_spec = task_spec.delete(tool_call_output_field_name)
                task_spec = task_spec.delete(tool_call_input_field_name)

        for name, field in task_spec.output_fields.items():
            field_type = field.type_
            if not (
                isinstance(field_type, type)
                and field_type in self.native_response_types
                and issubclass(field_type, Type)
            ):
                continue
            self._ensure_native_response_type_parses_output(field_type)
            if field_type is Reasoning:
                task_spec = self._adapt_reasoning_native(task_spec=task_spec, field_name=name, lm=lm, config=config)
            elif field_type is Citations:
                task_spec = self._adapt_citations_native(task_spec=task_spec, field_name=name, lm=lm)
            else:
                task_spec = task_spec.delete(name)

        return task_spec, tools, config

    def _call_postprocess(
        self,
        processed_task_spec: TaskSpec,
        original_task_spec: TaskSpec,
        response: LMResponse,
        _lm: BaseLM,
        _config: LMConfig | Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_task_spec)

        for output in response.outputs:
            output_logprobs = output.logprobs
            tool_calls = output.tool_calls
            text = output.text

            if text is not None and not (tool_calls and tool_call_output_field_name):
                value = self.parse(task_spec=processed_task_spec, completion=text)
            elif tool_calls and tool_call_output_field_name:
                try:
                    value = (
                        self.parse(task_spec=processed_task_spec, completion=text)
                        if text and processed_task_spec.output_fields
                        else {}
                    )
                except AdapterParseError:
                    value = {}
            elif text is None and not processed_task_spec.output_fields:
                value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    task_spec=original_task_spec,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            # Fields removed for native features are absent from the processed parse.
            for field_name in original_task_spec.output_fields:
                value.setdefault(field_name, None)

            if tool_calls and tool_call_output_field_name:
                tool_calls = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            for name, field in original_task_spec.output_fields.items():
                field_type = field.type_
                if (
                    isinstance(field_type, type)
                    and field_type in self.native_response_types
                    and issubclass(field_type, Type)
                ):
                    parsed_value = field_type.parse_lm_output(output)
                    if parsed_value is not None:
                        value[name] = parsed_value

            if output_logprobs:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values

    def _render_request(
        self,
        lm: BaseLM,
        config: LMConfig,
        tools: list[LMToolSpec],
        messages: Sequence[LMMessage],
    ) -> LMRequest:
        """Build the normalized LM request for the current adapter call path."""
        return LMRequest(
            model=lm.model,
            messages=list(messages),
            tools=tools,
            config=merge_lm_request_config(lm=lm, config=config),
        )

    async def _call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        return await lm(request)

    async def acall(
        self,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        from dspy.compile.resolve import resolve_call, resolve_lm_config
        from dspy.core.types.history import _history_request_messages_as_openai
        from dspy.dsp.utils.settings import settings
        from dspy.utils.transparency import (
            ACTIVE_CALL_METADATA,
            ACTIVE_COMPILED_CALL,
            validate_compiled_call,
        )

        resolved_config = coerce_lm_config(config)
        original_field_names = set(task_spec.fields.keys())
        processed_task_spec, tools, resolved_config = self._call_preprocess(
            lm=lm, config=resolved_config, task_spec=task_spec, inputs=inputs
        )
        mutations = [
            f"removed field {name}" for name in sorted(original_field_names - set(processed_task_spec.fields.keys()))
        ]
        messages = self.format(task_spec=processed_task_spec, demos=demos, inputs=inputs)
        request = self._render_request(lm=lm, config=resolved_config, tools=tools, messages=messages)
        merged_config, provenance = resolve_lm_config(lm, resolved_config)
        metadata = ACTIVE_CALL_METADATA.get()
        compiled = resolve_call(
            lm=lm,
            adapter=cast("Adapter", self),
            task_spec=task_spec,
            processed_task_spec=processed_task_spec,
            config=merged_config,
            config_provenance=provenance,
            messages=_history_request_messages_as_openai(request),
            task_spec_mutations=mutations,
            module=metadata.get("module", type(self).__name__),
            phase=metadata.get("phase", "adapter"),
            lm_role=metadata.get("lm_role", "default"),
        )
        transparency = settings.get("transparency", "strict")
        validate_compiled_call(compiled, transparency)
        token = ACTIVE_COMPILED_CALL.set(compiled)
        try:
            response = await self._call_lm(lm=lm, request=request)
        finally:
            ACTIVE_COMPILED_CALL.reset(token)
        return self._call_postprocess(
            processed_task_spec=processed_task_spec,
            original_task_spec=task_spec,
            response=response,
            _lm=lm,
            _config=resolved_config,
        )

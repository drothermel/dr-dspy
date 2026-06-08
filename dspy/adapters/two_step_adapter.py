from collections.abc import Mapping
from typing import Any

import json_repair
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import build_lm_message
from dspy.clients.base_lm import BaseLM
from dspy.compile.resolve import resolve_adapter, resolve_call, resolve_lm_config
from dspy.core.types import LMConfig, LMMessage, LMRequest, LMToolCallPart, coerce_lm_config, merge_lm_request_config
from dspy.core.types.history import _history_request_messages_as_openai
from dspy.dsp.utils.settings import settings
from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec
from dspy.task_spec.formatting import get_field_spec_description_string
from dspy.utils.exceptions import AdapterParseError, LMError
from dspy.utils.transparency import (
    ACTIVE_COMPILED_CALL,
    reset_active_call_metadata,
    set_active_call_metadata,
    validate_compiled_call,
)


class FrameworkTwoStepExtractorTaskSpec(TaskSpec):
    name: str = "framework.two_step.extractor"
    instructions: str = "The input is text that should contain all information needed to produce the requested output fields. Extract each output field verbatim from the text."
    inputs: tuple[FieldSpec, ...] = (
        input_field(
            "text", str, desc="Raw completion text from the main language model to extract structured fields from."
        ),
    )
    outputs: tuple[FieldSpec, ...] = ()


class TwoStepAdapter(Adapter):
    def __init__(self, extraction_model: BaseLM, extraction_adapter: Adapter | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(extraction_model, BaseLM):
            raise ValueError("extraction_model must be an instance of dspy.clients.base_lm.BaseLM")
        self.extraction_model = extraction_model
        self.extraction_adapter = extraction_adapter

    @override
    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        messages: list[LMMessage] = []
        task_description = self.format_task_description(task_spec)
        messages.append(build_lm_message(role="system", content=task_description))
        messages.extend(self.format_demos(task_spec=task_spec, demos=demos))
        messages.append(
            build_lm_message(role="user", content=self.format_user_message_content(task_spec=task_spec, inputs=inputs))
        )
        return messages

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        raise NotImplementedError(
            "TwoStepAdapter.parse is not supported. Structured extraction runs in TwoStepAdapter.acall."
        )

    async def _run_extraction(self, *, original_task_spec: TaskSpec, text: str) -> dict[str, Any]:
        transparency = settings.get("transparency", "strict")
        extraction_adapter, _adapter_notes = resolve_adapter(
            self.extraction_adapter or settings.adapter, transparency=transparency
        )
        extractor_task_spec = self._create_extractor_task_spec(original_task_spec)
        config, _provenance = resolve_lm_config(self.extraction_model, LMConfig())
        metadata_token = set_active_call_metadata(
            module="TwoStepAdapter", phase="two_step.extraction", lm_role="extraction_model"
        )
        try:
            results = await extraction_adapter.acall(
                lm=self.extraction_model, config=config, task_spec=extractor_task_spec, demos=[], inputs={"text": text}
            )
        finally:
            reset_active_call_metadata(metadata_token)
        return results[0]

    @override
    async def acall(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resolved_config = coerce_lm_config(config)
        messages = self.format(task_spec=task_spec, demos=demos, inputs=inputs)
        merged_config, provenance = resolve_lm_config(lm, resolved_config)
        request = LMRequest(
            model=lm.model, messages=messages, config=merge_lm_request_config(lm=lm, config=merged_config)
        )
        transparency = settings.get("transparency", "strict")
        main_compiled = resolve_call(
            lm=lm,
            adapter=self,
            task_spec=task_spec,
            config=merged_config,
            config_provenance=provenance,
            messages=_history_request_messages_as_openai(request),
            module="TwoStepAdapter",
            phase="two_step.main",
            lm_role="default",
        )
        validate_compiled_call(main_compiled, transparency)
        main_token = ACTIVE_COMPILED_CALL.set(main_compiled)
        metadata_token = set_active_call_metadata(module="TwoStepAdapter", phase="two_step.main", lm_role="default")
        try:
            response = await lm.acall(request)
        finally:
            ACTIVE_COMPILED_CALL.reset(main_token)
            reset_active_call_metadata(metadata_token)
        extractor_task_spec = self._create_extractor_task_spec(task_spec)
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(task_spec)
        for output in response.outputs:
            output_logprobs = output.logprobs
            tool_calls = output.tool_calls
            text = output.text
            try:
                value = await self._run_extraction(original_task_spec=task_spec, text=text or "")
            except LMError:
                raise
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="TwoStepAdapter",
                    task_spec=extractor_task_spec,
                    lm_response=str(output),
                    message=f"Failed to parse response from the original completion: {e}",
                ) from e
            if tool_calls and tool_call_output_field_name:
                normalized_tool_calls = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, LMToolCallPart):
                        normalized_tool_calls.append(
                            {"name": tool_call.name, "args": dict(tool_call.args), "id": tool_call.id}
                        )
                    else:
                        normalized_tool_calls.append(
                            {
                                "name": tool_call["function"]["name"],
                                "args": json_repair.loads(tool_call["function"]["arguments"]),
                            }
                        )
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(normalized_tool_calls)
            if output_logprobs is not None:
                value["logprobs"] = output_logprobs
            values.append(value)
        return values

    @override
    def format_task_description(self, task_spec: TaskSpec) -> str:
        parts = []
        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append(
            "As input, you will be provided with:\n" + get_field_spec_description_string(task_spec.input_fields)
        )
        parts.append("Your outputs must contain:\n" + get_field_spec_description_string(task_spec.output_fields))
        parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")
        if task_spec.instructions:
            parts.append(f"Specific instructions: {task_spec.instructions}")
        return "\n".join(parts)

    @override
    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        _ = main_request
        parts = [prefix]
        parts.extend(f"{name}: {inputs.get(name, '')}" for name in task_spec.input_fields if name in inputs)
        parts.append(suffix)
        return "\n\n".join(parts).strip()

    @override
    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        parts = [
            f"{name}: {outputs.get(name, missing_field_message)}" for name in task_spec.output_fields if name in outputs
        ]
        return "\n\n".join(parts).strip()

    def _create_extractor_task_spec(self, original_task_spec: TaskSpec) -> TaskSpec:
        new_fields = {
            "text": input_field(
                "text", str, desc="Raw completion text from the main language model to extract structured fields from."
            ),
            **dict(original_task_spec.output_fields),
        }
        outputs_str = ", ".join(f"`{field}`" for field in original_task_spec.output_fields)
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."
        return make_task_spec(new_fields, instructions=instructions, name="framework.two_step.extractor")

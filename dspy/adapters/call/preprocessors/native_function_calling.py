from __future__ import annotations

from typing import TYPE_CHECKING, cast

from dspy.core.types import LMConfig, LMToolChoice, coerce_tool_spec

if TYPE_CHECKING:
    from dspy.adapters.call.preprocessors.context import PreprocessState


class NativeFunctionCallingPreprocessor:
    def run(self, state: PreprocessState) -> PreprocessState:
        adapter = state.adapter
        config = cast("LMConfig", state.config)
        if not adapter.use_native_function_calling:
            if config.tool_choice is not None:
                state.config = config.model_copy(update={"tool_choice": None})
            return state
        tool_call_input_field_name = adapter._get_tool_call_input_field_name(state.task_spec)
        tool_call_output_field_name = adapter._get_tool_call_output_field_name(state.task_spec)
        if not tool_call_output_field_name:
            return state
        if tool_call_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, but did not provide any tools as the input. Please provide a list of tools as the input by adding an input field with type `list[dspy.adapters.types.tool.Tool]`."
            )
        if not state.lm.supports_function_calling:
            raise ValueError(
                f"Adapter {type(adapter).__name__} has use_native_function_calling=True but "
                f"model {state.lm.model!r} does not support function calling. "
                "Use an LM with supports_function_calling=True or disable native function calling."
            )
        input_tools = state.inputs[tool_call_input_field_name]
        input_tools = input_tools if isinstance(input_tools, list) else [input_tools]
        state.tools = [coerce_tool_spec(tool) for tool in input_tools]
        if adapter.parallel_tool_calls is not None:
            if config.tool_choice is None:
                state.config = config.model_copy(
                    update={"tool_choice": LMToolChoice(mode="auto", parallel=adapter.parallel_tool_calls)}
                )
            elif config.tool_choice.parallel is None:
                state.config = config.model_copy(
                    update={
                        "tool_choice": config.tool_choice.model_copy(update={"parallel": adapter.parallel_tool_calls})
                    }
                )
        state.task_spec = state.task_spec.delete(tool_call_output_field_name)
        state.task_spec = state.task_spec.delete(tool_call_input_field_name)
        return state

from __future__ import annotations

from typing import Any, get_args, get_origin

from dspy.adapters.base.protocols import ComposedAdapterT
from dspy.adapters.base.task_specs import ToolCallResultsTaskSpec
from dspy.adapters.base.tool_calls import (
    _tool_call_as_openai_message_tool_call,
    _tool_calls_from_message,
    _tool_result_content,
)
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import build_lm_message
from dspy.core.types import LMMessage
from dspy.history.turn_event import TurnEvent
from dspy.history.turn_log import is_turn_log_type
from dspy.task_spec import TaskSpec


class AdapterConversationMixin:
    def _get_turn_log_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.input_fields.items():
            if is_turn_log_type(field.type_):
                return name
        return None

    def _get_tool_call_input_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.input_fields.items():
            field_type = field.type_
            origin = get_origin(field_type)
            if origin is list and get_args(field_type)[0] == Tool:
                return name
            if field_type == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.output_fields.items():
            if field.type_ == ToolCalls:
                return name
        return None

    def format_conversation_history(
        self: ComposedAdapterT, task_spec: TaskSpec, turn_log_field_name: str, inputs: dict[str, Any]
    ) -> list[LMMessage]:
        turn_log = inputs.get(turn_log_field_name)
        conversation_history = turn_log.turns if turn_log is not None else None
        if conversation_history is None:
            return []
        messages = []
        for turn in conversation_history:
            message = turn.to_dict() if isinstance(turn, TurnEvent) else turn
            tool_call_field_name, tool_calls = _tool_calls_from_message(message)
            tool_call_results = (
                ToolCallResults.model_validate(tool_calls.tool_call_results)
                if tool_calls is not None and tool_calls.tool_call_results is not None
                else None
            )
            user_content = self.format_user_message_content(task_spec=task_spec, inputs=message)
            if user_content:
                messages.append(build_lm_message(role="user", content=user_content))
            if self.use_native_function_calling and tool_calls is not None:
                content_task_spec = task_spec
                for name, field in task_spec.output_fields.items():
                    if field.type_ == ToolCalls or message.get(name) is None:
                        content_task_spec = content_task_spec.delete(name)
                content = (
                    self.format_assistant_message_content(task_spec=content_task_spec, outputs=message)
                    if content_task_spec.output_fields
                    else ""
                )
                if tool_call_results is not None:
                    tool_call_ids = [tool_call.id for tool_call in tool_calls.tool_calls]
                    result_ids = [result.call_id for result in tool_call_results.tool_call_results]
                    if tool_call_ids != result_ids or not all(tool_call_ids):
                        tool_call_results = None
                if content or tool_call_results is not None:
                    assistant_kwargs: dict[str, Any] = {"content": content or None}
                    if tool_call_results is not None:
                        assistant_kwargs["tool_calls"] = [
                            _tool_call_as_openai_message_tool_call(tool_call) for tool_call in tool_calls.tool_calls
                        ]
                    messages.append(build_lm_message("assistant", **assistant_kwargs))
                if tool_call_results is not None:
                    for result in tool_call_results.tool_call_results:
                        content = _tool_result_content(result.value)
                        messages.append(
                            build_lm_message(
                                role="tool", content=content, tool_call_id=result.call_id, name=result.name
                            )
                        )
                continue
            assistant_values = message
            if tool_call_field_name is not None and tool_calls is not None and (tool_call_results is not None):
                assistant_values = dict(message)
                assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})
            assistant_content = self.format_assistant_message_content(task_spec=task_spec, outputs=assistant_values)
            if assistant_content:
                messages.append(build_lm_message(role="assistant", content=assistant_content))
            if tool_call_results is not None:
                result_input = {"tool_call_results": tool_call_results}
                content = self.format_user_message_content(task_spec=ToolCallResultsTaskSpec(), inputs=result_input)
                messages.append(build_lm_message(role="user", content=content))
        inputs.pop(turn_log_field_name, None)
        return messages

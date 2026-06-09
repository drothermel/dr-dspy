"""Adapter message assembly: system, demos, conversation history, current user."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args, get_origin

from dspy.adapters.base.task_specs import ToolCallResultsTaskSpec
from dspy.adapters.base.tool_calls import (
    _tool_call_as_openai_message_tool_call,
    _tool_calls_from_message,
    _tool_result_content,
)
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import build_lm_message
from dspy.history.discovery import is_conversation_turn_log_type

if TYPE_CHECKING:
    from dspy.adapters.base.protocols import MessageAssemblerHost
    from dspy.core.types import LMMessage
    from dspy.task_spec import TaskSpec

MESSAGE_BUILD_ORDER = ("system", "demos", "conversation_history", "current_user")
INCOMPLETE_DEMO_PREFIX = "This is an example of the task, though some input or output fields are not supplied."


class MessageAssembler:
    def __init__(self, host: MessageAssemblerHost) -> None:
        self._host = host

    def get_turn_log_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.input_fields.items():
            if is_conversation_turn_log_type(field.type_):
                return name
        return None

    def get_tool_call_input_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.input_fields.items():
            field_type = field.type_
            origin = get_origin(field_type)
            if origin is list and get_args(field_type)[0] == Tool:
                return name
            if field_type == Tool:
                return name
        return None

    def get_tool_call_output_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.output_fields.items():
            if field.type_ == ToolCalls:
                return name
        return None

    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        inputs_copy = dict(inputs)
        turn_log_field_name = self.get_turn_log_field_name(task_spec)
        task_spec_without_history = task_spec
        conversation_history: list[LMMessage] = []
        if turn_log_field_name:
            task_spec_without_history = task_spec.delete(turn_log_field_name)
            conversation_history = self._host.format_conversation_history(
                task_spec=task_spec, turn_log_field_name=turn_log_field_name, inputs=inputs_copy
            )
        messages: list[LMMessage] = []
        self._append_system_message(messages=messages, task_spec=task_spec)
        self._append_demos(messages=messages, task_spec=task_spec, demos=demos)
        if turn_log_field_name:
            self._append_conversation_history(messages=messages, conversation_history=conversation_history)
            self._append_current_user_message(
                messages=messages,
                task_spec=task_spec_without_history,
                inputs=inputs_copy,
            )
        else:
            self._append_current_user_message(messages=messages, task_spec=task_spec, inputs=inputs_copy)
        return messages

    def format_conversation_history(
        self,
        task_spec: TaskSpec,
        turn_log_field_name: str,
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
        turn_log = inputs.get(turn_log_field_name)
        conversation_history = turn_log.turns if turn_log is not None else None
        if conversation_history is None:
            return []
        messages = []
        host = self._host
        for turn in conversation_history:
            message = turn.model_dump(mode="json", exclude_none=True)
            if turn.tool_calls is not None:
                tool_call_field_name, tool_calls = "tool_calls", turn.tool_calls
            else:
                tool_call_field_name, tool_calls = _tool_calls_from_message(message)
            tool_call_results = (
                ToolCallResults.model_validate(tool_calls.tool_call_results)
                if tool_calls is not None and tool_calls.tool_call_results is not None
                else None
            )
            user_content = host.format_user_message_content(task_spec=task_spec, inputs=message)
            if user_content:
                messages.append(build_lm_message(role="user", content=user_content))
            if host.use_native_function_calling and tool_calls is not None:
                content_task_spec = task_spec
                for name, field in task_spec.output_fields.items():
                    if field.type_ == ToolCalls or message.get(name) is None:
                        content_task_spec = content_task_spec.delete(name)
                content = (
                    host.format_assistant_message_content(task_spec=content_task_spec, outputs=message)
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
            assistant_content = host.format_assistant_message_content(task_spec=task_spec, outputs=assistant_values)
            if assistant_content:
                messages.append(build_lm_message(role="assistant", content=assistant_content))
            if tool_call_results is not None:
                result_input = {"tool_call_results": tool_call_results}
                content = host.format_user_message_content(task_spec=ToolCallResultsTaskSpec(), inputs=result_input)
                messages.append(build_lm_message(role="user", content=content))
        inputs.pop(turn_log_field_name, None)
        return messages

    def format_demos(self, task_spec: TaskSpec, demos: list[dict[str, Any]]) -> list[LMMessage]:
        host = self._host
        complete_demos = []
        incomplete_demos = []
        for demo in demos:
            is_complete = all(k in demo and demo[k] is not None for k in task_spec.fields)
            has_input = any(k in demo for k in task_spec.input_fields)
            has_output = any(k in demo for k in task_spec.output_fields)
            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                incomplete_demos.append(demo)
        messages = []
        for demo in incomplete_demos:
            messages.append(
                build_lm_message(
                    role="user",
                    content=host.format_user_message_content(
                        task_spec=task_spec, inputs=demo, prefix=INCOMPLETE_DEMO_PREFIX
                    ),
                )
            )
            messages.append(
                build_lm_message(
                    role="assistant",
                    content=host.format_assistant_message_content(
                        task_spec=task_spec,
                        outputs=demo,
                        missing_field_message="Not supplied for this particular example. ",
                    ),
                )
            )
        for demo in complete_demos:
            messages.append(
                build_lm_message(
                    role="user", content=host.format_user_message_content(task_spec=task_spec, inputs=demo)
                )
            )
            messages.append(
                build_lm_message(
                    role="assistant",
                    content=host.format_assistant_message_content(
                        task_spec=task_spec,
                        outputs=demo,
                        missing_field_message="Not supplied for this conversation history message. ",
                    ),
                )
            )
        return messages

    def _append_system_message(self, *, messages: list[LMMessage], task_spec: TaskSpec) -> None:
        system_message = self._host.format_system_message(task_spec)
        messages.append(build_lm_message(role="system", content=system_message))

    def _append_demos(self, *, messages: list[LMMessage], task_spec: TaskSpec, demos: list[dict[str, Any]]) -> None:
        messages.extend(self.format_demos(task_spec=task_spec, demos=demos))

    def _append_conversation_history(self, *, messages: list[LMMessage], conversation_history: list[LMMessage]) -> None:
        messages.extend(conversation_history)

    def _append_current_user_message(
        self,
        *,
        messages: list[LMMessage],
        task_spec: TaskSpec,
        inputs: dict[str, Any],
    ) -> None:
        content = self._host.format_user_message_content(task_spec=task_spec, inputs=inputs, main_request=True)
        if content:
            messages.append(build_lm_message(role="user", content=content))

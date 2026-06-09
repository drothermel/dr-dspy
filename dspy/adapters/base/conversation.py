from __future__ import annotations

from typing import Any

from dspy.adapters.base.protocols import MessageAssemblerHost
from dspy.core.types import LMMessage
from dspy.task_spec import TaskSpec


class AdapterConversationMixin:
    def _get_turn_log_field_name(self: MessageAssemblerHost, task_spec: TaskSpec) -> str | None:
        return self.message_assembler.get_turn_log_field_name(task_spec)

    def _get_tool_call_input_field_name(self: MessageAssemblerHost, task_spec: TaskSpec) -> str | None:
        return self.message_assembler.get_tool_call_input_field_name(task_spec)

    def _get_tool_call_output_field_name(self: MessageAssemblerHost, task_spec: TaskSpec) -> str | None:
        return self.message_assembler.get_tool_call_output_field_name(task_spec)

    def format_conversation_history(
        self: MessageAssemblerHost,
        task_spec: TaskSpec,
        turn_log_field_name: str,
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
        return self.message_assembler.format_conversation_history(
            task_spec=task_spec, turn_log_field_name=turn_log_field_name, inputs=inputs
        )

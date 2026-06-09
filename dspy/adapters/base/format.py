"""Adapter message formatting.

Canonical message ordering: system, few-shot demos, live conversation history,
then the current user request (see ``MESSAGE_BUILD_ORDER``).

``TurnLog`` fields expand into interleaved user/assistant messages.
``REPLHistory`` fields stay inline in the current user block as a formatted
string via Pydantic serialization (not conversation expansion).
"""

from __future__ import annotations

from typing import Any

from dspy.adapters.base.conversation import AdapterConversationMixin
from dspy.adapters.base.protocols import MessageAssemblerHost
from dspy.adapters.format.message_assembler import MESSAGE_BUILD_ORDER
from dspy.core.types import LMMessage, UserMessageContent
from dspy.task_spec import TaskSpec

__all__ = ["MESSAGE_BUILD_ORDER", "AdapterFormatMixin"]


class AdapterFormatMixin(AdapterConversationMixin):
    def format(
        self: MessageAssemblerHost, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[LMMessage]:
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

    def format_demos(self: MessageAssemblerHost, task_spec: TaskSpec, demos: list[dict[str, Any]]) -> list[LMMessage]:
        return self.message_assembler.format_demos(task_spec=task_spec, demos=demos)

"""Adapter message formatting.

Canonical message ordering: system, few-shot demos, live conversation history,
then the current user request (see ``MESSAGE_BUILD_ORDER``).

``TurnLog`` fields expand into interleaved user/assistant messages.
``REPLHistory`` fields stay inline in the current user block as a formatted
string via Pydantic serialization (not conversation expansion).
"""

from __future__ import annotations

from typing import Any, cast

from dspy.adapters.base.conversation import AdapterConversationMixin
from dspy.adapters.base.protocols import ConversationFormattingAdapter
from dspy.adapters.utils import build_lm_message
from dspy.core.types import LMMessage, UserMessageContent
from dspy.task_spec import TaskSpec

MESSAGE_BUILD_ORDER = ("system", "demos", "conversation_history", "current_user")


class AdapterFormatMixin(AdapterConversationMixin):
    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        inputs_copy = dict(inputs)
        turn_log_field_name = self._get_turn_log_field_name(task_spec)
        task_spec_without_history = task_spec
        conversation_history: list[LMMessage] = []
        if turn_log_field_name:
            task_spec_without_history = task_spec.delete(turn_log_field_name)
            conversation_history = cast("ConversationFormattingAdapter", self).format_conversation_history(
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

    def _append_system_message(self, *, messages: list[LMMessage], task_spec: TaskSpec) -> None:
        system_message = self.format_system_message(task_spec)
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
        content = self.format_user_message_content(task_spec=task_spec, inputs=inputs, main_request=True)
        if content:
            messages.append(build_lm_message(role="user", content=content))

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
        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.append(
                build_lm_message(
                    role="user",
                    content=self.format_user_message_content(
                        task_spec=task_spec, inputs=demo, prefix=incomplete_demo_prefix
                    ),
                )
            )
            messages.append(
                build_lm_message(
                    role="assistant",
                    content=self.format_assistant_message_content(
                        task_spec=task_spec,
                        outputs=demo,
                        missing_field_message="Not supplied for this particular example. ",
                    ),
                )
            )
        for demo in complete_demos:
            messages.append(
                build_lm_message(
                    role="user", content=self.format_user_message_content(task_spec=task_spec, inputs=demo)
                )
            )
            messages.append(
                build_lm_message(
                    role="assistant",
                    content=self.format_assistant_message_content(
                        task_spec=task_spec,
                        outputs=demo,
                        missing_field_message="Not supplied for this conversation history message. ",
                    ),
                )
            )
        return messages

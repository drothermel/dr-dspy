from __future__ import annotations

from typing import Any

from dspy.adapters.base.native import AdapterMixinBase
from dspy.adapters.utils import build_lm_message
from dspy.core.types import LMMessage
from dspy.task_spec import TaskSpec


class AdapterFormatMixin(AdapterMixinBase):
    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        inputs_copy = dict(inputs)
        turn_log_field_name = self._get_turn_log_field_name(task_spec)
        task_spec_without_history = task_spec
        conversation_history: list[LMMessage] = []
        if turn_log_field_name:
            task_spec_without_history = task_spec.delete(turn_log_field_name)
            conversation_history = self.format_conversation_history(
                task_spec=task_spec, turn_log_field_name=turn_log_field_name, inputs=inputs_copy
            )
        messages: list[LMMessage] = []
        system_message = self.format_system_message(task_spec)
        messages.append(build_lm_message(role="system", content=system_message))
        messages.extend(self.format_demos(task_spec=task_spec, demos=demos))
        if turn_log_field_name:
            content = self.format_user_message_content(
                task_spec=task_spec_without_history, inputs=inputs_copy, main_request=True
            )
            messages.extend(conversation_history)
            if content:
                messages.append(build_lm_message(role="user", content=content))
        else:
            content = self.format_user_message_content(task_spec=task_spec, inputs=inputs_copy, main_request=True)
            if content:
                messages.append(build_lm_message(role="user", content=content))
        return messages

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
    ) -> str | list[dict[str, Any]]:
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

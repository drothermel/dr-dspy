from __future__ import annotations

from typing import Any

from dspy.adapters.base.native import AdapterMixinBase
from dspy.adapters.utils import build_lm_message
from dspy.core.types import LMMessage
from dspy.task_spec import TaskSpec


class AdapterFormatMixin(AdapterMixinBase):
    def format(
        self,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
        """Format the input messages for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
        the formatting of the input messages.

        In general we recommend the messages to have the following structure:
        ```
        [
            {"role": "system", "content": system_message},
            # Begin few-shot examples
            {"role": "user", "content": few_shot_example_1_input},
            {"role": "assistant", "content": few_shot_example_1_output},
            {"role": "user", "content": few_shot_example_2_input},
            {"role": "assistant", "content": few_shot_example_2_output},
            ...
            # End few-shot examples
            # Begin conversation history
            {"role": "user", "content": conversation_history_1_input},
            {"role": "assistant", "content": conversation_history_1_output},
            {"role": "user", "content": conversation_history_2_input},
            {"role": "assistant", "content": conversation_history_2_output},
            ...
            # End conversation history
            {"role": "user", "content": current_input},
        ]

        And system message should contain the field description, field structure, and task description.
        ```


        Args:
            task_spec: The DSPy task spec for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        inputs_copy = dict(inputs)

        # Render conversation history as prior messages; omit the History field from history/current user content while keeping the original task spec for system instructions.
        history_field_name = self._get_history_field_name(task_spec)
        task_spec_without_history = task_spec
        conversation_history: list[LMMessage] = []
        if history_field_name:
            task_spec_without_history = task_spec.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                task_spec=task_spec,
                history_field_name=history_field_name,
                inputs=inputs_copy,
            )

        messages: list[LMMessage] = []
        system_message = self.format_system_message(task_spec)
        messages.append(build_lm_message(role="system", content=system_message))
        messages.extend(self.format_demos(task_spec=task_spec, demos=demos))
        if history_field_name:
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
        """Format the system message for the LM call.


        Args:
            task_spec: The DSPy task spec for which to format the system message.
        """
        return (
            f"{self.format_field_description(task_spec)}\n"
            f"{self.format_field_structure(task_spec)}\n"
            f"{self.format_task_description(task_spec)}"
        )

    def format_field_description(self, task_spec: TaskSpec) -> str:
        """Format the field description for the system message.

        This method formats the field description for the system message. It should return a string that contains
        the field description for the input fields and the output fields.

        Args:
            task_spec: The DSPy task spec for which to format the field description.

        Returns:
            A string that contains the field description for the input fields and the output fields.
        """
        raise NotImplementedError

    def format_field_structure(self, task_spec: TaskSpec) -> str:
        """Format the field structure for the system message.

        This method formats the field structure for the system message. It should return a string that dictates the
        format the input fields should be provided to the LM, and the format the output fields will be in the response.
        Refer to the ChatAdapter and JsonAdapter for an example.

        Args:
            task_spec: The DSPy task spec for which to format the field structure.
        """
        raise NotImplementedError

    def format_task_description(self, task_spec: TaskSpec) -> str:
        """Format the task description for the system message.

        This method formats the task description for the system message. In most cases this is just a thin wrapper
        over `signature.instructions`.

        Args:
            task_spec: The DSPy task spec of the DSpy module.

        Returns:
            A string that describes the task.
        """
        raise NotImplementedError

    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | list[dict[str, Any]]:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            task_spec: The DSPy task spec for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            User message content as a string or OpenAI-style content blocks when inputs include custom types.
        """
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        task_spec: TaskSpec,
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format the assistant message content.

        This method formats the assistant message content, which can be used in formatting few-shot examples,
        conversation history.

        Args:
            task_spec: The DSPy task spec for which to format the assistant message content.
            outputs: The output fields to be formatted.
            missing_field_message: A message to be used when a field is missing.

        Returns:
            A string that contains the assistant message content.
        """
        raise NotImplementedError

    def format_demos(self, task_spec: TaskSpec, demos: list[dict[str, Any]]) -> list[LMMessage]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            task_spec: The DSPy task spec for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the task spec.

        Returns:
            A list of multiturn messages.
        """
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            is_complete = all(k in demo and demo[k] is not None for k in task_spec.fields)

            has_input = any(k in demo for k in task_spec.input_fields)
            has_output = any(k in demo for k in task_spec.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                # We only keep incomplete demos that have at least one input and one output field
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
                    role="user",
                    content=self.format_user_message_content(task_spec=task_spec, inputs=demo),
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

import asyncio
from collections.abc import Mapping
from typing import Any

import json_repair
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import build_lm_message
from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMConfig, LMMessage, LMRequest, LMToolCallPart, merge_lm_request_config
from dspy.task_spec import FieldSpec, TaskSpec, make_task_spec
from dspy.task_spec.formatting import get_field_spec_description_string
from dspy.utils.exceptions import AdapterParseError, LMError

"""
NOTE/TODO/FIXME:

The main issue below is that the second step's task spec is entirely created on the fly and is invoked with a chat
adapter explicitly constructed with no demonstrations. This means that it cannot "learn" or get optimized.
"""


class TwoStepAdapter(Adapter):
    """
    A two-stage adapter that:
        1. Uses a simpler, more natural prompt for the main LM
        2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
    This adapter uses a common __call__ logic defined in base Adapter class.
    This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
    are known to struggle with structured outputs.

    Examples:
    ```
    from dspy.adapters.two_step_adapter import TwoStepAdapter
    from dspy.clients.lm import LM
    from dspy.dsp.utils.settings import settings
    from dspy.predict.chain_of_thought import ChainOfThought

    lm = LM(model="openai/o3-mini", max_tokens=16000, temperature = 1.0)
    adapter = TwoStepAdapter(LM("openai/gpt-4o-mini"))
    settings.configure(lm=lm, adapter=adapter)
    program = ChainOfThought("question->answer")
    result = program("What is the capital of France?")
    print(result)
    ```
    """

    def __init__(self, extraction_model: BaseLM, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(extraction_model, BaseLM):
            raise ValueError("extraction_model must be an instance of dspy.clients.base_lm.BaseLM")
        self.extraction_model = extraction_model

    @override
    def format(self, task_spec: TaskSpec, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[LMMessage]:
        """
        Format a prompt for the first stage with the main LM.
        This no specific structure is required for the main LM, we customize the format method
        instead of format_field_description or format_field_structure.

        Args:
            task_spec: The task spec of the original task
            demos: A list of demo examples
            inputs: The current input

        Returns:
            A list of messages to be passed to the main LM.
        """
        messages: list[LMMessage] = []

        task_description = self.format_task_description(task_spec)
        messages.append(build_lm_message(role="system", content=task_description))

        messages.extend(self.format_demos(task_spec=task_spec, demos=demos))

        messages.append(
            build_lm_message(
                role="user",
                content=self.format_user_message_content(task_spec=task_spec, inputs=inputs),
            )
        )

        return messages

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        """
        Use a smaller LM (extraction_model) with chat adapter to extract structured data
        from the raw completion text of the main LM.

        Args:
            task_spec: The task spec of the original task
            completion: The completion from the main LM

        Returns:
            A dictionary containing the extracted structured data.
        """
        extractor_task_spec = self._create_extractor_task_spec(task_spec)

        try:
            parsed_result = asyncio.run(
                ChatAdapter().acall(
                    lm=self.extraction_model,
                    config=LMConfig(),
                    task_spec=extractor_task_spec,
                    demos=[],
                    inputs={"text": completion},
                )
            )
            return parsed_result[0]

        except LMError:
            raise
        except Exception as e:
            raise AdapterParseError(
                adapter_name="TwoStepAdapter",
                task_spec=task_spec,
                lm_response=completion,
                message=f"Failed to parse response from the original completion: {e}",
            ) from e

    @override
    async def acall(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        from dspy.core.types import coerce_lm_config

        messages = self.format(task_spec=task_spec, demos=demos, inputs=inputs)
        request = LMRequest(
            model=lm.model,
            messages=messages,
            config=merge_lm_request_config(lm=lm, config=coerce_lm_config(config)),
        )
        response = await lm.acall(request)
        extractor_task_spec = self._create_extractor_task_spec(task_spec)

        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(task_spec)
        for output in response.outputs:
            output_logprobs = output.logprobs
            tool_calls = output.tool_calls
            text = output.text

            try:
                value = await ChatAdapter().acall(
                    lm=self.extraction_model,
                    config=LMConfig(),
                    task_spec=extractor_task_spec,
                    demos=[],
                    inputs={"text": text or ""},
                )
                value = value[0]

            except LMError:
                raise
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="TwoStepAdapter",
                    task_spec=task_spec,
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
        """Create a description of the task based on the task spec"""
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
        self,
        task_spec: TaskSpec,
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        parts = [
            f"{name}: {outputs.get(name, missing_field_message)}" for name in task_spec.output_fields if name in outputs
        ]

        return "\n\n".join(parts).strip()

    def _create_extractor_task_spec(
        self,
        original_task_spec: TaskSpec,
    ) -> TaskSpec:
        """Create a new task spec containing a new 'text' input field and all output fields.

        Args:
            original_task_spec: The original task spec to extract output fields from

        Returns:
            A new TaskSpec with a text input field and all output fields
        """
        new_fields = {
            "text": FieldSpec.input("text"),
            **dict(original_task_spec.output_fields),
        }

        outputs_str = ", ".join([f"`{field}`" for field in original_task_spec.output_fields])
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. \
            Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."

        return make_task_spec(new_fields, instructions=instructions, name="Extractor")

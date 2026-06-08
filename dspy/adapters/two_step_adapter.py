from typing import Any, cast

import json_repair
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import get_field_description_string
from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMMessage, LMRequest, LMToolCallPart
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature, make_signature
from dspy.utils.exceptions import AdapterParseError, LMError

"""
NOTE/TODO/FIXME:

The main issue below is that the second step's signature is entirely created on the fly and is invoked with a chat
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
    def format(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Format a prompt for the first stage with the main LM.
        This no specific structure is required for the main LM, we customize the format method
        instead of format_field_description or format_field_structure.

        Args:
            signature: The signature of the original task
            demos: A list of demo examples
            inputs: The current input

        Returns:
            A list of messages to be passed to the main LM.
        """
        messages = []

        task_description = self.format_task_description(signature)
        messages.append({"role": "system", "content": task_description})

        messages.extend(self.format_demos(signature, demos))

        messages.append({"role": "user", "content": self.format_user_message_content(signature, inputs)})

        return messages

    @override
    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """
        Use a smaller LM (extraction_model) with chat adapter to extract structured data
        from the raw completion text of the main LM.

        Args:
            signature: The signature of the original task
            completion: The completion from the main LM

        Returns:
            A dictionary containing the extracted structured data.
        """
        extractor_signature = self._create_extractor_signature(signature)

        try:
            parsed_result = ChatAdapter()(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": completion},
            )
            return parsed_result[0]

        except LMError:
            raise
        except Exception as e:
            raise AdapterParseError(
                adapter_name="TwoStepAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse response from the original completion: {e}",
            ) from e

    @override
    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages = self.format(signature, demos, inputs)
        request = LMRequest.from_call(
            model=lm.model,
            messages=cast("list[dict[str, Any] | LMMessage]", messages),
            **{**lm.kwargs, **lm_kwargs},
        )
        response = await lm.acall(request)
        extractor_signature = self._create_extractor_signature(signature)

        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
        for output in response.outputs:
            output_logprobs = output.logprobs
            tool_calls = output.tool_calls
            text = output.text

            try:
                value = await ChatAdapter().acall(
                    lm=self.extraction_model,
                    lm_kwargs={},
                    signature=extractor_signature,
                    demos=[],
                    inputs={"text": text or ""},
                )
                value = value[0]

            except LMError:
                raise
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="TwoStepAdapter",
                    signature=signature,
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
    def format_task_description(self, signature: type[Signature]) -> str:
        """Create a description of the task based on the signature"""
        parts = []

        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append("As input, you will be provided with:\n" + get_field_description_string(signature.input_fields))
        parts.append("Your outputs must contain:\n" + get_field_description_string(signature.output_fields))
        parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")

        if signature.instructions:
            parts.append(f"Specific instructions: {signature.instructions}")

        return "\n".join(parts)

    @override
    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        _ = main_request
        parts = [prefix]

        parts.extend(f"{name}: {inputs.get(name, '')}" for name in signature.input_fields if name in inputs)

        parts.append(suffix)
        return "\n\n".join(parts).strip()

    @override
    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        parts = [
            f"{name}: {outputs.get(name, missing_field_message)}" for name in signature.output_fields if name in outputs
        ]

        return "\n\n".join(parts).strip()

    def _create_extractor_signature(
        self,
        original_signature: type[Signature],
    ) -> type[Signature]:
        """Create a new signature containing a new 'text' input field and all output fields.

        Args:
            original_signature: The original signature to extract output fields from

        Returns:
            A new Signature type with a text input field and all output fields
        """
        new_fields = {
            "text": (str, InputField()),
            **{name: (field.annotation, field) for name, field in original_signature.output_fields.items()},
        }

        outputs_str = ", ".join([f"`{field}`" for field in original_signature.output_fields])
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. \
            Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."

        return make_signature(new_fields, instructions)  # ty: ignore[invalid-argument-type]

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin

import json_repair

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_chat_message,
    _expand_legacy_custom_type_markers_in_lm_message,
)
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.history import History
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import serialize_for_json
from dspy.core.types import (
    LMConfig,
    LMMessage,
    LMRequest,
    LMResponse,
    LMToolCallPart,
    LMToolChoice,
    LMToolSpec,
    _coerce_tool_spec,
    coerce_lm_config,
    merge_lm_request_config,
)
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature, make_signature
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dspy.clients.base_lm import BaseLM
    from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]
_TOOL_CALL_RESULTS_SIGNATURE = make_signature({"tool_call_results": (ToolCallResults, InputField())})


class Adapter:
    """Base Adapter class.

    The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs). It handles the
    complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.

    Key responsibilities:
        - Transform user inputs and signatures into properly formatted LM prompts, which also instructs the LM to format
            the response in a specific format.
        - Parse LM outputs into dictionaries matching the signature's output fields.
        - Enable/disable native LM features (function calling, citations, etc.) based on configuration.
        - Handle conversation history, few-shot examples, and custom type processing.

    The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming
    model for users.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> None:
        """
        Args:
            callbacks: List of callback functions to execute during `format()` and `parse()` methods. Callbacks can be
                used for logging, monitoring, or custom processing. Defaults to None (empty list).
            use_native_function_calling: Whether to enable native function calling capabilities when the LM supports it.
                If True, the adapter will automatically configure function calling when input fields contain
                `dspy.adapters.types.tool.Tool` or `list[dspy.adapters.types.tool.Tool]` types. Defaults to False.
            native_response_types: List of output field types that should be handled by native LM features rather than
                adapter parsing. For example, `dspy.adapters.types.citation.Citations` can be populated directly by citation APIs
                (e.g., Anthropic's citation feature). Defaults to `[Citations]`.
            parallel_tool_calls: Whether to request provider-side parallel tool-call generation when native function
                calling is active. If None, the adapter does not set the provider option. Defaults to None.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.parallel_tool_calls = parallel_tool_calls
        self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        from dspy.utils.callback import with_callbacks

        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def _call_preprocess(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> tuple[type[Signature], list[LMToolSpec], LMConfig]:
        # TODO(adapters-plan): This remains the pre-normalized planning hook. It
        # mutates `config` and returns only the render signature, which loses
        # information we will need for plan-driven rendering/parsing. The next
        # stacked PR should replace this with an explicit `_AdapterPlan` that
        # records deleted fields, native tools, native output fields, inserted
        # messages/parts, and LM config patches.
        if not isinstance(config, LMConfig):
            config = coerce_lm_config(config)
        tools: list[LMToolSpec] = []
        if not self.use_native_function_calling:
            if config.tool_choice is not None:
                config = config.model_copy(update={"tool_choice": None})
        else:
            tool_call_input_field_name = self._get_tool_call_input_field_name(signature)
            tool_call_output_field_name = self._get_tool_call_output_field_name(signature)

            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.adapters.types.tool.Tool]`."
                )

            if tool_call_output_field_name and lm.supports_function_calling:
                if tool_call_input_field_name is None:
                    raise ValueError("Tool call input field is required when native function calling is enabled.")
                input_tools = inputs[tool_call_input_field_name]
                input_tools = input_tools if isinstance(input_tools, list) else [input_tools]

                tools = [_coerce_tool_spec(tool.format_as_litellm_function_call()) for tool in input_tools]
                if self.parallel_tool_calls is not None:
                    if config.tool_choice is None:
                        config = config.model_copy(
                            update={"tool_choice": LMToolChoice(mode="auto", parallel=self.parallel_tool_calls)}
                        )
                    elif config.tool_choice.parallel is None:
                        config = config.model_copy(
                            update={
                                "tool_choice": config.tool_choice.model_copy(
                                    update={"parallel": self.parallel_tool_calls}
                                )
                            }
                        )

                signature = signature.delete(tool_call_output_field_name)
                signature = signature.delete(tool_call_input_field_name)

        # TODO(adapters-plan): Built-in/native response planning should move out
        # of `Type.adapt_to_native_lm_feature()` and into adapter-owned planning
        # renderers. Keep this compatibility hook for this boundary-only PR.
        for name, field in signature.output_fields.items():
            if (
                isinstance(field.annotation, type)
                and field.annotation in self.native_response_types
                and issubclass(field.annotation, Type)
            ):
                signature = field.annotation.adapt_to_native_lm_feature(signature, name, lm, config)

        return signature, tools, config

    def _call_postprocess(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        response: LMResponse,
        _lm: BaseLM,
        _config: LMConfig | Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in response.outputs:
            output_logprobs = output.logprobs
            tool_calls = output.tool_calls
            text = output.text

            if text is not None and not (tool_calls and tool_call_output_field_name):
                value = self.parse(processed_signature, text)
            elif tool_calls and tool_call_output_field_name:
                try:
                    value = self.parse(processed_signature, text) if text and processed_signature.output_fields else {}
                except AdapterParseError:
                    value = {}
            elif text is None and not processed_signature.output_fields:
                value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            # Fields removed for native features are absent from the processed parse.
            for field_name in original_signature.output_fields:
                value.setdefault(field_name, None)

            if tool_calls and tool_call_output_field_name:
                tool_calls = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            # TODO(adapter-types): Once `Type.parse_lm_output(context, output)` is
            # the real normalized hook, this should not call the legacy
            # provider-shaped `parse_lm_response()` directly.
            for name, field in original_signature.output_fields.items():
                if (
                    isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)
                ):
                    parsed_value = field.annotation.parse_lm_output(output)
                    if parsed_value is not None:
                        value[name] = parsed_value

            if output_logprobs:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values

    def _render_request(
        self,
        lm: BaseLM,
        config: LMConfig,
        tools: list[LMToolSpec],
        messages: Sequence[LMMessage | dict[str, Any]],
    ) -> LMRequest:
        """Build the normalized LM request for the current adapter call path.

        TODO(adapters-plan): This currently receives already-rendered messages.
        Once planning lands, this should render from `_AdapterPlan` and apply
        planned message/part insertions before creating `LMRequest`.
        """
        return LMRequest(
            model=lm.model,
            messages=self._coerce_lm_messages(messages),
            tools=tools,
            config=merge_lm_request_config(lm, config),
        )

    def _call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        return lm(request)

    async def _acall_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        return await lm.acall(request)

    def _coerce_lm_messages(self, messages: Sequence[LMMessage | dict[str, Any]]) -> list[LMMessage]:
        """Normalize subclass `format()` output before the LM boundary.

        TODO(adapters-normalized-rendering): Adapter `format()` methods still
        return OpenAI-chat-shaped dictionaries. This coercion is the bridge until
        adapters render `LMMessage` / `LMPart` directly.
        """
        return [
            _expand_legacy_custom_type_markers_in_lm_message(
                message if isinstance(message, LMMessage) else self._chat_dict_to_lm_message(message)
            )
            for message in messages
        ]

    def _chat_dict_to_lm_message(self, message: dict[str, Any]) -> LMMessage:
        try:
            return LMMessage(**message)
        except Exception:
            # TODO(legacy-custom-types): Unknown OpenAI content blocks are
            # temporarily preserved as `legacy_content_block` metadata on an
            # empty text part so `openai_format` can round-trip them back to
            # current BaseLM calls. Replace this with either explicit opaque
            # provider parts or remove it when marker-based custom type
            # serialization is retired.
            message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                sanitized = []
                supported = {"text", "image_url", "input_audio", "file", "document", "video"}
                for block in content:
                    if isinstance(block, dict) and block.get("type") in supported:
                        sanitized.append(block)
                    elif isinstance(block, dict):
                        sanitized.append({"type": "text", "text": "", "metadata": {"legacy_content_block": block}})
                    else:
                        sanitized.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
                message["content"] = sanitized
            return LMMessage(**message)

    def __call__(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute the adapter pipeline: format inputs, call LM, and parse outputs.

        Args:
            lm: The Language Model instance to use for generation. Must be an instance of
                `dspy.clients.base_lm.BaseLM`.
            config: Generation controls for the LM call (e.g., temperature, max_tokens).
            signature: The DSPy signature associated with this LM call.
            demos: List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the
                signature's input and output field names. Examples are formatted as user/assistant message pairs.
            inputs: The current input values for this call. Keys must match the signature's input field names.

        Returns:
            List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the
            signature's output field names. For multiple generations (n > 1), returns multiple dictionaries.
        """
        resolved_config = coerce_lm_config(config)
        processed_signature, tools, resolved_config = self._call_preprocess(lm, resolved_config, signature, inputs)
        messages = self.format(processed_signature, demos, inputs)
        request = self._render_request(lm, resolved_config, tools, messages)
        response = self._call_lm(lm, request)
        # TODO(adapters-response): We normalize at the LM boundary, but still
        # convert back to legacy postprocess dictionaries here to keep this PR
        # behavior-preserving. Replace with direct `LMResponse` parsing once the
        # explicit adapter plan exists.
        return self._call_postprocess(processed_signature, signature, response, lm, resolved_config)

    async def acall(
        self,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resolved_config = coerce_lm_config(config)
        processed_signature, tools, resolved_config = self._call_preprocess(lm, resolved_config, signature, inputs)
        messages = self.format(processed_signature, demos, inputs)
        request = self._render_request(lm, resolved_config, tools, messages)
        response = await self._acall_lm(lm, request)
        # TODO(adapters-response): Keep in sync with `__call__()` until both use
        # direct `LMResponse` parsing.
        return self._call_postprocess(processed_signature, signature, response, lm, resolved_config)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
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
            signature: The DSPy signature for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        inputs_copy = dict(inputs)

        # Render conversation history as prior messages; omit the History field from history/current user content while keeping the original signature for system instructions.
        history_field_name = self._get_history_field_name(signature)
        signature_without_history = signature
        conversation_history: list[dict[str, Any]] = []
        if history_field_name:
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )

        messages: list[dict[str, Any]] = []
        system_message = self.format_system_message(signature)
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        if history_field_name:
            content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
            messages.extend(conversation_history)
            if content:
                messages.append({"role": "user", "content": content})
        else:
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
            if content:
                messages.append({"role": "user", "content": content})

        return [_expand_legacy_custom_type_markers_in_chat_message(message) for message in messages]

    def format_system_message(self, signature: type[Signature]) -> str:
        """Format the system message for the LM call.


        Args:
            signature: The DSPy signature for which to format the system message.
        """
        return (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format the field description for the system message.

        This method formats the field description for the system message. It should return a string that contains
        the field description for the input fields and the output fields.

        Args:
            signature: The DSPy signature for which to format the field description.

        Returns:
            A string that contains the field description for the input fields and the output fields.
        """
        raise NotImplementedError

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the field structure for the system message.

        This method formats the field structure for the system message. It should return a string that dictates the
        format the input fields should be provided to the LM, and the format the output fields will be in the response.
        Refer to the ChatAdapter and JsonAdapter for an example.

        Args:
            signature: The DSPy signature for which to format the field structure.
        """
        raise NotImplementedError

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description for the system message.

        This method formats the task description for the system message. In most cases this is just a thin wrapper
        over `signature.instructions`.

        Args:
            signature: The DSPy signature of the DSpy module.

        Returns:
            A string that describes the task.
        """
        raise NotImplementedError

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            signature: The DSPy signature for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            A string that contains the user message content.
        """
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format the assistant message content.

        This method formats the assistant message content, which can be used in formatting few-shot examples,
        conversation history.

        Args:
            signature: The DSPy signature for which to format the assistant message content.
            outputs: The output fields to be formatted.
            missing_field_message: A message to be used when a field is missing.

        Returns:
            A string that contains the assistant message content.
        """
        raise NotImplementedError

    def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the signature.

        Returns:
            A list of multiturn messages.
        """
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                # We only keep incomplete demos that have at least one input and one output field
                incomplete_demos.append(demo)

        messages = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this particular example. "
                    ),
                }
            )

        for demo in complete_demos:
            messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this conversation history message. "
                    ),
                }
            )

        return messages

    def _get_history_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def _get_tool_call_input_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.input_fields.items():
            origin = get_origin(field.annotation)
            if origin is list and get_args(field.annotation)[0] == Tool:
                return name
            if field.annotation == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.output_fields.items():
            if field.annotation == ToolCalls:
                return name
        return None

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the conversation history.
            history_field_name: The name of the history field in the signature.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

        if conversation_history is None:
            return []

        messages = []
        for message in conversation_history:
            tool_call_field_name, tool_calls = _tool_calls_from_message(message)
            tool_call_results = (
                ToolCallResults.model_validate(tool_calls.tool_call_results)
                if tool_calls is not None and tool_calls.tool_call_results is not None
                else None
            )

            user_content = self.format_user_message_content(signature, message)
            if user_content:
                messages.append({"role": "user", "content": user_content})

            if self.use_native_function_calling and tool_calls is not None:
                content_signature = signature
                for name, field in signature.output_fields.items():
                    if field.annotation == ToolCalls or message.get(name) is None:
                        content_signature = content_signature.delete(name)

                content = (
                    self.format_assistant_message_content(content_signature, message)
                    if content_signature.output_fields
                    else ""
                )

                if tool_call_results is not None:
                    tool_call_ids = [tool_call.id for tool_call in tool_calls.tool_calls]
                    result_ids = [result.call_id for result in tool_call_results.tool_call_results]
                    if tool_call_ids != result_ids or not all(tool_call_ids):
                        tool_call_results = None

                if content or tool_call_results is not None:
                    assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
                    if tool_call_results is not None:
                        assistant_message["tool_calls"] = [
                            _tool_call_as_openai_message_tool_call(tool_call) for tool_call in tool_calls.tool_calls
                        ]
                    messages.append(assistant_message)

                if tool_call_results is not None:
                    for result in tool_call_results.tool_call_results:
                        content = _tool_result_content(result.value)
                        messages.append(
                            {"role": "tool", "tool_call_id": result.call_id, "name": result.name, "content": content}
                        )
                continue

            assistant_values = message
            if tool_call_field_name is not None and tool_calls is not None and tool_call_results is not None:
                assistant_values = dict(message)
                assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

            assistant_content = self.format_assistant_message_content(signature, assistant_values)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            if tool_call_results is not None:
                result_input = {"tool_call_results": tool_call_results}
                content = self.format_user_message_content(_TOOL_CALL_RESULTS_SIGNATURE, result_input)
                messages.append({"role": "user", "content": content})

        del inputs[history_field_name]

        return messages

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            signature: The DSPy signature for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError


def _provider_value(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        value = cast("dict[str, object]", value)
        return value.get(key, default)
    return getattr(value, key, default)


def _provider_tool_call_to_tool_call_dict(tool_call: object) -> dict[str, Any]:
    if isinstance(tool_call, LMToolCallPart):
        args = dict(tool_call.args)
        if not args:
            raw_arguments = tool_call.provider_data.get("raw_arguments") or tool_call.provider_data.get("arguments")
            if isinstance(raw_arguments, str):
                args = json_repair.loads(raw_arguments)
        return {"id": tool_call.id, "name": tool_call.name, "args": args}

    function = _provider_value(tool_call, "function", {}) or {}
    arguments = _provider_value(function, "arguments", {})
    if isinstance(arguments, str):
        parsed_arguments = json_repair.loads(arguments)
    elif isinstance(arguments, dict):
        parsed_arguments = arguments
    else:
        parsed_arguments = {}

    return {
        "id": _provider_value(tool_call, "id") or _provider_value(tool_call, "call_id"),
        "name": _provider_value(function, "name") or _provider_value(tool_call, "name"),
        "args": parsed_arguments,
    }


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    for name, value in message.items():
        if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value):
            return name, ToolCalls.model_validate(value)
    return None, None


def _tool_result_content(value: object) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(serialize_for_json(cast("Any", value)), ensure_ascii=False)


def _tool_call_as_openai_message_tool_call(tool_call: ToolCalls.ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(serialize_for_json(tool_call.args), ensure_ascii=False),
        },
    }

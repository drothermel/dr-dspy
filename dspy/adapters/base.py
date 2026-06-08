from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin

import json_repair

from dspy.adapters.types.base_type import Type
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.history import History
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import build_lm_message, serialize_for_json
from dspy.core.types import (
    LMConfig,
    LMMessage,
    LMReasoningConfig,
    LMRequest,
    LMResponse,
    LMToolCallPart,
    LMToolChoice,
    LMToolSpec,
    _coerce_tool_spec,
    coerce_lm_config,
    merge_lm_request_config,
)
from dspy.task_spec import FieldSpec, TaskSpec, make_task_spec
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dspy.clients.base_lm import BaseLM
    from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]
_TOOL_CALL_RESULTS_TASK_SPEC = make_task_spec(
    {"tool_call_results": FieldSpec.input("tool_call_results", type_=ToolCallResults)},
    instructions="Tool call results from conversation history.",
)


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
        task_spec: TaskSpec,
        inputs: dict[str, Any],
    ) -> tuple[TaskSpec, list[LMToolSpec], LMConfig]:
        if not isinstance(config, LMConfig):
            config = coerce_lm_config(config)
        tools: list[LMToolSpec] = []
        if not self.use_native_function_calling:
            if config.tool_choice is not None:
                config = config.model_copy(update={"tool_choice": None})
        else:
            tool_call_input_field_name = self._get_tool_call_input_field_name(task_spec)
            tool_call_output_field_name = self._get_tool_call_output_field_name(task_spec)

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

                tools = [_coerce_tool_spec(tool) for tool in input_tools]
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

                task_spec = task_spec.delete(tool_call_output_field_name)
                task_spec = task_spec.delete(tool_call_input_field_name)

        for name, field in task_spec.output_fields.items():
            field_type = field.type_
            if not (
                isinstance(field_type, type)
                and field_type in self.native_response_types
                and issubclass(field_type, Type)
            ):
                continue
            self._ensure_native_response_type_parses_output(field_type)
            if field_type is Reasoning:
                task_spec = self._adapt_reasoning_native(task_spec=task_spec, field_name=name, lm=lm, config=config)
            elif field_type is Citations:
                task_spec = self._adapt_citations_native(task_spec=task_spec, field_name=name, lm=lm)
            else:
                task_spec = task_spec.delete(name)

        return task_spec, tools, config

    @staticmethod
    def _ensure_native_response_type_parses_output(native_type: type[Type]) -> None:
        if native_type.parse_lm_output.__func__ is Type.parse_lm_output.__func__:
            raise TypeError(
                f"{native_type.__name__} is listed in native_response_types but does not implement "
                "parse_lm_output(). Native response fields must parse typed LMOutput values."
            )

    def _adapt_reasoning_native(
        self,
        task_spec: TaskSpec,
        field_name: str,
        lm: BaseLM,
        config: LMConfig,
    ) -> TaskSpec:
        if "reasoning" in config.model_fields_set and config.reasoning is None:
            return task_spec

        if config.reasoning is not None and config.reasoning.effort is not None:
            reasoning_effort = config.reasoning.effort
        elif isinstance(lm.kwargs.get("reasoning"), Mapping):
            reasoning_effort = lm.kwargs["reasoning"].get("effort")
        elif lm.kwargs.get("reasoning_effort") is not None:
            reasoning_effort = lm.kwargs["reasoning_effort"]
        else:
            reasoning_effort = "low"

        if reasoning_effort is None or not lm.supports_reasoning:
            return task_spec

        if "gpt-5" in lm.model and lm.model_type == "chat":
            return task_spec

        config.reasoning = LMReasoningConfig(effort=reasoning_effort)
        return task_spec.delete(field_name)

    def _adapt_citations_native(
        self,
        task_spec: TaskSpec,
        field_name: str,
        lm: BaseLM,
    ) -> TaskSpec:
        if lm.model.startswith("anthropic/"):
            return task_spec.delete(field_name)
        return task_spec

    def _call_postprocess(
        self,
        processed_task_spec: TaskSpec,
        original_task_spec: TaskSpec,
        response: LMResponse,
        _lm: BaseLM,
        _config: LMConfig | Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_task_spec)

        for output in response.outputs:
            output_logprobs = output.logprobs
            tool_calls = output.tool_calls
            text = output.text

            if text is not None and not (tool_calls and tool_call_output_field_name):
                value = self.parse(task_spec=processed_task_spec, completion=text)
            elif tool_calls and tool_call_output_field_name:
                try:
                    value = (
                        self.parse(task_spec=processed_task_spec, completion=text)
                        if text and processed_task_spec.output_fields
                        else {}
                    )
                except AdapterParseError:
                    value = {}
            elif text is None and not processed_task_spec.output_fields:
                value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    task_spec=original_task_spec,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            # Fields removed for native features are absent from the processed parse.
            for field_name in original_task_spec.output_fields:
                value.setdefault(field_name, None)

            if tool_calls and tool_call_output_field_name:
                tool_calls = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            for name, field in original_task_spec.output_fields.items():
                field_type = field.type_
                if (
                    isinstance(field_type, type)
                    and field_type in self.native_response_types
                    and issubclass(field_type, Type)
                ):
                    parsed_value = field_type.parse_lm_output(output)
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
        messages: Sequence[LMMessage],
    ) -> LMRequest:
        """Build the normalized LM request for the current adapter call path."""
        return LMRequest(
            model=lm.model,
            messages=list(messages),
            tools=tools,
            config=merge_lm_request_config(lm=lm, config=config),
        )

    async def _call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        return await lm(request)

    async def acall(
        self,
        *,
        lm: BaseLM,
        config: LMConfig | Mapping[str, Any] | None,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resolved_config = coerce_lm_config(config)
        processed_task_spec, tools, resolved_config = self._call_preprocess(
            lm=lm, config=resolved_config, task_spec=task_spec, inputs=inputs
        )
        messages = self.format(task_spec=processed_task_spec, demos=demos, inputs=inputs)
        request = self._render_request(lm=lm, config=resolved_config, tools=tools, messages=messages)
        response = await self._call_lm(lm=lm, request=request)
        return self._call_postprocess(
            processed_task_spec=processed_task_spec,
            original_task_spec=task_spec,
            response=response,
            _lm=lm,
            _config=resolved_config,
        )

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

    def _get_history_field_name(self, task_spec: TaskSpec) -> str | None:
        for name, field in task_spec.input_fields.items():
            if field.type_ == History:
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
        self,
        task_spec: TaskSpec,
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            task_spec: The DSPy task spec for which to format the conversation history.
            history_field_name: The name of the history field in the task spec.
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
                                role="tool",
                                content=content,
                                tool_call_id=result.call_id,
                                name=result.name,
                            )
                        )
                continue

            assistant_values = message
            if tool_call_field_name is not None and tool_calls is not None and tool_call_results is not None:
                assistant_values = dict(message)
                assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

            assistant_content = self.format_assistant_message_content(task_spec=task_spec, outputs=assistant_values)
            if assistant_content:
                messages.append(build_lm_message(role="assistant", content=assistant_content))
            if tool_call_results is not None:
                result_input = {"tool_call_results": tool_call_results}
                content = self.format_user_message_content(task_spec=_TOOL_CALL_RESULTS_TASK_SPEC, inputs=result_input)
                messages.append(build_lm_message(role="user", content=content))

        del inputs[history_field_name]

        return messages

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            task_spec: The DSPy task spec for which to parse the LM output.
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

    function = _provider_value(value=tool_call, key="function", default={}) or {}
    arguments = _provider_value(value=function, key="arguments", default={})
    if isinstance(arguments, str):
        parsed_arguments = json_repair.loads(arguments)
    elif isinstance(arguments, dict):
        parsed_arguments = arguments
    else:
        parsed_arguments = {}

    return {
        "id": _provider_value(value=tool_call, key="id") or _provider_value(value=tool_call, key="call_id"),
        "name": _provider_value(value=function, key="name") or _provider_value(value=tool_call, key="name"),
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

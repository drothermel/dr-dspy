"""Test-only LM double for unit and integration tests."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

from typing_extensions import override

from dspy._legacy.dotdict import dotdict
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.format_shared import FIELD_HEADER_PATTERN
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import provider_tool_call_to_part
from dspy.core.types import LMOutput, LMPart, LMRequest, LMResponse, LMTextPart, LMThinkingPart
from dspy.core.types.lm_provider import LMProviderOptions
from dspy.task_spec import FieldBinding, output_field


class DummyLM(BaseLM):
    def __init__(
        self,
        answers: list[dict[str, Any]] | dict[str, dict[str, Any]],
        follow_examples: bool = False,
        reasoning: bool = False,
        adapter=None,
    ) -> None:
        super().__init__(
            "dummy",
            "chat",
            temperature=0.0,
            max_tokens=1000,
            provider_options=LMProviderOptions(cache=False),
        )
        self.answers = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples
        self.reasoning = reasoning
        if adapter is None:
            adapter = ChatAdapter()
        self.adapter = adapter

    def _use_example(self, messages):
        fields = defaultdict(int)
        for message in messages:
            content = getattr(message, "text", None)
            if content and (ma := FIELD_HEADER_PATTERN.match(content)):
                fields[content[ma.start() : ma.end()]] += 1
        max_count = max(fields.values())
        output_fields = [field for field, count in fields.items() if count != max_count]
        final_input = (messages[-1].text or "").split("\n\n")[0]
        for input, output in zip(reversed(messages[:-1]), reversed(messages), strict=False):
            input_content = getattr(input, "text", "") or ""
            output_content = getattr(output, "text", "") or ""
            if any(field in output_content for field in output_fields) and final_input in input_content:
                return output_content
        return None

    @staticmethod
    def _field_spec_for_dummy_value(field_name: str, value: object):
        if isinstance(value, bool):
            return output_field(field_name, bool, desc="dummy")
        if isinstance(value, int):
            return output_field(field_name, int, desc="dummy")
        if isinstance(value, float):
            return output_field(field_name, float, desc="dummy")
        if isinstance(value, list):
            if value and all(isinstance(item, str) for item in value):
                return output_field(field_name, list[str], desc="dummy")
            return output_field(field_name, list[Any], desc="dummy")
        if isinstance(value, dict):
            return output_field(field_name, dict[str, Any], desc="dummy")
        return output_field(field_name, str, desc="dummy")

    def _format_answer_fields(self, field_names_and_values: dict[str, Any]):
        fields_with_values = {
            FieldBinding(name=field_name, field=self._field_spec_for_dummy_value(field_name, value)): value
            for field_name, value in field_names_and_values.items()
        }
        adapter = self.adapter
        role = adapter.capabilities.field_value_role
        if role == "assistant":
            return cast("Any", adapter).format_field_with_value(fields_with_values=fields_with_values, role="assistant")
        return adapter.format_field_with_value(fields_with_values=fields_with_values)

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        messages = request.messages
        kwargs = {**self.kwargs, **request.config.model_dump(exclude_none=True)}
        outputs = []
        for _ in range(kwargs.get("n", 1)):
            if self.follow_examples:
                current_output = self._use_example(messages)
            elif isinstance(self.answers, dict):
                answers = cast("dict[str, dict[str, Any]]", self.answers)
                last_message = messages[-1]
                last_content = getattr(last_message, "text", None)
                if last_content is None and isinstance(last_message, dict):
                    last_content = last_message.get("content")
                last_content_str = last_content if isinstance(last_content, str) else ""
                current_output = next(
                    (self._format_answer_fields(v) for k, v in answers.items() if k in last_content_str),
                    "No more responses",
                )
            else:
                answer_iter = cast("Iterator[dict[str, Any]]", self.answers)
                current_output = self._format_answer_fields(next(answer_iter, {"answer": "No more responses"}))
            outputs.append(self._to_output(current_output))
        return LMResponse(
            model="dummy", outputs=outputs, usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

    def _to_output(self, current_output: Any) -> LMOutput:
        if isinstance(current_output, dict):
            parts: list[LMPart] = []
            text = current_output.get("text")
            if isinstance(text, str):
                parts.append(LMTextPart(text=text))
            if self.reasoning and (not any(isinstance(part, LMThinkingPart) for part in parts)):
                parts.append(LMThinkingPart(text="Some reasoning"))
            reasoning_content = current_output.get("reasoning_content")
            if isinstance(reasoning_content, str):
                parts.append(LMThinkingPart(text=reasoning_content))
            parts.extend(provider_tool_call_to_part(tool_call) for tool_call in current_output.get("tool_calls") or [])
            return LMOutput(parts=parts, provider_output=current_output)
        if current_output is None:
            return LMOutput(parts=[])
        parts: list[LMPart] = [LMTextPart(text=str(current_output))]
        if self.reasoning:
            parts.append(LMThinkingPart(text="Some reasoning"))
        return LMOutput(parts=parts, provider_output=current_output)

    def get_convo(self, index):
        entry = self.call_log[index]
        return (entry.messages_as_openai, entry.outputs)

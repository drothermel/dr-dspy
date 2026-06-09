"""Test-only LM double for unit and integration tests."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format.chat_request import request_messages_as_openai
from dspy.core.types import LMRequest, LMResponse, LMUsage
from dspy.core.types.lm_provider import LMProviderOptions
from tests.test_utils.lm.answer_routing import DummyAnswers, resolve_answer
from tests.test_utils.lm.field_formatting import format_answer_fields
from tests.test_utils.lm.output_assembly import build_lm_output


class DummyLM(BaseLM):
    def __init__(
        self,
        answers: DummyAnswers,
        follow_examples: bool = False,
        reasoning: bool = False,
        supports_reasoning: bool = False,
        adapter=None,
    ) -> None:
        super().__init__(
            "dummy",
            "chat",
            temperature=0.0,
            max_tokens=1000,
            provider_options=LMProviderOptions(cache=False),
        )
        self.answers: DummyAnswers | Any = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples
        self.reasoning = reasoning
        self._supports_reasoning = supports_reasoning
        if adapter is None:
            adapter = ChatAdapter()
        self.adapter = adapter

    @property
    @override
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    def _format_fields(self, field_names_and_values: dict[str, Any]) -> Any:
        return format_answer_fields(self.adapter, field_names_and_values)

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        messages = request.messages
        kwargs = {**self.kwargs, **request.config.model_dump(exclude_none=True)}
        outputs = []
        for _ in range(kwargs.get("n", 1)):
            current_output = resolve_answer(
                answers=self.answers,
                messages=messages,
                follow_examples=self.follow_examples,
                format_fields=self._format_fields,
            )
            outputs.append(build_lm_output(current_output, reasoning=self.reasoning))
        return LMResponse(
            model="dummy", outputs=outputs, usage=LMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

    def get_convo(self, index):
        entry = self.call_log[index]
        return (request_messages_as_openai(entry.request), entry.outputs)

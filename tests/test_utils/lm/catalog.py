"""Shared LM doubles for common test scenarios."""

from __future__ import annotations

import json
from typing import Any

from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format.chat_request import to_openai_chat_request
from dspy.core.types import (
    LMOutput,
    LMRequest,
    LMResponse,
    LMToolCallPart,
    LMUsage,
    NativeAdaptationMode,
)


def captured_lm_kwargs(request: LMRequest) -> dict:
    data = to_openai_chat_request(request)
    data.pop("model", None)
    data.pop("messages", None)
    return data


class SequentialTextLM(BaseLM):
    def __init__(self, texts: list[str], *, model: str = "openai/gpt-4o-mini", temperature: float = 1.0):
        super().__init__(model, "chat", temperature=temperature, max_tokens=1000)
        self.texts = list(texts)
        self.requests: list[LMRequest] = []

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        self.requests.append(request)
        text = self.texts.pop(0) if self.texts else "No more responses"
        return LMResponse.from_text(text, model=self.model)


class FailingLM(BaseLM):
    def __init__(self, *, error: BaseException | None = None) -> None:
        super().__init__(
            "fail-lm",
            "chat",
            temperature=0.0,
            max_tokens=1000,
        )
        self.error = error or RuntimeError("LM failed")

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        raise self.error


class CapabilityStubLM(BaseLM):
    def __init__(
        self,
        *,
        reasoning_adaptation_mode: NativeAdaptationMode = NativeAdaptationMode.ADAPT,
        citations_adaptation_mode: NativeAdaptationMode = NativeAdaptationMode.ADAPT,
        supports_reasoning: bool = True,
    ) -> None:
        super().__init__(model="stub/test")
        self._reasoning_adaptation_mode = reasoning_adaptation_mode
        self._citations_adaptation_mode = citations_adaptation_mode
        self._supports_reasoning = supports_reasoning

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_adaptation_mode(self) -> NativeAdaptationMode:
        return self._reasoning_adaptation_mode

    @property
    def citations_adaptation_mode(self) -> NativeAdaptationMode:
        return self._citations_adaptation_mode


class NativeToolCallLM(BaseLM):
    def __init__(self, *, parallel_first_turn: bool = False):
        model = "parallel-native-tool-lm" if parallel_first_turn else "native-tool-lm"
        super().__init__(model, "chat", temperature=0.0, max_tokens=1000)
        self.parallel_first_turn = parallel_first_turn
        self.calls: list[dict[str, Any]] = []

    @property
    @override
    def supports_function_calling(self):
        return True

    def _tool_calls_for_turn(self, turn_index: int) -> list[dict[str, Any]]:
        if turn_index == 1:
            if self.parallel_first_turn:
                return [
                    {
                        "id": "call_provider_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query":"cats"}'},
                    },
                    {
                        "id": "call_provider_2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query":"dogs"}'},
                    },
                ]
            return [
                {
                    "id": "call_provider_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"cats"}'},
                }
            ]
        if self.parallel_first_turn:
            return [
                {
                    "id": "call_submit",
                    "type": "function",
                    "function": {"name": "submit", "arguments": '{"answer":"found cats and found dogs"}'},
                }
            ]
        return [
            {
                "id": "call_submit",
                "type": "function",
                "function": {"name": "submit", "arguments": '{"answer":"found cats"}'},
            }
        ]

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        self.calls.append({"messages": request.messages, "kwargs": captured_lm_kwargs(request)})
        tool_calls = self._tool_calls_for_turn(len(self.calls))
        return LMResponse(
            model=self.model,
            outputs=[
                LMOutput(
                    parts=[
                        LMToolCallPart(
                            id=tool_call["id"],
                            name=tool_call["function"]["name"],
                            args=json.loads(tool_call["function"]["arguments"]),
                        )
                        for tool_call in tool_calls
                    ],
                    finish_reason="tool_calls",
                )
            ],
            usage=LMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

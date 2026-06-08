from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.core.types import LMRequest, LMResponse


def make_response(output_blocks):
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="openai/dspy-test-model",
        object="response",
        output=output_blocks,
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        user=None,
    )


def _request(lm: BaseLM, *items: object, prompt: str | None = None, messages=None, **kwargs: Any) -> LMRequest:
    return LMRequest.from_call(model=lm.model, items=items, prompt=prompt, messages=messages, **kwargs)


def _model_response(text: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(role="assistant", content=text))], usage={}, model="custom-model"
    )


class _TypedContractLM(BaseLM):
    def __init__(self, *args: object, outputs: list[str], **kwargs: object):
        super().__init__(*args, **kwargs)
        self.outputs = outputs
        self.requests = []

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        assert isinstance(request, LMRequest)
        self.requests.append(request)
        return LMResponse.from_text(self.outputs[len(self.requests) - 1], model=request.model)


def _direct_lm_case(lm_kind: str, outputs: list[str]):
    if lm_kind == "current_lm":
        patcher = mock.patch(
            "dspy.clients.lm.alitellm_completion", side_effect=[_model_response(output) for output in outputs]
        )
        completion = patcher.start()
        lm = LM("custom-model")

        def get_messages(index: int) -> list[dict[str, object]]:
            return completion.call_args_list[index].kwargs["request"]["messages"]

        def get_request(index: int):
            return None

        return (lm, get_messages, get_request, patcher)
    if lm_kind == "typed_lm":
        lm = _TypedContractLM("custom-model", outputs=outputs)

        def get_messages(index: int) -> list[dict[str, object]]:
            from dspy.clients.openai_format import to_openai_chat_request

            return to_openai_chat_request(lm.requests[index])["messages"]

        def get_request(index: int):
            return lm.requests[index]

        return (lm, get_messages, get_request, None)
    raise ValueError(f"Unknown lm_kind: {lm_kind}")

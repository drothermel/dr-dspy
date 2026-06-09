import asyncio
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar
from unittest import mock

import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.core.types import LMConfig, LMRequest, LMResponse, coerce_lm_config, merge_lm_config

_T = TypeVar("_T")


async def _flush_litellm_logging_worker() -> None:
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except ImportError:
        return
    worker = GLOBAL_LOGGING_WORKER
    try:
        await asyncio.wait_for(worker.clear_queue(), timeout=2.0)
    except TimeoutError:
        pass


def run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    async def wrapped() -> _T:
        try:
            return await coro
        finally:
            await _flush_litellm_logging_worker()

    return asyncio.run(wrapped())


def make_response(output_blocks):
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
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


def _request(
    lm: BaseLM,
    *items: object,
    prompt: str | None = None,
    messages=None,
    config: LMConfig | None = None,
    **kwargs: Any,
) -> LMRequest:
    override = coerce_lm_config(kwargs) if kwargs else None
    merged_config = config
    if override is not None and override.model_fields_set:
        merged_config = merge_lm_config(config or LMConfig(), override) or (config or LMConfig())
    return LMRequest.from_call(
        model=lm.model,
        items=items,
        prompt=prompt,
        messages=messages,
        config=merged_config,
    )


def _model_response(text: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(role="assistant", content=text))], usage={}, model="custom-model"
    )


class _TypedContractLM(BaseLM):
    def __init__(
        self,
        model: str,
        *,
        outputs: list[str],
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
    ):
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
        )
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
            "dspy.clients.lm.transport.alitellm_completion", side_effect=[_model_response(output) for output in outputs]
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
            from dspy.clients.openai_format.chat_request import to_openai_chat_request

            return to_openai_chat_request(lm.requests[index])["messages"]

        def get_request(index: int):
            return lm.requests[index]

        return (lm, get_messages, get_request, None)
    raise ValueError(f"Unknown lm_kind: {lm_kind}")

import asyncio
from collections.abc import Coroutine, Iterator
from typing import TYPE_CHECKING, Any, TypeVar
from unittest import mock

import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    import litellm

    litellm.disable_aiohttp_transport = True
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.core.types import LMConfig, LMRequest, LMResponse, coerce_lm_config, merge_lm_config

_T = TypeVar("_T")


async def _cleanup_litellm_after_async_run() -> None:
    try:
        from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
    except ImportError:
        close_clients = None
    else:
        close_clients = close_litellm_async_clients

    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except ImportError:
        worker = None
    else:
        worker = GLOBAL_LOGGING_WORKER

    if worker is not None:
        try:
            await asyncio.wait_for(worker.clear_queue(), timeout=2.0)
        except TimeoutError:
            pass

    if close_clients is not None:
        try:
            await asyncio.wait_for(close_clients(), timeout=2.0)
        except TimeoutError:
            pass

    try:
        import litellm
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        cache_dict = getattr(litellm.in_memory_llm_clients_cache, "cache_dict", {})
        try:
            from openai import AsyncOpenAI
        except ImportError:
            async_openai = None
        else:
            async_openai = AsyncOpenAI

        for handler in list(cache_dict.values()):
            if isinstance(handler, AsyncHTTPHandler) or (
                async_openai is not None and isinstance(handler, async_openai)
            ):
                try:
                    await handler.close()
                except Exception:
                    pass

        litellm.in_memory_llm_clients_cache.flush_cache()
    except (ImportError, AttributeError):
        pass


def cleanup_litellm_after_test() -> None:
    asyncio.run(_cleanup_litellm_after_async_run())


@pytest.fixture(autouse=True)
def _cleanup_litellm_after_async_lm_test(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if asyncio.iscoroutinefunction(request.function):
        cleanup_litellm_after_test()


def run_async(*coros: Coroutine[Any, Any, Any]) -> Any:
    if not coros:
        msg = "run_async requires at least one coroutine"
        raise ValueError(msg)

    async def run_sequential() -> Any:
        results = [await coro for coro in coros]
        return results[0] if len(results) == 1 else tuple(results)

    async def wrapped() -> Any:
        try:
            return await run_sequential()
        finally:
            await _cleanup_litellm_after_async_run()

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

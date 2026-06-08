import json
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock
from unittest.mock import patch

import pydantic
import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from litellm.utils import ModelResponse

try:
    import litellm
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]
from openai import RateLimitError
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

import dspy.clients as dspy_clients
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.core.types import Assistant, LMHistoryEntry, LMRequest, LMResponse, System, ToolCall, ToolResult, User
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMConfigurationError,
    LMError,
    LMRateLimitError,
    LMUnexpectedError,
)
from dspy.utils.usage_tracker import track_usage


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


def test_chat_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    openai_lm = LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert openai_lm(_request(openai_lm, prompt="openai query")).text == "Hi!"

    azure_openai_lm = LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert azure_openai_lm(_request(azure_openai_lm, prompt="azure openai query")).text == "Hi!"


def test_dspy_cache(litellm_test_server, tmp_path):
    api_base, _ = litellm_test_server

    original_cache = dspy_clients.DSPY_CACHE
    dspy_clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=tmp_path / ".disk_cache",
    )
    cache = dspy_clients.DSPY_CACHE

    lm = LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    with track_usage() as usage_tracker:
        lm(_request(lm, prompt="Query"))

    assert len(cache.memory_cache) == 1
    cache_key = next(iter(cache.memory_cache.keys()))
    assert cache_key in cache.disk_cache
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(_request(lm, prompt="Query"))

    assert len(usage_tracker.usage_data) == 0

    dspy_clients.DSPY_CACHE = original_cache


def test_disabled_cache_skips_cache_key(monkeypatch):
    original_cache = dspy_clients.DSPY_CACHE
    dspy_clients.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    cache = dspy_clients.DSPY_CACHE

    try:
        with (
            mock.patch.object(cache, "cache_key", wraps=cache.cache_key) as cache_key_spy,
            mock.patch.object(cache, "get", wraps=cache.get) as cache_get_spy,
            mock.patch.object(cache, "put", wraps=cache.put) as cache_put_spy,
        ):

            def fake_completion(*, cache, num_retries, retry_strategy, **request: object):
                return ModelResponse(
                    choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    model="dummy",
                )

            monkeypatch.setattr(litellm, "completion", fake_completion)

            lm = LM("dummy", model_type="chat")
            lm(_request(lm, messages=[{"role": "user", "content": "Hello"}]))

            cache_key_spy.assert_not_called()
            cache_get_spy.assert_called_once()
            cache_put_spy.assert_called_once()
    finally:
        dspy_clients.DSPY_CACHE = original_cache


def test_rollout_id_bypasses_cache(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_completion(*, cache, num_retries, retry_strategy, **request: object):
        calls.append(request)
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/dspy-test-model",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    original_cache = dspy_clients.DSPY_CACHE
    dspy_clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=tmp_path / ".disk_cache",
    )

    lm = LM(model="openai/dspy-test-model", model_type="chat")

    with track_usage() as usage_tracker:
        lm(_request(lm, messages=[{"role": "user", "content": "Query"}], rollout_id=1))
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(_request(lm, messages=[{"role": "user", "content": "Query"}], rollout_id=1))
    assert len(usage_tracker.usage_data) == 0

    with track_usage() as usage_tracker:
        lm(_request(lm, messages=[{"role": "user", "content": "Query"}], rollout_id=2))
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(_request(lm, messages=[{"role": "user", "content": "NoRID"}]))
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(_request(lm, messages=[{"role": "user", "content": "NoRID"}], rollout_id=None))
    assert len(usage_tracker.usage_data) == 0

    assert len(dspy_clients.DSPY_CACHE.memory_cache) == 3
    assert all("rollout_id" not in r for r in calls)
    dspy_clients.DSPY_CACHE = original_cache


def test_zero_temperature_rollout_warns_once(monkeypatch):
    def fake_completion(*, cache, num_retries, retry_strategy, **request: object):
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/dspy-test-model",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    lm = LM(model="openai/dspy-test-model", model_type="chat", temperature=0)
    with pytest.warns(UserWarning, match="rollout_id has no effect"):
        lm(_request(lm, prompt="Query", rollout_id=1))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm(_request(lm, prompt="Query", rollout_id=2))
        assert len(record) == 0


def test_rollout_id_with_default_temperature_does_not_warn(monkeypatch):
    def fake_completion(*, cache, num_retries, retry_strategy, **request: object):
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/gpt-5-nano",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm = LM(model="openai/gpt-5-nano", model_type="chat", rollout_id=1)
        lm(_request(lm, prompt="Query"))
        assert len(record) == 0


def test_text_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    openai_lm = LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert openai_lm(_request(openai_lm, prompt="openai query")).text == "Hi!"

    azure_openai_lm = LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert azure_openai_lm(_request(azure_openai_lm, prompt="azure openai query")).text == "Hi!"


def test_lm_calls_support_callables(litellm_test_server):
    api_base, _ = litellm_test_server

    with mock.patch("litellm.completion", autospec=True, wraps=litellm.completion) as spy_completion:

        def azure_ad_token_provider(*args: object, **kwargs: object):
            return None

        lm_with_callable = LM(
            model="openai/dspy-test-model",
            api_base=api_base,
            api_key="fakekey",
            azure_ad_token_provider=azure_ad_token_provider,
            cache=False,
        )

        lm_with_callable(_request(lm_with_callable, prompt="Query"))

        spy_completion.assert_called_once()
        call_args = spy_completion.call_args.kwargs
        assert call_args["model"] == "openai/dspy-test-model"
        assert call_args["api_base"] == api_base
        assert call_args["api_key"] == "fakekey"
        assert call_args["azure_ad_token_provider"] is azure_ad_token_provider


def test_lm_calls_support_pydantic_models(litellm_test_server):
    api_base, _ = litellm_test_server

    class ResponseFormat(pydantic.BaseModel):
        response: str

    lm = LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        response_format=ResponseFormat,
    )
    lm(_request(lm, prompt="Query"))


def test_lm_wraps_litellm_errors_with_metadata():
    lm = LM("openai/gpt-4o-mini")
    response = mock.Mock()
    response.status_code = 429
    response.headers = {"x-request-id": "req-123", "retry-after": "2.5"}

    error = litellm.RateLimitError(
        message="too many requests", llm_provider="openai", model="gpt-4o", response=response
    )
    wrapped = lm._wrap_litellm_exception(error)

    assert isinstance(wrapped, LMRateLimitError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"
    assert wrapped.status == 429
    assert wrapped.request_id == "req-123"
    assert wrapped.retry_after == 2.5


def test_lm_wraps_litellm_context_window_error():
    lm = LM("openai/gpt-4o-mini")
    error = litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model="gpt-4o")
    wrapped = lm._wrap_litellm_exception(error)

    assert isinstance(wrapped, ContextWindowExceededError)
    assert isinstance(wrapped, LMError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"


def test_lm_wraps_unknown_boundary_error_as_unexpected_error():
    lm = LM("openai/gpt-4o-mini")
    wrapped = lm._wrap_litellm_exception(RuntimeError("local boundary failure"))

    assert isinstance(wrapped, LMUnexpectedError)
    assert wrapped.code == "unexpected"
    assert wrapped.model == "openai/gpt-4o-mini"


def test_lm_preserves_existing_lm_error_without_self_cause():
    error = LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = LM("openai/gpt-4o-mini", cache=False)

    with mock.patch("dspy.clients.lm.litellm_completion", side_effect=error):  # noqa: SIM117
        with pytest.raises(LMRateLimitError) as exc_info:
            lm(_request(lm, prompt="question"))

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_lm_preserves_existing_lm_error_without_self_cause_async():
    error = LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = LM("openai/gpt-4o-mini", cache=False)

    with mock.patch("dspy.clients.lm.alitellm_completion", side_effect=error):  # noqa: SIM117
        with pytest.raises(LMRateLimitError) as exc_info:
            await lm.acall(_request(lm, prompt="question"))

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


def test_retry_number_set_correctly():
    lm = LM("openai/gpt-4o-mini", num_retries=3)
    with mock.patch("litellm.completion") as mock_completion:
        lm(_request(lm, prompt="query"))

    assert mock_completion.call_args.kwargs["num_retries"] == 3


def test_retry_made_on_system_errors():
    retry_tracking = [0]  # Using a list to track retries

    def mock_create(*args: object, **kwargs: object):
        retry_tracking[0] += 1
        # LiteLLM RateLimitError handling expects response.status_code and response.headers.
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = LM(model="openai/gpt-4o-mini", max_tokens=250, num_retries=3)
    with mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create):  # noqa: SIM117
        with pytest.raises(LMRateLimitError):
            lm(_request(lm, prompt="question"))

    assert retry_tracking[0] == 4


def test_reasoning_model_token_parameter():
    test_cases = [
        ("openai/o1", True),
        ("openai/o1-mini", True),
        ("openai/o1-2023-01-01", True),
        ("openai/o3", True),
        ("openai/o3-mini-2023-01-01", True),
        ("openai/gpt-5", True),
        ("openai/gpt-5-mini", True),
        ("openai/gpt-5-nano", True),
        ("azure/gpt-5-chat", False),  # gpt-5-chat is NOT a reasoning model
        ("openai/gpt-4", False),
        ("anthropic/claude-2", False),
    ]

    for model_name, is_reasoning_model in test_cases:
        lm = LM(
            model=model_name,
            temperature=1.0 if is_reasoning_model else 0.7,
            max_tokens=16_000 if is_reasoning_model else 1000,
        )
        if is_reasoning_model:
            assert "max_completion_tokens" in lm.kwargs
            assert "max_tokens" not in lm.kwargs
            assert lm.kwargs["max_completion_tokens"] == 16_000
        else:
            assert "max_completion_tokens" not in lm.kwargs
            assert "max_tokens" in lm.kwargs
            assert lm.kwargs["max_tokens"] == 1000


@pytest.mark.parametrize("model_name", ["openai/o1", "openai/gpt-5-nano", "openai/gpt-5-mini"])
def test_reasoning_model_requirements(model_name):
    # Should raise LMConfigurationError when reasoning-model temperature or max_tokens constraints are violated.
    with pytest.raises(
        LMConfigurationError,
        match=r"reasoning models require passing temperature=1\.0 or None and max_tokens >= 16000 or None",
    ):
        LM(
            model=model_name,
            temperature=0.7,
            max_tokens=1000,
        )

    lm = LM(
        model=model_name,
        temperature=1.0,
        max_tokens=16_000,
    )
    assert lm.kwargs["max_completion_tokens"] == 16_000

    lm = LM(
        model=model_name,
    )
    assert lm.kwargs["temperature"] is None
    assert lm.kwargs["max_completion_tokens"] is None


def test_gpt_5_chat_not_reasoning_model():
    """Test that gpt-5-chat is NOT treated as a reasoning model."""
    lm = LM(
        model="openai/gpt-5-chat",
        temperature=0.7,
        max_tokens=1000,
    )
    assert "max_completion_tokens" not in lm.kwargs
    assert "max_tokens" in lm.kwargs
    assert lm.kwargs["max_tokens"] == 1000
    assert lm.kwargs["temperature"] == 0.7


def test_base_lm_init_uses_lm_defaults_and_isolates_callback_list():
    callbacks = [object()]
    lm = BaseLM("custom-model", callbacks=callbacks)  # ty:ignore[invalid-argument-type]

    assert lm.kwargs == {"temperature": None, "max_tokens": None}
    assert lm.num_retries == 3
    assert lm.callbacks == callbacks
    assert lm.callbacks is not callbacks


def _request(
    lm: BaseLM,
    *items: object,
    prompt: str | None = None,
    messages=None,
    **kwargs: Any,
) -> LMRequest:
    return LMRequest.from_call(
        model=lm.model,
        items=items,
        prompt=prompt,
        messages=messages,
        **kwargs,
    )


def test_base_lm_requires_lm_request():
    class CustomLM(BaseLM):
        @override
        def forward(self, request: LMRequest) -> LMResponse:
            return LMResponse.from_text("ok", model=request.model)

    with pytest.raises(TypeError, match="expects dspy\\.core\\.types\\.LMRequest"):
        CustomLM("custom-model")("Query")


def test_base_lm_typed_call_returns_lm_response_and_records_history():
    class CustomLM(BaseLM):
        @override
        def forward(self, request: LMRequest) -> LMResponse:
            assert request.model == "custom-model"
            assert request.messages[0].text == "Query"
            return LMResponse.from_text(
                "Hi!",
                model="custom-model",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            )

    lm = CustomLM("custom-model")
    request = _request(lm, prompt="Query")

    with track_usage() as usage_tracker:
        response = lm(request)

    assert isinstance(response, LMResponse)
    assert response.text == "Hi!"
    assert len(lm.history) == 1
    assert lm.history[0].request == request
    assert lm.history[0].response == response
    total_usage = usage_tracker.get_total_tokens()["custom-model"]
    assert total_usage["prompt_tokens"] == 1
    assert total_usage["completion_tokens"] == 2
    assert total_usage["total_tokens"] == 3


def test_base_lm_rejects_non_lm_response():
    class CustomLM(BaseLM):
        @override
        def forward(self, request: LMRequest):
            return ["not typed"]

    with pytest.raises(TypeError, match="must return dspy\\.core\\.types\\.LMResponse"):
        CustomLM("custom-model")(_request(BaseLM("custom-model"), prompt="Query"))


def _model_response(text: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(role="assistant", content=text))],
        usage={},
        model="custom-model",
    )


class _TypedContractLM(BaseLM):
    """Test double that records normalized requests received through the typed LM contract."""

    def __init__(self, *args: object, outputs: list[str], **kwargs: object):
        super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        self.outputs = outputs
        self.requests = []

    @override
    def forward(self, request: LMRequest) -> LMResponse:
        assert isinstance(request, LMRequest)
        self.requests.append(request)
        return LMResponse.from_text(self.outputs[len(self.requests) - 1], model=request.model)


def _direct_lm_case(lm_kind: str, outputs: list[str]):
    """Return a direct-call test double and helpers for inspecting normalized messages."""
    if lm_kind == "current_lm":
        patcher = mock.patch(
            "dspy.clients.lm.litellm_completion",
            side_effect=[_model_response(output) for output in outputs],
        )
        completion = patcher.start()
        lm = LM("custom-model", cache=False)

        def get_messages(index: int) -> list[dict[str, object]]:
            return completion.call_args_list[index].kwargs["request"]["messages"]

        def get_request(index: int):
            return None

        return lm, get_messages, get_request, patcher

    if lm_kind == "typed_lm":
        lm = _TypedContractLM("custom-model", outputs=outputs)

        def get_messages(index: int) -> list[dict[str, object]]:
            from dspy.clients.openai_format import to_openai_chat_request

            return to_openai_chat_request(lm.requests[index])["messages"]

        def get_request(index: int):
            return lm.requests[index]

        return lm, get_messages, get_request, None

    raise ValueError(f"Unknown lm_kind: {lm_kind}")


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_support_system_user_and_assistant_turns(lm_kind):
    lm, get_messages, get_request, patcher = _direct_lm_case(lm_kind, ["Five-word answer."])
    try:
        request = _request(
            lm,
            System("Be concise."),
            User("What is DSPy?"),
            Assistant("DSPy is a framework for programming LM pipelines."),
            User("Say that in five words."),
            temperature=0.2,
        )
        response = lm(request)
    finally:
        if patcher is not None:
            patcher.stop()

    assert isinstance(response, LMResponse)
    assert response.text == "Five-word answer."
    assert get_messages(0) == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is DSPy?"},
        {"role": "assistant", "content": "DSPy is a framework for programming LM pipelines."},
        {"role": "user", "content": "Say that in five words."},
    ]
    if lm_kind == "typed_lm":
        assert get_request(0).config.temperature == 0.2


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_support_tool_call_transcripts(lm_kind):
    lm, get_messages, get_request, patcher = _direct_lm_case(lm_kind, ["It is 22 C in Paris."])
    try:
        request = _request(
            lm,
            User("What is the weather in Paris?"),
            Assistant(ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            User("Summarize the result."),
        )
        response = lm(request)
    finally:
        if patcher is not None:
            patcher.stop()

    assert response.text == "It is 22 C in Paris."
    assert get_messages(0) == [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                    "id": "call_1",
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": "22 C"}', "tool_call_id": "call_1", "name": "get_weather"},
        {"role": "user", "content": "Summarize the result."},
    ]
    if lm_kind == "typed_lm":
        assert isinstance(get_request(0), LMRequest)


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_can_reuse_lm_response_as_assistant_turn(lm_kind):
    lm, get_messages, get_request, patcher = _direct_lm_case(
        lm_kind,
        ["DSPy programs LM pipelines.", "DSPy programs pipelines."],
    )
    try:
        first = lm(_request(lm, prompt="Explain DSPy in one sentence."))
        follow_up = lm(_request(lm, User("Explain DSPy in one sentence."), first, User("Now make it even shorter.")))
    finally:
        if patcher is not None:
            patcher.stop()

    assert first.text == "DSPy programs LM pipelines."
    assert follow_up.text == "DSPy programs pipelines."
    assert get_messages(0) == [{"role": "user", "content": "Explain DSPy in one sentence."}]
    assert get_messages(1) == [
        {"role": "user", "content": "Explain DSPy in one sentence."},
        {"role": "assistant", "content": "DSPy programs LM pipelines."},
        {"role": "user", "content": "Now make it even shorter."},
    ]
    if lm_kind == "typed_lm":
        assert isinstance(get_request(1), LMRequest)


@pytest.mark.asyncio
async def test_base_lm_async_explicit_lm_request_returns_lm_response():
    class CustomLM(BaseLM):
        @override
        async def aforward(self, request: LMRequest) -> LMResponse:
            assert request.model == "custom-model"
            return LMResponse.from_text("Hi async!", model=request.model)

    request = LMRequest.from_call(model="custom-model", prompt="Query")
    response = await CustomLM("custom-model").acall(request)

    assert isinstance(response, LMResponse)
    assert response.text == "Hi async!"


def test_base_lm_tracks_usage_for_custom_subclasses():
    class CustomLM(BaseLM):
        @override
        def forward(self, request: LMRequest) -> LMResponse:
            assert request.model == "custom-model"
            return LMResponse.from_text(
                "Hi!",
                model="custom-model",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    lm = CustomLM(model="custom-model")

    with track_usage() as usage_tracker:
        lm(_request(lm, prompt="Query"))

    total_usage = usage_tracker.get_total_tokens()["custom-model"]
    assert total_usage["prompt_tokens"] == 1
    assert total_usage["completion_tokens"] == 1
    assert total_usage["total_tokens"] == 2


def test_base_lm_copy_is_shallow_runtime_copy_with_isolated_dspy_state():
    class CustomLM(BaseLM):
        pass

    callback = object()
    client = object()
    lm = CustomLM(model="custom-model", callbacks=[callback], temperature=0.1)  # ty:ignore[invalid-argument-type]
    lm.client = client  # ty:ignore[unresolved-attribute]
    lm.extra_state = {"mutable": []}  # ty:ignore[unresolved-attribute]
    lm.history = [
        LMHistoryEntry(
            request=LMRequest.from_call(model="custom-model", prompt="original"),
            response=LMResponse.from_text("ok"),
            timestamp="timestamp",
            uuid="uuid",
        )
    ]

    copied_lm = lm.copy(temperature=0.2, rollout_id=1)

    assert copied_lm is not lm
    assert copied_lm.client is client
    assert copied_lm.extra_state is lm.extra_state  # ty:ignore[unresolved-attribute]
    assert copied_lm.history == []
    assert copied_lm.history is not lm.history
    assert copied_lm.callbacks == [callback]
    assert copied_lm.callbacks is not lm.callbacks
    assert copied_lm.kwargs == {"temperature": 0.2, "max_tokens": None, "rollout_id": 1}
    assert lm.kwargs == {"temperature": 0.1, "max_tokens": None}


def test_dump_state():
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
        launch_kwargs={"temperature": 1},
        train_kwargs={"temperature": 5},
    )

    assert lm.dump_state() == {
        "_dspy_lm_class": "dspy.clients.lm.LM",
        "model": "openai/gpt-4o-mini",
        "model_type": "chat",
        "temperature": 1,
        "max_tokens": 100,
        "num_retries": 10,
        "cache": True,
        "finetuning_model": None,
        "launch_kwargs": {"temperature": 1},
        "train_kwargs": {"temperature": 5},
    }


def test_reasoning_model_dump_state_uses_constructor_max_tokens():
    lm = LM(
        model="openai/gpt-5-nano",
        temperature=1.0,
        max_tokens=16_000,
        cache=False,
        num_retries=1,
    )

    state = lm.dump_state()

    assert lm.kwargs["max_completion_tokens"] == 16_000
    assert "max_completion_tokens" not in state
    assert state["max_tokens"] == 16_000


def test_dump_state_preserves_enabled_developer_role():
    lm = LM("openai/gpt-4o-mini", use_developer_role=True)

    assert lm.dump_state()["use_developer_role"] is True
    assert LM.load_state(lm.dump_state()).use_developer_role is True


def test_dump_state_ignores_internal_class_marker_kwarg():
    lm = LM(
        model="openai/gpt-4o-mini",
        _dspy_lm_class="malicious.module.LM",
    )

    dumped_state = lm.dump_state()

    assert dumped_state["_dspy_lm_class"] == "dspy.clients.lm.LM"
    assert lm.kwargs["_dspy_lm_class"] == "malicious.module.LM"


def test_load_state():
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
        launch_kwargs={"temperature": 1},
        train_kwargs={"temperature": 5},
    )

    loaded_lm = LM.load_state(lm.dump_state())

    assert isinstance(loaded_lm, LM)
    assert loaded_lm.dump_state() == lm.dump_state()


def test_reasoning_model_load_state_round_trips_canonical_state():
    lm = LM(
        model="openai/gpt-5-nano",
        temperature=1.0,
        max_tokens=16_000,
        cache=False,
        num_retries=1,
    )

    loaded_lm = BaseLM.load_state(lm.dump_state())

    assert isinstance(loaded_lm, LM)
    assert loaded_lm.kwargs["max_completion_tokens"] == 16_000
    assert loaded_lm.dump_state() == lm.dump_state()


def test_reasoning_model_load_state_accepts_max_completion_tokens_alias():
    state = {
        "_dspy_lm_class": "dspy.clients.lm.LM",
        "model": "openai/gpt-5-nano",
        "model_type": "chat",
        "cache": False,
        "num_retries": 1,
        "temperature": 1.0,
        "max_completion_tokens": 16_000,
        "finetuning_model": None,
        "launch_kwargs": {},
        "train_kwargs": {},
    }

    loaded_lm = BaseLM.load_state(state)

    assert isinstance(loaded_lm, LM)
    assert loaded_lm.kwargs["max_completion_tokens"] == 16_000
    assert "max_completion_tokens" not in loaded_lm.dump_state()
    assert loaded_lm.dump_state()["max_tokens"] == 16_000


def test_lm_load_state_forwards_allow_custom_lm_class(monkeypatch):
    calls = []
    original_load_state = BaseLM.load_state.__func__

    def spy_load_state(cls, state, *, allow_custom_lm_class=False):
        calls.append(allow_custom_lm_class)
        return original_load_state(cls, state, allow_custom_lm_class=allow_custom_lm_class)

    monkeypatch.setattr(BaseLM, "load_state", classmethod(spy_load_state))

    LM.load_state(LM("openai/gpt-4o-mini").dump_state(), allow_custom_lm_class=True)

    assert calls == [True]


def test_exponential_backoff_retry():
    time_counter = []

    def mock_create(*args: object, **kwargs: object):
        time_counter.append(time.time())
        # LiteLLM RateLimitError handling expects response.status_code and response.headers.
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = LM(model="openai/gpt-3.5-turbo", max_tokens=250, num_retries=3)
    with mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create):  # noqa: SIM117
        with pytest.raises(LMRateLimitError):
            lm(_request(lm, prompt="question"))

    # The first retry happens immediately regardless of the configuration
    for i in range(1, len(time_counter) - 1):
        assert time_counter[i + 1] - time_counter[i] >= 2 ** (i - 1)


def test_logprobs_included_when_requested():
    lm = LM(model="dspy-test-model", logprobs=True, cache=False)
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(content="test answer"),
                    logprobs={
                        "content": [
                            {"token": "test", "logprob": 0.1, "top_logprobs": [{"token": "test", "logprob": 0.1}]},
                            {"token": "answer", "logprob": 0.2, "top_logprobs": [{"token": "answer", "logprob": 0.2}]},
                        ]
                    },
                )
            ],
            model="dspy-test-model",
        )
        result = lm(_request(lm, prompt="question"))
        assert result.text == "test answer"
        assert result.outputs[0].logprobs.model_dump() == {
            "content": [
                {
                    "token": "test",
                    "bytes": None,
                    "logprob": 0.1,
                    "top_logprobs": [{"token": "test", "bytes": None, "logprob": 0.1}],
                },
                {
                    "token": "answer",
                    "bytes": None,
                    "logprob": 0.2,
                    "top_logprobs": [{"token": "answer", "bytes": None, "logprob": 0.2}],
                },
            ]
        }
        assert mock_completion.call_args.kwargs["logprobs"]


@pytest.mark.asyncio
async def test_async_lm_call():
    from litellm.utils import Choices, Message, ModelResponse

    mock_response = ModelResponse(choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini")

    with patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = mock_response

        lm = LM(model="openai/gpt-4o-mini", cache=False)
        result = await lm.acall(_request(lm, prompt="question"))

        assert result.text == "answer"
        mock_acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_async_lm_call_with_cache(tmp_path):
    """Test the async LM call with caching."""
    original_cache = dspy_clients.DSPY_CACHE
    dspy_clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=tmp_path / ".disk_cache",
    )
    cache = dspy_clients.DSPY_CACHE

    lm = LM(model="openai/gpt-4o-mini")

    with mock.patch("dspy.clients.lm.alitellm_completion") as mock_alitellm_completion:
        mock_alitellm_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini"
        )
        mock_alitellm_completion.__qualname__ = "alitellm_completion"
        await lm.acall(_request(lm, prompt="Query"))

        assert len(cache.memory_cache) == 1
        cache_key = next(iter(cache.memory_cache.keys()))
        assert cache_key in cache.disk_cache
        assert mock_alitellm_completion.call_count == 1

        await lm.acall(_request(lm, prompt="Query"))
        assert mock_alitellm_completion.call_count == 1

        await lm.acall(_request(lm, prompt="New query"))

        assert len(cache.memory_cache) == 2
        assert mock_alitellm_completion.call_count == 2

    dspy_clients.DSPY_CACHE = original_cache


def test_lm_history_size_limit():
    lm = LM(model="openai/gpt-4o-mini")
    with settings.context(max_history_size=5), mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="test answer"))],
            model="openai/gpt-4o-mini",
        )

        for _ in range(10):
            lm(_request(lm, prompt="query"))

    assert len(lm.history) == 5


def test_disable_history():
    lm = LM(model="openai/gpt-4o-mini")
    with settings.context(disable_history=True), mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="test answer"))],
            model="openai/gpt-4o-mini",
        )
        for _ in range(10):
            lm(_request(lm, prompt="query"))

    assert len(lm.history) == 0

    with settings.context(disable_history=False), mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="test answer"))],
            model="openai/gpt-4o-mini",
        )


def test_responses_api():
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    {"type": "output_text", "text": "This is a test answer from responses API.", "annotations": []}
                ],  # ty:ignore[invalid-argument-type]
            ),
            ResponseReasoningItem(
                id="reasoning_1",
                type="reasoning",
                summary=[Summary(type="summary_text", text="This is a dummy reasoning.")],
            ),
        ]
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = LM(
            model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )
        lm_result = lm(_request(lm, prompt="openai query"))

        assert lm_result.text == "This is a test answer from responses API."
        assert lm_result.reasoning_content == "This is a dummy reasoning."

        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/gpt-5-mini"


def test_lm_replaces_system_with_developer_role():
    with mock.patch("dspy.clients.lm.litellm_responses_completion", return_value={"choices": []}) as mock_completion:
        lm = LM(
            "openai/gpt-4o-mini",
            cache=False,
            model_type="responses",
            use_developer_role=True,
        )
        lm(_request(lm, messages=[{"role": "system", "content": "hi"}]))
        assert mock_completion.call_args.kwargs["request"]["input"][0]["role"] == "developer"


def test_responses_api_tool_calls(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_tool_call = {
        "type": "function_call",
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris"}),
        "call_id": "call_1",
        "status": "completed",
        "id": "call_1",
    }
    api_response = make_response(
        output_blocks=[expected_tool_call],
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = LM(
            model="openai/dspy-test-model",
            api_base=api_base,
            api_key="fakekey",
            model_type="responses",
            cache=False,
        )
        lm_result = lm(_request(lm, prompt="openai query"))
        tool_call = lm_result.outputs[0].tool_calls[0]
        assert tool_call.name == expected_tool_call["name"]
        assert tool_call.args == {"city": "Paris"}
        assert tool_call.id == expected_tool_call["call_id"]

        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/dspy-test-model"


def test_reasoning_effort_responses_api():
    """Test that reasoning_effort gets normalized to reasoning format for Responses API."""
    with mock.patch("litellm.responses") as mock_responses:
        # OpenAI model with Responses API - should normalize
        lm = LM(model="openai/gpt-5", model_type="responses", reasoning_effort="low", max_tokens=16000, temperature=1.0)
        lm(_request(lm, prompt="openai query"))
        call_kwargs = mock_responses.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert call_kwargs["reasoning"]["effort"] == "low"


def test_call_reasoning_model_with_chat_api():
    """Test that Chat API properly handles reasoning models and returns data in correct format."""
    # Create message with reasoning_content attribute
    message = Message(content="The answer is 4", role="assistant")
    # Add reasoning_content attribute
    message.reasoning_content = "Step 1: I need to add 2 + 2\nStep 2: 2 + 2 = 4\nTherefore, the answer is 4"

    # Create choice with the message
    mock_choice = Choices(message=message)

    # Mock response with reasoning content for chat completion
    mock_response = ModelResponse(
        choices=[mock_choice],
        model="anthropic/claude-3-7-sonnet-20250219",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )

    with mock.patch("litellm.completion", return_value=mock_response) as mock_completion:  # noqa: SIM117
        with mock.patch("litellm.supports_reasoning", return_value=True):
            # Create reasoning model with chat API
            lm = LM(
                model="anthropic/claude-3-7-sonnet-20250219",
                model_type="chat",
                temperature=1.0,
                max_tokens=16000,
                reasoning_effort="low",
                cache=False,
            )

            # Test the call
            result = lm(_request(lm, prompt="What is 2 + 2?"))

            # Verify the response format
            assert result.text == "The answer is 4"
            assert result.reasoning_content is not None
            assert "Step 1" in result.reasoning_content

            # Verify mock was called with correct parameters
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args.kwargs
            assert call_kwargs["model"] == "anthropic/claude-3-7-sonnet-20250219"
            assert call_kwargs["reasoning_effort"] == "low"


def test_api_key_not_saved_in_json():
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1.0,
        max_tokens=100,
        api_key="sk-test-api-key-12345",
    )

    predict = Predict("question -> answer")
    predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "program.json"
        predict.save(json_path)

        with open(json_path) as f:
            saved_state = json.load(f)

        # Verify API key is not in the saved state
        assert "api_key" not in saved_state.get("lm", {}), "API key should not be saved in JSON"

        # Verify other attributes are saved
        assert saved_state["lm"]["model"] == "openai/gpt-4o-mini"
        assert saved_state["lm"]["temperature"] == 1.0
        assert saved_state["lm"]["max_tokens"] == 100


def test_responses_api_converts_images_correctly():
    from dspy.clients.lm import _convert_chat_request_to_responses_request

    # Test with base64 image
    request_with_base64_image = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_base64_image)

    assert "input" in result
    assert len(result["input"]) == 1
    assert result["input"][0]["role"] == "user"

    content = result["input"][0]["content"]
    assert len(content) == 2

    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "What's in this image?"

    assert content[1]["type"] == "input_image"
    assert (
        content[1]["image_url"]
        == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    request_with_url_image = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_url_image)

    content = result["input"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"] == "https://example.com/image.jpg"


def test_responses_api_converts_files_correctly():
    from dspy.clients.lm import _convert_chat_request_to_responses_request

    request_with_file = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this file"},
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
                            "filename": "test.txt",
                        },
                    },
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_file)

    assert "input" in result
    assert len(result["input"]) == 1
    assert result["input"][0]["role"] == "user"

    content = result["input"][0]["content"]
    assert len(content) == 2

    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "Analyze this file"

    assert content[1]["type"] == "input_file"
    assert content[1]["file_data"] == "data:text/plain;base64,SGVsbG8gV29ybGQ="
    assert content[1]["filename"] == "test.txt"

    request_with_file_id = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_id": "file-abc123",
                            "filename": "document.pdf",
                        },
                    }
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_file_id)

    content = result["input"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_file"
    assert content[0]["file_id"] == "file-abc123"
    assert content[0]["filename"] == "document.pdf"

    # Test with all file fields
    request_with_all_fields = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                            "file_id": "file-xyz789",
                            "filename": "report.pdf",
                        },
                    }
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_all_fields)

    content = result["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"] == "data:application/pdf;base64,JVBERi0xLjQ="
    assert content[0]["file_id"] == "file-xyz789"
    assert content[0]["filename"] == "report.pdf"


def test_responses_api_preserves_multi_message_structure():
    from dspy.clients.lm import _convert_chat_request_to_responses_request

    request = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
    }

    result = _convert_chat_request_to_responses_request(request)

    assert "input" in result
    assert len(result["input"]) == 4

    assert result["input"][0]["role"] == "system"
    assert result["input"][0]["content"] == [{"type": "input_text", "text": "You are a helpful assistant."}]

    assert result["input"][1]["role"] == "user"
    assert result["input"][1]["content"] == [{"type": "input_text", "text": "What is 2+2?"}]

    assert result["input"][2]["role"] == "assistant"
    assert result["input"][2]["content"] == [{"type": "input_text", "text": "4"}]

    assert result["input"][3]["role"] == "user"
    assert result["input"][3]["content"] == [{"type": "input_text", "text": "And 3+3?"}]


def test_responses_api_with_image_input():
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "This is a test answer with image input.", "annotations": []}],  # ty:ignore[invalid-argument-type]
            ),
        ]
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = LM(
            model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        # Test with messages containing an image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ]

        lm_result = lm(_request(lm, messages=messages))

        assert lm_result.text == "This is a test answer with image input."

        dspy_responses.assert_called_once()
        call_args = dspy_responses.call_args.kwargs

        # Verify the request was converted correctly
        assert "input" in call_args
        content = call_args["input"][0]["content"]

        # Check that image was converted to input_image format
        image_content = [c for c in content if c.get("type") == "input_image"]
        assert len(image_content) == 1
        assert (
            image_content[0]["image_url"]
            == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )


def test_responses_api_with_pydantic_model_input():
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    {
                        "type": "output_text",
                        "text": '{"answer" : "This is a good test answer", "number" : 42}',
                        "annotations": [],
                    }
                ],  # ty:ignore[invalid-argument-type]
            ),
        ]
    )

    lm = LM(
        model="openai/gpt-5-mini",
        model_type="responses",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )

    class TestModel(pydantic.BaseModel):
        answer: str
        number: int

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        # Test with messages containing a Pydantic model as response format
        lm_result = lm(_request(lm, prompt="What is a good test answer?", response_format=TestModel))

    # Try to validate to Pydantic model
    TestModel.model_validate_json(lm_result.text)

    dspy_responses.assert_called_once()
    call_args = dspy_responses.call_args.kwargs

    # Verify the request was converted correctly
    assert "text" in call_args
    response_format = call_args["text"]["format"]

    assert response_format == {
        "name": TestModel.__name__,
        "type": "json_schema",
        "schema": TestModel.model_json_schema(),
    }


def test_responses_api_with_none_usage():
    """Responses API returns usage=None for incomplete/truncated responses (e.g. max_output_tokens hit)."""
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details={"reason": "max_output_tokens"},  # ty:ignore[invalid-argument-type]
        instructions=None,
        model="openai/gpt-5-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="incomplete",
                content=[{"type": "output_text", "text": "Partial response that was truncated", "annotations": []}],  # ty:ignore[invalid-argument-type]
            ),
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=100,
        previous_response_id=None,
        reasoning=None,
        status="incomplete",
        text=None,
        truncation="disabled",
        usage=None,
        user=None,
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response):
        lm = LM(
            model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        with track_usage() as tracker:
            result = lm(_request(lm, prompt="test query"))

        assert result.text == "Partial response that was truncated"
        assert lm.history[-1].usage == {}
        assert tracker.get_total_tokens() == {}


@pytest.mark.asyncio
async def test_responses_api_with_none_usage_async():
    """Async path: Responses API returns usage=None for incomplete/truncated responses."""
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details={"reason": "max_output_tokens"},  # ty:ignore[invalid-argument-type]
        instructions=None,
        model="openai/gpt-5-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="incomplete",
                content=[{"type": "output_text", "text": "Partial async response", "annotations": []}],  # ty:ignore[invalid-argument-type]
            ),
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=100,
        previous_response_id=None,
        reasoning=None,
        status="incomplete",
        text=None,
        truncation="disabled",
        usage=None,
        user=None,
    )

    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response):
        lm = LM(
            model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        with track_usage() as tracker:
            result = await lm.acall(_request(lm, prompt="test query"))

        assert result.text == "Partial async response"
        assert lm.history[-1].usage == {}
        assert tracker.get_total_tokens() == {}


@pytest.mark.asyncio
async def test_streaming_passes_headers_correctly():
    from dspy.clients.lm import _get_stream_completion_fn

    custom_headers = {"Authorization": "Bearer my-custom-token"}
    request = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "test"}],
    }

    mock_stream = mock.AsyncMock()
    mock_stream.send = mock.AsyncMock()

    async def empty_async_generator():
        return
        yield  # Make it a generator

    with mock.patch("dspy.clients.lm.settings") as mock_settings:
        mock_settings.send_stream = mock_stream
        mock_settings.caller_predict = None
        mock_settings.track_usage = False

        with mock.patch("litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = empty_async_generator()

            stream_fn = _get_stream_completion_fn(request, {}, sync=False, headers=custom_headers)
            assert stream_fn is not None

            with mock.patch("litellm.stream_chunk_builder", return_value={}):
                await stream_fn()

            # Verify headers were passed to litellm.acompletion
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer my-custom-token"

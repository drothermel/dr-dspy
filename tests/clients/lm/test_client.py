import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest import mock
from unittest.mock import patch

import pydantic
import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    import litellm
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, BaseLM
from dspy.clients.lm import LM
from dspy.clients.lm_registry import BUILTIN_LM_CLASS_PATH, get_lm_class
from dspy.clients.lm_strict import LegacyLMKeyError
from dspy.core.types import (
    Assistant,
    CallRecord,
    LMProviderOptions,
    LMRequest,
    LMResponse,
    System,
    ToolCall,
    ToolResult,
    User,
)
from dspy.errors import LMConfigurationError
from dspy.predict.predict import Predict
from dspy.runtime import CallLogMode, TelemetryConfig
from dspy.runtime.usage_tracker import track_usage
from tests.clients.lm.conftest import _direct_lm_case, _request, run_async
from tests.task_spec.helpers import ts


def test_chat_lms_can_be_queried(litellm_test_server, make_run):
    api_base, _ = litellm_test_server
    provider_options = LMProviderOptions(api_base=api_base, api_key="fakekey")
    openai_lm = LM(model="openai/dspy-test-model", provider_options=provider_options, model_type="chat")
    azure_openai_lm = LM(model="azure/dspy-test-model", provider_options=provider_options, model_type="chat")
    openai_result, azure_result = run_async(
        openai_lm(_request(openai_lm, prompt="openai query"), run=make_run(lm=openai_lm)),
        azure_openai_lm(_request(azure_openai_lm, prompt="azure openai query"), run=make_run(lm=azure_openai_lm)),
    )
    assert openai_result.text == "Hi!"
    assert azure_result.text == "Hi!"


def test_text_lms_can_be_queried(litellm_test_server, make_run):
    api_base, _ = litellm_test_server
    provider_options = LMProviderOptions(api_base=api_base, api_key="fakekey")
    openai_lm = LM(model="openai/dspy-test-model", provider_options=provider_options, model_type="text")
    azure_openai_lm = LM(model="azure/dspy-test-model", provider_options=provider_options, model_type="text")
    openai_result, azure_result = run_async(
        openai_lm(_request(openai_lm, prompt="openai query"), run=make_run(lm=openai_lm)),
        azure_openai_lm(_request(azure_openai_lm, prompt="azure openai query"), run=make_run(lm=azure_openai_lm)),
    )
    assert openai_result.text == "Hi!"
    assert azure_result.text == "Hi!"


def test_lm_calls_support_callables(litellm_test_server, make_run):
    api_base, _ = litellm_test_server
    with mock.patch("litellm.acompletion", autospec=True, wraps=litellm.acompletion) as spy_completion:

        def azure_ad_token_provider(*args: object, **kwargs: object):
            return None

        lm_with_callable = LM(
            model="openai/dspy-test-model",
            provider_options=LMProviderOptions(
                api_base=api_base,
                api_key="fakekey",
                extensions={"azure_ad_token_provider": azure_ad_token_provider},
            ),
        )
        run_async(lm_with_callable(_request(lm_with_callable, prompt="Query"), run=make_run(lm=lm_with_callable)))
        spy_completion.assert_called_once()
        call_args = spy_completion.call_args.kwargs
        assert call_args["model"] == "openai/dspy-test-model"
        assert call_args["api_base"] == api_base
        assert call_args["api_key"] == "fakekey"
        assert call_args["azure_ad_token_provider"] is azure_ad_token_provider


def test_lm_calls_support_pydantic_models(litellm_test_server, make_run):
    api_base, _ = litellm_test_server

    class ResponseFormat(pydantic.BaseModel):
        response: str

    lm = LM(
        model="openai/dspy-test-model",
        provider_options=LMProviderOptions(
            api_base=api_base,
            api_key="fakekey",
            response_format=ResponseFormat,
        ),
    )
    run_async(lm(_request(lm, prompt="Query"), run=make_run(lm=lm)))


def test_reasoning_model_token_parameter(make_run):
    test_cases = [
        ("openai/o1", True),
        ("openai/o1-mini", True),
        ("openai/o1-2023-01-01", True),
        ("openai/o3", True),
        ("openai/o3-mini-2023-01-01", True),
        ("openai/gpt-5", True),
        ("openai/gpt-5-mini", True),
        ("openai/gpt-5-nano", True),
        ("azure/gpt-5-chat", False),
        ("openai/gpt-4", False),
        ("anthropic/claude-2", False),
    ]
    for model_name, is_reasoning_model in test_cases:
        lm = LM(
            model=model_name,
            temperature=1.0 if is_reasoning_model else 0.7,
            max_tokens=16000 if is_reasoning_model else 1000,
        )
        if is_reasoning_model:
            assert lm.kwargs["max_tokens"] == 16000
        else:
            assert "max_completion_tokens" not in lm.kwargs
            assert lm.kwargs["max_tokens"] == 1000


@pytest.mark.parametrize("model_name", ["openai/o1", "openai/gpt-5-nano", "openai/gpt-5-mini"])
def test_reasoning_model_requirements(model_name, make_run):
    with pytest.raises(
        LMConfigurationError,
        match="reasoning models require passing temperature=1\\.0 or None and max_tokens >= 16000 or None",
    ):
        LM(model=model_name, temperature=0.7, max_tokens=1000)
    lm = LM(model=model_name, temperature=1.0, max_tokens=16000)
    assert lm.kwargs["max_tokens"] == 16000
    lm = LM(model=model_name)
    assert lm.kwargs.get("temperature") is None
    assert lm.kwargs.get("max_tokens") is None


def test_gpt_5_chat_not_reasoning_model(make_run):
    lm = LM(model="openai/gpt-5-chat", temperature=0.7, max_tokens=1000)
    assert "max_completion_tokens" not in lm.kwargs
    assert "max_tokens" in lm.kwargs
    assert lm.kwargs["max_tokens"] == 1000
    assert lm.kwargs["temperature"] == 0.7


def test_base_lm_init_uses_lm_defaults_and_isolates_callback_list(make_run):
    callbacks = cast("list[Any]", [object()])
    lm = BaseLM("custom-model", callbacks=callbacks)
    assert lm.kwargs == {}
    assert lm.num_retries == 3
    assert lm.callbacks == callbacks
    assert lm.callbacks is not callbacks


def test_base_lm_requires_lm_request(make_run):

    class CustomLM(BaseLM):
        @override
        async def aforward(self, request: LMRequest) -> LMResponse:
            return LMResponse.from_text("ok", model=request.model)

    custom_lm = CustomLM("custom-model")
    with pytest.raises(TypeError, match="expects dspy\\.core\\.types\\.LMRequest"):
        run_async(custom_lm("Query", run=make_run(lm=custom_lm)))


def test_base_lm_typed_call_returns_lm_response_and_records_history(make_run):

    class CustomLM(BaseLM):
        @override
        async def aforward(self, request: LMRequest) -> LMResponse:
            assert request.model == "custom-model"
            assert request.messages[0].text == "Query"
            return LMResponse.from_text(
                "Hi!", model="custom-model", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            )

    lm = CustomLM("custom-model")
    run = make_run(lm=lm)
    request = _request(lm, prompt="Query")
    with track_usage(run) as usage_tracker:
        response = run_async(lm(request, run=run))
    assert isinstance(response, LMResponse)
    assert response.text == "Hi!"
    assert len(lm.call_log) == 1
    assert lm.call_log[0].request == request
    assert lm.call_log[0].response == response
    total_usage = usage_tracker.get_total_tokens()["custom-model"]
    assert total_usage["prompt_tokens"] == 1
    assert total_usage["completion_tokens"] == 2
    assert total_usage["total_tokens"] == 3


def test_base_lm_rejects_non_lm_response(make_run):

    class CustomLM(BaseLM):
        @override
        async def aforward(self, request: LMRequest):
            return ["not typed"]

    with pytest.raises(TypeError, match="must return dspy\\.core\\.types\\.LMResponse"):
        run_async(
            CustomLM("custom-model")(
                _request(BaseLM("custom-model"), prompt="Query"), run=make_run(lm=CustomLM("custom-model"))
            )
        )


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_support_system_user_and_assistant_turns(lm_kind, make_run):
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
        response = run_async(lm(request, run=make_run(lm=lm)))
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
def test_base_lm_experimental_direct_messages_support_tool_call_transcripts(lm_kind, make_run):
    lm, get_messages, get_request, patcher = _direct_lm_case(lm_kind, ["It is 22 C in Paris."])
    try:
        request = _request(
            lm,
            User("What is the weather in Paris?"),
            Assistant(ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            User("Summarize the result."),
        )
        response = run_async(lm(request, run=make_run(lm=lm)))
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
def test_base_lm_experimental_direct_messages_can_reuse_lm_response_as_assistant_turn(lm_kind, make_run):
    lm, get_messages, get_request, patcher = _direct_lm_case(
        lm_kind, ["DSPy programs LM pipelines.", "DSPy programs pipelines."]
    )
    try:

        async def _query_twice():
            first = await lm(_request(lm, prompt="Explain DSPy in one sentence."), run=make_run(lm=lm))
            follow_up = await lm(
                _request(lm, User("Explain DSPy in one sentence."), first, User("Now make it even shorter.")),
                run=make_run(lm=lm),
            )
            return first, follow_up

        first, follow_up = run_async(_query_twice())
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
async def test_base_lm_async_explicit_lm_request_returns_lm_response(make_run):

    class CustomLM(BaseLM):
        @override
        async def aforward(self, request: LMRequest) -> LMResponse:
            assert request.model == "custom-model"
            return LMResponse.from_text("Hi async!", model=request.model)

    custom_lm = CustomLM("custom-model")
    request = LMRequest.from_call(model="custom-model", prompt="Query")
    response = await custom_lm(request, run=make_run(lm=custom_lm))
    assert isinstance(response, LMResponse)
    assert response.text == "Hi async!"


def test_base_lm_tracks_usage_for_custom_subclasses(make_run):

    class CustomLM(BaseLM):
        @override
        async def aforward(self, request: LMRequest) -> LMResponse:
            assert request.model == "custom-model"
            return LMResponse.from_text(
                "Hi!", model="custom-model", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )

    lm = CustomLM(model="custom-model")
    run = make_run(lm=lm)
    with track_usage(run) as usage_tracker:
        run_async(lm(_request(lm, prompt="Query"), run=run))
    total_usage = usage_tracker.get_total_tokens()["custom-model"]
    assert total_usage["prompt_tokens"] == 1
    assert total_usage["completion_tokens"] == 1
    assert total_usage["total_tokens"] == 2


def test_base_lm_copy_is_shallow_runtime_copy_with_isolated_dspy_state():

    class CustomLM(BaseLM):
        pass

    callback = cast("Any", object())
    client = object()
    lm = CustomLM(model="custom-model", callbacks=[callback], temperature=0.1)
    cast("Any", lm).client = client
    cast("Any", lm).extra_state = {"mutable": []}
    lm.call_log = [
        CallRecord(
            request=LMRequest.from_call(model="custom-model", prompt="original"),
            response=LMResponse.from_text("ok"),
            timestamp="timestamp",
            uuid="uuid",
        )
    ]
    copied_lm = lm.copy(temperature=0.2)
    assert copied_lm is not lm
    assert cast("Any", copied_lm).client is client
    assert cast("Any", copied_lm).extra_state is cast("Any", lm).extra_state
    assert copied_lm.call_log == []
    assert copied_lm.call_log is not lm.call_log
    assert copied_lm.callbacks == [callback]
    assert copied_lm.callbacks is not lm.callbacks
    assert copied_lm.kwargs == {"temperature": 0.2}
    assert lm.kwargs == {"temperature": 0.1}


def test_base_lm_copy_rejects_legacy_kwargs_in_existing_state() -> None:
    class CustomLM(BaseLM):
        pass

    lm = CustomLM(model="custom-model", temperature=0.1)
    lm.kwargs["reasoning_effort"] = "high"
    with pytest.raises(LegacyLMKeyError, match="reasoning_effort"):
        lm.copy(temperature=0.5)


def test_base_lm_copy_with_temperature_still_validates() -> None:
    lm = LM(model="openai/gpt-4o-mini", temperature=0.1)
    copied = lm.copy(temperature=0.5)
    assert copied.kwargs["temperature"] == 0.5
    assert lm.kwargs["temperature"] == 0.1


def test_get_lm_class_unknown_path_lists_builtins() -> None:
    with pytest.raises(LMConfigurationError, match=r"bogus\.path") as exc_info:
        get_lm_class("bogus.path")
    message = str(exc_info.value)
    assert BUILTIN_LM_CLASS_PATH in message
    assert "dspy.clients.dr_llm.direct.DrLlmDirectLM" in message


def test_dump_state():
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
    )
    assert lm.dump_state() == {
        "_dspy_lm_class": "dspy.clients.lm.LM",
        "model": "openai/gpt-4o-mini",
        "model_type": "chat",
        "temperature": 1,
        "max_tokens": 100,
        "num_retries": 10,
        "_dspy_provider_options": {"extensions": {}},
    }


def test_reasoning_model_dump_state_uses_constructor_max_tokens():
    lm = LM(model="openai/gpt-5-nano", temperature=1.0, max_tokens=16000, num_retries=1)
    state = lm.dump_state()
    assert lm.kwargs["max_tokens"] == 16000
    assert state["max_tokens"] == 16000


def test_dump_state_preserves_enabled_developer_role():
    lm = LM("openai/gpt-4o-mini", use_developer_role=True)
    assert lm.dump_state()["use_developer_role"] is True


def test_dump_state_ignores_internal_class_marker_kwarg(make_run):
    lm = LM(
        model="openai/gpt-4o-mini",
        provider_options=LMProviderOptions(extensions={"_dspy_lm_class": "malicious.module.LM"}),
    )
    dumped_state = lm.dump_state()
    assert dumped_state["_dspy_lm_class"] == "dspy.clients.lm.LM"
    assert lm.kwargs["_dspy_lm_class"] == "malicious.module.LM"


def test_load_state(make_run):
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
    )
    loaded_lm = LM.load_state(lm.dump_state())
    assert isinstance(loaded_lm, LM)
    assert loaded_lm.dump_state() == lm.dump_state()


def test_load_state_round_trips_developer_role(make_run):
    lm = LM("openai/gpt-4o-mini", use_developer_role=True)
    loaded_lm = LM.load_state(lm.dump_state())
    assert loaded_lm.use_developer_role is True
    assert loaded_lm.dump_state() == lm.dump_state()


def test_reasoning_model_load_state_round_trips_canonical_state(make_run):
    lm = LM(model="openai/gpt-5-nano", temperature=1.0, max_tokens=16000, num_retries=1)
    loaded_lm = BaseLM.load_state(lm.dump_state())
    assert isinstance(loaded_lm, LM)
    assert loaded_lm.kwargs["max_tokens"] == 16000
    assert loaded_lm.dump_state() == lm.dump_state()


def test_reasoning_model_load_state_rejects_max_completion_tokens_alias(make_run):
    state = {
        "_dspy_lm_class": "dspy.clients.lm.LM",
        "model": "openai/gpt-5-nano",
        "model_type": "chat",
        "num_retries": 1,
        "temperature": 1.0,
        "max_completion_tokens": 16000,
    }
    with pytest.raises(ValueError, match="max_completion_tokens"):
        BaseLM.load_state(state)


def test_lm_load_state_forwards_allow_custom_lm_class(monkeypatch, make_run):
    calls = []
    original_load_state = BaseLM.load_state.__func__

    def spy_load_state(cls, state, *, allow_custom_lm_class=False):
        calls.append(allow_custom_lm_class)
        return original_load_state(cls, state, allow_custom_lm_class=allow_custom_lm_class)

    monkeypatch.setattr(BaseLM, "load_state", classmethod(spy_load_state))
    LM.load_state(LM("openai/gpt-4o-mini").dump_state(), allow_custom_lm_class=True)
    assert calls == [True]


def test_base_lm_load_state_error_names_allow_custom_lm_class() -> None:
    state = {
        LM_CLASS_STATE_KEY: "tests.predict.test_predict.CustomStateLM",
        "model": "custom-model",
        "model_type": "chat",
    }
    with pytest.raises(ValueError, match="allow_custom_lm_class"):
        BaseLM.load_state(state)


def test_logprobs_included_when_requested(make_run):
    lm = LM(model="dspy-test-model")
    with mock.patch("litellm.acompletion") as mock_completion:
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
        result = run_async(lm(_request(lm, prompt="question", logprobs=True), run=make_run(lm=lm)))
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
async def test_async_lm_call(make_run):
    from litellm.utils import Choices, Message, ModelResponse

    mock_response = ModelResponse(choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini")
    with patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = mock_response
        lm = LM(model="openai/gpt-4o-mini")
        result = await lm(_request(lm, prompt="question"), run=make_run(lm=lm))
        assert result.text == "answer"
        mock_acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_lm_history_size_limit(make_run):
    lm = LM(model="openai/gpt-4o-mini")
    run = make_run(lm=lm, telemetry=TelemetryConfig(max_call_log_entries=5))
    with mock.patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="test answer"))], model="openai/gpt-4o-mini"
        )
        for _ in range(10):
            await lm(_request(lm, prompt="query"), run=run)
    assert len(lm.call_log) == 5


def test_disable_history(make_run):
    lm = LM(model="openai/gpt-4o-mini")
    run = make_run(lm=lm, telemetry=TelemetryConfig(call_log=CallLogMode.off))
    with mock.patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="test answer"))], model="openai/gpt-4o-mini"
        )
        for _ in range(10):
            run_async(lm(_request(lm, prompt="query"), run=run))
    assert len(lm.call_log) == 0
    with mock.patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="test answer"))], model="openai/gpt-4o-mini"
        )


def test_call_reasoning_model_with_chat_api(make_run):
    message = Message(content="The answer is 4", role="assistant")
    message.reasoning_content = "Step 1: I need to add 2 + 2\nStep 2: 2 + 2 = 4\nTherefore, the answer is 4"
    mock_choice = Choices(message=message)
    mock_response = ModelResponse(
        choices=[mock_choice],
        model="anthropic/claude-3-7-sonnet-20250219",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    with (
        mock.patch("litellm.acompletion", return_value=mock_response) as mock_completion,
        mock.patch("litellm.supports_reasoning", return_value=True),
    ):
        lm = LM(
            model="anthropic/claude-3-7-sonnet-20250219",
            model_type="chat",
            temperature=1.0,
            max_tokens=16000,
            provider_options=LMProviderOptions(extensions={"reasoning": {"effort": "low"}}),
        )
        result = run_async(lm(_request(lm, prompt="What is 2 + 2?"), run=make_run(lm=lm)))
        assert result.text == "The answer is 4"
        assert result.reasoning_content is not None
        assert "Step 1" in result.reasoning_content
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
        provider_options=LMProviderOptions(api_key="sk-test-api-key-12345"),
    )
    predict = Predict(ts("question -> answer"))
    predict.lm = lm
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "program.json"
        predict.save(json_path)
        with open(json_path) as f:
            saved_state = json.load(f)
        lm_state = saved_state.get("lm", {})
        assert "api_key" not in lm_state, "API key should not be saved in JSON"
        assert "api_key" not in lm_state.get("_dspy_provider_options", {}), "API key should not be saved in JSON"
        assert saved_state["lm"]["model"] == "openai/gpt-4o-mini"
        assert saved_state["lm"]["temperature"] == 1.0
        assert saved_state["lm"]["max_tokens"] == 100

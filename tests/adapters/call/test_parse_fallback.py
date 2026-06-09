from typing import Any, cast
from unittest import mock

import pytest

from dspy.adapters.call.policies.parse_fallback import NoOpParseFallbackPolicy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.errors import AdapterParseError, LMError
from tests.adapters.conftest import Choices, Message, ModelResponse, make_adapter_run
from tests.task_spec.helpers import ts


@pytest.mark.asyncio
async def test_chat_adapter_fallback_to_json_adapter_on_parse_error():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter()
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            model="openai/gpt-4o-mini",
        )
        lm = LM("openai/gpt-4o-mini")
        result = await adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "What is the capital of France?"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
        assert result == [{"answer": "Paris"}]
        assert mock_completion.call_count == 2


@pytest.mark.asyncio
async def test_chat_adapter_fallback_preserves_native_function_calling_flag():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter(use_native_function_calling=False)
    seen = {}

    original_factory = cast("Any", adapter.parse_fallback_policy)._fallback_factory

    def tracking_factory():
        fallback = original_factory()
        seen["use_native_function_calling"] = fallback.use_native_function_calling
        return fallback

    cast("Any", adapter.parse_fallback_policy)._fallback_factory = tracking_factory
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            model="openai/gpt-4o-mini",
        )
        lm = LM("openai/gpt-4o-mini")
        result = await adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "What is the capital of France?"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    assert result == [{"answer": "Paris"}]
    assert seen["use_native_function_calling"] is False


@pytest.mark.asyncio
async def test_chat_adapter_respects_disabled_parse_fallback():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter(parse_fallback_policy=NoOpParseFallbackPolicy())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="nonsense"))],
            model="openai/gpt-4o-mini",
        )
        lm = LM("openai/gpt-4o-mini")
        with pytest.raises(AdapterParseError):
            await adapter(
                lm=lm,
                config={},
                task_spec=signature,
                demos=[],
                inputs={"question": "What is the capital of France?"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        assert mock_completion.call_count == 1


@pytest.mark.asyncio
async def test_fallback_does_not_run_on_lm_error():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = ChatAdapter()
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.side_effect = LMError("rate limited", provider="openai")
        lm = LM("openai/gpt-4o-mini")
        with pytest.raises(LMError):
            await adapter(
                lm=lm,
                config={},
                task_spec=signature,
                demos=[],
                inputs={"question": "What is the capital of France?"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        assert mock_completion.call_count == 1

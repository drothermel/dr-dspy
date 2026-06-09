import asyncio
import time
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    import litellm
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from openai import RateLimitError

from dspy.clients.lm import LM
from dspy.errors import ContextWindowExceededError, LMError, LMRateLimitError, LMUnexpectedError
from tests.clients.lm.conftest import _request


def test_lm_wraps_litellm_errors_with_metadata(make_run):
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


def test_lm_wraps_litellm_context_window_error(make_run):
    lm = LM("openai/gpt-4o-mini")
    error = litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model="gpt-4o")
    wrapped = lm._wrap_litellm_exception(error)
    assert isinstance(wrapped, ContextWindowExceededError)
    assert isinstance(wrapped, LMError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"


def test_lm_wraps_unknown_boundary_error_as_unexpected_error(make_run):
    lm = LM("openai/gpt-4o-mini")
    wrapped = lm._wrap_litellm_exception(RuntimeError("local boundary failure"))
    assert isinstance(wrapped, LMUnexpectedError)
    assert wrapped.code == "unexpected"
    assert wrapped.model == "openai/gpt-4o-mini"


def test_lm_preserves_existing_lm_error_without_self_cause(make_run):
    error = LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = LM("openai/gpt-4o-mini")
    with (
        mock.patch("dspy.clients.lm.alitellm_completion", side_effect=error),
        pytest.raises(LMRateLimitError) as exc_info,
    ):
        asyncio.run(lm(_request(lm, prompt="question"), run=make_run(lm=lm)))
    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


async def test_lm_preserves_existing_lm_error_without_self_cause_async(make_run):
    error = LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = LM("openai/gpt-4o-mini")
    with (
        mock.patch("dspy.clients.lm.alitellm_completion", side_effect=error),
        pytest.raises(LMRateLimitError) as exc_info,
    ):
        await lm(_request(lm, prompt="question"), run=make_run(lm=lm))
    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


def test_retry_number_set_correctly(make_run):
    lm = LM("openai/gpt-4o-mini", num_retries=3)
    mock_response = ModelResponse(choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", mock.AsyncMock(return_value=mock_response)) as mock_completion:
        asyncio.run(lm(_request(lm, prompt="query"), run=make_run(lm=lm)))
    assert mock_completion.call_args.kwargs["num_retries"] == 3


def test_retry_made_on_system_errors(make_run):
    retry_tracking = [0]

    def mock_create(*args: object, **kwargs: object):
        retry_tracking[0] += 1
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = LM(model="openai/gpt-4o-mini", max_tokens=250, num_retries=3)
    with (
        mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create),
        pytest.raises(LMRateLimitError),
    ):
        asyncio.run(lm(_request(lm, prompt="question"), run=make_run(lm=lm)))
    assert retry_tracking[0] == 4


def test_exponential_backoff_retry(make_run):
    time_counter = []

    def mock_create(*args: object, **kwargs: object):
        time_counter.append(time.time())
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = LM(model="openai/gpt-3.5-turbo", max_tokens=250, num_retries=3)
    with (
        mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create),
        pytest.raises(LMRateLimitError),
    ):
        asyncio.run(lm(_request(lm, prompt="question"), run=make_run(lm=lm)))
    for i in range(1, len(time_counter) - 1):
        assert time_counter[i + 1] - time_counter[i] >= 2 ** (i - 1)

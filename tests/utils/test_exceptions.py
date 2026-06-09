from dspy.utils.exceptions import (
    AdapterParseError,
    ContextWindowExceededError,
    DSPyError,
    LMAuthError,
    LMError,
    LMInvalidRequestError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnexpectedError,
    is_retryable_lm_error,
)
from tests.task_spec.helpers import ts


def test_lm_errors_are_exported_from_dspy():
    assert DSPyError is not None
    assert LMError is LMError
    assert LMUnexpectedError is not None
    assert AdapterParseError is AdapterParseError
    assert is_retryable_lm_error is not None


def test_retryable_lm_errors_classification():
    assert is_retryable_lm_error(LMRateLimitError())
    assert is_retryable_lm_error(LMTimeoutError())
    assert is_retryable_lm_error(LMServerError())
    assert is_retryable_lm_error(LMTransportError())
    assert not is_retryable_lm_error(LMAuthError())
    assert not is_retryable_lm_error(LMInvalidRequestError())
    assert not is_retryable_lm_error(LMUnexpectedError())
    assert not is_retryable_lm_error(ValueError("not an LM error"))


def test_lm_error_metadata():
    error = LMRateLimitError(
        "rate limited", model="openai/gpt-4o", provider="openai", status=429, request_id="req-123", retry_after=2.5
    )
    assert error.code == "rate_limit"
    assert error.model == "openai/gpt-4o"
    assert error.provider == "openai"
    assert error.status == 429
    assert error.request_id == "req-123"
    assert error.retry_after == 2.5


def test_context_window_exceeded_error():
    error = ContextWindowExceededError(message="Too many tokens")
    assert str(error) == "Too many tokens"


def test_adapter_parse_error_basic():
    adapter_name = "ChatAdapter"
    task_spec = ts("question->answer1, answer2")
    lm_response = "[[ ## answer1 ## ]]\nanswer1"
    error = AdapterParseError(adapter_name=adapter_name, task_spec=task_spec, lm_response=lm_response)
    assert isinstance(error, DSPyError)
    assert error.code == "adapter_parse_error"
    assert error.adapter_name == adapter_name
    assert error.task_spec == task_spec
    assert error.lm_response == lm_response
    error_message = str(error)
    assert (
        error_message
        == "Adapter ChatAdapter failed to parse the LM response. \n\nLM Response: [[ ## answer1 ## ]]\nanswer1 \n\nExpected to find output fields in the LM response: [answer1, answer2] \n\n"
    )


def test_adapter_parse_error_with_message():
    adapter_name = "ChatAdapter"
    task_spec = ts("question->answer1, answer2")
    lm_response = "[[ ## answer1 ## ]]\nanswer1"
    message = "Critical error, please fix!"
    error = AdapterParseError(adapter_name=adapter_name, task_spec=task_spec, lm_response=lm_response, message=message)
    assert error.adapter_name == adapter_name
    assert error.task_spec == task_spec
    assert error.lm_response == lm_response
    error_message = str(error)
    assert (
        error_message
        == "Critical error, please fix!\n\nAdapter ChatAdapter failed to parse the LM response. \n\nLM Response: [[ ## answer1 ## ]]\nanswer1 \n\nExpected to find output fields in the LM response: [answer1, answer2] \n\n"
    )


def test_adapter_parse_error_truncates_long_lm_response_in_message():
    adapter_name = "ChatAdapter"
    task_spec = ts("question->answer")
    lm_response = "x" * 5000
    error = AdapterParseError(adapter_name=adapter_name, task_spec=task_spec, lm_response=lm_response)
    assert error.lm_response == lm_response
    error_message = str(error)
    assert "[truncated" in error_message
    assert len(error_message) < len(lm_response)


def test_adapter_parse_error_with_parsed_result():
    adapter_name = "ChatAdapter"
    task_spec = ts("question->answer1, answer2")
    lm_response = "[[ ## answer1 ## ]]\nanswer1"
    parsed_result = {"answer1": "value1"}
    error = AdapterParseError(
        adapter_name=adapter_name, task_spec=task_spec, lm_response=lm_response, parsed_result=parsed_result
    )
    assert error.parsed_result == parsed_result
    error_message = str(error)
    assert (
        error_message
        == "Adapter ChatAdapter failed to parse the LM response. \n\nLM Response: [[ ## answer1 ## ]]\nanswer1 \n\nExpected to find output fields in the LM response: [answer1, answer2] \n\nActual output fields parsed from the LM response: [answer1] \n\n"
    )

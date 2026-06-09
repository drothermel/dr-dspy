from __future__ import annotations

from dspy.clients.errors import (
    _exception_message,
    _exception_provider_code,
    _exception_request_id,
    _exception_retry_after,
    _exception_status,
    _lm_error_class_from_status,
)
from dspy.errors import LMInvalidRequestError, LMRateLimitError, LMServerError


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response: _Response | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response


def test_exception_status_from_status_code() -> None:
    assert _exception_status(_StatusError("err", status_code=429)) == 429


def test_exception_status_from_response() -> None:
    exc = _StatusError("err", response=_Response(503))
    assert _exception_status(exc) == 503


def test_exception_message_prefers_message_attr() -> None:
    assert _exception_message(_StatusError("explicit")) == "explicit"


def test_exception_request_id_from_header() -> None:
    exc = _StatusError("err", response=_Response(500, headers={"x-request-id": "req-abc"}))
    assert _exception_request_id(exc) == "req-abc"


def test_exception_retry_after_from_header() -> None:
    exc = _StatusError("err", response=_Response(429, headers={"retry-after": "2.5"}))
    assert _exception_retry_after(exc) == 2.5


def test_exception_provider_code_from_body() -> None:
    class _BodyError(Exception):
        def __init__(self) -> None:
            super().__init__()
            self.body = {"error": {"code": "rate_limit_exceeded"}}

    assert _exception_provider_code(_BodyError()) == "rate_limit_exceeded"


def test_lm_error_class_from_status() -> None:
    assert _lm_error_class_from_status(429) is LMRateLimitError
    assert _lm_error_class_from_status(400) is LMInvalidRequestError
    assert _lm_error_class_from_status(503) is LMServerError

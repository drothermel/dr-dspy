from dspy.errors import (
    LMAuthError,
    LMBillingError,
    LMError,
    LMInvalidRequestError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMUnexpectedError,
    LMUnsupportedModelError,
)


def _lm_error_class_from_status(status: int | None) -> type[LMError]:
    if status in (401, 403):
        return LMAuthError
    if status == 402:
        return LMBillingError
    if status == 404:
        return LMUnsupportedModelError
    if status == 408:
        return LMTimeoutError
    if status == 429:
        return LMRateLimitError
    if status is not None and 400 <= status < 500:
        return LMInvalidRequestError
    if status is not None and status >= 500:
        return LMServerError
    return LMUnexpectedError if status is None else LMProviderError


def _exception_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _exception_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if message is None:
        message = str(exc)
    return str(message)


def _exception_headers(exc: Exception):
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or getattr(exc, "headers", None) or {}


def _exception_header(exc: Exception, name: str) -> str | None:
    headers = _exception_headers(exc)
    if not headers:
        return None
    try:
        return headers.get(name) or headers.get(name.lower())
    except AttributeError:
        return None


def _exception_request_id(exc: Exception) -> str | None:
    return (
        _exception_header(exc, "x-request-id")
        or _exception_header(exc, "request-id")
        or _exception_header(exc, "x-amzn-requestid")
        or _exception_header(exc, "x-ms-request-id")
    )


def _exception_retry_after(exc: Exception) -> float | None:
    retry_after = _exception_header(exc, "retry-after")
    try:
        return float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        return None


def _exception_provider_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") is not None:
            return str(error["code"])
        if body.get("code") is not None:
            return str(body["code"])
    return None

from dspy.clients.errors import (
    _exception_message,
    _exception_status,
)
from dspy.errors import (
    LMAuthError,
    LMBillingError,
    LMError,
    LMInvalidRequestError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnsupportedFeatureError,
    LMUnsupportedModelError,
)


def _safe_litellm_exception_class(name: str) -> type[Exception] | None:
    from dspy.clients.lm.litellm_access import _get_litellm

    cls = getattr(_get_litellm(), name, None)
    return cls if isinstance(cls, type) and issubclass(cls, Exception) else None


def _lm_error_class_from_litellm_exception(exc: Exception) -> type[LMError] | None:
    message = _exception_message(exc).lower()
    class_name = type(exc).__name__.lower()
    if _exception_status(exc) is None and any(
        phrase in message for phrase in ("api key", "apikey", "credentials", "environment variable")
    ):
        return LMNotConfiguredError
    if "timeout" in class_name or "timed out" in message or "timeout" in message:
        return LMTimeoutError
    if "connection" in class_name or "network" in message or "connection" in message:
        return LMTransportError
    mappings = [
        ("AuthenticationError", LMAuthError),
        ("RateLimitError", LMRateLimitError),
        ("NotFoundError", LMUnsupportedModelError),
        ("UnsupportedParamsError", LMUnsupportedFeatureError),
        ("UnprocessableEntityError", LMInvalidRequestError),
        ("ContentPolicyViolationError", LMInvalidRequestError),
        ("BadRequestError", LMInvalidRequestError),
        ("InvalidRequestError", LMInvalidRequestError),
        ("InternalServerError", LMServerError),
        ("ServiceUnavailableError", LMServerError),
        ("APIConnectionError", LMTransportError),
        ("APIResponseValidationError", LMProviderError),
        ("BudgetExceededError", LMBillingError),
        ("RouterRateLimitError", LMRateLimitError),
    ]
    for litellm_name, dspy_cls in mappings:
        litellm_cls = _safe_litellm_exception_class(litellm_name)
        if litellm_cls is not None and isinstance(exc, litellm_cls):
            return dspy_cls
    return None

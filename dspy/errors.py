from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.task_spec import TaskSpec


class DSPyError(Exception):
    """Base exception for DSPy runtime and adapter failures."""

    default_code: str | None = None

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        provider_code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.message = message
        self.code = code or self.default_code
        self.model = model
        self.provider = provider
        self.provider_code = provider_code
        self.status = status
        self.request_id = request_id
        self.retry_after = retry_after
        prefix = f"[{model}] " if model else ""
        super().__init__(f"{prefix}{message}" if message else prefix.rstrip())


class LMError(DSPyError):
    """Raised when an LM call fails before a successful response is returned."""

    default_code = "lm_error"


class LMTransportError(LMError):
    """Raised for network or client transport failures reaching the provider."""

    default_code = "transport"


class LMConfigurationError(LMError):
    """Raised when LM configuration is invalid or incomplete."""

    default_code = "configuration"


class LMNotConfiguredError(LMConfigurationError):
    """Raised when no LM is configured on the active run context."""

    default_code = "not_configured"


class LMUnsupportedFeatureError(LMError):
    """Raised when a request uses features unsupported by the active LM backend."""

    default_code = "unsupported_feature"

    def __init__(
        self, message: str = "", *, features: list[str] | None = None, issues: list[str] | None = None, **kwargs: Any
    ) -> None:
        self.features = list(features or [])
        self.issues = list(issues or [])
        super().__init__(message, **kwargs)


class LMProviderError(LMError):
    """Raised when the upstream provider returns an error response."""

    default_code = "provider"


class LMUnexpectedError(LMError):
    """Raised for unexpected LM failures that do not match a known provider category."""

    default_code = "unexpected"


class LMAuthError(LMProviderError):
    """Raised for authentication or authorization failures from the provider."""

    default_code = "auth"


class LMBillingError(LMProviderError):
    """Raised when the provider rejects a request due to billing or quota limits."""

    default_code = "billing"


class LMRateLimitError(LMProviderError):
    """Raised when the provider rate-limits the request."""

    default_code = "rate_limit"


class LMInvalidRequestError(LMProviderError):
    """Raised when the provider rejects the request as malformed or invalid."""

    default_code = "invalid_request"


class ContextWindowExceededError(LMInvalidRequestError):
    """Raised when the provider reports the prompt exceeds the model context window."""

    default_code = "context_window_exceeded"

    def __init__(self, *, model: str | None = None, message: str = "Context window exceeded", **kwargs: Any) -> None:
        super().__init__(message, model=model, **kwargs)


class LMUnsupportedModelError(LMInvalidRequestError):
    """Raised when the requested model is unknown or unsupported by the provider."""

    default_code = "unsupported_model"


class LMTimeoutError(LMProviderError):
    """Raised when the provider or client times out waiting for a response."""

    default_code = "timeout"


class LMServerError(LMProviderError):
    """Raised for transient upstream server errors from the provider."""

    default_code = "server"


_RETRYABLE_LM_ERRORS = (LMRateLimitError, LMTimeoutError, LMServerError, LMTransportError)


def is_retryable_lm_error(error: Exception) -> bool:
    return isinstance(error, _RETRYABLE_LM_ERRORS)


def _format_adapter_parse_message(
    *,
    adapter_name: str,
    task_spec: TaskSpec,
    lm_response: str,
    message: str | None = None,
    parsed_result: dict[str, Any] | None = None,
    max_response_chars: int = 4096,
) -> str:
    prefix = f"{message}\n\n" if message else ""
    if len(lm_response) > max_response_chars:
        truncated = lm_response[:max_response_chars]
        response_text = f"{truncated}… [truncated, {len(lm_response)} chars total]"
    else:
        response_text = lm_response
    text = (
        f"{prefix}Adapter {adapter_name} failed to parse the LM response. \n\n"
        f"LM Response: {response_text} \n\n"
        f"Expected to find output fields in the LM response: [{', '.join(task_spec.output_fields.keys())}] \n\n"
    )
    if parsed_result is not None:
        text += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"
    return text


class AdapterParseError(DSPyError):
    """Raised when an adapter cannot parse structured fields from an LM response."""

    default_code = "adapter_parse_error"

    def __init__(
        self,
        adapter_name: str,
        task_spec: TaskSpec,
        lm_response: str,
        message: str | None = None,
        parsed_result: dict[str, Any] | None = None,
    ) -> None:
        self.adapter_name = adapter_name
        self.task_spec = task_spec
        self.lm_response = lm_response
        self.parsed_result = parsed_result
        super().__init__(
            _format_adapter_parse_message(
                adapter_name=adapter_name,
                task_spec=task_spec,
                lm_response=lm_response,
                message=message,
                parsed_result=parsed_result,
            )
        )

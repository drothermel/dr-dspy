from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.task_spec import TaskSpec


class DSPyError(Exception):
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
    default_code = "lm_error"


class LMTransportError(LMError):
    default_code = "transport"


class LMConfigurationError(LMError):
    default_code = "configuration"


class LMNotConfiguredError(LMConfigurationError):
    default_code = "not_configured"


class LMUnsupportedFeatureError(LMError):
    default_code = "unsupported_feature"

    def __init__(
        self, message: str = "", *, features: list[str] | None = None, issues: list[str] | None = None, **kwargs: Any
    ) -> None:
        self.features = list(features or [])
        self.issues = list(issues or [])
        super().__init__(message, **kwargs)


class LMProviderError(LMError):
    default_code = "provider"


class LMUnexpectedError(LMError):
    default_code = "unexpected"


class LMAuthError(LMProviderError):
    default_code = "auth"


class LMBillingError(LMProviderError):
    default_code = "billing"


class LMRateLimitError(LMProviderError):
    default_code = "rate_limit"


class LMInvalidRequestError(LMProviderError):
    default_code = "invalid_request"


class ContextWindowExceededError(LMInvalidRequestError):
    default_code = "context_window_exceeded"

    def __init__(self, *, model: str | None = None, message: str = "Context window exceeded", **kwargs: Any) -> None:
        super().__init__(message, model=model, **kwargs)


class LMUnsupportedModelError(LMInvalidRequestError):
    default_code = "unsupported_model"


class LMTimeoutError(LMProviderError):
    default_code = "timeout"


class LMServerError(LMProviderError):
    default_code = "server"


_RETRYABLE_LM_ERRORS = (LMRateLimitError, LMTimeoutError, LMServerError, LMTransportError)


def is_retryable_lm_error(error: Exception) -> bool:
    return isinstance(error, _RETRYABLE_LM_ERRORS)


class AdapterParseError(DSPyError):
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
        message = f"{message}\n\n" if message else ""
        message = f"{message}Adapter {adapter_name} failed to parse the LM response. \n\nLM Response: {lm_response} \n\nExpected to find output fields in the LM response: [{', '.join(task_spec.output_fields.keys())}] \n\n"
        if parsed_result is not None:
            message += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"
        super().__init__(message)

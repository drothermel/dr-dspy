from __future__ import annotations

from dr_llm.backends.errors import (
    BackendAcquireTimeoutError,
    BackendDrainTimeoutError,
    BackendGenerationError,
    BackendSchemaError,
    BackendUnsupportedFeatureError,
    BackendValidationError,
)
from dr_llm.errors import ProviderSemanticError, ProviderTransportError

from dspy.clients.lm.errors import (
    _exception_message,
    _exception_status,
    _lm_error_class_from_status,
)
from dspy.utils.exceptions import (
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMProviderError,
    LMTimeoutError,
    LMTransportError,
    LMUnsupportedFeatureError,
)


def wrap_backend_exception(exc: Exception, *, model: str | None = None) -> LMError:
    if isinstance(exc, LMError):
        return exc
    if isinstance(exc, BackendUnsupportedFeatureError):
        return LMUnsupportedFeatureError(
            str(exc),
            model=model,
            features=["dr_llm_backend_v1"],
        )
    if isinstance(exc, (BackendAcquireTimeoutError, BackendDrainTimeoutError)):
        return LMTimeoutError(str(exc), model=model)
    if isinstance(exc, BackendGenerationError):
        return LMProviderError(str(exc), model=model)
    if isinstance(exc, (BackendSchemaError, BackendValidationError)):
        return LMConfigurationError(str(exc), model=model)
    if isinstance(exc, ProviderTransportError):
        return LMTransportError(str(exc), model=model)
    if isinstance(exc, ProviderSemanticError):
        return LMInvalidRequestError(str(exc), model=model)
    status = _exception_status(exc)
    message = _exception_message(exc)
    exc_cls = _lm_error_class_from_status(status)
    return exc_cls(message or str(exc), model=model, status=status)

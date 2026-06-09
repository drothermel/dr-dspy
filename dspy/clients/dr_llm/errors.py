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
from dspy.errors import (
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
        features = getattr(exc, "features", None) or getattr(exc, "feature", None)
        if isinstance(features, str):
            feature_list = [features]
        elif isinstance(features, (list, tuple, set)):
            feature_list = [str(item) for item in features]
        else:
            feature_list = ["dr_llm_backend_v1"]
        return LMUnsupportedFeatureError(
            str(exc),
            model=model,
            features=feature_list,
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

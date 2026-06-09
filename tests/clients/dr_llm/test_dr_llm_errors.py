from __future__ import annotations

from dr_llm.backends.errors import BackendAcquireTimeoutError, BackendUnsupportedFeatureError
from dr_llm.errors import ProviderTransportError

from dspy.clients.dr_llm.errors import wrap_backend_exception
from dspy.errors import LMTimeoutError, LMTransportError, LMUnsupportedFeatureError


def test_wrap_backend_unsupported_feature() -> None:
    exc = wrap_backend_exception(BackendUnsupportedFeatureError("no tools"), model="openai/m")
    assert isinstance(exc, LMUnsupportedFeatureError)


def test_wrap_backend_acquire_timeout() -> None:
    exc = wrap_backend_exception(BackendAcquireTimeoutError("timed out"), model="openai/m")
    assert isinstance(exc, LMTimeoutError)


def test_wrap_provider_transport() -> None:
    exc = wrap_backend_exception(
        ProviderTransportError("network"),
        model="openai/m",
    )
    assert isinstance(exc, LMTransportError)

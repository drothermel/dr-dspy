from __future__ import annotations

from dr_llm.backends.errors import BackendAcquireTimeoutError, BackendUnsupportedFeatureError
from dr_llm.errors import ProviderTransportError

from dspy.clients.dr_llm.errors import wrap_backend_exception
from dspy.errors import LMServerError, LMTimeoutError, LMTransportError, LMUnexpectedError, LMUnsupportedFeatureError


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


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_wrap_backend_exception_maps_status_to_server_error() -> None:
    exc = wrap_backend_exception(_StatusError("upstream down", 503), model="openai/m")
    assert isinstance(exc, LMServerError)
    assert exc.status == 503


def test_wrap_backend_exception_maps_unknown_exception_without_status() -> None:
    exc = wrap_backend_exception(RuntimeError("opaque failure"), model="openai/m")
    assert isinstance(exc, LMUnexpectedError)

"""Shared test helpers and doubles for the DSPy test suite."""

from tests.test_utils.lm import (
    CapabilityStubLM,
    DummyLM,
    FailingLM,
    NativeToolCallLM,
    SequentialTextLM,
    SpyLM,
)
from tests.test_utils.require_env import require_env
from tests.test_utils.retrieval.dummy_vectorizer import DummyVectorizer

__all__ = [
    "CapabilityStubLM",
    "DummyLM",
    "DummyVectorizer",
    "FailingLM",
    "NativeToolCallLM",
    "SequentialTextLM",
    "SpyLM",
    "require_env",
]

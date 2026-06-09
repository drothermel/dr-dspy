"""LM test doubles for DSPy unit and integration tests."""

from tests.test_utils.lm.catalog import (
    CapabilityStubLM,
    FailingLM,
    NativeToolCallLM,
    SequentialTextLM,
)
from tests.test_utils.lm.dummy_lm import DummyLM
from tests.test_utils.lm.spy_lm import SpyLM

__all__ = [
    "CapabilityStubLM",
    "DummyLM",
    "FailingLM",
    "NativeToolCallLM",
    "SequentialTextLM",
    "SpyLM",
]

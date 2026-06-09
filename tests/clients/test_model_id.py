from __future__ import annotations

from dspy.clients.model_id import split_provider_model


def test_split_provider_model() -> None:
    assert split_provider_model("openai/gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert split_provider_model("gpt-4.1-mini") == ("openai", "gpt-4.1-mini")

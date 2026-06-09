from __future__ import annotations

from dspy.clients._litellm import get_litellm


def _get_litellm():
    return get_litellm(feature="dspy.clients.lm.LM")

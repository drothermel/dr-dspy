"""LMOutput assembly helpers for DummyLM."""

from __future__ import annotations

from typing import Any

from dspy.clients.openai_format.parse import provider_tool_call_to_part
from dspy.core.types import LMOutput, LMPart, LMTextPart, LMThinkingPart


def build_lm_output(current_output: Any, *, reasoning: bool) -> LMOutput:
    if isinstance(current_output, dict):
        parts: list[LMPart] = []
        text = current_output.get("text")
        if isinstance(text, str):
            parts.append(LMTextPart(text=text))
        if reasoning and (not any(isinstance(part, LMThinkingPart) for part in parts)):
            parts.append(LMThinkingPart(text="Some reasoning"))
        reasoning_content = current_output.get("reasoning_content")
        if isinstance(reasoning_content, str):
            parts.append(LMThinkingPart(text=reasoning_content))
        parts.extend(provider_tool_call_to_part(tool_call) for tool_call in current_output.get("tool_calls") or [])
        return LMOutput(parts=parts, provider_output=current_output)
    if current_output is None:
        return LMOutput(parts=[])
    parts: list[LMPart] = [LMTextPart(text=str(current_output))]
    if reasoning:
        parts.append(LMThinkingPart(text="Some reasoning"))
    return LMOutput(parts=parts, provider_output=current_output)

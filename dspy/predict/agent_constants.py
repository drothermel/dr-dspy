"""Shared agent loop constants: terminal tools."""

from __future__ import annotations

from dspy.history.agent_constants import (
    FIELD_CODE,
    FIELD_CODE_OUTPUT,
    FIELD_GENERATED_CODE,
    FIELD_NEXT_THOUGHT,
    FIELD_OBSERVATION,
    FIELD_OUTPUT,
    FIELD_REASONING,
    FIELD_THOUGHT,
    FIELD_TOOL_ARGS,
    FIELD_TOOL_CALLS,
    FIELD_TOOL_NAME,
    AgentKind,
)

__all__ = [
    "AgentKind",
    "AVATAR_TERMINAL_TOOL",
    "FIELD_CODE",
    "FIELD_CODE_OUTPUT",
    "FIELD_GENERATED_CODE",
    "FIELD_NEXT_THOUGHT",
    "FIELD_OBSERVATION",
    "FIELD_OUTPUT",
    "FIELD_REASONING",
    "FIELD_THOUGHT",
    "FIELD_TOOL_ARGS",
    "FIELD_TOOL_CALLS",
    "FIELD_TOOL_NAME",
    "REACT_TERMINAL_TOOL",
    "REACT_V2_TERMINAL_TOOL",
    "RLM_SUBMIT_TOOL",
]

REACT_TERMINAL_TOOL = "finish"
REACT_V2_TERMINAL_TOOL = "submit"
AVATAR_TERMINAL_TOOL = "Finish"
RLM_SUBMIT_TOOL = "SUBMIT"

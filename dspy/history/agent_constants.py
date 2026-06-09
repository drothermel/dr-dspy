"""Agent turn event kinds and shared field key literals."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AgentKind",
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
]


class AgentKind(StrEnum):
    REACT = "react"
    REACT_V2 = "react_v2"
    CODE_ACT = "code_act"
    AVATAR = "avatar"
    RLM = "rlm"
    TASK_IO = "task_io"


FIELD_THOUGHT = "thought"
FIELD_TOOL_NAME = "tool_name"
FIELD_TOOL_ARGS = "tool_args"
FIELD_OBSERVATION = "observation"
FIELD_NEXT_THOUGHT = "next_thought"
FIELD_TOOL_CALLS = "tool_calls"
FIELD_GENERATED_CODE = "generated_code"
FIELD_CODE_OUTPUT = "code_output"
FIELD_REASONING = "reasoning"
FIELD_CODE = "code"
FIELD_OUTPUT = "output"

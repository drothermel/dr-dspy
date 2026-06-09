"""Helpers for constructing discriminated turn dicts in tests."""

from __future__ import annotations

from typing import Any


def task_io_turn(**fields: Any) -> dict[str, Any]:
    return {"agent": "task_io", "fields": fields}


def react_v2_turn(
    *,
    pending_inputs: dict[str, Any] | None = None,
    next_thought: Any = None,
    tool_calls: Any = None,
    submit_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turn: dict[str, Any] = {"agent": "react_v2"}
    if pending_inputs is not None:
        turn["pending_inputs"] = pending_inputs
    if next_thought is not None:
        turn["next_thought"] = next_thought
    if tool_calls is not None:
        turn["tool_calls"] = tool_calls
    if submit_outputs is not None:
        turn["submit_outputs"] = submit_outputs
    return turn

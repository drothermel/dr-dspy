"""Serialization helpers for LM content parts."""

from __future__ import annotations

import json
from typing import Any

from dspy.core.types.parts.models import (
    LMPart,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
)


def _part_to_value(part: LMPart) -> Any:
    if isinstance(part, LMTextPart):
        return part.text
    if isinstance(part, LMThinkingPart):
        return part
    if isinstance(part, LMToolCallPart):
        return part
    if isinstance(part, LMRefusalPart):
        return part.text
    return part


def _finalize_stream_part(part: LMPart) -> LMPart:
    if isinstance(part, LMToolCallPart) and "args_buffer" in part.provider_data:
        return part.model_copy(update={"args": _parse_json_object_strict(part.provider_data["args_buffer"])})
    return part


def _tool_call_to_provider_dict(call: LMToolCallPart) -> dict[str, Any]:
    data = {
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args),
        },
    }
    if call.id is not None:
        data["id"] = call.id
    return data


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_object_strict(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Streamed tool-call arguments must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Streamed tool-call arguments must be a JSON object.")
    return parsed

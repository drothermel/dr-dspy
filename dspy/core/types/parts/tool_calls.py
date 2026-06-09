from __future__ import annotations

import json
from typing import Any

from dspy.core.types.parts.models import LMToolCallPart


def tool_call_part_to_openai(call: LMToolCallPart, *, include_provider_data: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if include_provider_data:
        data.update(call.provider_data)
    data["type"] = "function"
    data["function"] = {"name": call.name, "arguments": json.dumps(call.args)}
    if call.id is not None:
        data["id"] = call.id
    return data

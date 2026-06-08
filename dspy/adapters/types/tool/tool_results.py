from typing import Any

import pydantic

from .tool_calls import ToolCalls


class ToolCallResults(pydantic.BaseModel):
    class ToolCallResult(pydantic.BaseModel):
        call_id: str | None = None
        name: str
        value: Any
        is_error: bool = False

    tool_call_results: list[ToolCallResult]

    @classmethod
    def from_tool_calls_and_values(
        cls, tool_calls: list[ToolCalls.ToolCall] | ToolCalls, values: list[Any], is_errors: list[bool] | None = None
    ) -> "ToolCallResults":
        if isinstance(tool_calls, ToolCalls):
            tool_calls = tool_calls.tool_calls
        if len(tool_calls) != len(values):
            raise ValueError("`tool_calls` and `values` must have the same length.")
        if is_errors is None:
            is_errors = [False] * len(tool_calls)
        elif len(is_errors) != len(tool_calls):
            raise ValueError("`is_errors` must have the same length as `tool_calls` when provided.")
        return cls(
            tool_call_results=[
                cls.ToolCallResult(call_id=tool_call.id, name=tool_call.name, value=value, is_error=is_error)
                for tool_call, value, is_error in zip(tool_calls, values, is_errors, strict=True)
            ]
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data
        if isinstance(data, list):
            return {"tool_call_results": data}
        if isinstance(data, dict):
            if "tool_call_results" in data:
                return data
            if {"name", "value"}.issubset(data):
                return {"tool_call_results": [data]}
        raise ValueError(f"Received invalid value for `dspy.adapters.types.tool.ToolCallResults`: {data}")

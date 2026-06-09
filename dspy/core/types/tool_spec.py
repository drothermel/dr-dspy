from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dspy.core.types._from_value import _MISSING, config_data


class LMToolSpec(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class LMToolChoice(BaseModel):
    mode: Literal["auto", "required", "none"] = "auto"
    allowed: list[str] | None = None
    parallel: bool | None = None
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_value(cls, value: Any = None, **overrides: Any) -> LMToolChoice:
        data = config_data(value, str_field="mode")
        data.update({key: value for key, value in overrides.items() if value is not _MISSING})
        return cls(**data)


def coerce_tool_spec(tool: Any) -> LMToolSpec:
    if isinstance(tool, LMToolSpec):
        return tool
    if hasattr(tool, "to_lm_tool_spec"):
        return tool.to_lm_tool_spec()
    if isinstance(tool, dict):
        if "function" in tool:
            function = tool["function"]
            provider_data = {key: value for key, value in tool.items() if key not in {"type", "function"}}
            return LMToolSpec(
                name=function.get("name"),
                description=function.get("description"),
                parameters=function.get("parameters", {}),
                provider_data=provider_data,
            )
        return LMToolSpec(**tool)
    raise TypeError(f"Cannot convert {type(tool)!r} to LMToolSpec.")

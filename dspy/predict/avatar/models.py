from typing import Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    tool_name: Any = Field(..., description="Name of the tool to use.")
    tool_args: dict[str, Any] = Field(..., description="JSON arguments to pass to the tool.")


class ActionOutput(BaseModel):
    tool_name: str
    tool_args: dict[str, Any]
    tool_output: str | None

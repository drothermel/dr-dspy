from typing import Any

from pydantic import BaseModel

from dspy.history.avatar_action import Action

__all__ = ["Action", "ActionOutput"]


class ActionOutput(BaseModel):
    tool_name: str
    tool_args: dict[str, Any]
    tool_output: str | None

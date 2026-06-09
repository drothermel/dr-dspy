"""Avatar actor action shape (shared with turn events and predict.avatar)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["Action"]


class Action(BaseModel):
    tool_name: Any = Field(..., description="Name of the tool to use.")
    tool_args: dict[str, Any] = Field(..., description="JSON arguments to pass to the tool.")

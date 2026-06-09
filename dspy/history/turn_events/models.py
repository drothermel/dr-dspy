"""Per-agent turn event models and discriminated union."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dspy.history.agent_constants import AgentKind
from dspy.history.avatar_action import Action  # noqa: TC001 — runtime field type for AvatarTurnEvent

__all__ = [
    "AvatarTurnEvent",
    "CodeActTurnEvent",
    "ReActTurnEvent",
    "ReActV2TurnEvent",
    "RlmTurnEvent",
    "TaskIOTurnEvent",
    "TurnEvent",
]


class ReActTurnEvent(BaseModel):
    agent: Literal[AgentKind.REACT] = AgentKind.REACT
    thought: Any
    tool_name: str
    tool_args: dict[str, Any]
    observation: Any
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReActV2TurnEvent(BaseModel):
    agent: Literal[AgentKind.REACT_V2] = AgentKind.REACT_V2
    next_thought: Any | None = None
    tool_calls: Any | None = None
    pending_inputs: dict[str, Any] | None = None
    submit_outputs: dict[str, Any] | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _coerce_tool_calls(cls, value: Any) -> Any:
        if value is None:
            return None
        from dspy.adapters.types.tool import ToolCalls

        if isinstance(value, ToolCalls):
            return value
        return ToolCalls.model_validate(value)

    @model_validator(mode="after")
    def _require_content(self) -> ReActV2TurnEvent:
        if any(
            value is not None
            for value in (self.next_thought, self.tool_calls, self.pending_inputs, self.submit_outputs)
        ):
            return self
        raise ValueError(
            "ReActV2TurnEvent requires at least one of next_thought, tool_calls, pending_inputs, submit_outputs."
        )


class CodeActTurnEvent(BaseModel):
    agent: Literal[AgentKind.CODE_ACT] = AgentKind.CODE_ACT
    generated_code: str | None = None
    code_output: str | None = None
    observation: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _require_content(self) -> CodeActTurnEvent:
        if any(value is not None for value in (self.generated_code, self.code_output, self.observation)):
            return self
        raise ValueError("CodeActTurnEvent requires at least one of generated_code, code_output, observation.")


class AvatarTurnEvent(BaseModel):
    agent: Literal[AgentKind.AVATAR] = AgentKind.AVATAR
    action: Action
    result: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class RlmTurnEvent(BaseModel):
    agent: Literal[AgentKind.RLM] = AgentKind.RLM
    reasoning: str = ""
    code: str
    output: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskIOTurnEvent(BaseModel):
    agent: Literal[AgentKind.TASK_IO] = AgentKind.TASK_IO
    fields: dict[str, Any]
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _require_fields(self) -> TaskIOTurnEvent:
        if self.fields:
            return self
        raise ValueError("TaskIOTurnEvent.fields must be non-empty.")


TurnEvent = Annotated[
    ReActTurnEvent | ReActV2TurnEvent | CodeActTurnEvent | AvatarTurnEvent | RlmTurnEvent | TaskIOTurnEvent,
    Field(discriminator="agent"),
]

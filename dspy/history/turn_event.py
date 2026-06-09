"""Turn log event model.

Per-agent field contract (one turn):

| Agent   | Fields written per turn |
|---------|-------------------------|
| ReAct   | ``thought``, ``tool_name``, ``tool_args``, ``observation`` |
| ReActV2 | ``next_thought``, ``tool_calls`` (+ task extras in ``__pydantic_extra__``) |
| CodeAct | ``generated_code``, ``code_output``, ``observation`` |
| Avatar  | ``action``, ``result`` |
| RLM     | ``reasoning``, ``code``, ``output`` (via ``REPLHistory``; formatted string replay only) |
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TurnEvent(BaseModel):
    """Single agent turn log entry.

    Known agent fields are typed; task-specific input/output keys may appear as
    extra fields (e.g. ReActV2 pending inputs merged into the event).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    thought: Any | None = None
    next_thought: Any | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_calls: Any | None = None
    observation: Any | None = None
    generated_code: str | None = None
    code_output: str | None = None
    action: Any | None = None
    result: str | None = None
    reasoning: str | None = None
    code: str | None = None
    output: str | None = None

"""Turn event serialization helpers for adapter prompt replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.history.turn_events.models import ReActV2TurnEvent, TaskIOTurnEvent

if TYPE_CHECKING:
    from dspy.history.turn_events.models import TurnEvent

__all__ = ["turn_to_format_dict"]


def turn_to_format_dict(turn: TurnEvent) -> dict[str, Any]:
    """Flatten a discriminated turn to a task-field dict for prompt formatting."""
    if isinstance(turn, TaskIOTurnEvent):
        return dict(turn.fields)
    if isinstance(turn, ReActV2TurnEvent):
        message: dict[str, Any] = {}
        if turn.pending_inputs:
            message.update(turn.pending_inputs)
        if turn.next_thought is not None:
            message["next_thought"] = turn.next_thought
        if turn.tool_calls is not None:
            message["tool_calls"] = turn.tool_calls
        if turn.submit_outputs:
            message.update(turn.submit_outputs)
        return message
    return turn.model_dump(mode="json", exclude={"agent"}, exclude_none=True)

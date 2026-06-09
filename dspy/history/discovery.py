from __future__ import annotations

from typing import Any, get_args, get_origin

from dspy.history.repl_history import REPLHistory
from dspy.history.turn_log import TurnLog


def _is_type_or_optional(annotation: Any, target: type[object]) -> bool:
    if annotation is target:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return any(_is_type_or_optional(arg, target) for arg in get_args(annotation))
    return False


def is_conversation_turn_log_type(annotation: Any) -> bool:
    return _is_type_or_optional(annotation, TurnLog)


def is_repl_history_type(annotation: Any) -> bool:
    return _is_type_or_optional(annotation, REPLHistory)


def is_agent_history_type(annotation: Any) -> bool:
    return is_conversation_turn_log_type(annotation) or is_repl_history_type(annotation)

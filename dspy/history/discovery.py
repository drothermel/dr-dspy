from __future__ import annotations

from typing import Any, get_args, get_origin

from dspy.history.repl_history import REPLHistory
from dspy.history.turn_log import TurnLog


def is_conversation_turn_log_type(annotation: Any) -> bool:
    if annotation is TurnLog:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return any(is_conversation_turn_log_type(arg) for arg in get_args(annotation))
    return False


def is_repl_history_type(annotation: Any) -> bool:
    if annotation is REPLHistory:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return any(is_repl_history_type(arg) for arg in get_args(annotation))
    return False


def is_agent_history_type(annotation: Any) -> bool:
    return is_conversation_turn_log_type(annotation) or is_repl_history_type(annotation)

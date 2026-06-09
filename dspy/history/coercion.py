from __future__ import annotations

from typing import Any

from dspy.history.turn_log import TurnLog


def coerce_turn_log(turn_log: Any) -> TurnLog:
    if turn_log is None:
        return TurnLog.empty()
    if isinstance(turn_log, TurnLog):
        return turn_log
    if isinstance(turn_log, dict) and "messages" in turn_log:
        return TurnLog(turns=tuple(turn_log["messages"]))
    return TurnLog.model_validate(turn_log)

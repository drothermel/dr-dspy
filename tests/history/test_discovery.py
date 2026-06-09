from typing import Optional

from dspy.history import REPLHistory, TurnLog
from dspy.history.discovery import (
    is_agent_history_type,
    is_conversation_turn_log_type,
    is_repl_history_type,
)


def test_is_conversation_turn_log_type_detects_turn_log():
    assert is_conversation_turn_log_type(TurnLog)
    assert is_conversation_turn_log_type(Optional[TurnLog])


def test_is_repl_history_type_detects_repl_history():
    assert is_repl_history_type(REPLHistory)
    assert is_repl_history_type(Optional[REPLHistory])


def test_is_agent_history_type_detects_both():
    assert is_agent_history_type(TurnLog)
    assert is_agent_history_type(REPLHistory)
    assert not is_agent_history_type(str)

"""Turn log event models.

Per-agent field contract (one turn):

| Agent    | ``agent`` value | Fields |
|----------|-----------------|--------|
| ReAct    | ``react``       | ``thought``, ``tool_name``, ``tool_args``, ``observation`` |
| ReActV2  | ``react_v2``    | ``next_thought``, ``tool_calls``, optional ``pending_inputs``, ``submit_outputs`` |
| CodeAct  | ``code_act``    | ``generated_code``, ``code_output``, ``observation`` (at least one) |
| Avatar   | ``avatar``      | ``action``, ``result`` |
| RLM      | ``rlm``         | ``reasoning``, ``code``, ``output`` |
| Task I/O | ``task_io``     | ``fields`` (task input/output replay for demos and history) |
"""

from dspy.history.turn_events.models import (
    AvatarTurnEvent,
    CodeActTurnEvent,
    ReActTurnEvent,
    ReActV2TurnEvent,
    RlmTurnEvent,
    TaskIOTurnEvent,
    TurnEvent,
)

__all__ = [
    "AvatarTurnEvent",
    "CodeActTurnEvent",
    "ReActTurnEvent",
    "ReActV2TurnEvent",
    "RlmTurnEvent",
    "TaskIOTurnEvent",
    "TurnEvent",
]

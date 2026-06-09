from __future__ import annotations

from enum import StrEnum


class AgentTerminationReason(StrEnum):
    SUBMIT = "submit"
    FORCED_SUBMIT = "forced_submit"
    MAX_ITERS = "max_iters"
    PARSE_ERROR = "parse_error"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    EMPTY_TOOL_CALLS = "empty_tool_calls"
    EXTRACT_FAILED = "extract_failed"
    FAILED = "failed"

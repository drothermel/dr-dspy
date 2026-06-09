from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TextIO

from dspy.clients.openai_format.chat_request import request_messages_as_openai
from dspy.core.types import CallRecord
from dspy.runtime.config import disk_call_log_enabled

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dspy.runtime.run_context import RunContext
    from dspy.runtime.run_log_session import RunLogSession


def read_call_log_entries(call_log: Sequence[CallRecord], *, n: int) -> list[dict[str, Any]]:
    records = call_log[-n:]
    for entry in records:
        if not isinstance(entry, CallRecord):
            raise TypeError(f"call_log entry must be CallRecord, got {type(entry)!r}")
    return [{**entry.to_dict(), "messages": request_messages_as_openai(entry.request)} for entry in records]


def tail_disk_call_log(session: RunLogSession, *, n: int) -> list[dict[str, Any]]:
    if not session.calls_path.exists():
        return []
    lines = session.calls_path.read_text(encoding="utf-8").splitlines()
    tail = lines[-n:] if n > 0 else []
    return [json.loads(line) for line in tail if line.strip()]


def read_call_log_for_run(run: RunContext, *, n: int = 10) -> list[dict[str, Any]]:
    if run.call_log:
        return read_call_log_entries(run.call_log, n=n)
    if disk_call_log_enabled(run.telemetry) and run.log_session is not None:
        return tail_disk_call_log(run.log_session, n=n)
    return []


def inspect_call_log_for_run(run: RunContext, *, n: int = 1, file: TextIO | None = None) -> None:
    from dspy.runtime.inspect_call_log import pretty_print_call_log, pretty_print_disk_call_log

    if run.call_log:
        pretty_print_call_log(call_log=run.call_log, n=n, file=file)
        return
    if disk_call_log_enabled(run.telemetry) and run.log_session is not None:
        records = tail_disk_call_log(run.log_session, n=n)
        pretty_print_disk_call_log(records, n=n, file=file)

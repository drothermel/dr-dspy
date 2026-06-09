from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.runtime.active_run import get_caller_modules
from dspy.runtime.call_log.disk_record import build_disk_call_record
from dspy.runtime.config import memory_call_log_enabled
from dspy.runtime.run_log_session import append_call_record

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import CallRecord, LMRequest, LMResponse
    from dspy.runtime.run_context import RunContext
    from dspy.runtime.transparency.types import CompiledCall


def _append_bounded(entry_list: list[CallRecord], entry: CallRecord, max_entries: int) -> None:
    if len(entry_list) >= max_entries:
        entry_list.pop(0)
    entry_list.append(entry)


def record_call(*, entry: CallRecord, run: RunContext, lm: BaseLM) -> None:
    if not memory_call_log_enabled(run.telemetry):
        return
    max_entries = run.telemetry.max_call_log_entries
    _append_bounded(run.call_log, entry, max_entries)
    _append_bounded(lm.call_log, entry, max_entries)
    for module in get_caller_modules():
        _append_bounded(module.call_log, entry, max_entries)


def append_disk_call(
    *,
    request: LMRequest,
    response: LMResponse,
    call_record: CallRecord | None,
    run: RunContext,
    lm: BaseLM,
    compiled: CompiledCall | None = None,
) -> None:
    record = build_disk_call_record(
        request=request,
        response=response,
        call_record=call_record,
        lm=lm,
        compiled=compiled,
    )
    append_call_record(record, session=run.log_session)

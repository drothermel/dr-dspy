from __future__ import annotations

import datetime
import json
import threading
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime for session directory creation
from typing import TYPE_CHECKING, Any

from dspy.runtime.config import disk_call_log_enabled
from dspy.runtime.log_paths import resolve_log_root, resolve_run_bucket
from dspy.serialization.json import to_jsonable

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext

_SESSION_LOCK = threading.Lock()


@dataclass(frozen=True)
class RunLogSession:
    run_dir: Path
    calls_path: Path
    timestamp: str


def create_run_log_session(
    *, call_log_dir: str | None, settings_snapshot: dict[str, Any] | None = None
) -> RunLogSession:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = resolve_log_root(call_log_dir) / resolve_run_bucket() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = to_jsonable(settings_snapshot or {})
    run_json = {
        "timestamp": timestamp,
        "run_id": resolve_run_bucket(),
        "log_root": str(resolve_log_root(call_log_dir)),
        "settings": snapshot,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return RunLogSession(run_dir=run_dir, calls_path=run_dir / "calls.jsonl", timestamp=timestamp)


def append_call_record(record: dict[str, Any], *, session: RunLogSession | None) -> None:
    if session is None:
        return
    line = json.dumps(to_jsonable(record), ensure_ascii=False) + "\n"
    with _SESSION_LOCK, session.calls_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def build_settings_snapshot(run: RunContext) -> dict[str, Any]:
    return {
        "lm": run.lm.model if hasattr(run.lm, "model") else repr(run.lm),
        "adapter": type(run.adapter).__name__,
        "retrieval": repr(run.retrieval) if run.retrieval is not None else None,
        "execution": run.execution.model_dump(),
        "telemetry": run.telemetry.model_dump(),
    }


def init_log_session(run: RunContext) -> None:
    if not disk_call_log_enabled(run.telemetry):
        run.log_session = None
        return
    run.log_session = create_run_log_session(
        call_log_dir=run.telemetry.call_log_dir,
        settings_snapshot=build_settings_snapshot(run),
    )


def ensure_log_session(run: RunContext, *, explicit_log_session: bool) -> None:
    if not disk_call_log_enabled(run.telemetry):
        run.log_session = None
        return
    if explicit_log_session:
        return
    if run.log_session is None:
        init_log_session(run)
        return
    expected_root = resolve_log_root(run.telemetry.call_log_dir)
    if run.log_session.run_dir.parent.parent != expected_root / resolve_run_bucket():
        init_log_session(run)

import json

from dspy.adapters.json_adapter import JSONAdapter
from dspy.core.types import CallRecord, LMRequest, LMResponse
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig
from dspy.runtime.call_log.coordinator import append_disk_call, record_call
from dspy.runtime.call_log.disk_record import build_disk_call_record
from dspy.testing import DummyLM


class _StubModule:
    def __init__(self) -> None:
        self.call_log: list[CallRecord] = []


def _sample_record(*, timestamp: str = "2026-01-01T00:00:00+00:00") -> CallRecord:
    request = LMRequest(model="test-model", messages=[])
    response = LMResponse.from_text("ok")
    return CallRecord(
        request=request,
        response=response,
        timestamp=timestamp,
        uuid="record-uuid",
        model_type="chat",
    )


def test_record_call_bounded_append_truncates_oldest():
    lm = DummyLM([{"answer": "ok"}])
    run = RunContext.create(
        lm=lm,
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory, max_call_log_entries=2),
    )
    first = _sample_record(timestamp="t1")
    second = _sample_record(timestamp="t2")
    third = _sample_record(timestamp="t3")
    record_call(entry=first, run=run, lm=lm)
    record_call(entry=second, run=run, lm=lm)
    record_call(entry=third, run=run, lm=lm)
    assert [entry.timestamp for entry in run.call_log] == ["t2", "t3"]
    assert [entry.timestamp for entry in lm.call_log] == ["t2", "t3"]


def test_record_call_fan_out_shares_entry_identity():
    lm = DummyLM([{"answer": "ok"}])
    run = RunContext.create(
        lm=lm,
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    module = _StubModule()
    run.caller_modules.append(module)  # type: ignore[arg-type]
    entry = _sample_record()
    record_call(entry=entry, run=run, lm=lm)
    assert id(run.call_log[0]) == id(lm.call_log[0]) == id(module.call_log[0])


def test_build_disk_call_record_uses_call_record_timestamp():
    lm = DummyLM([{"answer": "ok"}])
    request = LMRequest(model="test-model", messages=[])
    response = LMResponse.from_text("ok")
    call_record = _sample_record(timestamp="shared-ts")
    record = build_disk_call_record(request=request, response=response, call_record=call_record, lm=lm)
    assert record["timestamp"] == "shared-ts"
    assert record["call_id"] == "record-uuid"


def test_append_disk_call_noops_when_session_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    lm = DummyLM([{"answer": "ok"}])
    run = RunContext.create(
        lm=lm,
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    run.log_session = None
    request = LMRequest(model="test-model", messages=[])
    response = LMResponse.from_text("ok")
    append_disk_call(request=request, response=response, call_record=None, run=run, lm=lm)
    assert list(tmp_path.rglob("calls.jsonl")) == []


def test_append_disk_call_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    lm = DummyLM([{"answer": "ok"}])
    run = RunContext.create(
        lm=lm,
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.disk),
    )
    request = LMRequest(model="test-model", messages=[])
    response = LMResponse.from_text("ok")
    call_record = _sample_record()
    append_disk_call(request=request, response=response, call_record=call_record, run=run, lm=lm)
    assert run.log_session is not None
    lines = run.log_session.calls_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["call_id"] == "record-uuid"
    assert payload["timestamp"] == call_record.timestamp

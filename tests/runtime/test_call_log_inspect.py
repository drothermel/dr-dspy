import asyncio
import json
from pathlib import Path

from dspy.adapters.json_adapter import JSONAdapter
from dspy.predict.predict import Predict
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode
from dspy.runtime.call_log.inspect import read_call_log_for_run
from dspy.testing import DummyLM
from tests.runtime.test_transparency import SampleTaskSpec


def test_read_call_log_disk_only_returns_jsonl_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    json_adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=json_adapter)
    run = RunContext.create(
        lm=lm,
        adapter=json_adapter,
        telemetry=TelemetryConfig(call_log=CallLogMode.disk),
        init_run_log=True,
    )
    predict = Predict(SampleTaskSpec())
    asyncio.run(predict(question="What is the capital of France?", run=run))
    assert run.call_log == []
    records = read_call_log_for_run(run, n=1)
    assert len(records) == 1
    assert records[0]["response"]["outputs"][0]["text"]


def test_read_call_log_memory_prefers_memory_over_disk(tmp_path, monkeypatch, make_run):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    json_adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=json_adapter)
    run = make_run(
        lm=lm,
        adapter=json_adapter,
        telemetry=TelemetryConfig(transparency=TransparencyMode.strict, call_log=CallLogMode.both),
    )
    predict = Predict(SampleTaskSpec())
    asyncio.run(predict(question="What is the capital of France?", run=run))
    records = read_call_log_for_run(run, n=1)
    assert len(records) == 1
    assert "messages" in records[0]
    assert run.log_session is not None
    calls_files = list(Path(tmp_path).rglob("calls.jsonl"))
    assert len(calls_files) == 1
    disk_record = json.loads(calls_files[0].read_text(encoding="utf-8").strip())
    assert disk_record["caller"]["phase"] == "predict"

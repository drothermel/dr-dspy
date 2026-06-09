from typing import TYPE_CHECKING, cast

import pytest

from dspy.adapters.json_adapter import JSONAdapter
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode, resolve_run
from dspy.runtime.callback import NoOpCallback
from dspy.testing import DummyLM

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.core.types import CallRecord

_MEMORY_TELEMETRY = TelemetryConfig(call_log=CallLogMode.memory)


class _EchoCallback(NoOpCallback):
    pass


def test_create_requires_lm_and_adapter():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, telemetry=_MEMORY_TELEMETRY)
    assert run.lm is lm
    assert run.adapter is adapter


def test_create_rejects_missing_adapter():
    lm = DummyLM([{"answer": "ok"}])
    with pytest.raises(ValueError, match="adapter"):
        RunContext.create(lm=lm, adapter=cast("Adapter", None), telemetry=_MEMORY_TELEMETRY)


def test_fork_replaces_lm_and_clears_optimization_trace_override():
    lm = DummyLM([{"answer": "ok"}])
    other_lm = DummyLM([{"answer": "other"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, telemetry=_MEMORY_TELEMETRY)
    forked = run.fork(lm=other_lm, optimization_trace=[])
    assert forked.lm is other_lm
    assert forked.optimization_trace == []
    assert run.optimization_trace == []


def test_fork_copies_trace_by_default():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(
        lm=lm,
        adapter=adapter,
        optimization_trace=[("a",)],
        telemetry=_MEMORY_TELEMETRY,
    )
    forked = run.fork()
    assert forked.optimization_trace == [("a",)]
    forked.optimization_trace.append(("b",))
    assert run.optimization_trace == [("a",)]


def test_fork_nested_config_updates():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(
        lm=lm,
        adapter=adapter,
        telemetry=TelemetryConfig(transparency=TransparencyMode.off, call_log=CallLogMode.memory),
    )
    forked = run.fork(telemetry=TelemetryConfig(transparency=TransparencyMode.strict))
    assert forked.telemetry.transparency == TransparencyMode.strict
    assert run.telemetry.transparency == TransparencyMode.off


def test_resolve_run_prefers_call_override():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    bound = RunContext.create(lm=lm, adapter=adapter, telemetry=_MEMORY_TELEMETRY)
    override = bound.fork(optimization_trace=["override"])
    assert resolve_run(run=override, bound_run=bound) is override


def test_resolve_run_uses_bound_run():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    bound = RunContext.create(lm=lm, adapter=adapter, telemetry=_MEMORY_TELEMETRY)
    assert resolve_run(run=None, bound_run=bound) is bound


def test_resolve_run_raises_when_missing():
    with pytest.raises(RuntimeError, match="RunContext"):
        resolve_run(run=None, bound_run=None)


def test_default_telemetry_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    run = RunContext.create(lm=DummyLM([{"answer": "ok"}]), adapter=JSONAdapter())
    assert run.telemetry.transparency == TransparencyMode.strict
    assert run.telemetry.call_log == CallLogMode.both


def test_fork_callbacks_and_trace():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, callbacks=[_EchoCallback()], telemetry=_MEMORY_TELEMETRY)
    forked = run.fork(lm=DummyLM([{"answer": "other"}]), callbacks=[], optimization_trace=[1])
    assert len(forked.callbacks) == 0
    assert forked.optimization_trace == [1]
    assert len(run.callbacks) == 1


def test_default_execution_config():
    lm = DummyLM([{"answer": "ok"}])
    run = RunContext.create(lm=lm, adapter=JSONAdapter(), telemetry=_MEMORY_TELEMETRY)
    assert run.execution.max_concurrency == 8
    assert run.execution.max_errors == 10


def test_fork_rejects_unknown_kwargs():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, telemetry=_MEMORY_TELEMETRY)
    with pytest.raises(TypeError, match="telemtry"):
        run.fork(telemtry="strict")


def test_fork_enables_disk_logging_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(
        lm=lm,
        adapter=adapter,
        telemetry=TelemetryConfig(call_log=CallLogMode.off),
    )
    assert run.log_session is None
    forked = run.fork(telemetry=TelemetryConfig(call_log=CallLogMode.disk))
    assert forked.log_session is not None
    assert forked.log_session.run_dir.exists()
    assert (forked.log_session.run_dir / "run.json").exists()


def test_read_call_log_rejects_non_call_record():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, telemetry=_MEMORY_TELEMETRY)
    run.call_log.append(cast("CallRecord", object()))
    with pytest.raises(TypeError, match="call_log entry must be CallRecord"):
        run.read_call_log()


def test_fork_disables_disk_logging_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(
        lm=lm,
        adapter=adapter,
        telemetry=TelemetryConfig(call_log=CallLogMode.disk),
    )
    assert run.log_session is not None
    forked = run.fork(telemetry=TelemetryConfig(call_log=CallLogMode.off))
    assert forked.log_session is None

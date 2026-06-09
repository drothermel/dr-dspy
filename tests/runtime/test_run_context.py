import pytest

from dspy.adapters.json_adapter import JSONAdapter
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, resolve_run
from dspy.utils.dummies import DummyLM


def test_create_requires_lm_and_adapter():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, init_run_log=False)
    assert run.lm is lm
    assert run.adapter is adapter


def test_create_rejects_missing_adapter():
    lm = DummyLM([{"answer": "ok"}])
    with pytest.raises(ValueError, match="adapter"):
        RunContext.create(lm=lm, adapter=None, init_run_log=False)  # type: ignore[arg-type]


def test_fork_replaces_lm_and_clears_caller_modules():
    lm = DummyLM([{"answer": "ok"}])
    other_lm = DummyLM([{"answer": "other"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, init_run_log=False)
    run.caller_modules.append("parent")
    forked = run.fork(lm=other_lm, optimization_trace=[])
    assert forked.lm is other_lm
    assert forked.optimization_trace == []
    assert forked.caller_modules == []
    assert run.optimization_trace == []


def test_fork_copies_trace_by_default():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, optimization_trace=[("a",)], init_run_log=False)
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
        telemetry=TelemetryConfig(transparency="off"),
        init_run_log=False,
    )
    forked = run.fork(telemetry=TelemetryConfig(transparency="strict"))
    assert forked.telemetry.transparency == "strict"
    assert run.telemetry.transparency == "off"


def test_resolve_run_prefers_call_override():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    bound = RunContext.create(lm=lm, adapter=adapter, init_run_log=False)
    override = bound.fork(optimization_trace=["override"])
    assert resolve_run(run=override, bound_run=bound) is override


def test_resolve_run_uses_bound_run():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    bound = RunContext.create(lm=lm, adapter=adapter, init_run_log=False)
    assert resolve_run(run=None, bound_run=bound) is bound


def test_resolve_run_raises_when_missing():
    with pytest.raises(RuntimeError, match="RunContext"):
        resolve_run(run=None, bound_run=None)


def test_default_telemetry_config():
    run = RunContext.create(lm=DummyLM([{"answer": "ok"}]), adapter=JSONAdapter(), init_run_log=False)
    assert run.telemetry.transparency == "strict"
    assert run.telemetry.call_log == CallLogMode.both


def test_fork_callbacks_and_trace():
    lm = DummyLM([{"answer": "ok"}])
    adapter = JSONAdapter()
    run = RunContext.create(lm=lm, adapter=adapter, callbacks=[lambda x: x], init_run_log=False)
    forked = run.fork(lm=DummyLM([{"answer": "other"}]), callbacks=[], optimization_trace=[1])
    assert len(forked.callbacks) == 0
    assert forked.optimization_trace == [1]
    assert len(run.callbacks) == 1


def test_default_execution_config():
    lm = DummyLM([{"answer": "ok"}])
    run = RunContext.create(lm=lm, adapter=JSONAdapter(), init_run_log=False)
    assert run.execution.num_threads == 8
    assert run.execution.max_errors == 10

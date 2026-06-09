import asyncio

import pytest

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.tool import Tool
from dspy.predict.predict import Predict  # noqa: F401 — break primitives import cycle
from dspy.primitives import Module
from dspy.runtime import RunContext, TelemetryConfig
from dspy.runtime.active_run import call_scope, get_active_run, get_active_usage_tracker, get_caller_modules
from dspy.runtime.callback import NoOpCallback
from dspy.runtime.config import CallLogMode
from dspy.runtime.usage_tracker import UsageTracker, track_usage
from dspy.testing import DummyLM


class _OuterModule(Module):
    def __init__(self, inner: Module) -> None:
        super().__init__()
        self.inner = inner

    async def _aforward_impl(self, *, run, options=None, **inputs: object) -> object:
        return await self.inner(run=run, options=options, **inputs)


class _LeafModule(Module):
    async def _aforward_impl(self, *, run, options=None, **inputs: object) -> list[str]:
        return [type(module).__name__ for module in get_caller_modules()]


class _CallbackRecorder(NoOpCallback):
    def __init__(self) -> None:
        self.tool_runs: list[object] = []

    def on_tool_start(self, call_id: str, instance: object, inputs: dict[str, object]) -> None:
        self.tool_runs.append(get_active_run())


@pytest.mark.asyncio
async def test_nested_call_scope_tracks_caller_modules_lifo():
    run = RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    leaf = _LeafModule()
    outer = _OuterModule(leaf)
    result = await outer(run=run)
    assert result == ["_OuterModule", "_LeafModule"]


@pytest.mark.asyncio
async def test_concurrent_call_scopes_do_not_cross_contaminate():
    run = RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    module_a = _LeafModule()
    module_b = _LeafModule()

    async def invoke(module: Module) -> list[str]:
        async with call_scope(run=run, caller=module):
            await asyncio.sleep(0)
            return [type(module).__name__ for module in get_caller_modules()]

    results = await asyncio.gather(invoke(module_a), invoke(module_b))
    assert results[0] == ["_LeafModule"]
    assert results[1] == ["_LeafModule"]


@pytest.mark.asyncio
async def test_tool_callbacks_resolve_active_run_from_call_scope():
    callback = _CallbackRecorder()
    run = RunContext.create(
        lm=DummyLM([{}]),
        adapter=JSONAdapter(),
        callbacks=[callback],
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )

    def echo(query: str) -> str:
        return query

    class _ToolModule(Module):
        def __init__(self) -> None:
            self.tool = Tool(echo, description="Echo the query.")

        async def _aforward_impl(self, *, query: str, run, options=None, **kwargs: object) -> str:
            return await self.tool.acall(query=query)

    result = await _ToolModule()(query="ping", run=run)
    assert result == "ping"
    assert callback.tool_runs == [run]


def test_nested_track_usage_restores_outer_tracker():
    run = RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    with track_usage(run) as outer:
        outer.add_usage("model-a", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
        with track_usage(run) as inner:
            inner.add_usage("model-b", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        assert get_active_usage_tracker(run) is outer
    assert get_active_usage_tracker(run) is None


@pytest.mark.asyncio
async def test_concurrent_track_usage_does_not_corrupt_totals():
    run = RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )

    async def worker(prompt_tokens: int) -> int:
        with track_usage(run) as tracker:
            await asyncio.sleep(0)
            tracker.add_usage(
                "openai/gpt-4o-mini",
                {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1},
            )
            return tracker.get_total_tokens()["openai/gpt-4o-mini"]["prompt_tokens"]

    assert await asyncio.gather(worker(50), worker(80)) == [50, 80]


def test_configured_usage_sink_receives_usage_without_ambient_tracker():
    sink = UsageTracker()
    run = RunContext.create(
        lm=DummyLM([{"answer": "ok"}]),
        adapter=JSONAdapter(),
        usage_tracker=sink,
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    assert get_active_usage_tracker(run) is sink
    sink.add_usage("model-a", {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6})
    assert sink.get_total_tokens()["model-a"]["prompt_tokens"] == 5

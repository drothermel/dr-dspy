# RunContext migration guide

DSPy no longer uses the legacy global `settings` singleton. Pass an explicit `RunContext` via `run=` at call sites.

## Quick translation

| Legacy | RunContext |
| --- | --- |
| `settings.configure(lm=..., adapter=...)` | `run = RunContext.create(lm=..., adapter=..., telemetry=TelemetryConfig(call_log=CallLogMode.memory))` |
| `settings.configure(transparency="off")` | `telemetry=TelemetryConfig(transparency="off")` on `RunContext.create` |
| `settings.context(lm=..., adapter=...)` | `run = run.fork(lm=..., adapter=...)` |
| `settings.lm` | `run.lm` |
| `settings.adapter` | `run.adapter` |
| `settings.trace` | `run.optimization_trace` |
| `settings.callbacks` | `run.callbacks` |
| `settings.track_usage` | `run.telemetry.track_usage` |
| `settings.disable_history` | `run.telemetry.call_log=CallLogMode.off` |
| `settings.get("transparency", "strict")` | `run.telemetry.transparency` |
| `await module(...)` | `await module(..., run=run)` |
| `await predict(..., lm=..., config=...)` | `await predict(..., run=run, options=PredictOptions(lm=..., config=...))` |
| `await evaluator(program)` | `await evaluator(program, run=run)` |
| `await teleprompter.compile(student, trainset=...)` | `await teleprompter.compile(student, trainset=..., run=run)` |

## Configuring a run

```python
import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.core.types import LMConfig, LMProviderOptions
from dspy.predict.call_options import PredictOptions
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig

run = RunContext.create(
    lm=LM(
        "openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=4000,
        provider_options=LMProviderOptions(cache=False),
    ),
    adapter=JSONAdapter(),
    telemetry=TelemetryConfig(transparency="strict", call_log=CallLogMode.memory),
)
result = asyncio.run(program(question="What is DSPy?", run=run))
result = asyncio.run(
    program(
        question="What is DSPy?",
        run=run,
        options=PredictOptions(config=LMConfig(temperature=0.0), trace=True),
    )
)
```

Opt down for legacy behavior:

```python
from dspy.runtime import CallLogMode, TelemetryConfig

telemetry = TelemetryConfig(transparency="off", call_log=CallLogMode.off)
run = RunContext.create(lm=lm, adapter=adapter, telemetry=telemetry)
```

`adapter` is required on every `RunContext.create(...)`, including when `transparency="off"`. There is no implicit `ChatAdapter` fallback.

## Forking for scoped overrides

Replace `settings.context(...)` with `run.fork(...)` and pass the forked run to the call:

```python
from dspy.predict.call_options import PredictOptions

child = run.fork(lm=other_lm, optimization_trace=[], call_log=[])
result = await predict(question="...", run=child)
result = await predict(
    question="...",
    run=child,
    options=PredictOptions(lm=other_lm, trace=False),
)
```

For concurrent batch or trace capture, prefer `fork_worker_run` so each worker gets
isolated `optimization_trace` and `call_log` buffers:

```python
from dspy.runtime.run_fork import fork_worker_run

worker = fork_worker_run(run, lm=other_lm)
result = await predict(question="...", run=worker)
```

`Parallel` (`from dspy.runtime import Parallel`) and trace capture helpers use `fork_worker_run` internally.

Nested config updates accept model copies or dict patches:

```python
child = run.fork(telemetry={"transparency": "strict"})
```

## Ambient run scope

Module calls establish a task-local ambient scope via `call_scope` in
`dspy/runtime/active_run.py`. That scope owns:

- `ACTIVE_RUN` for tool callbacks that do not receive `run=`
- the caller-module stack used for call-log fan-out
- active usage trackers created by `track_usage`

`RunContext.usage_tracker` remains an optional configured sink set through
`create`/`fork`; it is not mutated during calls. Pass `run=` explicitly at spine
APIs; do not rely on implicit global run state outside module/tool execution.

## Tests

Use the `make_run` fixture when tests construct their own LM:

```python
def test_example(make_run):
    lm = DummyLM([{"answer": "ok"}])
    run = make_run(lm=lm)
    result = asyncio.run(predict(question="...", run=run))
```

## Audit logging

Environment variables:

- `DSPY_LOG_DIR` — root directory for run logs (default: `logs/` relative to cwd)
- `DSPY_RUN_ID` — experiment bucket name (default: `default_run`)

When `telemetry.call_log` is `disk` or `both`, `RunContext.create(...)` creates `{DSPY_LOG_DIR}/{DSPY_RUN_ID}/{timestamp}/` with `run.json` and append-only `calls.jsonl` for every LM call.

Inspect calls with `run.inspect_call_log()` or `run.read_call_log()` (RunContext only). When memory `call_log` is empty and disk logging is enabled, both APIs tail `run.log_session` `calls.jsonl`. For per-LM or per-module lists, use `pretty_print_call_log(lm.call_log)` from `dspy.runtime`. See `docs/migration/history.md` for the full agent vs call vs optimization vocabulary.

Optimizer/bootstrap teacher contexts must include a configured `adapter` (use `optimizer_lm_context` from `dspy.teleprompt.core`).

See `docs/migration/call-options.md` for `PredictOptions`, `LMProviderOptions`, and other strict kwargs.

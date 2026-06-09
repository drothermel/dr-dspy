# RunContext migration guide

DSPy no longer uses the legacy global `settings` singleton. Pass an explicit `RunContext` via `run=` at call sites.

## Quick translation

| Legacy | RunContext |
| --- | --- |
| `settings.configure(lm=..., adapter=...)` | `run = RunContext.create(lm=..., adapter=..., init_run_log=False)` |
| `settings.configure(transparency="off")` | `telemetry=TelemetryConfig(transparency="off")` on `RunContext.create` |
| `settings.context(lm=..., adapter=...)` | `run = run.fork(lm=..., adapter=...)` |
| `settings.lm` | `run.lm` |
| `settings.adapter` | `run.adapter` |
| `settings.trace` | `run.trace` |
| `settings.callbacks` | `run.callbacks` |
| `settings.track_usage` | `run.telemetry.track_usage` |
| `settings.disable_history` | `run.telemetry.disable_history` |
| `settings.get("transparency", "strict")` | `run.telemetry.transparency` |
| `await module(...)` | `await module(..., run=run)` |
| `await evaluator(program)` | `await evaluator(program, run=run)` |
| `await teleprompter.compile(student, trainset=...)` | `await teleprompter.compile(student, trainset=..., run=run)` |

## Configuring a run

```python
import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.runtime import RunContext, TelemetryConfig

run = RunContext.create(
    lm=LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=4000, cache=False),
    adapter=JSONAdapter(),
    telemetry=TelemetryConfig(transparency="strict"),
    init_run_log=False,
)
result = asyncio.run(program(question="What is DSPy?", run=run))
```

Opt down for legacy behavior:

```python
from dspy.runtime import TelemetryConfig

telemetry = TelemetryConfig(transparency="off", run_log_enabled=False)
run = RunContext.create(lm=lm, adapter=adapter, telemetry=telemetry, init_run_log=False)
```

## Forking for scoped overrides

Replace `settings.context(...)` with `run.fork(...)` and pass the forked run to the call:

```python
child = run.fork(lm=other_lm, trace=[])
result = await predict(question="...", run=child)
```

Nested config updates accept model copies or dict patches:

```python
child = run.fork(telemetry={"transparency": "strict"})
```

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

When `telemetry.run_log_enabled=True`, `RunContext.create(...)` creates `{DSPY_LOG_DIR}/{DSPY_RUN_ID}/{timestamp}/` with `run.json` and append-only `calls.jsonl` for every LM call.

Optimizer/bootstrap teacher contexts must include a configured `adapter` (use `optimizer_lm_context` from `dspy.teleprompt.utils`).

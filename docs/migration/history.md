# History system migration guide

DSPy separates three concepts that were previously conflated under "history":

| Layer | Type | Where it lives | Purpose |
| --- | --- | --- | --- |
| Agent turn state | `TurnLog` / `REPLHistory` | `turn_log` TaskSpec field | What the model sees on the next agent turn |
| LM observability | `CallRecord` | `run.call_log`, `lm.call_log`, disk `calls.jsonl` | What humans inspect after LM calls |
| Optimizer trace | `list` of trace entries | `run.optimization_trace` | Bootstrap / metric / optimizer debugging |

Public exports: `from dspy.history import AgentHistory, REPLHistory, TurnLog, is_agent_history_type, is_conversation_turn_log_type`.

Dict-shaped `turn_log` / `REPLHistory` values passed as task inputs are normalized to typed models when inputs are validated for adapter calls (`dspy.task_spec.validate_task_inputs` in `AdapterCallPipeline.execute`). There is no separate `coerce_turn_log` helper.

## Vocabulary (breaking renames)

| Old | New |
| --- | --- |
| `History` | `TurnLog` |
| TaskSpec / Prediction field `history` | `turn_log` |
| `LMHistoryEntry` | `CallRecord` |
| `lm.history`, `module.history` | `lm.call_log`, `module.call_log` |
| `GLOBAL_HISTORY`, `inspect_history()` | removed; use `run.inspect_call_log()` / `run.read_call_log()` |
| `disable_history`, `max_history_size`, `run_log_enabled` | `TelemetryConfig.call_log` (`CallLogMode`) |
| `run.trace`, `max_trace_size` | `run.optimization_trace`, `max_optimization_trace_entries` |
| ReAct/CodeAct `trajectory` dict | `TurnLog` via `turn_log` |
| `utils/inspect_history.py` | `dspy/runtime/inspect_call_log.py` (`pretty_print_call_log`) |

## Agent turn logs (`TurnLog`)

```python
from dspy.history import TurnLog

turn_log = TurnLog.empty()
turn_log = turn_log.append_turn({"thought": "...", "tool_name": "search", "tool_args": {"q": "cats"}, "observation": "..."})

# TaskSpec field
input_field("turn_log", TurnLog)
```

ReAct, ReActV2, CodeAct, Avatar, and RLM return `Prediction(..., turn_log=turn_log, termination_reason=...)`. Agent modules use `AgentTerminationReason` (`dspy.predict.agent_termination`) for `termination_reason`. RLM uses `REPLHistory` with the same `turn_log` field name.

Avatar uses canonical `dspy.adapters.types.tool.Tool` instances (same as other agent modules). The actor predictor outputs an `Action` with `tool_name` and structured `tool_args` (JSON dict), executed via `await tool.acall(**tool_args)`. `Prediction.actions` records `ActionOutput` entries with the same `tool_args` shape.

Immutability: `append_turn` returns a new instance; never mutate `.turns` in place.

## Call observability (`CallRecord` / `run.call_log`)

```python
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig

run = RunContext.create(
    lm=lm,
    adapter=adapter,
    telemetry=TelemetryConfig(call_log=CallLogMode.both, max_call_log_entries=10_000),
)

await program(question="...", run=run)

run.inspect_call_log(n=1)          # pretty-print last call (RunContext only)
records = run.read_call_log(n=10)  # notebook-friendly list[dict]; tails calls.jsonl when memory is empty and call_log is disk|both
lm.call_log[-1].request.messages   # typed LMRequest on CallRecord

# Scoped debugging (per-LM or per-module lists)
from dspy.runtime import pretty_print_call_log
pretty_print_call_log(lm.call_log, n=1)
```

`CallLogMode`: `off` | `memory` | `disk` | `both` (default). `max_call_log_entries=0` disables memory logging. With `call_log=disk`, `run.read_call_log()` and `run.inspect_call_log()` read from `run.log_session` disk records.

Disk logging is scoped to `RunContext.log_session` (no process-global session). Forked runs share the same disk session but isolate memory via `run.fork(call_log=[], optimization_trace=[])`.

## Optimizer trace

```python
from dspy.runtime import run_with_trace

# Predict appends (module, inputs, prediction) tuples when options.trace=True
prediction, trace = await run_with_trace(program, example, run)
assert trace

# Parse failures can be captured for bootstrap, GEPA, and GRPO
prediction, trace = await run_with_trace(
    program, example, run, capture_parse_failures=True
)

# Evaluate / Parallel fork isolated traces per item
item_run = run.fork(optimization_trace=[], call_log=[])
```

Metrics still use the third argument name `trace`; it receives `list(item_run.optimization_trace)`.

| Old | New |
| --- | --- |
| `run_program_with_trace` | `run_with_trace` (import from `dspy.runtime`) |
| `capture_failed_parses` | `capture_parse_failures` |

## Notebook recipe

```python
import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.runtime import RunContext

run = RunContext.create(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
result = asyncio.run(program(question="...", run=run))
print(run.read_call_log(n=3))
# or tail logs/{DSPY_RUN_ID}/.../calls.jsonl
```

See also: `docs/migration/runcontext.md`, `docs/migration/call-options.md`.

# Changelog

## [Unreleased]

### Breaking changes

- DSPy modules are async-only. Use `await program(...)` instead of `program(...)`.
- Subclasses must implement `async def aforward(...)`, not `forward`.
- `Evaluate`, `Parallel`, `Module.batch`, and teleprompter `compile` are async:
  `await evaluate(program, devset=...)`, `await parallel(pairs)`, `await module.batch(examples)`,
  `await teleprompter.compile(...)`.
- Removed streaming (`streamify`, `StreamListener`) and sync bridges (`asyncify`, `syncify`).
- Removed `ParallelExecutor` thread pools; batch execution uses `asyncio`-bounded concurrency.
- `BaseLM` and `Adapter` are async-only: `await lm(request, run=...)`, `await adapter(lm=..., run=...)`.
- History system refactor (see `docs/migration/history.md`):
  - `History` → `TurnLog`; agent field `history` → `turn_log`; ReAct/CodeAct `trajectory` → `turn_log`.
  - `LMHistoryEntry` → `CallRecord`; `lm/module.history` → `call_log`; removed `GLOBAL_HISTORY` / `inspect_history()`.
  - `run.trace` → `run.optimization_trace`; telemetry unified under `TelemetryConfig.call_log` (`CallLogMode`).
  - Disk logging scoped to `RunContext.log_session`; use `run.inspect_call_log()` / `run.read_call_log()`.

### Migration

```python
# Before
result = program(question="What is DSPy?")

# After
result = await program(question="What is DSPy?")

# Scripts without an event loop
import asyncio

async def main():
    result = await program(question="What is DSPy?")

asyncio.run(main())
```

## 3.3.0b1

### Breaking changes

Compatibility shims removed in the LM boundary and core API cutover. Users upgrading should expect:

1. `lm.history[i]["messages"]` → `lm.history[i].request.messages` (typed `LMHistoryEntry`)
2. `LM(..., reasoning_effort="low")` → `LM(..., reasoning={"effort": "low"})` or typed `LMConfig` fields
3. OpenAI-shaped `tool_choice` dicts are rejected at the LM boundary; use `LMToolChoice`
4. `InputField(prefix=…)` is rejected
5. `Example.toDict()` removed → use `Example.to_dict()`
6. `ChainOfThought(rationale_field=…)` and `rationale_field_type` removed; CoT always prepends `Reasoning`
7. `streamify()` without listeners no longer yields raw provider chunks
8. Metrics must accept `(example, prediction, trace)`; `Evaluate` passes `list(settings.trace)`
9. Custom `Type` values render as LM content blocks at format time, not marker strings
10. `Adapter.format()` returns `list[LMMessage]` instead of OpenAI chat dicts
11. `Type.parse_lm_response()` removed; implement `parse_lm_output(LMOutput)` for native response types
12. Cross-version JSON program state from DSPy 3.0.x is not supported; use pickle program saves or re-optimize
13. `named_parameters()` / `named_predictors()` paths are aligned with `named_sub_modules()` (e.g. `self.predict` instead of `predict`)

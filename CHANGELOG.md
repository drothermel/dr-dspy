# Changelog

## [Unreleased]

### Breaking changes

- TaskSpec boundary cutover (P0.2): adapter prompt formatting moved from `dspy.task_spec` to `dspy.adapters.prompt_format` (`format_field_value`, `translate_field_type`, `get_field_spec_description_string` no longer exported from `dspy.task_spec`); spine call validation is `dspy.task_spec.validate_task_inputs`; reserved predict kwargs and `PredictOptions` normalization live in `dspy.predict.options` (`dspy.predict.call_validation` removed); framework TaskSpecs colocated under `dspy.task_spec.framework/` and per-optimizer `task_specs.py` modules.
- Trace ownership cutover (P0.3): `run_with_trace`, `FailedPrediction`, and `TraceData` live in `dspy.runtime`; batch trace collection is `collect_trace_data` in `dspy.teleprompt.core` (via `TraceCapturingModule`, not `_aforward_impl` monkey-patching); compile-spine helpers moved to `dspy.teleprompt.core`; predictor context helpers moved to `dspy.task_spec.predictor_context`; `OptimizerMetric` types live in `dspy.evaluate.metric_contract`. Removed `run_program_with_trace`, `bootstrap_trace_data`, `capture_failed_parses`, `dspy.teleprompt.trace_helpers`, `dspy.teleprompt.bootstrap_trace`, `dspy.teleprompt.utils`, and `dspy.teleprompt.task_spec_context`.
- Legacy cleanup (11 phases): centralized LM field normalization (`reasoning_effort` removed from `LMProviderOptions`; use `reasoning={"effort": ...}`); explicit `adapter` required on all `RunContext` modes; `num_threads` renamed to `max_concurrency`; call provenance threaded via `RunContext.call_site` and explicit `compiled=` (context vars removed); `Tool` sync `__call__` no longer runs async tools; `acall` aliases removed from `Module`, `BaseLM`, `Embedding`, `KNN`, and `Adapter`; retrievers are async-only (`await retriever(query)`); teleprompters use nested `params=` (`XCompileParams`) with `run=` top-level.
- Adapter formatting/parsing cleanup: shared `AdapterFormatMixin` scaffolding (including `TwoStepAdapter` turn_log handling); centralized `validate_task_inputs` in `AdapterCallPipeline` (Predict no longer validates separately).
- Teleprompt cleanup: shared helpers in `dspy.teleprompt.utils` (`resolve_max_errors`, `make_optimizer_evaluator`, `run_program_with_trace`, `trace_to_demos`); Optuna integrations use native async `study.ask()` / `study.tell()` loops (removed `run_async_from_sync` / executor bridges); GEPA runs `gepa.optimize` in a worker thread and requires async instruction proposers (`AsyncProposalFn`).
- Teleprompt structural cutover (see `docs/migration/teleprompt.md`): `compile(...) -> CompileResult` with `.program`, `.candidates`, `.stats`; registry-backed `@register_teleprompter` params; typed candidate seed ladder; BetterTogether `strategy: list[str]`; removed module optimizer metadata attrs and deleted `TelepromptOptunaCompileParams` / `PassthroughCompileParams` / `demo_sets.py`.
- Dataset integrations moved under `dspy.integrations.datasets` (`HotPotQA`, `GSM8K`, `MATH`, `examples_from_huggingface`, `AlfWorld`); spine keeps `dspy.datasets.dataset` and `dspy.datasets.rows`. Removed spine `DataLoader`, `Colors`, and `Dataset.prepare_by_seed`. `HuggingFaceDataLoader` replaced by module functions (`examples_from_huggingface`, `examples_from_csv`, …). `AlfWorld` uses `.train` / `.dev` instead of `.trainset` / `.devset`.
- `Evaluate` moved to `dspy.evaluate.evaluator`; `dspy.clients.openai_format` package barrel removed (import submodules directly).
- Python interpreter sync tool path rejects coroutine returns (`await_in_sync` removed); use synchronous tool callables.
- Removed `responses_compat` OpenAI endpoint shims; use unified LM serialization.
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
- Primitives hard cutover (see `docs/migration/primitives.md`):
  - `Parameter` marker removed; `Predict(Module)` only; use `Predictor` protocol for isinstance checks.
  - `BaseModule` removed; graph introspection and persistence live on `Module`.
  - `Module.named_parameters()` / `parameters()` → `named_predictors()` / `predictors()`.
  - `BatchResult.failures` is always populated for failed indices; `return_failed_examples=`, `Parallel.failed_examples`, and `Parallel.exceptions` removed.

### Migration

```python
# Trace and teleprompter evaluation helpers
from dspy.runtime import run_with_trace
from dspy.teleprompt import collect_trace_data, make_optimizer_evaluator, resolve_max_errors

evaluate = make_optimizer_evaluator(
    run,
    devset=valset,
    metric=my_metric,
    max_concurrency=8,
    max_errors=resolve_max_errors(None, run),
)
prediction, trace = await run_with_trace(program, example, run)

# Teleprompter compile returns CompileResult
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotCompileParams

result = await teleprompter.compile(
    student,
    params=BootstrapFewShotCompileParams(trainset=trainset),
    run=run,
)
program = result.program

# GEPA custom instruction proposers must be async
from dspy.integrations.optimizers.gepa.adapter import AsyncProposalFn

# Dataset loaders (optional `datasets` extra for Hugging Face helpers)
from dspy.integrations.datasets.hotpotqa import HotPotQA
from dspy.integrations.datasets.huggingface import examples_from_huggingface
from dspy.integrations.datasets.alfworld.alfworld import AlfWorld

# Evaluate devsets
from dspy.evaluate.evaluator import Evaluate

# OpenAI wire-format helpers
from dspy.clients.openai_format.chat_request import to_openai_chat_request
from dspy.clients.openai_format.parse import completion_to_lm_response

class MyProposer(AsyncProposalFn):
    async def __call__(self, candidate, reflective_dataset, components_to_update):
        ...

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

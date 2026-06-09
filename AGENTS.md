## Commit gates (required)

Before committing, run these in order. **Each command must exit 0** — do not commit if any gate fails.

```
uv run ruff check --fix
uv run ty check --fix
uv run ruff format
```

`uv run ty check --fix` is a required commit gate: fix or resolve every reported diagnostic before committing. Re-run the full sequence after fixes.

## Async-only public API

DSPy modules, LMs, adapters, `Evaluate`, `Parallel`, and teleprompter `compile` are async.
Use `await` at call sites; in scripts use `asyncio.run(...)`.

```python
from dspy.core.types import LMConfig, PredictOptions

# Module invocation
result = await program(question="What is DSPy?", run=run)
result = await program(
    question="What is DSPy?",
    run=run,
    options=PredictOptions(config=LMConfig(temperature=0.0), trace=False),
)

# Evaluation
evaluator = Evaluate(devset=devset, metric=my_metric)
score = await evaluator(program, run=run)

# Parallel batch
parallel = Parallel(max_concurrency=8)
results = await parallel([(module, example), ...])

# Optimizers
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams

compiled = await teleprompter.compile(
    student,
    params=BootstrapFewShotCompileParams(trainset=trainset),
    run=run,
)
```

## TaskSpec (not Signature)

Define tasks as `TaskSpec` subclasses (or with `make_task_spec` for dynamic cases) and pass an instance to predictors. Do not pass strings or legacy `Signature` classes to `Predict`.

```python
import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.predict import ChainOfThought, Predict
from dspy.runtime import RunContext
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field

class QATaskSpec(TaskSpec):
    name: str = "QA"
    instructions: str = "Answer the question."
    inputs: tuple[FieldSpec, ...] = (input_field("question", desc="The user question"),)
    outputs: tuple[FieldSpec, ...] = (output_field("answer", desc="A concise answer"),)

qa = QATaskSpec()
run = RunContext.create(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter(), init_run_log=False)
predict = Predict(qa)
result = asyncio.run(predict(question="What is DSPy?", run=run))

cot = ChainOfThought(qa)
result = asyncio.run(cot(question="What is DSPy?", run=run))
```

For runtime-composed specs, use `make_task_spec` with `input_field` / `output_field` (or a spec string when field names are derived at runtime).

Tools require an explicit description:

```python
from dspy.adapters.types.tool import Tool

tool = Tool(my_func, description="Describe what the tool does.")
```

ReAct, CodeAct, RLM, and ReActV2 require `tools=[Tool(...)]` (raw callables are rejected). Agent modules return `turn_log` (`TurnLog` or `REPLHistory`) on predictions.

See `docs/migration/taskspec.md` for the full Signature → TaskSpec translation table.
See `docs/migration/history.md` for turn logs vs call logs vs optimization traces.

Field descriptions must be explicit under strict transparency (placeholder `${field}` descs are rejected).

Adapters read `FieldSpec` directly — there is no Pydantic `FieldInfo` bridge. Use `dspy.task_spec` for spine work:

```python
from dspy.task_spec import FieldBinding, field_bindings, format_field_value, validate_task_inputs_from_spec
from dspy.task_spec.field_spec import FieldRole

bindings = field_bindings(task_spec, role=FieldRole.INPUT)
for binding in bindings:
    text = format_field_value(field=binding.field, value=inputs[binding.name])
```

`TaskSpec.fingerprint()` returns a SHA-256 hex digest over `name`, `instructions`, and field specs. Compare specs with `==` (frozen Pydantic model equality).

## Core LM types

- `LMForward` (`dspy.core.types.lm`) — async `aforward(request) -> LMResponse` protocol for per-call `PredictOptions(lm=...)`.
- `LMMessageRole` — `StrEnum` on `LMMessage.role` (`user`, `assistant`, `system`, `tool`, …).
- `ReasoningEffort` — `StrEnum` on `LMReasoningConfig.effort` (`low`, `medium`, `high`).
- OpenAI-compat helpers live in `dspy.core.types.openai_compat` (`request_prompt`, `request_messages_as_openai`, `request_kwargs`). Import private part helpers from `dspy.core.types.parts.models` / `parts.openai` / `parts.serialize`, not the public `parts` barrel.

## Public `core/types` spine API

Cross-package code imports spine helpers from `dspy.core.types` only (not submodule paths). Symbols with a leading `_` under `dspy/core/types/` are internal.

- Config merge/coercion: `merge_lm_config`, `merge_lm_request_config`, `coerce_lm_config`, `merge_provider_options`
- Message/tool coercion: `coerce_tool_spec`
- OpenAI part parsing: `parts_from_openai_content`

## Strict transparency and audit logging

`transparency` defaults to `"strict"`. Create an explicit `RunContext` and pass `run=` to module, evaluation, and optimizer calls:

```python
import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.core.types import LMProviderOptions
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig

run = RunContext.create(
    lm=LM(
        "openai/gpt-4o-mini",
        temperature=0.0,
        max_tokens=4000,
        provider_options=LMProviderOptions(cache=False),
    ),
    adapter=JSONAdapter(),
    init_run_log=False,
)
result = asyncio.run(program(question="What is DSPy?", run=run))
```

Opt down for legacy behavior: `TelemetryConfig(transparency="off", call_log=CallLogMode.off)` on `RunContext.create`. An explicit `adapter=` is still required in all modes.

Environment variables:

- `DSPY_LOG_DIR` — root directory for run logs (default: `logs/` relative to cwd)
- `DSPY_RUN_ID` — experiment bucket name (default: `default_run`)

Each `RunContext.create(...)` with `call_log` in `(disk, both)` creates `{DSPY_LOG_DIR}/{DSPY_RUN_ID}/{timestamp}/` with `run.json` and append-only `calls.jsonl` for every LM call. Use `run.inspect_call_log()` or `run.read_call_log()` to inspect calls.

See `docs/migration/runcontext.md` for the full settings → RunContext translation table.
See `docs/migration/history.md` for turn logs, call logs, and optimization traces.

## dr-llm backends (optional)

`LM` (LiteLLM) remains the default builtin. For dr-llm 4.3.0 provider orchestration and Postgres-backed response pools, use `DrLlmDirectLM` or `DrLlmPoolLM`:

```python
import asyncio

from dr_llm.backends.models import PoolBackendConfig

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.dr_llm import DrLlmDirectLM, DrLlmPoolLM
from dspy.runtime import RunContext

direct = DrLlmDirectLM("openai/gpt-4.1-mini", temperature=0.0, max_tokens=4000)
run = RunContext.create(lm=direct, adapter=JSONAdapter(), init_run_log=False)
result = asyncio.run(program(question="What is DSPy?", run=run))

pool = DrLlmPoolLM(
    "openai/gpt-4.1-mini",
    pool_config=PoolBackendConfig(
        pool_name="my_exp",
        database_url="postgresql://user:pass@localhost/dr_llm",
    ),
    session_id="optimizer-session",  # optional override
)
samples = asyncio.run(pool.acquire_samples(request, n=10, run=run))
```

- **Direct** (`aforward`): calls `DirectBackend.acomplete` — one provider response per request.
- **Pool** (`aforward`): cache-first `PoolBackend.acomplete` (no session claims).
- **Pool acquire** (`acquire_samples`): session-scoped no-replacement sampling via `PoolBackend.aacquire(request, session_id, n)`. Session ID defaults to `{DSPY_RUN_ID}:{log_session.timestamp}` when disk logging is enabled; pass `session_id=` on the LM or to `acquire_samples` to override.
- **Auth/routing**: configure providers via the dr-llm registry and environment (for example `OPENAI_API_KEY`). `DrLlmDirectLM` / `DrLlmPoolLM` do **not** accept `provider_options`, `num_retries`, or LiteLLM-style passthrough kwargs — misconfiguration raises at construction.
- **v1 limits**: text-only; tools, multimodal parts, `response_format`, `stop`, `n`, `logprobs`, `tool_choice`, `prompt_cache`, `LMConfig.extensions`, and unsupported `reasoning` fields (`max_tokens`, `summary`) raise typed errors. Only `reasoning.effort` maps to `BackendRequest.effort`.
- **Lifecycle**: call `pool.close()` to tear down the pool consumer (tests); long-running notebooks may omit.

Integration tests (`pytest -m integration -n0 tests/clients/dr_llm/test_integration_pool.py`) require Postgres via `DR_LLM_TEST_DATABASE_URL` or `DR_LLM_DATABASE_URL`. Spin up a disposable database with `uv run dr-llm project create <name>` and export the returned DSN.

Live direct-provider smoke test (uses your `OPENAI_API_KEY` via dr-llm’s default registry):

```bash
uv run pytest tests/clients/dr_llm/test_integration_direct_live.py --llm_call -n0 -v
```

Override the model with `LM_FOR_TEST_DIRECT_DR_LLM=openai/gpt-4.1-mini`. Quick dr-llm-only sanity check without DSPy: `uv run dr-llm query --provider openai --model gpt-4.1-mini --message "ping"`.

## Strict call-site kwargs

Pass task inputs as keywords, `run=` for `RunContext`, and `options=PredictOptions(...)` for per-call overrides (`lm`, `config`, `demos`, `task_spec`, `trace`, `prediction`). Do not pass reserved names as flat task-input kwargs.

```python
from dspy.core.types import LMConfig, PredictOptions

result = await predict(
    question="What is DSPy?",
    run=run,
    options=PredictOptions(lm=other_lm, config=LMConfig(temperature=0.5)),
)
```

- `Example.from_record(record, input_keys=(...))` and `example.as_inputs()` replace `Example(**kwargs)` / `example.inputs()`.
- `LM(..., provider_options=LMProviderOptions(...))` replaces top-level provider kwargs such as `cache=` and `api_key=`.
- `await knn(inputs={...})` replaces positional KNN queries.

See `docs/migration/call-options.md` for before/after examples.

Optimizer/bootstrap teacher contexts must include a configured `adapter` (use `optimizer_lm_context` from `dspy.teleprompt.utils`).

Teleprompter evaluation and trace helpers live in `dspy.teleprompt.utils`:

```python
from dspy.teleprompt.utils import make_optimizer_evaluator, resolve_max_errors, run_program_with_trace, trace_to_demos

evaluate = make_optimizer_evaluator(
    run,
    devset=valset,
    metric=my_metric,
    max_concurrency=8,
    max_errors=resolve_max_errors(None, run),
)
prediction, trace = await run_program_with_trace(program, example, run)
demos_by_predictor = trace_to_demos(trace, predictor2name)
```

GEPA custom instruction proposers must implement `AsyncProposalFn` with `async def __call__(...)`. Sync proposers and `await_in_sync` interpreter tool bridges are removed.

Task input validation runs in `AdapterCallPipeline.execute`; do not rely on duplicate validation in `Predict`.

## Import tiers

1. **Public spine:** `dspy.runtime`, `dspy.core.types`, `dspy.task_spec`, `dspy.errors`, `dspy.persistence`, `dspy.serialization`
2. **Integrations:** `dspy.integrations.*` (optional extras: `mcp`, `langchain`)
3. **Internal / test-only:** `dspy._internal.*`, `dspy.testing.*`

## Do not import from (internal/legacy)

- `dspy._internal.*` — lazy import machinery, unbatchify batching helper
- `dspy.testing.*` — test doubles only

## Internal call-site conventions

- Use keyword arguments for multi-arg calls to DSPy-internal functions when meaning is not obvious from position.
- Do not add keyword-only `*` to public constructors or documented callback protocols (e.g. `metric(example, prediction, trace)`).
- Spine APIs require keywords at call sites: `run_bounded(items=..., fn=...)`, `await adapter(lm=..., config=..., task_spec=..., demos=..., inputs=..., run=...)`.

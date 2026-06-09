Before commiting run:
```
uv run ruff check --fix
uv run ty check --fix
uv run ruff format
```

Then fix any remaining issues and reformat before committing.

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
compiled = await teleprompter.compile(student, trainset=trainset, run=run)
```

`Module.acall` and `BaseLM.acall` are compatibility aliases for `__call__`.

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
    inputs: tuple[FieldSpec, ...] = (input_field("question"),)
    outputs: tuple[FieldSpec, ...] = (output_field("answer"),)

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

Opt down for legacy behavior: `TelemetryConfig(transparency="off", call_log=CallLogMode.off)` on `RunContext.create`.

Environment variables:

- `DSPY_LOG_DIR` — root directory for run logs (default: `logs/` relative to cwd)
- `DSPY_RUN_ID` — experiment bucket name (default: `default_run`)

Each `RunContext.create(...)` with `call_log` in `(disk, both)` creates `{DSPY_LOG_DIR}/{DSPY_RUN_ID}/{timestamp}/` with `run.json` and append-only `calls.jsonl` for every LM call. Use `run.inspect_call_log()` or `run.read_call_log()` to inspect calls.

See `docs/migration/runcontext.md` for the full settings → RunContext translation table.
See `docs/migration/history.md` for turn logs, call logs, and optimization traces.

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
- `await knn.acall(inputs={...})` replaces positional KNN queries.

See `docs/migration/call-options.md` for before/after examples.

Optimizer/bootstrap teacher contexts must include a configured `adapter` (use `optimizer_lm_context` from `dspy.teleprompt.utils`).

## Internal call-site conventions

- Use keyword arguments for multi-arg calls to DSPy-internal functions when meaning is not obvious from position.
- Do not add keyword-only `*` to public constructors or documented callback protocols (e.g. `metric(example, prediction, trace)`).
- Spine APIs require keywords at call sites: `run_bounded(items=..., fn=...)`, `adapter.acall(lm=..., config=..., task_spec=..., demos=..., inputs=...)`.

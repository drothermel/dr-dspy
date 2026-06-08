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
# Module invocation
result = await program(question="What is DSPy?")

# Evaluation
evaluator = Evaluate(devset=devset, metric=my_metric)
score = await evaluator(program)

# Parallel batch
parallel = Parallel(max_concurrency=8)
results = await parallel([(module, example), ...])

# Optimizers
compiled = await teleprompter.compile(student, trainset=trainset)
```

`Module.acall` and `BaseLM.acall` are compatibility aliases for `__call__`.

## TaskSpec (not Signature)

Define tasks as `TaskSpec` subclasses (or with `make_task_spec` for dynamic cases) and pass an instance to predictors. Do not pass strings or legacy `Signature` classes to `Predict`.

```python
import asyncio

from dspy.predict import ChainOfThought, Predict
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field

class QATaskSpec(TaskSpec):
    name: str = "QA"
    instructions: str = "Answer the question."
    inputs: tuple[FieldSpec, ...] = (input_field("question"),)
    outputs: tuple[FieldSpec, ...] = (output_field("answer"),)

qa = QATaskSpec()
predict = Predict(qa)
result = asyncio.run(predict(question="What is DSPy?"))

cot = ChainOfThought(qa)
result = asyncio.run(cot(question="What is DSPy?"))
```

For runtime-composed specs, use `make_task_spec` with `input_field` / `output_field` (or a spec string when field names are derived at runtime).

Tools require an explicit description:

```python
from dspy.adapters.types.tool import Tool

tool = Tool(my_func, description="Describe what the tool does.")
```

ReAct, CodeAct, RLM, and ReActV2 require `tools=[Tool(...)]` (raw callables are rejected).

See `docs/migration/taskspec.md` for the full Signature → TaskSpec translation table.

Field descriptions must be explicit under strict transparency (placeholder `${field}` descs are rejected).

## Strict transparency and audit logging

`transparency` defaults to `"strict"`. Create an explicit `RunContext` and pass `run=` to module, evaluation, and optimizer calls:

```python
import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.runtime import RunContext, TelemetryConfig

run = RunContext.create(
    lm=LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=4000, cache=False),
    adapter=JSONAdapter(),
    init_run_log=False,
)
result = asyncio.run(program(question="What is DSPy?", run=run))
```

Opt down for legacy behavior: `TelemetryConfig(transparency="off", run_log_enabled=False)` on `RunContext.create`.

Environment variables:

- `DSPY_LOG_DIR` — root directory for run logs (default: `logs/` relative to cwd)
- `DSPY_RUN_ID` — experiment bucket name (default: `default_run`)

Each `RunContext.create(...)` with `run_log_enabled=True` creates `{DSPY_LOG_DIR}/{DSPY_RUN_ID}/{timestamp}/` with `run.json` and append-only `calls.jsonl` for every LM call.

See `docs/migration/runcontext.md` for the full settings → RunContext translation table.

Optimizer/bootstrap teacher contexts must include a configured `adapter` (use `optimizer_lm_context` from `dspy.teleprompt.utils`).

## Internal call-site conventions

- Use keyword arguments for multi-arg calls to DSPy-internal functions when meaning is not obvious from position.
- Do not add keyword-only `*` to public constructors or documented callback protocols (e.g. `metric(example, prediction, trace)`).
- Spine APIs require keywords at call sites: `run_bounded(items=..., fn=...)`, `adapter.acall(lm=..., config=..., task_spec=..., demos=..., inputs=...)`.

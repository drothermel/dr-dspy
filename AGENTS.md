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

See `docs/migration/taskspec.md` for the full Signature → TaskSpec translation table.

## Internal call-site conventions

- Use keyword arguments for multi-arg calls to DSPy-internal functions when meaning is not obvious from position.
- Do not add keyword-only `*` to public constructors or documented callback protocols (e.g. `metric(example, prediction, trace)`).
- Spine APIs require keywords at call sites: `run_bounded(items=..., fn=...)`, `adapter.acall(lm=..., config=..., task_spec=..., demos=..., inputs=...)`.

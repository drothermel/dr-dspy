# TaskSpec migration guide

DSPy no longer exposes the legacy `Signature` API. Use `TaskSpec` subclasses and `input_field` / `output_field` instead.

## Quick translation

| Legacy | TaskSpec |
| --- | --- |
| `class QA(Signature): ...` | `class QATaskSpec(TaskSpec): ...` |
| `Signature("q -> a")` | `make_task_spec("q -> a", instructions="...")` (dynamic only) |
| `make_signature("q -> a")` | `make_task_spec("q -> a", instructions="...")` |
| `ensure_signature("q -> a")` | `make_task_spec("q -> a", instructions="...")` |
| `InputField(desc="...")` | `input_field("name", desc="...")` |
| `OutputField(desc="...")` | `output_field("name", desc="...")` |
| `Predict("q -> a")` | `Predict(QATaskSpec())` |
| `predictor.signature` | `predictor.task_spec` |
| `Tool(func)` | `Tool(func, description="...")` or `tool_from_callable(func)` |

## Defining a task

Preferred style for named, static specs:

```python
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field

class QATaskSpec(TaskSpec):
    name: str = "QA"
    instructions: str = "Answer the question accurately."
    inputs: tuple[FieldSpec, ...] = (
        input_field("question", desc="The user question"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("answer", desc="A concise answer"),
    )

qa = QATaskSpec()
```

Dynamic composition (runtime-built fields):

```python
from dspy.task_spec import input_field, make_task_spec, output_field

spec = make_task_spec(
    inputs=[input_field("question", desc="The user question")],
    outputs=[output_field("answer", desc="A concise answer")],
    instructions="Answer the question accurately.",
    name="QA",
)
```

String form is reserved for truly dynamic field names:

```python
qa = make_task_spec("question -> answer", instructions="Answer the question.")
```

## Calling predictors

All module and predictor calls are async:

```python
import asyncio

from dspy.predict import Predict

predict = Predict(QATaskSpec())
result = asyncio.run(predict(question="What is DSPy?"))
# or inside async code:
# result = await predict(question="What is DSPy?")
```

## Tools

`Tool` requires an explicit description:

```python
from dspy.adapters.types.tool import Tool, tool_from_callable

def search(query: str) -> str:
    return query

tool = Tool(search, description="Search documents by query.")
# ReAct/CodeAct/RLM accept raw callables and wrap them with tool_from_callable().
```

## TaskSpec transforms

| Legacy Signature method | TaskSpec method |
| --- | --- |
| `sig.append("field", OutputField(...))` | `spec.append(output_field("field", ...))` |
| `sig.prepend("field", InputField(...))` | `spec.prepend(input_field("field", ...))` |
| `sig.delete("field")` | `spec.delete("field")` |
| `sig.with_instructions("...")` | `spec.with_instructions("...")` |

## Saved program state

Saved programs now store `task_spec` instead of `signature`. Reload with the current DSPy version; legacy signature-only state is rejected.

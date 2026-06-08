# TaskSpec migration guide

DSPy no longer exposes the legacy `Signature` API. Use `TaskSpec` and `make_task_spec` instead.

## Quick translation

| Legacy | TaskSpec |
| --- | --- |
| `class QA(Signature): ...` | `make_task_spec({...}, instructions="...")` |
| `Signature("q -> a")` | `make_task_spec("q -> a", instructions="...")` |
| `make_signature("q -> a")` | `make_task_spec("q -> a", instructions="...")` |
| `ensure_signature("q -> a")` | `make_task_spec("q -> a", instructions="...")` |
| `InputField(desc="...")` | `FieldSpec.input("name", desc="...")` |
| `OutputField(desc="...")` | `FieldSpec.output("name", desc="...")` |
| `Predict("q -> a")` | `Predict(make_task_spec("q -> a", instructions="..."))` |
| `predictor.signature` | `predictor.task_spec` |
| `Tool(func)` | `Tool(func, description="...")` or `tool_from_callable(func)` |

## Defining a task

```python
from dspy.task_spec import FieldSpec, make_task_spec

qa = make_task_spec(
    {
        "question": FieldSpec.input("question", desc="The user question"),
        "answer": FieldSpec.output("answer", desc="A concise answer"),
    },
    instructions="Answer the question accurately.",
    name="QA",
)

# String form also works
qa = make_task_spec("question -> answer", instructions="Answer the question.")
```

## Calling predictors

All module and predictor calls are async:

```python
import asyncio

from dspy.predict import Predict

predict = Predict(qa)
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
| `sig.append("field", OutputField(...))` | `spec.append(FieldSpec.output("field", ...))` |
| `sig.prepend("field", InputField(...))` | `spec.prepend(FieldSpec.input("field", ...))` |
| `sig.delete("field")` | `spec.delete("field")` |
| `sig.with_instructions("...")` | `spec.with_instructions("...")` |

## Saved program state

Saved programs now store `task_spec` instead of `signature`. Reload with the current DSPy version; legacy signature-only state is rejected.

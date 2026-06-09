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
| `Tool(func)` | `Tool(func, description="...")` |

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

All module and predictor calls are async. Pass `run=` for `RunContext` and `options=PredictOptions(...)` for per-call overrides:

```python
import asyncio

from dspy.predict import Predict
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig

predict = Predict(QATaskSpec())
run = RunContext.create(lm=lm, adapter=adapter, telemetry=TelemetryConfig(call_log=CallLogMode.memory))
result = asyncio.run(predict(question="What is DSPy?", run=run))
# or inside async code:
# result = await predict(question="What is DSPy?", run=run)
```

## Task input validation

Predict validates task inputs before each call and raises `ValueError` (not warnings) for:

| Condition | Error |
| --- | --- |
| Unknown field names | `Unknown task input field(s) [...]` |
| Missing required fields | `Missing required task input field(s) [...]` |
| Type mismatch vs `FieldSpec` | `Type mismatch for task input field '...'` |
| Reserved kwargs as task inputs (`lm`, `config`, `demos`, `run`, `options`, etc.) | `Reserved keyword(s) [...] must not be passed as task inputs` |

Before (extra fields were ignored or warned):

```python
await predict(question="...", extra_field="oops", run=run)  # silently accepted or logged
```

After:

```python
await predict(question="...", extra_field="oops", run=run)
# ValueError: Unknown task input field(s) ['extra_field'] for task spec 'QA'.
```

Optional fields with defaults or `None`-able types do not need to be provided. Fields with `type_=str | None` may be omitted.

## Tools

`Tool` requires an explicit description:

```python
from dspy.adapters.types.tool import Tool

def search(query: str) -> str:
    return query

tool = Tool(search, description="Search documents by query.")
# ReAct, CodeAct, RLM, and ReActV2 require tools=[Tool(...)].
```

## TaskSpec transforms

| Legacy Signature method | TaskSpec method |
| --- | --- |
| `sig.append("field", OutputField(...))` | `spec.append(output_field("field", ...))` |
| `sig.prepend("field", InputField(...))` | `spec.prepend(input_field("field", ...))` |
| `sig.delete("field")` | `spec.delete("field")` |
| `sig.with_instructions("...")` | `spec.with_instructions("...")` |

## Adapters and field metadata

Adapters read `FieldSpec` directly from `dspy.task_spec` (`FieldBinding`, `field_bindings`, `validate_task_inputs`, `validate_task_inputs_from_spec`). Prompt rendering helpers (`format_field_value`, `translate_field_type`, `get_field_spec_description_string`) live in `dspy.adapters.prompt_format`. There is no Pydantic `FieldInfo` bridge layer.

Framework TaskSpecs are colocated under `dspy.task_spec.framework/`; optimizer-owned specs use per-package `task_specs.py` modules (for example `dspy.propose.task_specs`, `dspy.teleprompt.copro.task_specs`).

Field descriptions are required on `input_field` / `output_field` (no `${name}` placeholders). Use `field_desc_from_name(name)` when deriving descriptions from field names at parse time.

## Saved program state

Saved programs now store `task_spec` instead of `signature`. Reload with the current DSPy version; legacy signature-only state is rejected.

See `docs/migration/call-options.md` for `PredictOptions`, `Example.from_record`, and other strict kwargs.

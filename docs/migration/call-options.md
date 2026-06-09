# Call options migration guide

DSPy call sites now use strict keyword arguments. Task inputs, runtime context, and per-call overrides are separated:

- **Task inputs** — field names from your `TaskSpec` (e.g. `question="..."`)
- **`run=`** — `RunContext` (LM, adapter, telemetry, trace)
- **`options=`** — typed per-call overrides (`PredictOptions`, etc.)

Reserved names (`lm`, `config`, `demos`, `task_spec`, `trace`, `prediction`) must not be passed as flat task-input kwargs.

## Quick translation

| Legacy | Strict kwargs |
| --- | --- |
| `await predict(question="...", lm=other_lm)` | `await predict(question="...", run=run, options=PredictOptions(lm=other_lm))` |
| `await predict(question="...", config=cfg)` | `await predict(question="...", run=run, options=PredictOptions(config=cfg))` |
| `await predict(question="...", demos=demos)` | `await predict(question="...", run=run, options=PredictOptions(demos=demos))` |
| `await predict(question="...", trace=False)` | `await predict(question="...", run=run, options=PredictOptions(trace=False))` |
| `Predict(spec, config=LMConfig(...))` | `Predict(spec, config=LMConfig(...))` (unchanged at init) |
| `LM("model", cache=False, api_key="...")` | `LM("model", provider_options=LMProviderOptions(cache=False, api_key="..."))` |
| `Example(a=1, b=2)` / `Example({"a": 1})` | `Example.from_record({"a": 1, "b": 2}, input_keys=("a",))` |
| `example.inputs()` | `example.as_inputs()` |
| `await knn(query)` | `await knn.acall(inputs=query)` |
| `teleprompter.compile(student, trainset=..., num_threads=4, run=run)` | `teleprompter.compile(student, trainset=..., evaluate=EvaluateCompileParams(num_threads=4), run=run)` (COPRO) |

## PredictOptions

`PredictOptions` groups per-call overrides for `Predict` and composed modules (`ChainOfThought`, `ReAct`, etc.).

Fields:

| Field | Purpose |
| --- | --- |
| `lm` | Override the LM for this call |
| `config` | Merge an `LMConfig` patch for this call |
| `demos` | Override few-shot demos |
| `task_spec` | Override the task spec for this call |
| `trace` | Whether to append to `run.trace` (default `True`) |
| `prediction` | Provider predicted-output hint (OpenAI-style content prediction) |

Before:

```python
result = await predict(
    question="What is DSPy?",
    lm=other_lm,
    config=LMConfig(temperature=0.5),
    demos=my_demos,
    trace=False,
    run=run,
)
```

After:

```python
from dspy.core.types import LMConfig, PredictOptions

result = await predict(
    question="What is DSPy?",
    run=run,
    options=PredictOptions(
        lm=other_lm,
        config=LMConfig(temperature=0.5),
        demos=my_demos,
        trace=False,
    ),
)
```

Predicted-output hints (reserved from task inputs):

```python
result = await predict(
    question="Why did a chicken cross the kitchen?",
    run=run,
    options=PredictOptions(
        prediction={"type": "content", "content": "A chicken crossing the kitchen"},
    ),
)
```

## LMConfig on Predict init

Default LM call config is still set at construction via `config=LMConfig(...)`. Per-call patches go in `options.config`.

Before:

```python
predict = Predict(QATaskSpec(), config={"temperature": 0.7, "max_tokens": 500})
result = await predict(question="...", config={"temperature": 0.0}, run=run)
```

After:

```python
from dspy.core.types import LMConfig, PredictOptions

predict = Predict(QATaskSpec(), config=LMConfig(temperature=0.7, max_tokens=500))
result = await predict(
    question="...",
    run=run,
    options=PredictOptions(config=LMConfig(temperature=0.0)),
)
```

## LMProviderOptions

Provider connection and passthrough options moved off top-level `LM(...)` kwargs into `LMProviderOptions`.

Before:

```python
lm = LM(
    "openai/gpt-4o-mini",
    cache=False,
    api_key="...",
    api_base="https://custom.endpoint",
    timeout=30.0,
    max_retries=5,
)
```

After:

```python
from dspy.clients.lm import LM
from dspy.core.types import LMProviderOptions

lm = LM(
    "openai/gpt-4o-mini",
    provider_options=LMProviderOptions(
        cache=False,
        api_key="...",
        api_base="https://custom.endpoint",
        timeout=30.0,
        max_retries=5,
    ),
)
```

`base_url` is accepted as an alias for `api_base`. Use `lm.copy(provider_options=LMProviderOptions(timeout=60.0))` to patch provider options on a copy.

See `docs/migration/memoization.md` for how `cache` fits into the memoization removal.

## Example.from_record and as_inputs

`Example` no longer accepts arbitrary `**kwargs` at construction. Build records explicitly and declare input keys.

Before:

```python
example = Example(question="What is 1+1?", answer="2")
example = example.with_inputs("question")
inputs = example.inputs()
```

After:

```python
from dspy.primitives.example import Example

example = Example.from_record(
    {"question": "What is 1+1?", "answer": "2"},
    input_keys=("question",),
)
inputs = example.as_inputs()  # {"question": "What is 1+1?"}
labels = example.as_labels()  # {"answer": "2"}
```

`as_inputs()` raises `ValueError` when `input_keys` were not set. Use `with_input_keys("field")` to fork with different keys.

Passing examples to modules:

```python
result = await predict(**example.as_inputs(), run=run)
```

## KNN acall

`KNN` queries require the `inputs=` keyword.

Before:

```python
nearest = await knn({"question": "What is 3+3?"})
```

After:

```python
nearest = await knn.acall(inputs={"question": "What is 3+3?"})
# __call__ is an alias for acall
```

## Compile params

Typed compile params replace loose evaluate kwargs where enforced. `run=` remains a separate required keyword.

COPRO example — before:

```python
compiled = await copro.compile(
    student,
    trainset=trainset,
    num_threads=4,
    display_progress=True,
    run=run,
)
```

After:

```python
from dspy.teleprompt.compile_params import EvaluateCompileParams

compiled = await copro.compile(
    student,
    trainset=trainset,
    evaluate=EvaluateCompileParams(num_threads=4, display_progress=True),
    run=run,
)
```

`dspy.teleprompt.compile_params` also defines typed params for other teleprompters (`BootstrapFewShotCompileParams`, `MIPROv2CompileParams`, `BetterTogetherCompileParams`, etc.) for callers migrating off scattered kwargs. Teleprompters that have not yet adopted a typed params object still accept their existing keyword arguments alongside `run=`.

## Positional arguments rejected

Module and predictor calls do not accept positional task inputs:

```python
# TypeError — use keyword task fields
await predict("What is DSPy?", run=run)
```

## Related guides

- `docs/migration/runcontext.md` — `run=` and `RunContext`
- `docs/migration/taskspec.md` — task input validation
- `docs/migration/memoization.md` — `LMProviderOptions.cache`

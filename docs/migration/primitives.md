# Primitives migration guide

`dspy.primitives` now exposes a canonical public barrel. Prefer importing exported symbols from the package root; submodule paths remain valid for deep internals (for example `dspy.primitives.python_interpreter.jsonrpc`).

```python
from dspy.primitives import BatchFailure, BatchResult, Example, Module, Prediction, PythonInterpreter
```

## Breaking changes

| Old | New |
| --- | --- |
| `isinstance(pred, Example)` | `False` — `Prediction` is no longer an `Example` subclass |
| `hash(example)` / `example in {…}` | `TypeError` — `Example` is unhashable |
| `hash(prediction)` / `prediction in {…}` | `TypeError` — `Prediction` is unhashable |
| `Prediction.from_record(record, input_keys=…)` | `Prediction.from_record(record)` only |
| `to_repl_variable` | removed — use `build_repl_variable` |
| `named_sub_modules(skip_compiled=…)` | `skip_compiled` removed; compiled subgraphs are opaque by default |
| `async def aforward` on `Module` subclasses | implement `async def _aforward_impl` instead |
| `await module.batch(...)` / `await parallel(...)` return `list` or 3-tuple | returns `BatchResult`; use `.results` and `.failures` |
| `named_parameters()` / `parameters()` on `BaseModule` | `named_predictors()` / `predictors()` |
| `Predict(Module, Parameter)` / `Parameter` marker | `Predict(Module)`; use `Predictor` protocol for isinstance checks |
| `return_failed_examples=` on `Module.batch` / `Parallel` | removed; `BatchResult.failures` is always populated for failed indices |
| `Parallel.failed_examples` / `Parallel.exceptions` | removed; use `BatchResult.failures` |
| `sync_file` JSON-RPC notification | request/response; failures raise `CodeInterpreterError` |
| Save-time `.pkl` warning text | describes save semantics, not load |

## Module invocation

Call modules with `await module(..., run=run)`. Direct `await module.aforward(...)` still works but emits a one-time warning per class; subclasses should implement `_aforward_impl`.

## Interpreter and sandbox

- `PythonInterpreter`, `CodeInterpreterError`, and `FinalOutput` are exported from `dspy.primitives`.
- `PythonInterpreter.tools` is a read-only mapping view; mutate tools via the constructor or internal `_tools` during runtime injection (for example RLM execution setup).
- Sandbox tool registration advertises scalar types and homogeneous `list` / `dict` containers. Parameterized annotations such as `list[str]` map to their container origin.
- JSON-RPC application error codes are generated from `dspy/primitives/jsonrpc_app_errors.json` via `scripts/generate_jsonrpc_errors.py`.
- Import interpreter internals from `dspy.primitives.python_interpreter` when needed.

## Data types

- `Example` and `Prediction` compose a shared `RecordStore` for attribute and mapping access.
- Use `Example.from_record(record, input_keys=(...))` for labeled training rows; use `Prediction.from_record(record)` for model outputs.
- `Prediction` equality compares store fields and attached `Completions` objects (identity), not numeric scores.
- `Prediction` rich comparisons and arithmetic (`+`, `/`, `<`, etc.) coerce through `float(prediction["score"])`. Missing `score` raises `ValueError`.
- `Module.batch` and `Parallel(...)(pairs)` return `BatchResult` with `.results` and `.failures`. Failures are always populated for indices that raised an exception (`BatchFailure` entries with `.input` and `.exception`).

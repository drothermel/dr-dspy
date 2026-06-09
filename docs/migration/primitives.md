# Primitives migration guide

`dspy.primitives` now exposes a canonical public barrel. Prefer importing exported symbols from the package root; submodule paths remain valid for deep internals (for example `dspy.primitives.python_interpreter.jsonrpc`).

```python
from dspy.primitives import Example, Module, Prediction, PythonInterpreter
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
| `sync_file` JSON-RPC notification | request/response; failures raise `CodeInterpreterError` |
| Save-time `.pkl` warning text | describes save semantics, not load |

## Module invocation

Call modules with `await module(..., run=run)`. Direct `await module.aforward(...)` still works but emits a one-time warning per class; subclasses should implement `_aforward_impl`.

## Interpreter and sandbox

- `PythonInterpreter`, `CodeInterpreterError`, and `FinalOutput` are exported from `dspy.primitives`.
- JSON-RPC application error codes are generated from `dspy/primitives/jsonrpc_app_errors.json` via `scripts/generate_jsonrpc_errors.py`.
- Import interpreter internals from `dspy.primitives.python_interpreter` when needed.

## Data types

- `Example` and `Prediction` compose a shared `RecordStore` for attribute and mapping access.
- Use `Example.from_record(record, input_keys=(...))` for labeled training rows; use `Prediction.from_record(record)` for model outputs.

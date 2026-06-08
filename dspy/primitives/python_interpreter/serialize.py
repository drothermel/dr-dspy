"""Variable serialization and injection for sandbox code execution."""

import json
import keyword
from typing import TYPE_CHECKING, Any

from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter.deno_process import send_request

if TYPE_CHECKING:
    from dspy.primitives.python_interpreter.interpreter import PythonInterpreter

# Pyodide's FFI crashes at exactly 128MB (134,217,728 bytes). Use filesystem
# injection for strings above 100MB to stay safely below this limit.
LARGE_VAR_THRESHOLD = 100 * 1024 * 1024


def to_json_compatible(value: Any) -> Any:
    """Recursively convert Python values to JSON-compatible types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, set):
        try:
            return sorted(to_json_compatible(v) for v in value)
        except TypeError:
            return [to_json_compatible(v) for v in value]
    raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")


def serialize_value(value: Any) -> str:
    """Serialize a Python value to a Python literal string for injection.

    Sets and tuples are converted to lists for JSON round-trip compatibility,
    since the sandbox returns values via JSON which doesn't support these types.
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        # Must check bool before int since bool is a subclass of int
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        # Tuples become lists for JSON compatibility
        items = ", ".join(serialize_value(item) for item in value)
        return f"[{items}]"
    if isinstance(value, dict):
        items = ", ".join(f"{serialize_value(k)}: {serialize_value(v)}" for k, v in value.items())
        return f"{{{items}}}"
    if isinstance(value, set):
        # Sets become sorted lists (or unsorted if mixed types) for JSON compatibility
        try:
            sorted_items = sorted(value)
        except TypeError:
            sorted_items = list(value)
        items = ", ".join(serialize_value(item) for item in sorted_items)
        return f"[{items}]"
    raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")


def inject_variables(interpreter: "PythonInterpreter", code: str, variables: dict[str, Any]) -> str:
    """Insert Python assignments for each variable at the top of the code."""
    for key in variables:
        if not key.isidentifier() or keyword.iskeyword(key) or key == "json":
            raise CodeInterpreterError(f"Invalid variable name: '{key}'")

    large_vars = {}
    small_assignments = []
    for k, v in variables.items():
        serialized = serialize_value(v)
        if len(serialized) > LARGE_VAR_THRESHOLD:
            large_vars[k] = json.dumps(to_json_compatible(v))
        else:
            small_assignments.append(f"{k} = {serialized}")

    interpreter._pending_large_vars = large_vars

    if large_vars:
        large_assignments = [f"{k} = json.loads(open('/tmp/dspy_vars/{k}.json').read())" for k in large_vars]
        assignments = ["import json"] + small_assignments + large_assignments
    else:
        assignments = small_assignments

    return "\n".join(assignments) + "\n" + code if assignments else code


def inject_large_var(interpreter: "PythonInterpreter", name: str, value: str) -> None:
    """Inject a large variable via the virtual filesystem."""
    send_request(
        interpreter=interpreter,
        method="inject_var",
        params={"name": name, "value": value},
        context=f"injecting variable '{name}'",
    )

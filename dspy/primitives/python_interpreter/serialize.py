import json
import keyword
from typing import TYPE_CHECKING, Any, Literal

from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter.deno_process import send_request

if TYPE_CHECKING:
    from dspy.primitives.python_interpreter.interpreter import PythonInterpreter

LARGE_VAR_THRESHOLD = 100 * 1024 * 1024
DSPY_VARS_VPATH = "/tmp/dspy_vars"  # noqa: S108 — Pyodide virtual FS path, not host /tmp

JsonWalkMode = Literal["python_literal", "json"]


def _format_leaf(value: Any, *, mode: JsonWalkMode) -> Any:
    if mode == "json":
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")


def _sorted_set_items(value: set[Any]) -> list[Any]:
    try:
        return sorted(value)
    except TypeError:
        return list(value)


def _walk_jsonable(value: Any, *, mode: JsonWalkMode) -> Any:
    if isinstance(value, dict):
        if mode == "json":
            return {k: _walk_jsonable(v, mode=mode) for k, v in value.items()}
        items = ", ".join(f"{_walk_jsonable(k, mode=mode)}: {_walk_jsonable(v, mode=mode)}" for k, v in value.items())
        return f"{{{items}}}"
    if isinstance(value, (list, tuple)):
        if mode == "json":
            return [_walk_jsonable(item, mode=mode) for item in value]
        items = ", ".join(_walk_jsonable(item, mode=mode) for item in value)
        return f"[{items}]"
    if isinstance(value, set):
        sorted_items = _sorted_set_items(value)
        if mode == "json":
            try:
                return sorted(_walk_jsonable(item, mode=mode) for item in sorted_items)
            except TypeError:
                return [_walk_jsonable(item, mode=mode) for item in sorted_items]
        items = ", ".join(_walk_jsonable(item, mode=mode) for item in sorted_items)
        return f"[{items}]"
    return _format_leaf(value, mode=mode)


def to_json_compatible(value: Any) -> Any:
    return _walk_jsonable(value, mode="json")


def serialize_value(value: Any) -> str:
    result = _walk_jsonable(value, mode="python_literal")
    if isinstance(result, str):
        return result
    raise CodeInterpreterError(f"Unsupported value type: {type(value).__name__}")


def _reject_json_name_for_large_vars(*, large_vars: dict[str, str]) -> None:
    if "json" in large_vars:
        raise CodeInterpreterError("Invalid variable name: 'json'")


def inject_variables(interpreter: "PythonInterpreter", code: str, variables: dict[str, Any]) -> str:
    for key in variables:
        if not key.isidentifier() or keyword.iskeyword(key):
            raise CodeInterpreterError(f"Invalid variable name: '{key}'")

    large_vars: dict[str, str] = {}
    small_assignments: list[str] = []
    for key, value in variables.items():
        serialized = serialize_value(value)
        if len(serialized) > LARGE_VAR_THRESHOLD:
            large_vars[key] = json.dumps(to_json_compatible(value))
        else:
            small_assignments.append(f"{key} = {serialized}")

    _reject_json_name_for_large_vars(large_vars=large_vars)

    interpreter._pending_large_vars = large_vars
    if large_vars:
        large_assignments = [f"{k} = json.loads(open('{DSPY_VARS_VPATH}/{k}.json').read())" for k in large_vars]
        assignments = ["import json"] + small_assignments + large_assignments
    else:
        assignments = small_assignments
    return "\n".join(assignments) + "\n" + code if assignments else code


def inject_large_var(interpreter: "PythonInterpreter", name: str, value: str) -> None:
    send_request(
        interpreter=interpreter,
        method="inject_var",
        params={"name": name, "value": value},
        context=f"injecting variable '{name}'",
    )

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreterError
from dspy.primitives.python_interpreter.deno_process import deno_stdin, send_request
from dspy.primitives.python_interpreter.jsonrpc import JSONRPC_APP_ERRORS, jsonrpc_error, jsonrpc_result

if TYPE_CHECKING:
    from dspy.primitives.python_interpreter.interpreter import PythonInterpreter


def extract_parameters(fn: Callable) -> list[dict]:
    sig = inspect.signature(fn)
    params = []
    for name, param in sig.parameters.items():
        p = {"name": name}
        if param.annotation != inspect.Parameter.empty and param.annotation in SIMPLE_TYPES:
            p["type"] = param.annotation.__name__
        if param.default != inspect.Parameter.empty:
            p["default"] = param.default
        params.append(p)
    return params


def register_tools(interpreter: "PythonInterpreter") -> None:
    if interpreter._tools_registered:
        return
    params = {}
    if interpreter.tools:
        tools_info = []
        for name, fn in interpreter.tools.items():
            tools_info.append({"name": name, "parameters": extract_parameters(fn)})
        params["tools"] = tools_info
    if interpreter.output_fields:
        params["outputs"] = interpreter.output_fields
    if not params:
        interpreter._tools_registered = True
        return
    send_request(interpreter=interpreter, method="register", params=params, context="registering tools/outputs")
    interpreter._tools_registered = True


def handle_tool_call(interpreter: "PythonInterpreter", request: dict) -> None:
    request_id = request["id"]
    params = request.get("params", {})
    tool_name = params.get("name")
    kwargs = params.get("kwargs", {})
    try:
        if tool_name not in interpreter.tools:
            raise CodeInterpreterError(f"Unknown tool: {tool_name}")
        result = interpreter.tools[tool_name](**kwargs)
        if asyncio.iscoroutine(result):
            raise TypeError(
                "Python interpreter tools invoked from the sync JSON-RPC path must not return coroutines. "
                "Provide a synchronous callable."
            )
        is_json = isinstance(result, (list, dict))
        response = jsonrpc_result(
            result={
                "value": json.dumps(result) if is_json else str(result) if result is not None else "",
                "type": "json" if is_json else "string",
            },
            id=request_id,
        )
    except Exception as e:
        error_type = type(e).__name__
        error_code = JSONRPC_APP_ERRORS.get(error_type, JSONRPC_APP_ERRORS["Unknown"])
        response = jsonrpc_error(code=error_code, message=str(e), id=request_id, data={"type": error_type})
    stdin = deno_stdin(interpreter)
    stdin.write(response + "\n")
    stdin.flush()

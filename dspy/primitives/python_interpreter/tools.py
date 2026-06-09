import asyncio
import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from dspy.primitives.code_interpreter import CodeInterpreterError, annotation_to_sandbox_type
from dspy.primitives.python_interpreter.deno_process import deno_stdin, send_request
from dspy.primitives.python_interpreter.jsonrpc import JSONRPC_APP_ERRORS, jsonrpc_error, jsonrpc_result

if TYPE_CHECKING:
    from dspy.primitives.python_interpreter.interpreter import PythonInterpreter


class ToolParameterSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str | None = None
    default: Any | None = None
    include_default: bool = False

    def to_registration_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.type is not None:
            payload["type"] = self.type
        if self.include_default:
            payload["default"] = self.default
        return payload


def extract_parameters(fn: Callable) -> list[ToolParameterSpec]:
    sig = inspect.signature(fn)
    params: list[ToolParameterSpec] = []
    for name, param in sig.parameters.items():
        type_name = None
        if param.annotation != inspect.Parameter.empty:
            type_name = annotation_to_sandbox_type(param.annotation)
        include_default = param.default != inspect.Parameter.empty
        default = param.default if include_default else None
        params.append(
            ToolParameterSpec(
                name=name,
                type=type_name,
                default=default,
                include_default=include_default,
            )
        )
    return params


def register_tools(interpreter: "PythonInterpreter") -> None:
    if interpreter._tools_registered:
        return
    params = {}
    if interpreter._tools:
        tools_info = []
        for name, fn in interpreter._tools.items():
            tools_info.append(
                {
                    "name": name,
                    "parameters": [spec.to_registration_dict() for spec in extract_parameters(fn)],
                }
            )
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
        if tool_name not in interpreter._tools:
            raise CodeInterpreterError(f"Unknown tool: {tool_name}")
        result = interpreter._tools[tool_name](**kwargs)
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

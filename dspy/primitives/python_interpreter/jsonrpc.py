import asyncio
import json
import os
from os import PathLike
from pathlib import Path
from typing import Any

JSONRPC_APP_ERRORS = {
    "SyntaxError": -32000,
    "NameError": -32001,
    "TypeError": -32002,
    "ValueError": -32003,
    "AttributeError": -32004,
    "IndexError": -32005,
    "KeyError": -32006,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}


def canonicalize_path(path: PathLike | str) -> str:
    return str(Path(os.fspath(path)).expanduser().resolve())


def jsonrpc_request(method: str, params: dict[str, Any], id: int | str) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": id})


def jsonrpc_notification(method: str, params: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return json.dumps(msg)


def jsonrpc_result(result: Any, id: int | str) -> str:
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id})


def jsonrpc_error(code: int, message: str, id: int | str, data: dict[str, Any] | None = None) -> str:
    err: dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": err, "id": id})


def await_in_sync(coroutine: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coroutine)
    return loop.run_until_complete(coroutine)

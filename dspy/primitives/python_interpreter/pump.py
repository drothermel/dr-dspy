from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dspy.primitives.python_interpreter.deno_process import (
    MAX_SKIP_LINES,
    parse_response_line,
    read_response_line,
)
from dspy.primitives.python_interpreter.protocol import CodeInterpreterError

if TYPE_CHECKING:
    from dspy.primitives.python_interpreter.interpreter import PythonInterpreter

NotificationHandler = Callable[[dict[str, Any]], bool]
ResultHandler = Callable[[dict[str, Any]], Any]
ErrorHandler = Callable[[dict[str, Any]], Any]


def read_until_response(
    interpreter: PythonInterpreter,
    *,
    expected_id: int,
    context: str,
    on_notification: NotificationHandler | None = None,
    on_result: ResultHandler | None = None,
    on_error: ErrorHandler | None = None,
) -> Any:
    skipped = 0
    while skipped <= MAX_SKIP_LINES:
        output_line = read_response_line(interpreter, context)
        msg = parse_response_line(response_line=output_line, context=context)
        if msg is None:
            skipped += 1
            continue
        if on_notification is not None and "method" in msg:
            if on_notification(msg):
                continue
        if "result" in msg:
            if msg.get("id") != expected_id:
                raise CodeInterpreterError(
                    f"Response ID mismatch {context}: expected {expected_id}, got {msg.get('id')}"
                )
            if on_result is not None:
                return on_result(msg)
            return msg
        if "error" in msg:
            if msg.get("id") is not None and msg.get("id") != expected_id:
                raise CodeInterpreterError(
                    f"Response ID mismatch {context}: expected {expected_id}, got {msg.get('id')}"
                )
            if on_error is not None:
                return on_error(msg)
            error = msg["error"]
            raise CodeInterpreterError(f"Error {context}: {error.get('message', 'Unknown error')}")
        raise CodeInterpreterError(f"Unexpected message format from sandbox: {msg}")
    raise CodeInterpreterError(f"Too many non-JSON lines ({skipped}) {context}")

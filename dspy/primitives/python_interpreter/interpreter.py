import threading
from collections.abc import Callable
from os import PathLike
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import deno_process
from dspy.primitives.python_interpreter.deno_process import (
    MAX_SKIP_LINES,
    ensure_deno_process,
    get_deno_dir,
    get_runner_path,
    mount_files,
    read_response_line,
    sync_files,
)
from dspy.primitives.python_interpreter.jsonrpc import (
    JSONRPC_APP_ERRORS,
    canonicalize_path,
    jsonrpc_notification,
    jsonrpc_request,
)
from dspy.primitives.python_interpreter.serialize import inject_large_var, inject_variables
from dspy.primitives.python_interpreter.tools import handle_tool_call, register_tools


class PythonInterpreter:
    _MAX_SKIP_LINES = MAX_SKIP_LINES

    def __init__(
        self,
        deno_command: list[str] | None = None,
        enable_read_paths: list[PathLike | str] | None = None,
        enable_write_paths: list[PathLike | str] | None = None,
        enable_env_vars: list[str] | None = None,
        enable_network_access: list[str] | None = None,
        sync_files: bool = True,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
    ) -> None:
        if isinstance(deno_command, dict):
            raise TypeError("deno_command must be a list of strings, not a dict")
        self.enable_read_paths = enable_read_paths or []
        self.enable_write_paths = enable_write_paths or []
        self.enable_env_vars = enable_env_vars or []
        self.enable_network_access = enable_network_access or []
        self.sync_files = sync_files
        self.tools = dict(tools) if tools else {}
        self.output_fields = output_fields
        self._tools_registered = False
        if deno_command:
            self.deno_command = list(deno_command)
        else:
            args = ["deno", "run"]
            deno_dir = get_deno_dir()
            raw_read_paths = [
                get_runner_path(),
                *([deno_dir] if deno_dir else []),
                *self.enable_read_paths,
                *self.enable_write_paths,
            ]
            allowed_read_paths = [canonicalize_path(p) for p in raw_read_paths]
            args.append(f"--allow-read={','.join(allowed_read_paths)}")
            self._env_arg = ""
            if self.enable_env_vars:
                user_vars = [str(v).strip() for v in self.enable_env_vars]
                args.append("--allow-env=" + ",".join(user_vars))
                self._env_arg = ",".join(user_vars)
            if self.enable_network_access:
                args.append(f"--allow-net={','.join(str(x) for x in self.enable_network_access)}")
            if self.enable_write_paths:
                args.append(f"--allow-write={','.join(canonicalize_path(x) for x in self.enable_write_paths)}")
            args.append(canonicalize_path(get_runner_path()))
            if self._env_arg:
                args.append(self._env_arg)
            self.deno_command = args
        self.deno_process: subprocess.Popen[str] | None = None
        self._mounted_files = False
        self._sandbox_virtual_paths: dict[str, str] = {}
        self._request_id = 0
        self._owner_thread: int | None = None
        self._pending_large_vars = {}

    def _check_thread_ownership(self) -> None:
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "PythonInterpreter is not thread-safe and cannot be shared across threads. Create a separate interpreter instance for each thread."
            )

    def _prepare_sandbox(self) -> None:
        ensure_deno_process(self)
        mount_files(self)
        register_tools(self)
        for name, value in self._pending_large_vars.items():
            inject_large_var(interpreter=self, name=name, value=value)

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self._check_thread_ownership()
        variables = variables or {}
        code = inject_variables(interpreter=self, code=code, variables=variables)
        self._prepare_sandbox()
        self._request_id += 1
        execute_request_id = self._request_id
        input_data = jsonrpc_request(method="execute", params={"code": code}, id=execute_request_id)
        stdin = deno_process.deno_stdin(self)
        try:
            stdin.write(input_data + "\n")
            stdin.flush()
        except BrokenPipeError:
            self._prepare_sandbox()
            stdin = deno_process.deno_stdin(self)
            try:
                stdin.write(input_data + "\n")
                stdin.flush()
            except BrokenPipeError as exc:
                raise CodeInterpreterError("Deno process stdin unavailable during execution") from exc
        skipped = 0
        while skipped <= self._MAX_SKIP_LINES:
            output_line = read_response_line(self, "during execution")
            msg = deno_process.parse_response_line(response_line=output_line, context="during execution")
            if msg is None:
                skipped += 1
                continue
            if "method" in msg and msg["method"] == "tool_call":
                handle_tool_call(self, msg)
                continue
            if "result" in msg:
                if msg.get("id") != execute_request_id:
                    raise CodeInterpreterError(
                        f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}"
                    )
                result = msg["result"]
                sync_files(self)
                if "final" in result:
                    return FinalOutput(result["final"])
                return result.get("output", None)
            if "error" in msg:
                if msg.get("id") is not None and msg.get("id") != execute_request_id:
                    raise CodeInterpreterError(
                        f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}"
                    )
                error = msg["error"]
                error_code = error.get("code", JSONRPC_APP_ERRORS["Unknown"])
                error_message = error.get("message", "Unknown error")
                error_data = error.get("data", {})
                error_type = error_data.get("type", "Error")
                if error_code == JSONRPC_APP_ERRORS["SyntaxError"]:
                    raise SyntaxError(f"Invalid Python syntax. message: {error_message}")
                raise CodeInterpreterError(f"{error_type}: {error_data.get('args') or error_message}")
            raise CodeInterpreterError(f"Unexpected message format from sandbox: {msg}")
        raise CodeInterpreterError(f"Too many non-JSON lines ({skipped}) during execution")

    def start(self) -> None:
        ensure_deno_process(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code=code, variables=variables)

    def shutdown(self) -> None:
        if self.deno_process and self.deno_process.poll() is None:
            stdin = self.deno_process.stdin
            if stdin is not None:
                stdin.write(jsonrpc_notification("shutdown") + "\n")
                stdin.flush()
                stdin.close()
            self.deno_process.wait()
        self.deno_process = None
        self._owner_thread = None

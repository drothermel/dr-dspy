import functools
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import IO, TYPE_CHECKING

from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter.jsonrpc import canonicalize_path, jsonrpc_notification, jsonrpc_request

if TYPE_CHECKING:
    from dspy.primitives.python_interpreter.interpreter import PythonInterpreter
logger = logging.getLogger(__name__)
MAX_SKIP_LINES = 100


@functools.lru_cache(maxsize=1)
def get_deno_dir() -> str | None:
    if "DENO_DIR" in os.environ:
        return os.environ["DENO_DIR"]
    try:
        result = subprocess.run(["deno", "info", "--json"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return info.get("denoDir")
    except Exception:
        logger.warning("Unable to find the Deno cache dir.")
    return None


def get_runner_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "runner.js")


def mount_files(interpreter: "PythonInterpreter") -> None:
    if interpreter._mounted_files:
        return
    paths_to_mount = []
    if interpreter.enable_read_paths:
        paths_to_mount.extend(interpreter.enable_read_paths)
    if interpreter.enable_write_paths:
        paths_to_mount.extend(interpreter.enable_write_paths)
    if not paths_to_mount:
        return
    for path in paths_to_mount:
        if not path:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            if interpreter.enable_write_paths and path in interpreter.enable_write_paths:
                path_obj.open("a").close()
            else:
                raise FileNotFoundError(f"Cannot mount non-existent file: {path}")
        virtual_path = f"/sandbox/{path_obj.name}"
        host_path = canonicalize_path(path)
        send_request(
            interpreter=interpreter,
            method="mount_file",
            params={"host_path": host_path, "virtual_path": virtual_path},
            context=f"mounting {path}",
        )
    interpreter._mounted_files = True


def sync_files(interpreter: "PythonInterpreter") -> None:
    if not interpreter.enable_write_paths or not interpreter.sync_files:
        return
    for path in interpreter.enable_write_paths:
        virtual_path = f"/sandbox/{Path(path).name}"
        host_path = canonicalize_path(path)
        sync_msg = jsonrpc_notification("sync_file", {"virtual_path": virtual_path, "host_path": host_path})
        stdin = deno_stdin(interpreter)
        stdin.write(sync_msg + "\n")
        stdin.flush()


def ensure_deno_process(interpreter: "PythonInterpreter") -> None:
    if interpreter.deno_process is None or interpreter.deno_process.poll() is not None:
        interpreter._tools_registered = False
        interpreter._mounted_files = False
        try:
            interpreter.deno_process = subprocess.Popen(
                interpreter.deno_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="UTF-8",
                env=os.environ.copy(),
            )
        except FileNotFoundError as e:
            install_instructions = "Deno executable not found. Please install Deno to proceed.\nInstallation instructions:\n> curl -fsSL https://deno.land/install.sh | sh\n*or*, on macOS with Homebrew:\n> brew install deno\nFor additional configurations: https://docs.deno.com/runtime/getting_started/installation/"
            raise CodeInterpreterError(install_instructions) from e
        health_check(interpreter)


def require_deno_process(interpreter: "PythonInterpreter") -> subprocess.Popen[str]:
    ensure_deno_process(interpreter)
    if interpreter.deno_process is None:
        raise CodeInterpreterError("Deno process unavailable")
    return interpreter.deno_process


def deno_stdin(interpreter: "PythonInterpreter") -> IO[str]:
    stdin = require_deno_process(interpreter).stdin
    if stdin is None:
        raise CodeInterpreterError("Deno process stdin unavailable")
    return stdin


def deno_stdout(interpreter: "PythonInterpreter") -> IO[str]:
    stdout = require_deno_process(interpreter).stdout
    if stdout is None:
        raise CodeInterpreterError("Deno process stdout unavailable")
    return stdout


def read_response_line(interpreter: "PythonInterpreter", context: str) -> str:
    process = require_deno_process(interpreter)
    response_line = deno_stdout(interpreter).readline().strip()
    if response_line:
        return response_line
    exit_code = process.poll()
    if exit_code is not None:
        stderr = process.stderr.read() if process.stderr else ""
        raise CodeInterpreterError(f"Deno exited (code {exit_code}) {context}: {stderr}")
    raise CodeInterpreterError(f"No response {context}")


def parse_response_line(response_line: str, context: str) -> dict | None:
    if not response_line.startswith("{"):
        logger.debug("Skipping non-JSON output during %s: %s", context, response_line)
        return None
    try:
        return json.loads(response_line)
    except json.JSONDecodeError:
        logger.debug("Skipping malformed JSON during %s: %s", context, response_line[:100])
        return None


def send_request(interpreter: "PythonInterpreter", method: str, params: dict, context: str) -> dict:
    interpreter._request_id += 1
    request_id = interpreter._request_id
    msg = jsonrpc_request(method=method, params=params, id=request_id)
    stdin = deno_stdin(interpreter)
    stdin.write(msg + "\n")
    stdin.flush()
    skipped = 0
    while skipped <= MAX_SKIP_LINES:
        response_line = read_response_line(interpreter, context)
        response = parse_response_line(response_line=response_line, context=context)
        if response is None:
            skipped += 1
            continue
        if response.get("id") != request_id:
            raise CodeInterpreterError(
                f"Response ID mismatch {context}: expected {request_id}, got {response.get('id')}"
            )
        if "error" in response:
            raise CodeInterpreterError(f"Error {context}: {response['error'].get('message', 'Unknown error')}")
        return response
    raise CodeInterpreterError(f"Too many non-JSON lines ({skipped}) {context}")


def health_check(interpreter: "PythonInterpreter") -> None:
    response = send_request(
        interpreter=interpreter, method="execute", params={"code": "print(1+1)"}, context="during health check"
    )
    if response.get("result", {}).get("output", "").strip() != "2":
        raise CodeInterpreterError(f"Unexpected ping response: {response}")

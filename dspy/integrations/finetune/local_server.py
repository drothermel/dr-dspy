from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from dspy._internal.lazy_import import import_optional

if TYPE_CHECKING:
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)

_SERVER_READY_SLEEP_SECONDS = 5
_CONNECTION_RETRY_SLEEP_SECONDS = 1


class LmEndpointSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    api_base: str | None
    api_key: str | None
    provider_options: Any


@dataclass(frozen=True)
class LocalServerHandle:
    process: subprocess.Popen[str]
    thread: threading.Thread
    get_logs: Any
    port: int


def snapshot_lm_endpoint(lm: LM) -> LmEndpointSnapshot:
    return LmEndpointSnapshot(
        api_base=lm.kwargs.get("api_base"),
        api_key=lm.kwargs.get("api_key"),
        provider_options=lm.provider_options.model_copy(deep=True),
    )


def restore_lm_endpoint(lm: LM, snapshot: LmEndpointSnapshot) -> None:
    if snapshot.api_base is None:
        lm.kwargs.pop("api_base", None)
    else:
        lm.kwargs["api_base"] = snapshot.api_base
    if snapshot.api_key is None:
        lm.kwargs.pop("api_key", None)
    else:
        lm.kwargs["api_key"] = snapshot.api_key
    lm.provider_options = snapshot.provider_options.model_copy(deep=True)


def apply_local_server_endpoint(lm: LM, *, port: int) -> None:
    lm.kwargs["api_base"] = f"http://localhost:{port}/v1"
    lm.kwargs["api_key"] = "local"
    lm.provider_options = lm.provider_options.model_copy(
        update={"api_base": f"http://localhost:{port}/v1", "api_key": "local"}
    )


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def wait_for_server(base_url: str, timeout: int | None = None) -> None:
    import requests

    start_time = time.time()
    while True:
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError("Server did not become ready within timeout period")
        try:
            response = requests.get(f"{base_url}/v1/models", headers={"Authorization": "Bearer None"})
            if response.status_code == 200:
                time.sleep(_SERVER_READY_SLEEP_SECONDS)
                return
        except requests.exceptions.RequestException:
            time.sleep(_CONNECTION_RETRY_SLEEP_SECONDS)


def launch_local_server(*, model: str, timeout: int) -> LocalServerHandle:
    port = get_free_port()
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",  # noqa: S104
    ]
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logger.info("SGLang server process started with PID %s.", process.pid)
    stop_printing_event = threading.Event()
    logs_buffer: list[str] = []

    def _tail_process(proc: subprocess.Popen[str], buffer: list[str], stop_event: threading.Event) -> None:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                buffer.append(line)
                if not stop_event.is_set():
                    pass

    thread = threading.Thread(target=_tail_process, args=(process, logs_buffer, stop_printing_event), daemon=True)
    thread.start()
    base_url = f"http://localhost:{port}"
    try:
        wait_for_server(base_url, timeout=timeout)
    except TimeoutError:
        process.kill()
        raise
    stop_printing_event.set()

    def get_logs() -> str:
        return "".join(logs_buffer)

    return LocalServerHandle(process=process, thread=thread, get_logs=get_logs, port=port)


def attach_local_server(lm: LM, handle: LocalServerHandle) -> LmEndpointSnapshot:
    endpoint_snapshot = snapshot_lm_endpoint(lm)
    apply_local_server_endpoint(lm, port=handle.port)
    lm_attrs = cast("Any", lm)
    lm_attrs.get_logs = handle.get_logs
    lm_attrs.process = handle.process
    lm_attrs.thread = handle.thread
    lm_attrs._local_server_endpoint_snapshot = endpoint_snapshot
    return endpoint_snapshot


def kill_local_server(lm: LM) -> None:
    sglang_utils = import_optional(
        "sglang.utils",
        feature="local model launching",
        install_command="Navigate to https://docs.sglang.ai/start/install.html for the latest installation instructions.",
    )
    terminate_process = sglang_utils.terminate_process

    if not hasattr(lm, "process"):
        logger.info("No running server to kill.")
        return
    terminate_process(lm.process)
    thread = getattr(lm, "thread", None)
    if thread is not None:
        thread.join()
    snapshot = getattr(lm, "_local_server_endpoint_snapshot", None)
    if isinstance(snapshot, LmEndpointSnapshot):
        restore_lm_endpoint(lm, snapshot)
        lm_attrs = cast("Any", lm)
        del lm_attrs._local_server_endpoint_snapshot
    lm_attrs = cast("Any", lm)
    del lm_attrs.process
    if hasattr(lm, "thread"):
        del lm_attrs.thread
    if hasattr(lm, "get_logs"):
        del lm_attrs.get_logs
    logger.info("Server killed.")

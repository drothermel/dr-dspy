import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR = "LITELLM_TEST_SERVER_LOG_FILE_PATH"


@pytest.fixture
def litellm_test_server() -> tuple[str, str]:  # ty:ignore[invalid-return-type]
    """
    Start a LiteLLM test server for a DSPy integration test case, and tear down the
    server when the test case completes.
    """
    if importlib.util.find_spec("litellm") is None:
        pytest.skip("litellm is not installed")  # ty: ignore[too-many-positional-arguments]
    if sys.version_info[:2] == (3, 14):
        pytest.skip("Litellm proxy server is not supported on Python 3.14.")
    with tempfile.TemporaryDirectory() as server_log_dir_path:
        server_log_file_path = os.path.join(server_log_dir_path, "request_logs.jsonl")
        open(server_log_file_path, "a").close()

        port = _get_random_port()
        host = "127.0.0.1"

        litellm_cmd = shutil.which("litellm") or str(Path(sys.executable).with_name("litellm"))
        if not Path(litellm_cmd).exists():
            pytest.skip("litellm CLI is not installed")

        process = subprocess.Popen(
            [litellm_cmd, "--host", host, "--port", str(port), "--config", _get_litellm_config_path()],
            env={LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR: server_log_file_path, **os.environ.copy()},
            text=True,
        )

        try:
            _wait_for_port(host=host, port=port)
        except TimeoutError:
            process.terminate()
            raise

        server_url = f"http://{host}:{port}"
        yield server_url, server_log_file_path

        process.kill()
        process.wait()


def read_litellm_test_server_request_logs(server_log_file_path: str) -> list[dict[str, Any]]:
    """
    Read request logs from a LiteLLM server used during DSPy integration tests.

    Args:
        server_log_file_path: The filesystem path to the LiteLLM server request logs jsonlines file.
    Return:
        A list of log entries, where each entry corresponds to one request handled by the server.
    """
    data = []
    with open(server_log_file_path) as f:
        for line in f:
            data.append(json.loads(line))  # noqa: PERF401

    return data


def _get_litellm_config_path():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "litellm_server_config.yaml")


def _get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host, port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((host, port))
                return True
            except ConnectionRefusedError:
                time.sleep(0.5)
    raise TimeoutError(f"Server on port {port} did not become ready within {timeout} seconds.")

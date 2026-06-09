import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from dspy.primitives.python_interpreter import jsonrpc
from dspy.primitives.python_interpreter.deno_process import sync_files
from dspy.primitives.python_interpreter.jsonrpc import jsonrpc_notification

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_PATH = ROOT / "dspy/primitives/jsonrpc_app_errors.json"
RUNNER_PATH = ROOT / "dspy/primitives/runner.js"


def test_jsonrpc_notification_includes_empty_params():
    payload = json.loads(jsonrpc_notification("sync_file", {}))
    assert payload["params"] == {}


def test_jsonrpc_app_errors_match_canonical_json():
    canonical = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    assert {key: int(value) for key, value in canonical.items()} == jsonrpc.JSONRPC_APP_ERRORS


def test_runner_jsonrpc_app_errors_match_canonical_json():
    canonical = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    block_match = re.search(
        r"/\* JSONRPC_APP_ERRORS_BEGIN \*/(.*)/\* JSONRPC_APP_ERRORS_END \*/",
        runner_text,
        re.DOTALL,
    )
    assert block_match is not None
    block = block_match.group(1)
    for key, code in canonical.items():
        assert re.search(rf"^\s*{re.escape(key)}:\s*{code},?\s*$", block, re.MULTILINE)


def test_sync_files_raises_on_host_error(monkeypatch):
    from types import SimpleNamespace

    from dspy.primitives.code_interpreter import CodeInterpreterError
    from dspy.primitives.python_interpreter import deno_process

    host_path = "/sandbox-host/out.txt"
    interpreter = SimpleNamespace(
        enable_write_paths=[host_path],
        sync_files=True,
        _sandbox_virtual_paths={host_path: "/sandbox/out.txt"},
        _request_id=0,
    )

    def fake_send_request(*, interpreter, method, params, context):
        if method == "sync_file":
            raise CodeInterpreterError(f"Error {context}: disk full")
        return {"result": {}}

    monkeypatch.setattr(deno_process, "send_request", fake_send_request)
    with pytest.raises(CodeInterpreterError, match=r"syncing /sandbox-host/out\.txt"):
        sync_files(cast("Any", interpreter))

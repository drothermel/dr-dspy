import json

from dspy.primitives.python_interpreter.jsonrpc import jsonrpc_notification


def test_jsonrpc_notification_includes_empty_params():
    payload = json.loads(jsonrpc_notification("sync_file", {}))
    assert payload["params"] == {}

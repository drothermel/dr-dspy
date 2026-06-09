from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import requests

from dspy.clients.finetune.service import FinetuneService
from dspy.clients.finetune.utils import TrainDataFormat
from dspy.clients.lm import LM
from dspy.integrations.finetune import local_server
from dspy.integrations.finetune.local_server import (
    LmEndpointSnapshot,
    attach_local_server,
    kill_local_server,
    restore_lm_endpoint,
    snapshot_lm_endpoint,
    wait_for_server,
)
from dspy.integrations.finetune.openai import OpenAIProvider


def test_wait_for_server_times_out_on_connection_errors(monkeypatch: pytest.MonkeyPatch):
    times = iter([0.0, 2.0])

    def fake_time() -> float:
        return next(times)

    monkeypatch.setattr(local_server.time, "time", fake_time)
    monkeypatch.setattr(local_server.time, "sleep", lambda _seconds: None)

    def _raise_connection_error(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr("requests.get", _raise_connection_error)
    with pytest.raises(TimeoutError, match="Server did not become ready"):
        wait_for_server("http://localhost:9999", timeout=1)


def _lm_with_endpoint() -> LM:
    lm = LM("openai/gpt-4.1-mini")
    lm.kwargs["api_base"] = "https://example.com/v1"
    lm.kwargs["api_key"] = "secret"
    return lm


def test_snapshot_and_restore_lm_endpoint():
    lm = _lm_with_endpoint()
    snapshot = snapshot_lm_endpoint(lm)
    lm.kwargs["api_base"] = "http://localhost:8000/v1"
    lm.kwargs["api_key"] = "local"
    restore_lm_endpoint(lm, snapshot)
    assert lm.kwargs["api_base"] == "https://example.com/v1"
    assert lm.kwargs["api_key"] == "secret"


def test_kill_local_server_restores_endpoint():
    lm = cast("Any", _lm_with_endpoint())
    snapshot = snapshot_lm_endpoint(lm)
    process = MagicMock()
    thread = MagicMock()
    lm.process = process
    lm.thread = thread
    lm._local_server_endpoint_snapshot = snapshot
    lm.kwargs["api_base"] = "http://localhost:8000/v1"
    lm.kwargs["api_key"] = "local"

    with patch("dspy.integrations.finetune.local_server.import_optional") as import_optional:
        import_optional.return_value.terminate_process = MagicMock()
        kill_local_server(lm)

    assert lm.kwargs["api_base"] == "https://example.com/v1"
    assert lm.kwargs["api_key"] == "secret"
    assert not hasattr(lm, "process")


def test_attach_local_server_records_snapshot():
    lm = cast("Any", _lm_with_endpoint())
    handle = local_server.LocalServerHandle(
        process=MagicMock(),
        thread=MagicMock(),
        get_logs=lambda: "",
        port=8123,
    )
    snapshot = attach_local_server(lm, handle)
    assert isinstance(snapshot, LmEndpointSnapshot)
    assert lm.kwargs["api_base"] == "http://localhost:8123/v1"
    assert lm._local_server_endpoint_snapshot.api_base == "https://example.com/v1"


@patch.object(OpenAIProvider, "finetune", side_effect=RuntimeError("finetune failed"))
def test_finetune_service_failed_job_raises_on_result(mock_finetune: MagicMock):
    assert mock_finetune is not None
    lm = LM("openai/gpt-4.1-mini")
    service = FinetuneService(lm, finetune_provider=OpenAIProvider())
    train_data = [{"messages": [{"role": "user", "content": "hi"}]}]

    job = service.finetune(train_data=train_data, train_data_format=TrainDataFormat.CHAT)
    with pytest.raises(RuntimeError, match="finetune failed"):
        job.result()

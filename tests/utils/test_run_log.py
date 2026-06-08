import json
from pathlib import Path

from dspy.utils.run_log import append_call_record, init_run_session, redact_messages, slug_run_id


def test_slug_run_id_sanitizes_path_chars():
    assert slug_run_id("gepa/v2 test") == "gepa_v2_test"


def test_init_run_session_creates_timestamped_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("DSPY_RUN_ID", raising=False)
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    run_dir = init_run_session(run_log_enabled=True, run_log_dir=None, settings_snapshot={"transparency": "strict"})
    assert run_dir is not None
    assert run_dir.parent.name == "default_run"
    assert (run_dir / "run.json").exists()


def test_init_run_session_uses_dspy_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DSPY_RUN_ID", "my-experiment")
    run_dir = init_run_session(run_log_enabled=True, run_log_dir=None, settings_snapshot={})
    assert run_dir is not None
    assert run_dir.parent.name == "my-experiment"


def test_append_call_record_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    init_run_session(run_log_enabled=True, run_log_dir=None, settings_snapshot={})
    append_call_record({"call_id": "abc", "phase": "predict"})
    calls_files = list(Path(tmp_path).rglob("calls.jsonl"))
    assert len(calls_files) == 1
    lines = calls_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["call_id"] == "abc"


def test_redact_messages_image_data_url():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    redacted = redact_messages(messages)
    image_part = redacted[0]["content"][1]
    assert image_part["redacted"] is True
    assert "sha256" in image_part
    assert image_part["byte_length"] == 4

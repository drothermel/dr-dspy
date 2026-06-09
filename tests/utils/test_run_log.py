import json

from dspy.utils.run_log import append_call_record, create_run_log_session, redact_messages, slug_run_id


def test_slug_run_id():
    assert slug_run_id("my run!") == "my_run_"


def test_create_run_log_session_creates_timestamped_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    session = create_run_log_session(call_log_dir=None, settings_snapshot={"transparency": "strict"})
    assert session.run_dir.exists()
    assert (session.run_dir / "run.json").exists()


def test_create_run_log_session_uses_dspy_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DSPY_RUN_ID", "experiment_a")
    session = create_run_log_session(call_log_dir=None, settings_snapshot={})
    assert "experiment_a" in str(session.run_dir)


def test_append_call_record_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LOG_DIR", str(tmp_path))
    session = create_run_log_session(call_log_dir=None, settings_snapshot={})
    append_call_record({"call_id": "abc", "phase": "predict"}, session=session)
    lines = session.calls_path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["call_id"] == "abc"


def test_redact_messages_image_data_url():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ],
        }
    ]
    redacted = redact_messages(messages)
    assert redacted[0]["content"][0]["type"] == "text"
    assert redacted[0]["content"][1]["redacted"] is True

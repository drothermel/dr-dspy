from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dspy.utils.serialize import to_jsonable

logger = logging.getLogger(__name__)
_SESSION_LOCK = threading.Lock()
_SECRET_KEY_PATTERN = re.compile("(api[_-]?key|authorization|token|secret)", re.IGNORECASE)


@dataclass(frozen=True)
class RunLogSession:
    run_dir: Path
    calls_path: Path
    timestamp: str


def slug_run_id(raw: str) -> str:
    slug = re.sub("[^a-zA-Z0-9._-]+", "_", raw.strip())
    return slug or "default_run"


def resolve_log_root(call_log_dir: str | None) -> Path:
    if call_log_dir:
        return Path(call_log_dir)
    return Path(os.environ.get("DSPY_LOG_DIR", "logs"))


def resolve_run_bucket() -> str:
    return slug_run_id(os.environ.get("DSPY_RUN_ID", "default_run"))


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _redact_content_part(part: dict[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        return part
    if part_type == "image_url":
        url = part.get("image_url", {})
        url_value = url.get("url", "") if isinstance(url, dict) else str(url)
        if url_value.startswith("data:"):
            payload = url_value.split(",", 1)[-1]
            byte_length = len(payload.encode("utf-8"))
            return {
                "type": "image_url",
                "redacted": True,
                "sha256": _hash_bytes(payload.encode("utf-8")),
                "byte_length": byte_length,
            }
        return {"type": "image_url", "redacted": True, "uri": url_value}
    if part_type in {"input_audio", "audio"}:
        data = part.get("input_audio", part.get("audio", {}))
        raw = data.get("data", "") if isinstance(data, dict) else ""
        encoded = raw.encode("utf-8") if isinstance(raw, str) else b""
        return {"type": part_type, "redacted": True, "sha256": _hash_bytes(encoded), "byte_length": len(encoded)}
    return {"type": part_type, "redacted": True, "keys": sorted(part.keys())}


def redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted: list[dict[str, Any]] = []
    for message in messages:
        item = {"role": message.get("role")}
        content = message.get("content")
        if isinstance(content, str):
            item["content"] = content
        elif isinstance(content, list):
            item["content"] = [_redact_content_part(part) if isinstance(part, dict) else part for part in content]
        else:
            item["content"] = content
        if "tool_calls" in message:
            item["tool_calls"] = message["tool_calls"]
        redacted.append(item)
    return redacted


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        if _SECRET_KEY_PATTERN.search(str(key)):
            cleaned[key] = "<redacted>"
        else:
            cleaned[key] = to_jsonable(value)
    return cleaned


def create_run_log_session(
    *, call_log_dir: str | None, settings_snapshot: dict[str, Any] | None = None
) -> RunLogSession:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = resolve_log_root(call_log_dir) / resolve_run_bucket() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = to_jsonable(settings_snapshot or {})
    run_json = {
        "timestamp": timestamp,
        "run_id": resolve_run_bucket(),
        "log_root": str(resolve_log_root(call_log_dir)),
        "settings": snapshot,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return RunLogSession(run_dir=run_dir, calls_path=run_dir / "calls.jsonl", timestamp=timestamp)


def append_call_record(record: dict[str, Any], *, session: RunLogSession | None) -> None:
    if session is None:
        return
    line = json.dumps(to_jsonable(record), ensure_ascii=False) + "\n"
    with _SESSION_LOCK, session.calls_path.open("a", encoding="utf-8") as handle:
        handle.write(line)

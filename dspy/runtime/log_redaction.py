from __future__ import annotations

import hashlib
import re
from typing import Any

from dspy.serialization.json import to_jsonable

_SECRET_KEY_PATTERN = re.compile("(api[_-]?key|authorization|token|secret)", re.IGNORECASE)
_MESSAGE_PASSTHROUGH_KEYS = ("name", "tool_call_id")


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
        for key in _MESSAGE_PASSTHROUGH_KEYS:
            if key in message:
                item[key] = message[key]
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

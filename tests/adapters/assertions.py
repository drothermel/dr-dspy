from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any


def assert_messages_exact(*, messages: list[dict], expected: list[dict]) -> None:
    assert messages == expected


def assert_message_roles(*, messages: list[dict], roles: Iterable[str]) -> None:
    assert [message["role"] for message in messages] == list(roles)


def normalize_citations_schema_description(content: str) -> str:
    start = content.find('{"type": "object", "$defs":')
    if start == -1:
        return content
    prefix = content[:start]
    schema_part = content[start:]
    schema_part = re.sub(r'"description": "(?:[^"\\]|\\.)*"(?:, )?', "", schema_part)
    return prefix + schema_part


def normalize_citations_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            normalized.append(
                {
                    **message,
                    "content": normalize_citations_schema_description(message["content"]),
                }
            )
        else:
            normalized.append(message)
    return normalized


def content_blocks(message: dict) -> list[dict]:
    content = message.get("content")
    if isinstance(content, list):
        return content
    return []


def assert_multimodal_blocks(*, message: dict, expected_blocks: Iterable[dict[str, Any]]) -> None:
    blocks = content_blocks(message)
    for expected_block in expected_blocks:
        assert expected_block in blocks


def assert_content_contains(*, content: str, fragments: Iterable[str]) -> None:
    for fragment in fragments:
        assert fragment in content


NormalizeMessages = Callable[[list[dict]], list[dict]]

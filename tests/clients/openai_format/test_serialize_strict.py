from __future__ import annotations

import pytest

from dspy.clients.openai_format.serialize import part_to_openai_blocks
from dspy.core.types import LMToolCallPart
from dspy.errors import LMUnsupportedFeatureError


def test_part_to_openai_blocks_rejects_unknown_part_type() -> None:
    with pytest.raises(LMUnsupportedFeatureError, match="does not support message part type"):
        part_to_openai_blocks(object())


def test_part_to_openai_blocks_rejects_tool_call_part() -> None:
    part = LMToolCallPart(id="call-1", name="search", args={"q": "dspy"})
    with pytest.raises(LMUnsupportedFeatureError, match="message layer"):
        part_to_openai_blocks(part)

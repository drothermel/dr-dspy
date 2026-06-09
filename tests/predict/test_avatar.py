import asyncio
from typing import Any, cast

import pytest

from dspy.adapters.types.tool import Tool
from dspy.predict.agent_termination import AgentTerminationReason
from dspy.predict.avatar import Avatar
from dspy.predict.avatar.models import ActionOutput
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def test_avatar_requires_tool_instances():
    def lookup(query: str) -> str:
        return query

    with pytest.raises(TypeError, match="tools must be Tool instances"):
        Avatar(ts("question -> answer"), tools=cast("Any", [lookup]))


def test_avatar_invokes_tool_via_acall(make_run):
    calls: list[dict[str, Any]] = []

    def lookup(query: str) -> str:
        calls.append({"query": query})
        return f"found:{query}"

    lm = DummyLM(
        [
            {"action": {"tool_name": "lookup", "tool_args": {"query": "cats"}}},
            {"action": {"tool_name": "Finish", "tool_args": {}}},
            {"answer": "cats are great"},
        ]
    )
    run = make_run(lm=lm)
    avatar = Avatar(
        ts("question -> answer"),
        tools=[Tool(lookup, description="Look up information.")],
        max_iters=3,
    )
    result = asyncio.run(avatar(question="Tell me about cats", run=run))

    assert calls == [{"query": "cats"}]
    assert result.answer == "cats are great"
    assert len(result.actions) == 1
    action = result.actions[0]
    assert isinstance(action, ActionOutput)
    assert action.tool_name == "lookup"
    assert action.tool_args == {"query": "cats"}
    assert action.tool_output == "found:cats"


def test_avatar_finish_skips_tool_execution(make_run):
    lm = DummyLM(
        [
            {"action": {"tool_name": "Finish", "tool_args": {}}},
            {"answer": "done without tools"},
        ]
    )
    run = make_run(lm=lm)
    avatar = Avatar(ts("question -> answer"), tools=[], max_iters=3)
    result = asyncio.run(avatar(question="hi", run=run))

    assert result.answer == "done without tools"
    assert result.actions == []
    assert result.termination_reason == AgentTerminationReason.SUBMIT

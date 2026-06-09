import asyncio
import re
from typing import Any, cast

import pytest
from pydantic import BaseModel
from typing_extensions import override

import dspy.adapters.base as adapter_base
import dspy.adapters.utils as adapter_utils
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.errors import ContextWindowExceededError
from dspy.history import TurnEvent
from dspy.predict.react import ReAct
from dspy.primitives.prediction import Prediction
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def _turn_dict(turn: TurnEvent) -> dict:
    return turn.model_dump(mode="json", exclude_none=True)


def _turns_from_flat(flat: dict) -> tuple:
    turns = []
    i = 0
    while f"thought_{i}" in flat:
        turns.append(
            {
                "thought": flat[f"thought_{i}"],
                "tool_name": flat[f"tool_name_{i}"],
                "tool_args": flat[f"tool_args_{i}"],
                "observation": flat[f"observation_{i}"],
            }
        )
        i += 1
    return tuple(turns)


@pytest.mark.extra
def test_react_requires_tool_instances():

    def search(query: str) -> str:
        return query

    with pytest.raises(TypeError, match="tools must be Tool instances"):
        ReAct(ts("question -> answer"), tools=cast("Any", [search]))


def test_tool_observation_preserves_custom_type(make_run):
    pytest.importorskip("PIL.Image")
    from PIL import Image as PILImage

    from dspy.adapters.types.image import Image

    captured_calls = []

    class SpyChatAdapter(ChatAdapter):
        @override
        def format_user_message_content(
            self, task_spec, inputs, prefix: str = "", suffix: str = "", main_request: bool = False
        ) -> str | list[dict[str, Any]]:
            captured_calls.append((task_spec, dict(inputs)))
            return super().format_user_message_content(
                task_spec, inputs, prefix=prefix, suffix=suffix, main_request=main_request
            )

    def make_images():
        return (Image("https://example.com/test.png"), Image(PILImage.new("RGB", (100, 100), "red")))

    adapter = SpyChatAdapter()
    lm = DummyLM(
        [
            {"next_thought": "I should call the image tool.", "next_tool_name": "make_images", "next_tool_args": {}},
            {"next_thought": "I now have the image so I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "image ready", "answer": "done"},
        ],
        adapter=adapter,
    )
    run = make_run(lm=lm, adapter=adapter)
    react = ReAct(ts("question -> answer"), tools=[Tool(make_images, description="Create images.")])
    pred = asyncio.run(react(question="Draw me something red", run=run))
    observation = pred.turn_log.turns[0].observation
    assert isinstance(observation, tuple)
    assert len(observation) == 2
    assert all(hasattr(item, "url") or hasattr(item, "data") for item in observation)


def test_tool_calling_with_pydantic_args(make_run):

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    InvitationSignature = make_task_spec(
        {
            "participant_name": input_field("participant_name", desc="The name of the participant to invite"),
            "event_info": input_field("event_info", type_=CalendarEvent, desc="The information about the event"),
            "invitation_letter": output_field(
                "invitation_letter", desc="The invitation letter to be sent to the participant"
            ),
        },
        instructions="Write invitation letters.",
        name="InvitationSignature",
    )
    react = ReAct(
        InvitationSignature,
        tools=[Tool(write_invitation_letter, description="Write an invitation letter for a participant.")],
    )
    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    run = make_run(lm=lm)
    outputs = asyncio.run(
        react(
            participant_name="Alice",
            event_info=CalendarEvent(
                name="Science Fair", date="Friday", participants={"Alice": "female", "Bob": "male"}
            ),
            run=run,
        )
    )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."
    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert tuple(_turn_dict(t) for t in outputs.turn_log.turns) == _turns_from_flat(expected_trajectory)


def test_react_with_tools_skips_native_response_issubclass_for_generic_alias(monkeypatch, make_run):

    def get_user_info(name: str):
        return {"name": name}

    CustomerService = make_task_spec(
        {
            "user_request": input_field("user_request", desc="The user request."),
            "process_result": output_field("process_result", desc="The process result."),
        },
        instructions="Handle customer service requests.",
        name="CustomerService",
    )
    react = ReAct(CustomerService, tools=[Tool(get_user_info, description="Get user information by name.")])
    problem_annotation = react.react.task_spec.output_fields["next_tool_args"].type_

    def guarded_issubclass(cls, class_or_tuple):
        if cls == problem_annotation:
            raise TypeError("issubclass() arg 1 must be a class")
        return issubclass(cls, class_or_tuple)

    monkeypatch.setattr(adapter_base, "issubclass", guarded_issubclass, raising=False)
    monkeypatch.setattr(adapter_utils, "issubclass", guarded_issubclass, raising=False)
    lm = DummyLM(
        [
            {
                "next_thought": "I should look up the user first.",
                "next_tool_name": "get_user_info",
                "next_tool_args": {"name": "Adam"},
            },
            {
                "next_thought": "I have the information I need, so I can finish now.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "I fetched the user profile and can answer the request.",
                "process_result": "Resolved Adam's request.",
            },
        ]
    )
    run = make_run(lm=lm)
    result = asyncio.run(react(user_request="Help me, my name is Adam", run=run))
    assert result.process_result == "Resolved Adam's request."
    assert result.turn_log.turns[0].tool_name == "get_user_info"
    assert result.turn_log.turns[0].tool_args == {"name": "Adam"}


def test_tool_calling_without_typehint(make_run):

    def foo(a, b):
        return a + b

    react = ReAct(ts("a, b -> c:int"), tools=[Tool(foo, description="Combine inputs.")])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"next_thought": "I have the sum, now I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    run = make_run(lm=lm)
    outputs = asyncio.run(react(a=1, b=2, run=run))
    expected_trajectory = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "observation_0": 3,
        "thought_1": "I have the sum, now I can finish.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert tuple(_turn_dict(t) for t in outputs.turn_log.turns) == _turns_from_flat(expected_trajectory)


def test_trajectory_truncation(make_run):

    run = make_run(lm=DummyLM([{}]))

    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = ReAct(ts("input_text -> output_text"), tools=[Tool(echo, description="Echo input text.")])
    call_count = 0

    async def mock_react(**kwargs: object):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return Prediction.from_record(
                {
                    "next_thought": f"Thought {call_count}",
                    "next_tool_name": "echo",
                    "next_tool_args": {"text": f"Text {call_count}"},
                }
            )
        if call_count == 3:
            raise ContextWindowExceededError
        return Prediction.from_record(
            {"next_thought": "Final thought", "next_tool_name": "finish", "next_tool_args": {}}
        )

    cast("Any", react).react = mock_react

    async def mock_extract(**kwargs: object):
        return Prediction.from_record({"output_text": "Final output"})

    cast("Any", react).extract = mock_extract
    result = asyncio.run(react(input_text="test input", run=run))
    assert result.output_text == "Final output"
    assert len(result.turn_log.turns) >= 1


@pytest.mark.asyncio
async def test_context_window_exceeded_after_retries(make_run):

    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = ReAct(ts("input_text -> output_text"), tools=[Tool(echo, description="Echo input text.")])

    async def mock_react(**kwargs: object):
        raise ContextWindowExceededError

    extract_calls = []

    async def mock_extract(**kwargs: object):
        extract_calls.append(kwargs)
        return Prediction.from_record({"output_text": "Fallback output"})

    cast("Any", react).react = mock_react
    cast("Any", react).extract = mock_extract
    run = make_run(lm=DummyLM([{}]))
    result = await react(input_text="test input", run=run)
    assert result.turn_log.turns == ()
    assert result.output_text == "Fallback output"
    assert len(extract_calls) == 1
    assert extract_calls[0]["input_text"] == "test input"
    assert "turn_log" in extract_calls[0]


def test_error_retry(make_run):

    def foo(a, b):
        raise Exception("tool error")

    react = ReAct(ts("a, b -> c:int"), tools=[Tool(foo, description="Combine inputs.")])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    run = make_run(lm=lm)
    outputs = asyncio.run(react(a=1, b=2, max_iters=2, run=run))
    turns = outputs.turn_log.turns
    control_expected = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "thought_1": "I need to add two numbers.",
        "tool_name_1": "foo",
        "tool_args_1": {"a": 1, "b": 2},
    }
    assert turns[0].thought == control_expected["thought_0"]
    assert turns[0].tool_name == control_expected["tool_name_0"]
    assert turns[0].tool_args == control_expected["tool_args_0"]
    assert turns[1].thought == control_expected["thought_1"]
    assert turns[1].tool_name == control_expected["tool_name_1"]
    assert turns[1].tool_args == control_expected["tool_args_1"]
    for i in range(2):
        obs = turns[i].observation
        assert re.search("\\btool error\\b", obs), f"unexpected observation_{i!r}: {obs}"


@pytest.mark.asyncio
async def test_async_tool_calling_with_pydantic_args(make_run):

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    async def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    InvitationSignature = make_task_spec(
        {
            "participant_name": input_field("participant_name", desc="The name of the participant to invite"),
            "event_info": input_field("event_info", type_=CalendarEvent, desc="The information about the event"),
            "invitation_letter": output_field(
                "invitation_letter", desc="The invitation letter to be sent to the participant"
            ),
        },
        instructions="Write invitation letters.",
        name="InvitationSignature",
    )
    react = ReAct(
        InvitationSignature,
        tools=[Tool(write_invitation_letter, description="Write an invitation letter for a participant.")],
    )
    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    run = make_run(lm=lm)
    outputs = await react(
        participant_name="Alice",
        event_info=CalendarEvent(name="Science Fair", date="Friday", participants={"Alice": "female", "Bob": "male"}),
        run=run,
    )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."
    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert tuple(_turn_dict(t) for t in outputs.turn_log.turns) == _turns_from_flat(expected_trajectory)


@pytest.mark.asyncio
async def test_async_error_retry(make_run):

    async def foo(a, b):
        raise Exception("tool error")

    react = ReAct(ts("a, b -> c:int"), tools=[Tool(foo, description="Combine inputs.")])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    run = make_run(lm=lm)
    outputs = await react(a=1, b=2, max_iters=2, run=run)
    turns = outputs.turn_log.turns
    control_expected = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "thought_1": "I need to add two numbers.",
        "tool_name_1": "foo",
        "tool_args_1": {"a": 1, "b": 2},
    }
    assert turns[0].thought == control_expected["thought_0"]
    assert turns[0].tool_name == control_expected["tool_name_0"]
    assert turns[0].tool_args == control_expected["tool_args_0"]
    assert turns[1].thought == control_expected["thought_1"]
    assert turns[1].tool_name == control_expected["tool_name_1"]
    assert turns[1].tool_args == control_expected["tool_args_1"]
    for i in range(2):
        obs = turns[i].observation
        assert re.search("\\btool error\\b", obs), f"unexpected observation_{i!r}: {obs}"

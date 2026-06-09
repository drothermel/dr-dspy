from __future__ import annotations

from typing import Literal

from dspy.adapters.types.audio import Audio
from dspy.adapters.types.document import Document
from dspy.adapters.types.file import File
from dspy.adapters.types.image import Image
from dspy.adapters.types.tool import Tool
from dspy.history import TurnLog
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.adapters.scenarios.chat_cases import Event
from tests.adapters.scenarios.pydantic_models import AnswerCard, Location, Profile
from tests.adapters.scenarios.tools import search_tool
from tests.history.turn_fixtures import task_io_turn


def kitchen_sink_case() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "image": input_field("image", type_=Image, desc="The image."),
            "audio": input_field("audio", type_=Audio, desc="The audio."),
            "file": input_field("file", type_=File, desc="The file."),
            "document": input_field("document", type_=Document, desc="The document."),
            "event": input_field("event", type_=Event, desc="The event."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "profile": input_field("profile", type_=Profile, desc="The profile."),
            "context": input_field("context", desc="The context."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", type_=AnswerCard, desc="The answer."),
            "verdict": output_field("verdict", type_=Literal["yes", "no"], desc="The verdict."),
            "confidence": output_field("confidence", type_=float, desc="The confidence."),
        },
        instructions="Answer carefully using every available signal.",
    )
    tool = search_tool()
    demo_profile = Profile(name="Ada", location=Location(city="London", country="UK"), interests=["math", "machines"])
    current_profile = Profile(
        name="Grace", location=Location(city="Arlington", country="USA"), interests=["compilers", "navy"]
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                task_io_turn(
                    profile=demo_profile,
                    context=["old note", "older note"],
                    question="Who is Ada?",
                    answer=AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
                    verdict="yes",
                    confidence=0.8,
                )
            ],
        }
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(
            {
                "image": Image("https://example.com/demo.png"),
                "audio": Audio(data="REVNTw==", audio_format="wav"),
                "file": File.from_file_id("file-demo", filename="demo.txt"),
                "document": Document(data="Demo document", title="Demo Doc"),
                "event": Event(label="demo-event"),
                "tools": [tool],
                "profile": demo_profile,
                "context": ["demo context one", "demo context two"],
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
                "verdict": "yes",
                "confidence": 0.9,
            },
            {
                "question": "Incomplete example question",
                "answer": AnswerCard(answer="Partial answer.", sources=["partial"]),
            },
        ),
        inputs={
            "turn_log": history,
            "image": Image("https://example.com/current.png"),
            "audio": Audio(data="Q1VSUkVOVA==", audio_format="wav"),
            "file": File.from_file_id("file-current", filename="current.txt"),
            "document": Document(data="Current document", title="Current Doc"),
            "event": Event(label="current-event"),
            "tools": [tool],
            "profile": current_profile,
            "context": ["current context one", "current context two"],
            "question": "What should the answer include?",
        },
    )

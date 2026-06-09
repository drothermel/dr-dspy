from dspy.adapters.types.image import Image
from dspy.history import TurnLog
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.adapters.scenarios.pydantic_models import AnswerCard, Location, Profile
from tests.adapters.scenarios.tools import search_tool
from tests.history.turn_fixtures import task_io_turn


def rich_rendering_history() -> TurnLog:
    demo_profile = Profile(name="Ada", location=Location(city="London", country="UK"), interests=["math", "machines"])
    return TurnLog.model_validate(
        {
            "turns": [
                task_io_turn(
                    profile=demo_profile,
                    question="Who is Ada?",
                    answer=AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
                )
            ],
        }
    )


def rich_rendering_case() -> FormatScenarioCase:
    from dspy.adapters.types.tool import Tool
    from dspy.history import TurnLog

    tool = search_tool()
    demo_profile = Profile(name="Ada", location=Location(city="London", country="UK"), interests=["math", "machines"])
    current_profile = Profile(
        name="Grace", location=Location(city="Arlington", country="USA"), interests=["compilers", "navy"]
    )
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "image": input_field("image", type_=Image, desc="The image."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "profile": input_field("profile", type_=Profile, desc="The profile."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", type_=AnswerCard, desc="The answer."),
        },
        instructions="Answer using all supplied context.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(
            {
                "image": Image("https://example.com/demo.png"),
                "tools": [tool],
                "profile": demo_profile,
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
            },
        ),
        inputs={
            "turn_log": rich_rendering_history(),
            "image": Image("https://example.com/current.png"),
            "tools": [tool],
            "profile": current_profile,
            "question": "What should the answer include?",
        },
    )

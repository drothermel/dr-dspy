import pydantic

from dspy.adapters.types.base_type import Type as DSPyType
from dspy.task_spec import FieldSpec, make_task_spec


def test_basic_extract_custom_type_from_annotation():
    class Event(DSPyType):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    extract_event = make_task_spec(
        {"email": FieldSpec.input("email"), "event": FieldSpec.output("event", type_=Event)},
        instructions="Extract all events from the email content.",
    )

    assert DSPyType.extract_custom_type_from_annotation(extract_event.output_fields["event"].type_) == [Event]

    extract_events = make_task_spec(
        {"email": FieldSpec.input("email"), "events": FieldSpec.output("events", type_=list[Event])},
        instructions="Extract all events from the email content.",
    )

    assert DSPyType.extract_custom_type_from_annotation(extract_events.output_fields["events"].type_) == [Event]


def test_extract_custom_type_from_annotation_with_nested_type():
    class Event(DSPyType):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    class EventIdentifier(DSPyType):
        model_config = pydantic.ConfigDict(frozen=True)  # Make it hashable
        event_id: str
        event_name: str

    extract_events = make_task_spec(
        {
            "email": FieldSpec.input("email"),
            "events": FieldSpec.output("events", type_=list[dict[EventIdentifier, Event]]),
        },
        instructions="Extract all events from the email content.",
    )

    assert DSPyType.extract_custom_type_from_annotation(extract_events.output_fields["events"].type_) == [
        EventIdentifier,
        Event,
    ]

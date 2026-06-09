import pydantic

from dspy.adapters.types.base_type import Type as DSPyType
from dspy.task_spec import input_field, make_task_spec, output_field


def test_basic_extract_custom_type_from_annotation():

    class Event(DSPyType):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    extract_event = make_task_spec(
        {
            "email": input_field("email", desc="The email."),
            "event": output_field("event", type_=Event, desc="The event."),
        },
        instructions="Extract all events from the email content.",
    )
    assert DSPyType.extract_custom_type_from_annotation(extract_event.output_fields["event"].type_) == [Event]
    extract_events = make_task_spec(
        {
            "email": input_field("email", desc="The email."),
            "events": output_field("events", type_=list[Event], desc="The events."),
        },
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
        model_config = pydantic.ConfigDict(frozen=True)
        event_id: str
        event_name: str

    extract_events = make_task_spec(
        {
            "email": input_field("email", desc="The email."),
            "events": output_field("events", type_=list[dict[EventIdentifier, Event]], desc="The events."),
        },
        instructions="Extract all events from the email content.",
    )
    assert DSPyType.extract_custom_type_from_annotation(extract_events.output_fields["events"].type_) == [
        EventIdentifier,
        Event,
    ]

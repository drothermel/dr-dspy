import pydantic

from dspy.adapters.types.field_type import FieldTypeMixin, extract_field_types_from_annotation, is_field_type
from dspy.task_spec import input_field, make_task_spec, output_field


def test_is_field_type_rejects_strings_with_format_method():
    assert not is_field_type("plain text")
    assert not is_field_type('[{"type": "file", "file": {"file_id": "file-test"}}]')


def test_basic_extract_field_types_from_annotation():

    class Event(FieldTypeMixin):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

        def format(self) -> str:
            return self.event_name

    extract_event = make_task_spec(
        {
            "email": input_field("email", desc="The email."),
            "event": output_field("event", type_=Event, desc="The event."),
        },
        instructions="Extract all events from the email content.",
    )
    assert extract_field_types_from_annotation(extract_event.output_fields["event"].type_) == [Event]
    extract_events = make_task_spec(
        {
            "email": input_field("email", desc="The email."),
            "events": output_field("events", type_=list[Event], desc="The events."),
        },
        instructions="Extract all events from the email content.",
    )
    assert extract_field_types_from_annotation(extract_events.output_fields["events"].type_) == [Event]


def test_extract_field_types_from_annotation_with_nested_type():

    class Event(FieldTypeMixin):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

        def format(self) -> str:
            return self.event_name

    class EventIdentifier(FieldTypeMixin):
        model_config = pydantic.ConfigDict(frozen=True)
        event_id: str
        event_name: str

        def format(self) -> str:
            return self.event_id

    extract_events = make_task_spec(
        {
            "email": input_field("email", desc="The email."),
            "events": output_field("events", type_=list[dict[EventIdentifier, Event]], desc="The events."),
        },
        instructions="Extract all events from the email content.",
    )
    assert extract_field_types_from_annotation(extract_events.output_fields["events"].type_) == [
        EventIdentifier,
        Event,
    ]

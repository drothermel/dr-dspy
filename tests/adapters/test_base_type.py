import pydantic

from dspy.adapters.types.base_type import Type as DSPyType
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature


def test_basic_extract_custom_type_from_annotation():
    class Event(DSPyType):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    class ExtractEvent(Signature):
        """Extract all events from the email content."""

        email: str = InputField()
        event: Event = OutputField()

    assert DSPyType.extract_custom_type_from_annotation(ExtractEvent.output_fields["event"].annotation) == [Event]

    class ExtractEvents(Signature):
        """Extract all events from the email content."""

        email: str = InputField()
        events: list[Event] = OutputField()

    assert DSPyType.extract_custom_type_from_annotation(ExtractEvents.output_fields["events"].annotation) == [Event]


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

    class ExtractEvents(Signature):
        """Extract all events from the email content."""

        email: str = InputField()
        events: list[dict[EventIdentifier, Event]] = OutputField()

    assert DSPyType.extract_custom_type_from_annotation(ExtractEvents.output_fields["events"].annotation) == [
        EventIdentifier,
        Event,
    ]

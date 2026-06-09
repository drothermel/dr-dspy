from dspy.core.types import CallRecord, LMMessage, LMRequest, LMResponse


def history_entry(message: LMMessage) -> CallRecord:
    return CallRecord(
        request=LMRequest(model="model", messages=[message]),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

from dspy.core.types import LMHistoryEntry, LMMessage, LMRequest, LMResponse


def history_entry(message: LMMessage) -> LMHistoryEntry:
    return LMHistoryEntry(
        request=LMRequest(model="model", messages=[message]),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )

from typing import Any

from dspy.clients.openai_format.chat_request import request_messages_as_openai
from dspy.core.types import CallRecord, LMMessage, LMRequest, LMResponse


def history_entry(message: LMMessage) -> CallRecord:
    return CallRecord(
        request=LMRequest(model="model", messages=[message]),
        response=LMResponse.from_text("ok"),
        timestamp="timestamp",
        uuid="uuid",
    )


def history_messages_as_openai(message: LMMessage) -> list[dict[str, Any]]:
    return request_messages_as_openai(history_entry(message).request)

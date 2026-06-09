from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING, Any

from dspy.clients.openai_format.chat_request import request_messages_as_openai
from dspy.runtime.log_redaction import redact_config, redact_messages

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import CallRecord, LMRequest, LMResponse
    from dspy.runtime.transparency.types import CompiledCall


def build_disk_call_record(
    *,
    request: LMRequest,
    response: LMResponse,
    call_record: CallRecord | None,
    lm: BaseLM,
    compiled: CompiledCall | None = None,
) -> dict[str, Any]:
    call_id = compiled.call_id if compiled is not None else call_record.uuid if call_record else str(uuid.uuid4())
    timestamp = call_record.timestamp if call_record is not None else None
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    messages = request_messages_as_openai(request)
    outputs = [
        {
            "text": output.text,
            "tool_calls": [
                {"name": call.name, "args": dict(call.args), "id": call.id} for call in output.tool_calls or []
            ],
            "logprobs": output.logprobs,
        }
        for output in response.outputs
    ]
    return {
        "call_id": call_id,
        "timestamp": timestamp,
        "caller": {
            "module": compiled.module if compiled else "unknown",
            "phase": compiled.phase if compiled else "unknown",
            "lm_role": compiled.lm_role if compiled else "unknown",
        },
        "lm": {"model": lm.model, "model_type": getattr(lm, "model_type", None)},
        "adapter": {
            "class": compiled.adapter_class if compiled else None,
            "notes": compiled.adapter_notes if compiled else [],
        },
        "task_spec": compiled.original_task_spec.to_dict() if compiled and compiled.original_task_spec else None,
        "processed_task_spec": compiled.processed_task_spec.to_dict()
        if compiled and compiled.processed_task_spec
        else None,
        "task_spec_mutations": compiled.task_spec_mutations if compiled else [],
        "messages": redact_messages(messages),
        "config": redact_config(request.config.model_dump(exclude_none=True)),
        "config_provenance": compiled.config_provenance if compiled else {},
        "response": {
            "outputs": outputs,
            "usage": response.usage_as_dict(),
        },
    }

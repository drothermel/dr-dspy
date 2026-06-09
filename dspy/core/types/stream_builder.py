from __future__ import annotations

from typing import Any

from dspy.core.types.lm_output import LMOutput
from dspy.core.types.lm_response import LMResponse
from dspy.core.types.parts import (
    LMAudioPart,
    LMCitationPart,
    LMImagePart,
    LMPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
)
from dspy.core.types.parts.serialize import _finalize_stream_part, _parse_json_object
from dspy.core.types.stream_events import (
    LMAudioDelta,
    LMCitationDelta,
    LMImageDelta,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamErrorEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMToolCallDelta,
)
from dspy.core.types.usage import LMUsage


class LMOutputBuilder:
    def __init__(self) -> None:
        self.model: str | None = None
        self._parts: dict[int, list[LMPart | None]] = {}
        self._finish_reasons: dict[int, str | None] = {}
        self._truncated: dict[int, bool] = {}

    def apply(self, event: LMStreamEvent) -> LMResponse | None:
        if isinstance(event, LMStreamStartEvent):
            self.model = event.model
            return None
        if isinstance(event, LMStreamDeltaEvent):
            self._apply_delta(event)
            return None
        if isinstance(event, LMStreamOutputEndEvent):
            self._finish_reasons[event.output_index] = event.finish_reason
            self._truncated[event.output_index] = event.truncated
            return None
        if isinstance(event, LMStreamEndEvent):
            if event.response is not None:
                return event.response
            return self.to_response(usage=event.usage, cost=event.cost)
        if isinstance(event, LMStreamErrorEvent):
            raise event.error
        return None

    def to_response(self, *, usage: LMUsage | dict[str, Any] | None = None, cost: float | None = None) -> LMResponse:
        output_indices = set(self._parts) | set(self._finish_reasons) | set(self._truncated)
        if not output_indices:
            output_indices = {0}
        max_index = max(output_indices)
        expected_indices = set(range(max_index + 1))
        if output_indices != expected_indices:
            missing = sorted(expected_indices - output_indices)
            raise ValueError(f"Stream output indices must be contiguous from 0; missing indices: {missing}.")
        outputs = []
        for output_index in range(max_index + 1):
            part_buffer = self._parts.get(output_index, [])
            missing_part_indices = [index for index, part in enumerate(part_buffer) if part is None]
            if missing_part_indices:
                raise ValueError(
                    f"Stream part indices for output {output_index} must be contiguous; missing indices: {missing_part_indices}."
                )
            parts = [_finalize_stream_part(part) for part in part_buffer if part is not None]
            outputs.append(
                LMOutput(
                    parts=parts,
                    finish_reason=self._finish_reasons.get(output_index),
                    truncated=self._truncated.get(output_index, False),
                )
            )
        return LMResponse(model=self.model, outputs=outputs, usage=usage, cost=cost)

    def _apply_delta(self, event: LMStreamDeltaEvent) -> None:
        parts = self._parts.setdefault(event.output_index, [])
        while len(parts) <= event.part_index:
            parts.append(None)
        current = parts[event.part_index]
        delta = event.delta
        if isinstance(delta, LMThinkingDelta):
            if current is not None and (not isinstance(current, LMThinkingPart)):
                raise ValueError("Cannot apply thinking delta to a non-thinking stream part.")
            text = (current.text if isinstance(current, LMThinkingPart) else "") + delta.text
            parts[event.part_index] = LMThinkingPart(text=text)
        elif isinstance(delta, LMTextDelta):
            if current is not None and (not isinstance(current, LMTextPart)):
                raise ValueError("Cannot apply text delta to a non-text stream part.")
            text = (current.text if isinstance(current, LMTextPart) else "") + delta.text
            parts[event.part_index] = LMTextPart(text=text)
        elif isinstance(delta, LMToolCallDelta):
            if current is not None and (not isinstance(current, LMToolCallPart)):
                raise ValueError("Cannot apply tool-call delta to a non-tool-call stream part.")
            buffer = ""
            if isinstance(current, LMToolCallPart):
                buffer = current.provider_data.get("args_buffer", "")
            buffer += delta.args_delta or ""
            args = _parse_json_object(buffer)
            parts[event.part_index] = LMToolCallPart(
                id=delta.id if delta.id is not None else getattr(current, "id", None),
                name=delta.name if delta.name is not None else getattr(current, "name", ""),
                args=args,
                provider_data={"args_buffer": buffer},
            )
        elif isinstance(delta, LMCitationDelta):
            if current is not None and (not isinstance(current, LMCitationPart)):
                raise ValueError("Cannot apply citation delta to a different stream part type.")
            parts[event.part_index] = delta.citation
        elif isinstance(delta, LMImageDelta):
            if current is not None and (not isinstance(current, LMImagePart)):
                raise ValueError("Cannot apply image delta to a different stream part type.")
            parts[event.part_index] = delta.image
        elif isinstance(delta, LMAudioDelta):
            if current is not None and (not isinstance(current, LMAudioPart)):
                raise ValueError("Cannot apply audio delta to a different stream part type.")
            parts[event.part_index] = delta.audio

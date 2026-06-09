from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator

from dspy.core.types.lm_response import LMResponse
from dspy.core.types.request import LMRequest
from dspy.core.types.stream_builder import LMOutputBuilder
from dspy.core.types.stream_events import LMStreamEvent


class LMStream:
    def __init__(
        self,
        *,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        finalize: Callable[[LMRequest, LMResponse], LMResponse],
    ) -> None:
        self.request = request
        self._events = events
        self._finalize = finalize
        self._builder = LMOutputBuilder()
        self._result: LMResponse | None = None

    def __iter__(self) -> Iterator[LMStreamEvent]:
        for event in self._events:
            response = self._builder.apply(event)
            if response is not None:
                self._result = self._finalize(self.request, response)
            yield event

    def result(self) -> LMResponse:
        if self._result is None:
            raise RuntimeError("Stream has not completed yet.")
        return self._result


class AsyncLMStream:
    def __init__(
        self,
        *,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        finalize: Callable[[LMRequest, LMResponse], LMResponse],
    ) -> None:
        self.request = request
        self._events = events
        self._finalize = finalize
        self._builder = LMOutputBuilder()
        self._result: LMResponse | None = None

    async def __aiter__(self) -> AsyncIterator[LMStreamEvent]:
        async for event in self._events:
            response = self._builder.apply(event)
            if response is not None:
                self._result = self._finalize(self.request, response)
            yield event

    def result(self) -> LMResponse:
        if self._result is None:
            raise RuntimeError("Stream has not completed yet.")
        return self._result

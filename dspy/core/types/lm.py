from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dspy.core.types.request import LMRequest
    from dspy.core.types.response import LMResponse


class LMForward(Protocol):
    async def aforward(self, request: LMRequest) -> LMResponse: ...

    async def __call__(self, request: LMRequest) -> LMResponse: ...

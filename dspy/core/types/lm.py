from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dspy.core.types.request import LMRequest
    from dspy.core.types.response import LMResponse


@runtime_checkable
class LMForward(Protocol):
    """Async LM protocol for per-call overrides via ``dspy.predict.call_options.PredictOptions(lm=...)``.

    Implementations must provide ``aforward(request) -> LMResponse``.
    ``BaseLM.__call__(request, run=..., compiled=...)`` is the runtime entry
    point with logging and callbacks; it is not part of this protocol.
    """

    async def aforward(self, request: LMRequest) -> LMResponse: ...

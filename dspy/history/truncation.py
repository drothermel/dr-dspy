from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel, ConfigDict

from dspy.errors import ContextWindowExceededError
from dspy.history.protocol import H, HistoryModule

if TYPE_CHECKING:
    from dspy.runtime.call_options import ModuleCallOptions
    from dspy.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


class TruncationExhaustedError(ValueError):
    """Raised when context-window truncation retries are exhausted."""


class HistoryCallResult(BaseModel, Generic[H]):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    result: Any
    turn_log: H


async def call_with_history_truncation(
    module: HistoryModule[H],
    *,
    turn_log: H,
    run: RunContext,
    options: ModuleCallOptions | None = None,
    max_attempts: int = 3,
    **input_args: Any,
) -> HistoryCallResult[H]:
    for _ in range(max_attempts):
        try:
            result = await module(**input_args, turn_log=turn_log, run=run, options=options)
            return HistoryCallResult(result=result, turn_log=turn_log)
        except ContextWindowExceededError:
            logger.warning("History exceeded the context window, truncating the oldest entry.")
            turn_log = turn_log.truncate_oldest()
    raise TruncationExhaustedError(
        f"The context window was exceeded even after {max_attempts} attempts to truncate the history."
    )

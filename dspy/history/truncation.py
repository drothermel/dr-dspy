from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from dspy.errors import ContextWindowExceededError
from dspy.history.repl_history import REPLHistory  # noqa: TC001
from dspy.history.turn_log import TurnLog  # noqa: TC001

if TYPE_CHECKING:
    from dspy.core.types.call_options import ModuleCallOptions
    from dspy.history.protocol import REPLHistoryModule, TurnLogModule
    from dspy.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


class TruncationExhaustedError(ValueError):
    """Raised when context-window truncation retries are exhausted."""


class TurnLogCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    result: Any
    turn_log: TurnLog


class REPLHistoryCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    result: Any
    turn_log: REPLHistory


async def call_with_turn_log_truncation(
    module: TurnLogModule,
    *,
    turn_log: TurnLog,
    run: RunContext,
    options: ModuleCallOptions | None = None,
    max_attempts: int = 3,
    **input_args: Any,
) -> TurnLogCallResult:
    for _ in range(max_attempts):
        try:
            result = await module(**input_args, turn_log=turn_log, run=run, options=options)
            return TurnLogCallResult(result=result, turn_log=turn_log)
        except ContextWindowExceededError:
            logger.warning("Turn log exceeded the context window, truncating the oldest turn.")
            turn_log = turn_log.truncate_oldest()
    raise TruncationExhaustedError(
        f"The context window was exceeded even after {max_attempts} attempts to truncate the turn log."
    )


async def call_with_repl_history_truncation(
    module: REPLHistoryModule,
    *,
    turn_log: REPLHistory,
    run: RunContext,
    options: ModuleCallOptions | None = None,
    max_attempts: int = 3,
    **input_args: Any,
) -> REPLHistoryCallResult:
    for _ in range(max_attempts):
        try:
            result = await module(**input_args, turn_log=turn_log, run=run, options=options)
            return REPLHistoryCallResult(result=result, turn_log=turn_log)
        except ContextWindowExceededError:
            logger.warning("REPL history exceeded the context window, truncating the oldest entry.")
            turn_log = turn_log.truncate_oldest()
    raise TruncationExhaustedError(
        f"The context window was exceeded even after {max_attempts} attempts to truncate the REPL history."
    )

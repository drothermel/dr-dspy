from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dspy.errors import ContextWindowExceededError
from dspy.history.turn_log import TurnLog  # noqa: TC001

if TYPE_CHECKING:
    from dspy.core.types.call_options import ModuleCallOptions
    from dspy.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


async def call_with_turn_log_truncation(
    module: Any,
    *,
    turn_log: TurnLog,
    run: RunContext,
    options: ModuleCallOptions | None = None,
    max_attempts: int = 3,
    **input_args: Any,
) -> Any:
    for _ in range(max_attempts):
        try:
            return await module(**input_args, turn_log=turn_log, run=run, options=options)
        except ContextWindowExceededError:
            logger.warning("Turn log exceeded the context window, truncating the oldest turn.")
            turn_log = turn_log.truncate_oldest()
    raise ValueError("The context window was exceeded even after 3 attempts to truncate the turn log.")

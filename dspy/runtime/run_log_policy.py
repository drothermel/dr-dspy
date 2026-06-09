from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.runtime.run_log_session import ensure_log_session, init_log_session

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext


def apply_create_log_policy(run: RunContext) -> None:
    init_log_session(run)


def apply_fork_log_policy(forked: RunContext, _parent: RunContext, *, explicit_log_session: bool) -> None:
    ensure_log_session(forked, explicit_log_session=explicit_log_session)

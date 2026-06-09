from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


def rebuild_run_context_model(run_context_cls: type[BaseModel]) -> None:
    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.response import CallRecord
    from dspy.primitives.module import Module
    from dspy.runtime.callback import Callback
    from dspy.runtime.run_log import RunLogSession
    from dspy.runtime.usage_tracker import UsageTracker

    run_context_cls.model_rebuild(
        _types_namespace={
            "BaseLM": BaseLM,
            "Adapter": Adapter,
            "Callback": Callback,
            "CallRecord": CallRecord,
            "UsageTracker": UsageTracker,
            "Module": Module,
            "RunLogSession": RunLogSession,
        }
    )

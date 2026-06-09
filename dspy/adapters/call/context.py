from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types.config import LMConfig, LMToolSpec
    from dspy.task_spec import TaskSpec


@dataclass(frozen=True)
class CallContext:
    lm: BaseLM
    config: LMConfig
    task_spec: TaskSpec
    demos: list[dict[str, Any]]
    inputs: dict[str, Any]


@dataclass(frozen=True)
class ProcessedCall:
    processed_task_spec: TaskSpec
    tools: list[LMToolSpec]
    config: LMConfig
    mutations: list[str]

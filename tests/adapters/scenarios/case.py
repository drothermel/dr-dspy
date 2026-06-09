from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.clients.base_lm import BaseLM
    from dspy.task_spec import TaskSpec


@dataclass(frozen=True)
class FormatScenarioCase:
    task_spec: TaskSpec
    inputs: dict[str, Any]
    demos: tuple[dict[str, Any], ...] = ()
    lm: BaseLM | None = None
    config: Mapping[str, Any] | None = field(default=None)

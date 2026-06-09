from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dspy.adapters.base.protocols import ComposedAdapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMConfig, LMToolSpec
    from dspy.task_spec import TaskSpec


@dataclass
class PreprocessState:
    adapter: ComposedAdapter
    lm: BaseLM
    config: LMConfig | Mapping[str, Any]
    task_spec: TaskSpec
    inputs: dict[str, Any]
    tools: list[LMToolSpec] = field(default_factory=list)

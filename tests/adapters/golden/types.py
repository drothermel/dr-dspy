from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from dspy.adapters.base import Adapter
    from tests.adapters.assertions import NormalizeMessages
    from tests.adapters.scenarios.case import FormatScenarioCase


@dataclass(frozen=True)
class GoldenPromptCase:
    id: str
    adapter_builder: Callable[[], Adapter]
    scenario: FormatScenarioCase
    messages: list[dict[str, Any]]
    lm_kwargs: dict[str, Any] = field(default_factory=dict)
    normalize: NormalizeMessages | None = None

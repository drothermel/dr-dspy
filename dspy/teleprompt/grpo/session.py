from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from dspy.clients.finetune import GRPORolloutGroup  # noqa: TC001 — queue value type
from dspy.clients.lm import LM  # noqa: TC001 — job key tuple element


@dataclass
class GRPOCompileSession:
    """Per-compile mutable state for a single GRPO ``compile()`` invocation."""

    shuffled_trainset_ids: list[int] = field(default_factory=list)
    epoch: int = -1
    id_freqs: Counter[int] = field(default_factory=Counter)
    fulfilled_batch_ids: list[int] = field(default_factory=list)
    group_queues: dict[tuple[LM, Any], deque[GRPORolloutGroup]] = field(default_factory=dict)

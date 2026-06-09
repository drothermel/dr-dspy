from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.primitives import Example


def split_trainset_holdout(
    trainset: list[Example],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    if not trainset:
        raise ValueError("trainset cannot be empty")
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError(f"holdout_ratio must be in range (0, 1), got {holdout_ratio}")
    shuffled = trainset[:]
    random.Random(seed).shuffle(shuffled)
    num_holdout = int(holdout_ratio * len(shuffled))
    if num_holdout == 0:
        raise ValueError(
            f"holdout_ratio {holdout_ratio} yields zero holdout examples for trainset of size {len(shuffled)}"
        )
    return shuffled[num_holdout:], shuffled[:num_holdout]

from __future__ import annotations

import logging
import random  # noqa: TC003 — shuffle and sampling at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.primitives import Example
    from dspy.teleprompt.grpo.session import GRPOCompileSession

logger = logging.getLogger(__name__)


def update_shuffled_trainset(
    session: GRPOCompileSession,
    *,
    original_trainset: list[Example],
    num_dspy_examples_per_grpo_step: int,
    rng: random.Random,
) -> None:
    session.shuffled_trainset_ids = list(range(len(original_trainset)))
    rng.shuffle(session.shuffled_trainset_ids)
    for example_id in session.shuffled_trainset_ids:
        session.id_freqs[example_id] += 1
    num_to_pad = num_dspy_examples_per_grpo_step - len(original_trainset) % num_dspy_examples_per_grpo_step
    if num_to_pad > 0:
        for _ in range(num_to_pad):
            selected_id = session.id_freqs.most_common()[::-1][0][0]
            session.shuffled_trainset_ids.append(selected_id)
            session.id_freqs[selected_id] += 1


def select_training_sample(
    session: GRPOCompileSession,
    *,
    original_trainset: list[Example],
    train_step_idx: int,
    num_dspy_examples_per_grpo_step: int,
    rng: random.Random,
) -> list[Example]:
    base_idx = train_step_idx * num_dspy_examples_per_grpo_step
    curr_epoch = 0 if session.epoch == -1 else base_idx // len(session.shuffled_trainset_ids)
    if curr_epoch > session.epoch:
        logger.info(f"Updating shuffled trainset for epoch {curr_epoch}...")
        session.epoch = curr_epoch
        update_shuffled_trainset(
            session,
            original_trainset=original_trainset,
            num_dspy_examples_per_grpo_step=num_dspy_examples_per_grpo_step,
            rng=rng,
        )
    if len(session.shuffled_trainset_ids) < num_dspy_examples_per_grpo_step:
        raise ValueError(
            f"Shuffled trainset length {len(session.shuffled_trainset_ids)} is less than "
            f"num_dspy_examples_per_grpo_step {num_dspy_examples_per_grpo_step}"
        )
    if len(session.shuffled_trainset_ids) % num_dspy_examples_per_grpo_step != 0:
        raise ValueError(
            f"Shuffled trainset length {len(session.shuffled_trainset_ids)} is not divisible by "
            f"num_dspy_examples_per_grpo_step {num_dspy_examples_per_grpo_step}"
        )
    base_idx = base_idx % len(session.shuffled_trainset_ids)
    end_idx = base_idx + num_dspy_examples_per_grpo_step
    if end_idx > len(session.shuffled_trainset_ids):
        raise ValueError(
            f"End index {end_idx} is out of bounds for shuffled trainset length {len(session.shuffled_trainset_ids)}"
        )
    selected_ids = session.shuffled_trainset_ids[base_idx:end_idx]
    return [original_trainset[i] for i in selected_ids]

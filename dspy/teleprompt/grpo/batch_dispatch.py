from __future__ import annotations

import functools
import logging
import operator
import random  # noqa: TC003 — rng.sample at runtime
from collections import deque
from typing import Any

from dspy.clients.finetune import GRPOGroup, GRPORolloutGroup, GRPOStatus, TrainDataFormat
from dspy.clients.lm import LM  # noqa: TC001 — used at runtime in job_key tuples
from dspy.teleprompt.grpo.session import GRPOCompileSession  # noqa: TC001 — session mutation at runtime
from dspy.teleprompt.grpo.types import PredictorRolloutBatches  # noqa: TC001 — rollout batch typing

logger = logging.getLogger(__name__)


def pad_rollout_groups(
    train_data: list[GRPORolloutGroup],
    *,
    num_rollouts_per_grpo_step: int,
) -> None:
    for group in train_data:
        if len(group) != num_rollouts_per_grpo_step:
            while len(group) < num_rollouts_per_grpo_step:
                group.extend(group[: min(num_rollouts_per_grpo_step - len(group), len(group))])
        if len(group) != num_rollouts_per_grpo_step:
            raise ValueError(
                f"Number of completions {len(group)} does not match the expected number "
                f"num_rollouts_per_grpo_step={num_rollouts_per_grpo_step}"
            )


def dispatch_training_step(
    session: GRPOCompileSession,
    *,
    job_key: tuple[LM, Any],
    job: Any,
    train_data: list[GRPORolloutGroup],
    train_batch_per_predictor: PredictorRolloutBatches,
    num_rollouts_per_grpo_step: int,
    rng: random.Random,
) -> bool:
    """Dispatch one GRPO training step for a single job. Returns True if a step was submitted."""
    pad_rollout_groups(train_data, num_rollouts_per_grpo_step=num_rollouts_per_grpo_step)
    grpo_status: GRPOStatus = job.get_status()
    pending_batch_ids = grpo_status.pending_batch_ids
    available_batch_ids = list(set(pending_batch_ids) - set(session.fulfilled_batch_ids))
    if not available_batch_ids:
        return False
    q = session.group_queues.setdefault(job_key, deque())
    if len(q) < len(available_batch_ids) and len(train_data) > 0:
        need = len(available_batch_ids) - len(q)
        while need > 0:
            shuffled = rng.sample(train_data, k=len(train_data))
            q.extend(shuffled)
            need -= len(shuffled)
    final_train_data: list[GRPOGroup] = []
    for bid in available_batch_ids:
        if q:
            grp = q.popleft()
        else:
            fallback_pool = (
                train_data if len(train_data) > 0 else functools.reduce(operator.iadd, train_batch_per_predictor, [])
            )
            if len(fallback_pool) == 0:
                continue
            grp = rng.choice(fallback_pool)
        final_train_data.append(GRPOGroup(batch_id=bid, group=grp))
    if not final_train_data:
        return False
    session.fulfilled_batch_ids.extend(item.batch_id for item in final_train_data if item.batch_id is not None)
    job.step(train_data=final_train_data, train_data_format=TrainDataFormat.GRPO_CHAT)
    return True

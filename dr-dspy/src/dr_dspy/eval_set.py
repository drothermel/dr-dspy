"""Pinned train/val/test split over the seeded HumanEval shuffle.

An optimizer study must evaluate every candidate over the *same* fixed
task subset so scores are comparable and content-addressed reuse stays
valid. ``build_eval_split`` partitions the deterministic seeded shuffle
(``human_eval_sampling``) into three disjoint, contiguous slices: the
val set is selected on, the test set is held out and never selected on,
and the train set is available to a proposer. Pure (DB access is only the
dataset load, mirrored by a ``*_from_rows`` variant for tests).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt

from dr_dspy import human_eval_sampling as sampling

HumanEvalRow = Mapping[str, Any]


class EvalSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: StrictInt
    repetitions: StrictInt
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def all_ids(self) -> tuple[str, ...]:
        return self.train_ids + self.val_ids + self.test_ids


def build_eval_split_from_rows(
    rows: Sequence[HumanEvalRow],
    *,
    seed: int,
    train: int,
    val: int,
    test: int,
    repetitions: int,
) -> EvalSplit:
    if min(train, val, test) < 0:
        raise ValueError("split sizes must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    total = train + val + test
    sampled = sampling.sample_human_eval_tasks_from_rows(
        rows, seed=seed, sample_count=total
    )
    if len(sampled) < total:
        raise ValueError(
            f"requested {total} tasks for the split but only "
            f"{len(sampled)} are available"
        )
    ids = [sample.task.task_id for sample in sampled]
    return EvalSplit(
        seed=seed,
        repetitions=repetitions,
        train_ids=tuple(ids[:train]),
        val_ids=tuple(ids[train : train + val]),
        test_ids=tuple(ids[train + val : total]),
    )


def build_eval_split(
    *,
    seed: int,
    train: int,
    val: int,
    test: int,
    repetitions: int,
    dataset_name: str,
    dataset_split: str,
) -> EvalSplit:
    rows = sampling.load_human_eval_rows(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    return build_eval_split_from_rows(
        rows,
        seed=seed,
        train=train,
        val=val,
        test=test,
        repetitions=repetitions,
    )

"""Read-side ``config -> score`` contract for the optimizer.

A candidate is one ``GraphSpec`` (one ``dimensions_digest``). Its score
over a pinned task set is read back from ``dr_dspy_predictions``:
correctness from the typed ``score`` column, compression from
``metrics->>'best_compression_ratio'``. The objective combines them per
task into a single reward; the study selects on the mean over the val set
(the full distribution is preserved for variance-aware selection later).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import eval_records
from dr_dspy.prediction_status import GenerationStatus, ScoringStatus

PREDICTION_TABLE_NAME = eval_records.PREDICTIONS_TABLE_NAME

_TERMINAL_GENERATION = frozenset(
    {
        GenerationStatus.ERROR.value,
        GenerationStatus.RECOVERABLE_ERROR.value,
    }
)
_TERMINAL_SCORING = frozenset(
    {
        ScoringStatus.SCORED.value,
        ScoringStatus.ERROR.value,
        ScoringStatus.RECOVERABLE_ERROR.value,
    }
)


def combined_reward(
    score: float | None, best_compression_ratio: float | None
) -> float:
    """Per-task objective: correctness x compression.

    ``best_compression_ratio`` is the smallest compressed representation
    size relative to the ground-truth code (lower = better compression),
    so the compression reward is ``1 - ratio``. Only correct solutions
    earn compression credit. Clamped to ``[0, inf)``; if there is no
    compression signal the reward is just the correctness score.
    """
    correctness = score or 0.0
    if best_compression_ratio is None:
        return correctness
    return correctness * max(0.0, 1.0 - best_compression_ratio)


class TaskScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    repetition_seed: StrictInt
    generation_status: StrictStr
    scoring_status: StrictStr
    score: float | None = None
    best_compression_ratio: float | None = None

    def is_terminal(self) -> bool:
        return (
            self.scoring_status in _TERMINAL_SCORING
            or self.generation_status in _TERMINAL_GENERATION
        )

    def is_scored(self) -> bool:
        return self.scoring_status == ScoringStatus.SCORED.value

    def reward(self) -> float:
        return combined_reward(self.score, self.best_compression_ratio)


class CandidateScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions_digest: StrictStr
    tasks: list[TaskScore]

    def terminal_count(self) -> int:
        return sum(1 for task in self.tasks if task.is_terminal())

    def coverage(self) -> int:
        return sum(1 for task in self.tasks if task.is_scored())

    def reward_distribution(self) -> list[float]:
        return [task.reward() for task in self.tasks]

    def mean_reward(self) -> float | None:
        rewards = self.reward_distribution()
        return sum(rewards) / len(rewards) if rewards else None


def read_candidate_scores(
    database_url: str,
    *,
    experiment_name: str,
    dimensions_digests: Sequence[str],
    task_ids: Sequence[str],
) -> dict[str, CandidateScores]:
    """Group every prediction row for the given candidates + tasks by
    ``dimensions_digest``. Missing rows simply do not appear."""
    grouped: dict[str, list[TaskScore]] = {
        digest: [] for digest in dimensions_digests
    }
    if not dimensions_digests or not task_ids:
        return {
            digest: CandidateScores(dimensions_digest=digest, tasks=[])
            for digest in dimensions_digests
        }
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    dimensions_digest,
                    task_id,
                    repetition_seed,
                    generation_status,
                    scoring_status,
                    score,
                    (metrics->>'best_compression_ratio')::double precision
                FROM {PREDICTION_TABLE_NAME}
                WHERE experiment_name = %s
                  AND dimensions_digest = ANY(%s)
                  AND task_id = ANY(%s)
                """,
                (
                    experiment_name,
                    list(dimensions_digests),
                    list(task_ids),
                ),
            )
            rows = cur.fetchall()
    for row in rows:
        digest = row[0]
        grouped.setdefault(digest, []).append(
            TaskScore(
                task_id=row[1],
                repetition_seed=row[2],
                generation_status=row[3],
                scoring_status=row[4],
                score=row[5],
                best_compression_ratio=row[6],
            )
        )
    return {
        digest: CandidateScores(dimensions_digest=digest, tasks=tasks)
        for digest, tasks in grouped.items()
    }


def count_terminal(scores: dict[str, CandidateScores]) -> int:
    return sum(candidate.terminal_count() for candidate in scores.values())


def wait_for_scored(
    database_url: str,
    *,
    experiment_name: str,
    dimensions_digests: Sequence[str],
    task_ids: Sequence[str],
    expected_count: int,
    interval_seconds: float = 5.0,
    timeout_seconds: float = 3600.0,
    sleep: Callable[[float], Any] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll until every expected prediction reaches a terminal state.

    Returns ``True`` once ``expected_count`` rows are terminal (scored or
    failed), ``False`` if ``timeout_seconds`` elapses first.
    """
    deadline = monotonic() + timeout_seconds
    while True:
        scores = read_candidate_scores(
            database_url,
            experiment_name=experiment_name,
            dimensions_digests=dimensions_digests,
            task_ids=task_ids,
        )
        if count_terminal(scores) >= expected_count:
            return True
        if monotonic() >= deadline:
            return False
        sleep(interval_seconds)

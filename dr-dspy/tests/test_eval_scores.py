from __future__ import annotations

import pytest

from dr_dspy import eval_scores
from dr_dspy.eval_scores import CandidateScores, TaskScore, combined_reward


def test_combined_reward_rewards_smaller_descriptions() -> None:
    # Correct + heavily compressed -> high reward.
    assert combined_reward(1.0, 0.2) == pytest.approx(0.8)
    # Smaller representation (lower ratio) earns strictly more.
    assert combined_reward(1.0, 0.2) > combined_reward(1.0, 0.6)


def test_combined_reward_zero_for_incorrect() -> None:
    assert combined_reward(0.0, 0.1) == 0.0
    assert combined_reward(None, 0.1) == 0.0


def test_combined_reward_clamps_expansion() -> None:
    # Representation bigger than ground truth (ratio > 1) -> no credit.
    assert combined_reward(1.0, 1.5) == 0.0


def test_combined_reward_without_compression_is_correctness() -> None:
    assert combined_reward(1.0, None) == 1.0
    assert combined_reward(0.0, None) == 0.0


def _task(
    *,
    scoring: str = "scored",
    generation: str = "generated",
    score: float | None = 1.0,
    ratio: float | None = 0.5,
    rep: int = 0,
) -> TaskScore:
    return TaskScore(
        task_id="T0",
        repetition_seed=rep,
        generation_status=generation,
        scoring_status=scoring,
        score=score,
        best_compression_ratio=ratio,
    )


def test_candidate_scores_mean_and_coverage() -> None:
    candidate = CandidateScores(
        dimensions_digest="d0",
        tasks=[
            _task(score=1.0, ratio=0.5),  # reward 0.5
            _task(score=0.0, ratio=0.5),  # reward 0.0
        ],
    )
    assert candidate.coverage() == 2
    assert candidate.mean_reward() == 0.25
    assert candidate.terminal_count() == 2


def test_terminal_includes_failed_generation() -> None:
    failed = _task(
        scoring="pending",
        generation="generation_error",
        score=None,
        ratio=None,
    )
    assert failed.is_terminal()
    assert not failed.is_scored()
    assert failed.reward() == 0.0


def test_count_terminal_across_candidates() -> None:
    scores = {
        "d0": CandidateScores(
            dimensions_digest="d0",
            tasks=[_task(), _task(scoring="started", generation="started")],
        ),
        "d1": CandidateScores(dimensions_digest="d1", tasks=[_task()]),
    }
    # d0: one scored (terminal) + one in-flight; d1: one scored.
    assert eval_scores.count_terminal(scores) == 2


def test_empty_candidate_has_no_mean() -> None:
    candidate = CandidateScores(dimensions_digest="d0", tasks=[])
    assert candidate.mean_reward() is None
    assert candidate.coverage() == 0

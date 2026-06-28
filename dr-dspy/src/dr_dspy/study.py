"""Strategy-agnostic optimizer core (pure, DB-free, DBOS-free).

Turns instructions into addressable candidate graphs, scores them via the
``eval_scores`` contract, and selects the best by mean reward. The grid
and COPRO strategies both drive this same core; the DBOS study workflow
is a thin shell around it. Kept pure so the whole round loop is
unit-testable with injected scores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_dspy.eval_scores import CandidateScores
from dr_dspy.experiment_spec import (
    GraphSpec,
    dimensions_digest,
    with_node_instruction,
)


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    graph: GraphSpec
    dimensions_digest: str
    provenance: dict[str, Any] = {}


def make_candidate_graphs(
    base_graph: GraphSpec,
    node_id: str,
    instructions: Sequence[str],
    *,
    provenances: Sequence[Mapping[str, Any]] | None = None,
) -> list[CandidateSpec]:
    """One ``CandidateSpec`` per instruction, varying only ``node_id``'s
    instruction on ``base_graph``."""
    if provenances is not None and len(provenances) != len(instructions):
        raise ValueError("provenances length must match instructions")
    specs: list[CandidateSpec] = []
    for index, instruction in enumerate(instructions):
        graph = with_node_instruction(base_graph, node_id, instruction)
        specs.append(
            CandidateSpec(
                instruction=instruction,
                graph=graph,
                dimensions_digest=dimensions_digest(graph),
                provenance=(dict(provenances[index]) if provenances else {}),
            )
        )
    return specs


class ScoredCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: CandidateSpec
    scores: CandidateScores

    def mean_reward(self) -> float | None:
        return self.scores.mean_reward()


def select_best(scored: Sequence[ScoredCandidate]) -> ScoredCandidate:
    """Highest mean reward wins; unscored candidates rank last; ties break
    deterministically by ``dimensions_digest``."""
    if not scored:
        raise ValueError("cannot select from an empty candidate set")

    def key(item: ScoredCandidate) -> tuple[float, str]:
        mean = item.mean_reward()
        ranked = mean if mean is not None else -math.inf
        # Negate digest ordering by selecting min digest on ties: use the
        # digest directly with reversed reward so max() is well-defined.
        return (ranked, item.candidate.dimensions_digest)

    best = scored[0]
    best_key = key(best)
    for item in scored[1:]:
        item_key = key(item)
        if item_key[0] > best_key[0] or (
            item_key[0] == best_key[0] and item_key[1] < best_key[1]
        ):
            best, best_key = item, item_key
    return best


def history_entry(round_index: int, best: ScoredCandidate) -> dict[str, Any]:
    """One persisted round summary for the study ``history`` JSONB."""
    return {
        "round_index": round_index,
        "instruction": best.candidate.instruction,
        "dimensions_digest": best.candidate.dimensions_digest,
        "mean_reward": best.mean_reward(),
        "coverage": best.scores.coverage(),
    }


def proposer_history(
    history: Sequence[Mapping[str, Any]],
) -> list[tuple[str, float]]:
    """Sorted (instruction, reward) attempts for the COPRO
    propose-given-attempts op; best reward first, unscored dropped."""
    attempts = [
        (str(item["instruction"]), float(item["mean_reward"]))
        for item in history
        if item.get("instruction") is not None
        and item.get("mean_reward") is not None
    ]
    attempts.sort(key=lambda pair: pair[1], reverse=True)
    return attempts

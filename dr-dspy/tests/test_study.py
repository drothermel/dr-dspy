from __future__ import annotations

from dr_dspy import study
from dr_dspy.eval_scores import CandidateScores, TaskScore
from dr_dspy.experiment_spec import (
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    dimensions_digest,
    with_node_instruction,
)


def _base_graph(instruction: str = "solve it") -> GraphSpec:
    solve = NodeSpec(
        id="solve",
        config=NodeConfig(
            model="m",
            instruction=instruction,
            signature_name="Solve",
            fields=(
                FieldSpec(name="prompt", role="input"),
                FieldSpec(name="code", role="output", type_name="code"),
            ),
            output_field="code",
            input_bindings={"prompt": "task.prompt"},
        ),
    )
    return GraphSpec(
        nodes=(solve,),
        terminal_node_id="solve",
        compression_source="task.prompt",
    )


def test_with_node_instruction_changes_digest() -> None:
    base = _base_graph("A")
    changed = with_node_instruction(base, "solve", "B")
    assert dimensions_digest(base) != dimensions_digest(changed)
    assert changed.node("solve").config.instruction == "B"
    # Other fields untouched.
    assert changed.node("solve").config.model == "m"


def test_with_node_instruction_unknown_node() -> None:
    try:
        with_node_instruction(_base_graph(), "missing", "x")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_make_candidate_graphs_distinct_digests() -> None:
    base = _base_graph()
    candidates = study.make_candidate_graphs(
        base, "solve", ["one", "two", "three"]
    )
    assert [c.instruction for c in candidates] == ["one", "two", "three"]
    digests = {c.dimensions_digest for c in candidates}
    assert len(digests) == 3


def _scored(
    digest_instruction: str, mean: float | None
) -> study.ScoredCandidate:
    base = _base_graph()
    graph = with_node_instruction(base, "solve", digest_instruction)
    candidate = study.CandidateSpec(
        instruction=digest_instruction,
        graph=graph,
        dimensions_digest=dimensions_digest(graph),
    )
    if mean is None:
        tasks: list[TaskScore] = []
    else:
        tasks = [
            TaskScore(
                task_id="T0",
                repetition_seed=0,
                generation_status="generated",
                scoring_status="scored",
                score=mean,
                best_compression_ratio=None,
            )
        ]
    scores = CandidateScores(
        dimensions_digest=candidate.dimensions_digest, tasks=tasks
    )
    return study.ScoredCandidate(candidate=candidate, scores=scores)


def test_select_best_picks_highest_mean() -> None:
    best = study.select_best(
        [_scored("a", 0.1), _scored("b", 0.9), _scored("c", 0.5)]
    )
    assert best.candidate.instruction == "b"


def test_select_best_ranks_unscored_last() -> None:
    best = study.select_best([_scored("a", None), _scored("b", 0.2)])
    assert best.candidate.instruction == "b"


def test_select_best_empty_raises() -> None:
    try:
        study.select_best([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_proposer_history_sorted_and_filtered() -> None:
    history = [
        {"instruction": "a", "mean_reward": 0.2},
        {"instruction": "b", "mean_reward": 0.7},
        {"instruction": "c", "mean_reward": None},
        {"phase": "test", "test_mean_reward": 0.9},
    ]
    attempts = study.proposer_history(history)
    assert attempts == [("b", 0.7), ("a", 0.2)]


def test_history_entry_shape() -> None:
    best = _scored("a", 0.4)
    entry = study.history_entry(2, best)
    assert entry["round_index"] == 2
    assert entry["instruction"] == "a"
    assert entry["mean_reward"] == 0.4
    assert entry["coverage"] == 1

from __future__ import annotations

from dr_dspy.experiment_spec import (
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
)
from dr_dspy.humaneval_eval_dbos import (
    EvalSample,
    EvalSubmitSpec,
    build_prediction_jobs_for_offsets,
)


def _graph(instruction: str) -> GraphSpec:
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


def _spec() -> EvalSubmitSpec:
    return EvalSubmitSpec(
        pipeline="direct",
        script_kind="k",
        experiment_name="exp",
        seed=0,
        sample_count=2,
        repetitions=2,
        score_timeout=15.0,
        max_completion_tokens=1000,
        graphs=[_graph("g0"), _graph("g1")],
    )


def _samples() -> list[EvalSample]:
    return [
        EvalSample(
            task_id="t0",
            sample_index=0,
            prompt="p0",
            test="x",
            entry_point="e",
        ),
        EvalSample(
            task_id="t1",
            sample_index=1,
            prompt="p1",
            test="x",
            entry_point="e",
        ),
    ]


def test_total_and_per_sample_counts() -> None:
    spec = _spec()
    assert spec.jobs_per_sample() == 4
    assert spec.total_jobs() == 8


def test_mixed_radix_offset_mapping() -> None:
    spec = _spec()
    jobs = build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="s",
        samples=_samples(),
        start_offset=0,
        limit=8,
    )
    assert len(jobs) == 8
    # offset -> (repetition_seed, graph_index, sample_index)
    coords = [
        (
            job.repetition_seed,
            job.graph.nodes[0].config.instruction,
            job.sample_index,
        )
        for job in jobs
    ]
    assert coords[0] == (0, "g0", 0)
    assert coords[1] == (1, "g0", 0)
    assert coords[2] == (0, "g1", 0)
    assert coords[3] == (1, "g1", 0)
    assert coords[4] == (0, "g0", 1)
    assert coords[7] == (1, "g1", 1)
    assert len({job.prediction_id for job in jobs}) == 8


def test_partial_window() -> None:
    spec = _spec()
    jobs = build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="s",
        samples=_samples(),
        start_offset=2,
        limit=3,
    )
    assert [job.sample_index for job in jobs] == [0, 0, 1]


def test_prediction_id_is_graph_sensitive() -> None:
    spec = _spec()
    jobs = build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id="s",
        samples=_samples(),
        start_offset=0,
        limit=4,
    )
    # same sample/rep, different graph -> different id
    g0_job = next(
        j
        for j in jobs
        if j.sample_index == 0
        and j.repetition_seed == 0
        and j.graph.nodes[0].config.instruction == "g0"
    )
    g1_job = next(
        j
        for j in jobs
        if j.sample_index == 0
        and j.repetition_seed == 0
        and j.graph.nodes[0].config.instruction == "g1"
    )
    assert g0_job.prediction_id != g1_job.prediction_id

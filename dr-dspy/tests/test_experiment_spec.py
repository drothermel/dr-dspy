from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_dspy.experiment_spec import (
    ArtifactRecord,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    PredictionPayload,
    dimensions_digest,
    prediction_id,
)


def _node(
    node_id: str,
    *,
    instruction: str,
    in_field: str,
    out_field: str,
    binding: dict[str, str],
    reasoning: dict | None = None,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            model="openai/gpt-5.1-codex-mini",
            instruction=instruction,
            signature_name=node_id.title(),
            fields=(
                FieldSpec(name=in_field, role="input"),
                FieldSpec(name=out_field, role="output", type_name="code"),
            ),
            output_field=out_field,
            input_bindings=binding,
            reasoning=reasoning or {},
        ),
    )


def direct_graph(instruction: str = "Write code.") -> GraphSpec:
    solve = _node(
        "solve",
        instruction=instruction,
        in_field="prompt",
        out_field="code",
        binding={"prompt": "task.prompt"},
    )
    return GraphSpec(
        nodes=(solve,),
        terminal_node_id="solve",
        compression_source="task.prompt",
    )


def encdec_graph() -> GraphSpec:
    encode = _node(
        "encode",
        instruction="Encode the code.",
        in_field="code",
        out_field="description",
        binding={"code": "task.ground_truth_code"},
    )
    decode = _node(
        "decode",
        instruction="Decode the description.",
        in_field="description",
        out_field="code",
        binding={"description": "encode"},
    )
    return GraphSpec(
        nodes=(encode, decode),
        terminal_node_id="decode",
        compression_source="encode",
    )


def test_direct_and_encdec_validate_and_order() -> None:
    assert direct_graph().topological_order()[-1].id == "solve"
    order = [n.id for n in encdec_graph().topological_order()]
    assert order == ["encode", "decode"]


def test_terminal_must_exist() -> None:
    with pytest.raises(ValidationError):
        GraphSpec(
            nodes=direct_graph().nodes,
            terminal_node_id="nope",
            compression_source="task.prompt",
        )


def test_unknown_ref_rejected() -> None:
    bad = _node(
        "decode",
        instruction="d",
        in_field="description",
        out_field="code",
        binding={"description": "ghost"},
    )
    with pytest.raises(ValidationError):
        GraphSpec(
            nodes=(bad,),
            terminal_node_id="decode",
            compression_source="task.prompt",
        )


def test_cycle_rejected() -> None:
    a = _node(
        "a",
        instruction="a",
        in_field="x",
        out_field="code",
        binding={"x": "b"},
    )
    b = _node(
        "b",
        instruction="b",
        in_field="x",
        out_field="code",
        binding={"x": "a"},
    )
    with pytest.raises(ValidationError):
        GraphSpec(
            nodes=(a, b),
            terminal_node_id="a",
            compression_source="task.prompt",
        )


def test_unknown_field_type_rejected() -> None:
    with pytest.raises(ValidationError):
        FieldSpec(name="x", role="input", type_name="widget")


def test_output_field_must_be_output() -> None:
    with pytest.raises(ValidationError):
        NodeConfig(
            model="m",
            instruction="i",
            signature_name="S",
            fields=(FieldSpec(name="prompt", role="input"),),
            output_field="prompt",
        )


def test_prediction_id_is_deterministic() -> None:
    graph = direct_graph()
    first = prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph=graph,
        repetition_seed=0,
    )
    second = prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph=graph,
        repetition_seed=0,
    )
    assert first == second


def test_prediction_id_ignores_dict_key_order() -> None:
    g1 = direct_graph()
    g1.nodes[0].config.reasoning.update({"a": 1, "b": 2})
    g2 = direct_graph()
    g2.nodes[0].config.reasoning.update({"b": 2, "a": 1})
    id1 = prediction_id(
        experiment_name="e", task_id="t", graph=g1, repetition_seed=0
    )
    id2 = prediction_id(
        experiment_name="e", task_id="t", graph=g2, repetition_seed=0
    )
    assert id1 == id2


def test_prediction_id_changes_with_instruction() -> None:
    a = prediction_id(
        experiment_name="e",
        task_id="t",
        graph=direct_graph("v1"),
        repetition_seed=0,
    )
    b = prediction_id(
        experiment_name="e",
        task_id="t",
        graph=direct_graph("v2"),
        repetition_seed=0,
    )
    assert a != b


def test_dimensions_digest_tracks_graph() -> None:
    assert dimensions_digest(direct_graph()) == dimensions_digest(
        direct_graph()
    )
    assert dimensions_digest(direct_graph()) != dimensions_digest(
        encdec_graph()
    )


def test_prediction_payload_round_trip() -> None:
    payload = PredictionPayload(
        pipeline="direct",
        dimensions={"graph": direct_graph().model_dump(mode="json")},
        artifacts={
            "solve": ArtifactRecord(
                output="def f(): ...",
                usage={"completion_tokens": 5},
                cost=0.001,
            )
        },
        task_inputs={"prompt": "..."},
    )
    again = PredictionPayload.model_validate(payload.model_dump())
    assert again == payload
    assert again.artifacts["solve"].cost == 0.001

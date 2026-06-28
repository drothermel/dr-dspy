from __future__ import annotations

from typing import Any

from dr_dspy import dspy_runner, node_runner
from dr_dspy.experiment_spec import (
    ArtifactRecord,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
)


def _solve_node() -> NodeSpec:
    return NodeSpec(
        id="solve",
        config=NodeConfig(
            model="m",
            instruction="Write code.",
            signature_name="Solve",
            fields=(
                FieldSpec(name="prompt", role="input"),
                FieldSpec(name="code", role="output", type_name="code"),
            ),
            output_field="code",
            input_bindings={"prompt": "task.prompt"},
        ),
    )


def direct_graph() -> GraphSpec:
    return GraphSpec(
        nodes=(_solve_node(),),
        terminal_node_id="solve",
        compression_source="task.prompt",
    )


def encdec_graph(*, budget_ratio: float | None = None) -> GraphSpec:
    enc_fields = [
        FieldSpec(name="code", role="input"),
        FieldSpec(name="description", role="output"),
    ]
    extra: dict[str, Any] = {}
    if budget_ratio is not None:
        enc_fields.insert(
            1, FieldSpec(name="max_characters", role="input", type_name="int")
        )
        extra = {"budget_ratio": budget_ratio}
    encode = NodeSpec(
        id="encode",
        config=NodeConfig(
            model="m",
            instruction="Encode.",
            signature_name="Encode",
            fields=tuple(enc_fields),
            output_field="description",
            input_bindings={"code": "task.ground_truth_code"},
            extra=extra,
        ),
    )
    decode = NodeSpec(
        id="decode",
        config=NodeConfig(
            model="m",
            instruction="Decode.",
            signature_name="Decode",
            fields=(
                FieldSpec(name="description", role="input"),
                FieldSpec(name="code", role="output", type_name="code"),
            ),
            output_field="code",
            input_bindings={"description": "encode"},
        ),
    )
    return GraphSpec(
        nodes=(encode, decode),
        terminal_node_id="decode",
        compression_source="encode",
    )


def test_execute_direct_graph() -> None:
    def run_node(node: NodeSpec, inputs: dict[str, Any]) -> ArtifactRecord:
        return ArtifactRecord(output="CODE:" + inputs["prompt"])

    run = node_runner.execute_graph(
        direct_graph(), task_inputs={"prompt": "P"}, run_node=run_node
    )
    assert run.artifacts["solve"].output == "CODE:P"
    assert run.terminal_output == "CODE:P"


def test_execute_encdec_threads_outputs() -> None:
    def run_node(node: NodeSpec, inputs: dict[str, Any]) -> ArtifactRecord:
        if node.id == "encode":
            return ArtifactRecord(output="DESC:" + inputs["code"])
        return ArtifactRecord(output="CODE:" + inputs["description"])

    run = node_runner.execute_graph(
        encdec_graph(),
        task_inputs={"ground_truth_code": "GT"},
        run_node=run_node,
    )
    assert run.artifacts["encode"].output == "DESC:GT"
    assert run.artifacts["decode"].output == "CODE:DESC:GT"
    assert run.terminal_output == "CODE:DESC:GT"


def test_budget_ratio_injects_max_characters() -> None:
    seen: dict[str, Any] = {}

    def run_node(node: NodeSpec, inputs: dict[str, Any]) -> ArtifactRecord:
        if node.id == "encode":
            seen.update(inputs)
            return ArtifactRecord(output="DESC")
        return ArtifactRecord(output="CODE")

    node_runner.execute_graph(
        encdec_graph(budget_ratio=0.5),
        task_inputs={"ground_truth_code": "x" * 200},
        run_node=run_node,
    )
    assert seen["max_characters"] == 100


def test_budget_ratio_floored() -> None:
    seen: dict[str, Any] = {}

    def run_node(node: NodeSpec, inputs: dict[str, Any]) -> ArtifactRecord:
        if node.id == "encode":
            seen.update(inputs)
        return ArtifactRecord(output="x")

    node_runner.execute_graph(
        encdec_graph(budget_ratio=0.5),
        task_inputs={"ground_truth_code": "abc"},
        run_node=run_node,
    )
    assert seen["max_characters"] == node_runner.MIN_ENCODER_CHAR_BUDGET


def test_build_node_signature_injects_instruction_and_caches() -> None:
    config = _solve_node().config
    sig = node_runner.build_node_signature(config)
    assert sig.instructions == "Write code."
    assert node_runner.build_node_signature(config) is sig


def test_llm_run_node_uses_per_node_instruction(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_predictor(**kwargs: Any) -> str:
        captured["instructions"] = kwargs["signature"].instructions
        return "OK:" + kwargs["input_kwargs"].get("prompt", "")

    monkeypatch.setattr(dspy_runner, "build_logged_lm", lambda **kw: object())
    monkeypatch.setattr(dspy_runner, "run_predictor", fake_run_predictor)

    run_node = node_runner.make_llm_run_node(max_completion_tokens=100)
    run = node_runner.execute_graph(
        direct_graph(), task_inputs={"prompt": "P"}, run_node=run_node
    )
    assert run.terminal_output == "OK:P"
    assert captured["instructions"] == "Write code."


def test_compression_source_text() -> None:
    def run_node(node: NodeSpec, inputs: dict[str, Any]) -> ArtifactRecord:
        if node.id == "encode":
            return ArtifactRecord(output="DESC")
        return ArtifactRecord(output="CODE")

    graph = encdec_graph()
    run = node_runner.execute_graph(
        graph, task_inputs={"ground_truth_code": "GT"}, run_node=run_node
    )
    assert run.compression_source_text(graph, {}) == "DESC"
    direct = direct_graph()
    drun = node_runner.execute_graph(
        direct, task_inputs={"prompt": "P"}, run_node=run_node
    )
    assert drun.compression_source_text(direct, {"prompt": "P"}) == "P"

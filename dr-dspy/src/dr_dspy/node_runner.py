"""Generic executor for graph-shaped generation.

Walks a ``GraphSpec`` in deterministic topological order, running each
node and threading its output to downstream nodes. The single ``llm_call``
op builds its dspy signature *per node from the node config* — so the
instruction is a runtime parameter (the COPRO enabler) rather than baked
in at construction. ``direct`` is a one-node graph; ``enc-dec`` a
two-node chain; the same code runs any linear chain (DAG-ready).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

import dspy
from dr_dspy import dspy_runner
from dr_dspy.experiment_spec import (
    TASK_SOURCE,
    ArtifactRecord,
    GraphSpec,
    NodeConfig,
    NodeSpec,
)
from dr_dspy.lm_utils import LmEventBuffer, stable_json
from dspy.signatures.signature import make_signature

#: Floor for the budgeted-encoder character budget (mirrors v0).
MIN_ENCODER_CHAR_BUDGET = 50

_TYPE_MAP: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "code": dspy.Code,
}

RunNode = Callable[[NodeSpec, dict[str, Any]], ArtifactRecord]

_SIGNATURE_CACHE: dict[str, type[dspy.Signature]] = {}


def _signature_cache_key(config: NodeConfig) -> str:
    return stable_json(
        {
            "name": config.signature_name,
            "instruction": config.instruction,
            "fields": [[f.name, f.type_name, f.role] for f in config.fields],
        }
    )


def build_node_signature(config: NodeConfig) -> type[dspy.Signature]:
    """Build (and cache) a dspy signature from a node config, injecting
    the node's ``instruction``."""
    key = _signature_cache_key(config)
    cached = _SIGNATURE_CACHE.get(key)
    if cached is not None:
        return cached
    fields = {
        field.name: (
            _TYPE_MAP[field.type_name],
            dspy.InputField() if field.role == "input" else dspy.OutputField(),
        )
        for field in config.fields
    }
    signature = make_signature(
        fields,
        instructions=config.instruction,
        signature_name=config.signature_name,
    )
    _SIGNATURE_CACHE[key] = signature
    return signature


def _resolve_inputs(
    node: NodeSpec,
    *,
    task_inputs: Mapping[str, Any],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for field, ref in node.config.input_bindings.items():
        head, _, _tail = ref.partition(".")
        if head == TASK_SOURCE:
            resolved[field] = task_inputs[_tail]
        else:
            resolved[field] = outputs[head]
    ratio = node.config.extra.get("budget_ratio")
    if ratio is not None:
        source = node.config.extra.get("budget_source", "code")
        target = node.config.extra.get("budget_target", "max_characters")
        resolved[target] = max(
            MIN_ENCODER_CHAR_BUDGET,
            round(float(ratio) * len(resolved[source])),
        )
    return resolved


def make_llm_run_node(
    *, max_completion_tokens: int, client: Any = None
) -> RunNode:
    """A ``run_node`` that performs one logged OpenRouter predictor call."""

    def run_node(node: NodeSpec, inputs: dict[str, Any]) -> ArtifactRecord:
        config = node.config
        buffer = LmEventBuffer()
        lm = dspy_runner.build_logged_lm(
            model=config.model,
            reasoning=config.reasoning,
            temperature=config.temperature,
            event_buffer=buffer,
            max_completion_tokens=max_completion_tokens,
            client=client,
        )
        text = dspy_runner.run_predictor(
            signature=build_node_signature(config),
            input_kwargs=inputs,
            output_field=config.output_field,
            lm=lm,
            event_buffer=buffer,
        )
        result = dspy_runner.predictor_run_result(text, buffer)
        return ArtifactRecord(
            output=result.text,
            usage=result.usage_metadata,
            cost=result.provider_cost,
            response_metadata=result.response_metadata,
        )

    return run_node


class GraphRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts: dict[str, ArtifactRecord]
    terminal_output: str

    def total_cost(self) -> float | None:
        costs = [a.cost for a in self.artifacts.values() if a.cost is not None]
        return sum(costs) if costs else None

    def compression_source_text(
        self, graph: GraphSpec, task_inputs: Mapping[str, Any]
    ) -> str:
        ref = graph.compression_source
        head, _, tail = ref.partition(".")
        if head == TASK_SOURCE:
            return str(task_inputs[tail])
        return self.artifacts[head].output


def execute_graph(
    graph: GraphSpec,
    *,
    task_inputs: Mapping[str, Any],
    run_node: RunNode,
) -> GraphRun:
    """Run every node in topological order, threading outputs."""
    outputs: dict[str, str] = {}
    artifacts: dict[str, ArtifactRecord] = {}
    for node in graph.topological_order():
        inputs = _resolve_inputs(
            node, task_inputs=task_inputs, outputs=outputs
        )
        record = run_node(node, inputs)
        artifacts[node.id] = record
        outputs[node.id] = record.output
    return GraphRun(
        artifacts=artifacts,
        terminal_output=outputs[graph.terminal_node_id],
    )

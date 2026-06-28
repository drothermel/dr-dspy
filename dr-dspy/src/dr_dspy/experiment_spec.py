"""Pipeline-agnostic specs for graph-shaped generation experiments.

A generation pipeline is a DAG of typed nodes. ``direct`` is a one-node
graph (``prompt -> code``); ``enc-dec`` is a two-node chain
(``code -> description -> code``). Everything that varies between
experiments lives in the ``GraphSpec`` (topology + per-node config,
including the instruction), which is hashed into the prediction identity.
Execution of the graph lives in ``node_runner`` (Phase 1); this module is
pure data + validation + hashing so it stays dependency-light and
DB-free.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from dr_dspy.lm_utils import stable_json

SCHEMA_VERSION = 1
PREDICTION_ID_LENGTH = 32
DIMENSIONS_DIGEST_LENGTH = 16

#: Reserved source namespace for task-provided fields in input bindings.
TASK_SOURCE = "task"
TASK_FIELDS = frozenset(
    {
        "task_id",
        "prompt",
        "canonical_solution",
        "ground_truth_code",
        "test",
        "entry_point",
    }
)
ALLOWED_FIELD_TYPES = frozenset({"str", "int", "float", "bool", "code"})

FieldRole = Literal["input", "output"]


class FieldSpec(BaseModel):
    """A JSON-serializable description of one dspy signature field.

    ``type_name`` is mapped back to a concrete Python/dspy type by the
    executor (Phase 1); kept as a string so the spec is hashable.
    """

    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    role: FieldRole
    type_name: str = "str"
    description: str | None = None

    @field_validator("type_name")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in ALLOWED_FIELD_TYPES:
            allowed = ", ".join(sorted(ALLOWED_FIELD_TYPES))
            raise ValueError(
                f"unknown field type {value!r}; allowed: {allowed}"
            )
        return value


class NodeConfig(BaseModel):
    """Per-node configuration. The ``instruction`` is a runtime axis."""

    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    instruction: StrictStr
    signature_name: StrictStr
    fields: tuple[FieldSpec, ...]
    output_field: StrictStr
    temperature: float | None = None
    reasoning: dict[str, Any] = Field(default_factory=dict)
    #: Maps this node's input field name -> a source ref. A ref is either
    #: ``task.<field>`` or ``<node_id>`` / ``<node_id>.<output_field>``.
    input_bindings: dict[str, str] = Field(default_factory=dict)
    #: Op-specific extras (e.g. ``{"budget_ratio": 0.5}``).
    extra: dict[str, Any] = Field(default_factory=dict)

    def input_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.role == "input")

    def output_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(f for f in self.fields if f.role == "output")

    @model_validator(mode="after")
    def _check_fields(self) -> NodeConfig:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field names in node config")
        outputs = {f.name for f in self.output_fields()}
        if self.output_field not in outputs:
            raise ValueError(
                f"output_field {self.output_field!r} is not an output field"
            )
        inputs = {f.name for f in self.input_fields()}
        for bound in self.input_bindings:
            if bound not in inputs:
                raise ValueError(
                    f"input binding {bound!r} is not an input field"
                )
        return self


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    config: NodeConfig
    op: str = "llm_call"

    def dependencies(self) -> set[str]:
        """Upstream node ids this node consumes (excludes ``task``)."""
        deps: set[str] = set()
        for ref in self.config.input_bindings.values():
            head = ref.split(".", 1)[0]
            if head != TASK_SOURCE:
                deps.add(head)
        return deps


class GraphSpec(BaseModel):
    """A generation graph: nodes, the terminal (code) node, and the text
    used as the compression source for metrics."""

    model_config = ConfigDict(extra="forbid")

    nodes: tuple[NodeSpec, ...]
    terminal_node_id: StrictStr
    compression_source: StrictStr

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def topological_order(self) -> tuple[NodeSpec, ...]:
        order: list[NodeSpec] = []
        done: set[str] = set()
        remaining = list(self.nodes)
        while remaining:
            progressed = False
            for node in list(remaining):
                if node.dependencies() <= done:
                    order.append(node)
                    done.add(node.id)
                    remaining.remove(node)
                    progressed = True
            if not progressed:
                stuck = ", ".join(n.id for n in remaining)
                raise ValueError(f"graph has a cycle among: {stuck}")
        return tuple(order)

    @model_validator(mode="after")
    def _validate(self) -> GraphSpec:
        if not self.nodes:
            raise ValueError("graph must have at least one node")
        ids = self.node_ids()
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids")
        id_set = set(ids)
        if self.terminal_node_id not in id_set:
            raise ValueError(
                f"terminal_node_id {self.terminal_node_id!r} not in graph"
            )
        for node in self.nodes:
            for ref in node.config.input_bindings.values():
                _validate_ref(ref, id_set)
            missing = node.dependencies() - id_set
            if missing:
                raise ValueError(
                    f"node {node.id!r} depends on unknown nodes: {missing}"
                )
        _validate_ref(self.compression_source, id_set)
        self.topological_order()  # raises on cycle
        return self


def _validate_ref(ref: str, node_ids: set[str]) -> None:
    head, _, tail = ref.partition(".")
    if head == TASK_SOURCE:
        if tail and tail not in TASK_FIELDS:
            raise ValueError(f"unknown task field in ref {ref!r}")
        if not tail:
            raise ValueError(f"task ref {ref!r} needs a field")
        return
    if head not in node_ids:
        raise ValueError(f"ref {ref!r} points at unknown node {head!r}")


# --- identity & canonicalization -------------------------------------


def canonical_dimensions(graph: GraphSpec) -> dict[str, Any]:
    """The canonical, JSON-serializable axes of variation."""
    return {"graph": graph.model_dump(mode="json")}


def _digest(payload: dict[str, Any], length: int | None) -> str:
    raw = stable_json(payload)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def dimensions_digest(
    graph: GraphSpec, *, length: int = DIMENSIONS_DIGEST_LENGTH
) -> str:
    """Short, stable hash of the graph alone (for indexing/ordering)."""
    return _digest(canonical_dimensions(graph), length)


def prediction_id(
    *,
    experiment_name: str,
    task_id: str,
    graph: GraphSpec,
    repetition_seed: int,
    length: int | None = PREDICTION_ID_LENGTH,
) -> str:
    """Stable content-addressed id. Mirrors the algorithm in
    ``humaneval_dbos_flow.stable_prediction_id_from_dimensions`` so the
    instruction (and whole graph) is part of the identity."""
    return _digest(
        {
            "experiment_name": experiment_name,
            "task_id": task_id,
            **canonical_dimensions(graph),
            "repetition_seed": repetition_seed,
        },
        length,
    )


# --- graph mutation ---------------------------------------------------


def with_node_instruction(
    graph: GraphSpec, node_id: str, instruction: str
) -> GraphSpec:
    """Return a copy of ``graph`` with one node's ``instruction`` replaced.

    Everything else is held fixed, so the only axis that changes is the
    instruction — and because the instruction is part of the hashed
    dimensions, the returned graph has a different ``dimensions_digest``
    (the COPRO candidate addressing relies on this).
    """
    new_nodes: list[NodeSpec] = []
    found = False
    for node in graph.nodes:
        if node.id == node_id:
            new_config = node.config.model_copy(
                update={"instruction": instruction}
            )
            new_nodes.append(node.model_copy(update={"config": new_config}))
            found = True
        else:
            new_nodes.append(node)
    if not found:
        raise KeyError(node_id)
    return graph.model_copy(update={"nodes": tuple(new_nodes)})


# --- run-time payloads ------------------------------------------------


class ArtifactRecord(BaseModel):
    """One node's output + observability, stored in the artifacts bag."""

    model_config = ConfigDict(extra="forbid")

    output: str
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionPayload(BaseModel):
    """The JSONB payload columns of a unified prediction row."""

    model_config = ConfigDict(extra="forbid")

    pipeline: StrictStr
    dimensions: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, Any] = Field(default_factory=dict)
    task_inputs: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Minimal experiment-level config (stored as JSONB in the
    experiments table). The concrete graph(s) for a run are built by the
    submitting script (Phase 1)."""

    model_config = ConfigDict(extra="forbid")

    pipeline: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr
    metadata: dict[str, Any] = Field(default_factory=dict)

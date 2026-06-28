"""Unified DDL + row IO for the consolidated eval schema.

One predictions table and one experiments table serve every pipeline,
discriminated by a ``pipeline`` column. The experiment *axes* (the graph
spec, incl. per-node instruction) live in the ``dimensions`` JSONB; node
outputs live in the ``artifacts`` JSONB. A small typed control-plane
spine (ids, statuses, score, digest, timestamps) stays as columns so the
orchestrator/dispatcher/repair queries and indexes are fast.

This module is DB-free: it exposes DDL strings, the statement list, and
pure row<->payload (de)serialization. The actual ``create_schema`` /
INSERT execution lives in the pipeline module (Phase 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_dspy import batch_operation
from dr_dspy.experiment_spec import PredictionPayload

EXPERIMENTS_TABLE_NAME = "dr_dspy_experiments"
PREDICTIONS_TABLE_NAME = "dr_dspy_predictions"

EXPERIMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_experiments (
    experiment_name TEXT        PRIMARY KEY,
    pipeline        TEXT        NOT NULL,
    script_kind     TEXT        NOT NULL,
    seed            INTEGER     NOT NULL,
    sample_count    INTEGER     NOT NULL,
    config          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_predictions (
    prediction_id     TEXT        PRIMARY KEY,
    experiment_name   TEXT        NOT NULL
        REFERENCES dr_dspy_experiments(experiment_name),
    pipeline          TEXT        NOT NULL,
    schema_version    INTEGER     NOT NULL DEFAULT 1,
    script_kind       TEXT        NOT NULL,
    submission_id     TEXT        NOT NULL,
    task_id           TEXT        NOT NULL,
    sample_index      INTEGER     NOT NULL,
    repetition_seed   INTEGER     NOT NULL,
    dimensions_digest TEXT        NOT NULL,
    generation_status TEXT        NOT NULL DEFAULT 'pending',
    generation_failure_class TEXT,
    scoring_status    TEXT        NOT NULL DEFAULT 'pending',
    scoring_failure_class TEXT,
    score             DOUBLE PRECISION,
    provider_cost     DOUBLE PRECISION,
    raw_generation    TEXT,
    dimensions        JSONB       NOT NULL,
    task_inputs       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    artifacts         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    metrics           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    errors            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at      TIMESTAMPTZ,
    scored_at         TIMESTAMPTZ,
    CONSTRAINT dr_dspy_predictions_identity_key UNIQUE (
        experiment_name,
        task_id,
        dimensions_digest,
        repetition_seed
    )
)
"""

PREDICTION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_predictions_generation "
    "ON dr_dspy_predictions(experiment_name, generation_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_predictions_scoring "
    "ON dr_dspy_predictions(experiment_name, scoring_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_predictions_pipeline "
    "ON dr_dspy_predictions(experiment_name, pipeline)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_predictions_digest "
    "ON dr_dspy_predictions(experiment_name, dimensions_digest)",
)

#: Columns written when a prediction job is first created. JSONB columns
#: hold plain dicts here; the caller wraps them with ``psycopg ... Jsonb``.
CREATION_COLUMNS = (
    "prediction_id",
    "experiment_name",
    "pipeline",
    "schema_version",
    "script_kind",
    "submission_id",
    "task_id",
    "sample_index",
    "repetition_seed",
    "dimensions_digest",
    "dimensions",
    "task_inputs",
)


def eval_schema_statements() -> tuple[str, ...]:
    """Full ordered DDL for ``init-db`` (predictions + experiments +
    the shared batch-operation tables + indexes)."""
    return (
        EXPERIMENTS_TABLE_SQL,
        PREDICTIONS_TABLE_SQL,
        batch_operation.operation_table_sql(),
        batch_operation.operation_item_table_sql(),
        *PREDICTION_INDEX_SQL,
        *batch_operation.operation_index_sql(),
        *batch_operation.operation_item_index_sql(),
    )


class PredictionRow(BaseModel):
    """Typed control-plane spine + JSONB payload for one prediction."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    script_kind: StrictStr
    submission_id: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    repetition_seed: StrictInt
    dimensions_digest: StrictStr
    payload: PredictionPayload

    def creation_values(self) -> dict[str, Any]:
        """Column -> value for the creation INSERT. JSONB columns are
        plain dicts; the caller wraps with ``Jsonb``."""
        return {
            "prediction_id": self.prediction_id,
            "experiment_name": self.experiment_name,
            "pipeline": self.payload.pipeline,
            "schema_version": self.payload.schema_version,
            "script_kind": self.script_kind,
            "submission_id": self.submission_id,
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "repetition_seed": self.repetition_seed,
            "dimensions_digest": self.dimensions_digest,
            "dimensions": self.payload.dimensions,
            "task_inputs": self.payload.task_inputs,
        }


def parse_prediction_payload(columns: Mapping[str, Any]) -> PredictionPayload:
    """Rebuild a ``PredictionPayload`` from fetched columns."""
    return PredictionPayload.model_validate(
        {
            "pipeline": columns["pipeline"],
            "schema_version": columns.get("schema_version", 1),
            "dimensions": columns.get("dimensions") or {},
            "artifacts": columns.get("artifacts") or {},
            "metrics": columns.get("metrics") or {},
            "errors": columns.get("errors") or {},
            "task_inputs": columns.get("task_inputs") or {},
        }
    )

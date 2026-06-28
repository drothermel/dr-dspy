"""DDL + row IO for optimizer studies (the outer COPRO loop).

A *study* is one coordinate-ascent run over a pinned eval set under a
single ``experiment_name``. Each round proposes *candidates* (one per
instruction); a candidate is one ``GraphSpec`` addressed by its
``dimensions_digest``. The inner ``dr_dspy_predictions`` rows are
unchanged — these two tables only track the outer loop and its results
so the study is durable, resumable, and analyzable.

DB-free except for the IO helpers, mirroring ``eval_records``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_dspy import dbos_runtime as shared_dbos

STUDIES_TABLE_NAME = "dr_dspy_studies"
STUDY_CANDIDATES_TABLE_NAME = "dr_dspy_study_candidates"

STUDIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_studies (
    study_id        TEXT        PRIMARY KEY,
    experiment_name TEXT        NOT NULL,
    strategy        TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'pending',
    params          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    eval_set        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    history         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

STUDY_CANDIDATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_study_candidates (
    study_id          TEXT        NOT NULL
        REFERENCES dr_dspy_studies(study_id),
    round_index       INTEGER     NOT NULL,
    candidate_index   INTEGER     NOT NULL,
    instruction       TEXT        NOT NULL,
    dimensions_digest TEXT        NOT NULL,
    graph             JSONB       NOT NULL,
    provenance        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    val_mean_reward   DOUBLE PRECISION,
    val_coverage      INTEGER,
    val_scores        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    selected          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (study_id, round_index, candidate_index)
)
"""

STUDY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_study_candidates_digest "
    "ON dr_dspy_study_candidates(study_id, dimensions_digest)",
)


def study_schema_statements() -> tuple[str, ...]:
    return (
        STUDIES_TABLE_SQL,
        STUDY_CANDIDATES_TABLE_SQL,
        *STUDY_INDEX_SQL,
    )


class StudyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: StrictStr
    experiment_name: StrictStr
    strategy: StrictStr
    status: StrictStr = "pending"
    params: dict[str, Any] = {}
    eval_set: dict[str, Any] = {}
    history: list[dict[str, Any]] = []


class CandidateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: StrictStr
    round_index: StrictInt
    candidate_index: StrictInt
    instruction: StrictStr
    dimensions_digest: StrictStr
    graph: dict[str, Any]
    provenance: dict[str, Any] = {}
    val_mean_reward: float | None = None
    val_coverage: int | None = None
    val_scores: dict[str, Any] = {}
    selected: bool = False


# --- IO ---------------------------------------------------------------


def create_study_schema(database_url: str) -> None:
    shared_dbos.create_schema(
        database_url, statements=study_schema_statements()
    )


def upsert_study(database_url: str, study: StudyRow) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dr_dspy_studies (
                    study_id, experiment_name, strategy, status,
                    params, eval_set, history
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (study_id) DO UPDATE SET
                    experiment_name = EXCLUDED.experiment_name,
                    strategy = EXCLUDED.strategy,
                    status = EXCLUDED.status,
                    params = EXCLUDED.params,
                    eval_set = EXCLUDED.eval_set,
                    history = EXCLUDED.history,
                    updated_at = now()
                """,
                (
                    study.study_id,
                    study.experiment_name,
                    study.strategy,
                    study.status,
                    Jsonb(study.params),
                    Jsonb(study.eval_set),
                    Jsonb(study.history),
                ),
            )


def update_study_state(
    database_url: str,
    study_id: str,
    *,
    status: str | None = None,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    sets: list[str] = ["updated_at = now()"]
    params: list[Any] = []
    if status is not None:
        sets.append("status = %s")
        params.append(status)
    if history is not None:
        sets.append("history = %s")
        params.append(Jsonb([dict(item) for item in history]))
    params.append(study_id)
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"UPDATE dr_dspy_studies SET {', '.join(sets)} "
                    "WHERE study_id = %s",
                ),
                tuple(params),
            )


def upsert_candidate(database_url: str, candidate: CandidateRow) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dr_dspy_study_candidates (
                    study_id, round_index, candidate_index, instruction,
                    dimensions_digest, graph, provenance,
                    val_mean_reward, val_coverage, val_scores, selected
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (study_id, round_index, candidate_index)
                DO UPDATE SET
                    instruction = EXCLUDED.instruction,
                    dimensions_digest = EXCLUDED.dimensions_digest,
                    graph = EXCLUDED.graph,
                    provenance = EXCLUDED.provenance,
                    val_mean_reward = EXCLUDED.val_mean_reward,
                    val_coverage = EXCLUDED.val_coverage,
                    val_scores = EXCLUDED.val_scores,
                    selected = EXCLUDED.selected,
                    updated_at = now()
                """,
                (
                    candidate.study_id,
                    candidate.round_index,
                    candidate.candidate_index,
                    candidate.instruction,
                    candidate.dimensions_digest,
                    Jsonb(candidate.graph),
                    Jsonb(candidate.provenance),
                    candidate.val_mean_reward,
                    candidate.val_coverage,
                    Jsonb(candidate.val_scores),
                    candidate.selected,
                ),
            )


def read_study(database_url: str, study_id: str) -> StudyRow | None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT study_id, experiment_name, strategy, status,
                       params, eval_set, history
                FROM dr_dspy_studies WHERE study_id = %s
                """,
                (study_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return StudyRow(
        study_id=row[0],
        experiment_name=row[1],
        strategy=row[2],
        status=row[3],
        params=dict(row[4] or {}),
        eval_set=dict(row[5] or {}),
        history=list(row[6] or []),
    )


def read_candidates(
    database_url: str, study_id: str, *, round_index: int | None = None
) -> list[CandidateRow]:
    query = """
        SELECT study_id, round_index, candidate_index, instruction,
               dimensions_digest, graph, provenance, val_mean_reward,
               val_coverage, val_scores, selected
        FROM dr_dspy_study_candidates
        WHERE study_id = %s
    """
    params: list[Any] = [study_id]
    if round_index is not None:
        query += " AND round_index = %s"
        params.append(round_index)
    query += " ORDER BY round_index, candidate_index"
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
    return [
        CandidateRow(
            study_id=row[0],
            round_index=row[1],
            candidate_index=row[2],
            instruction=row[3],
            dimensions_digest=row[4],
            graph=dict(row[5] or {}),
            provenance=dict(row[6] or {}),
            val_mean_reward=row[7],
            val_coverage=row[8],
            val_scores=dict(row[9] or {}),
            selected=row[10],
        )
        for row in rows
    ]

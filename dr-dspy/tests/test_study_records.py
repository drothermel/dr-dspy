from __future__ import annotations

from dr_dspy import study_records
from dr_dspy.study_records import CandidateRow, StudyRow


def test_schema_statements_cover_tables() -> None:
    sql = "\n".join(study_records.study_schema_statements())
    assert "CREATE TABLE IF NOT EXISTS dr_dspy_studies" in sql
    assert "CREATE TABLE IF NOT EXISTS dr_dspy_study_candidates" in sql
    assert "REFERENCES dr_dspy_studies(study_id)" in sql
    assert "idx_dr_dspy_study_candidates_digest" in sql


def test_study_row_round_trip() -> None:
    row = StudyRow(
        study_id="s1",
        experiment_name="exp",
        strategy="copro",
        status="running",
        params={"breadth": 4, "depth": 3},
        eval_set={"val_ids": ["T0", "T1"]},
        history=[{"round_index": 0, "mean_reward": 0.5}],
    )
    assert StudyRow.model_validate(row.model_dump()) == row


def test_candidate_row_round_trip() -> None:
    row = CandidateRow(
        study_id="s1",
        round_index=0,
        candidate_index=2,
        instruction="do the thing",
        dimensions_digest="d33d",
        graph={"nodes": []},
        provenance={"strategy": "copro", "cost": 0.01},
        val_mean_reward=0.42,
        val_coverage=16,
        val_scores={"distribution": [0.5, 0.0]},
        selected=True,
    )
    assert CandidateRow.model_validate(row.model_dump()) == row


def test_candidate_row_defaults() -> None:
    row = CandidateRow(
        study_id="s1",
        round_index=0,
        candidate_index=0,
        instruction="x",
        dimensions_digest="d0",
        graph={},
    )
    assert row.val_mean_reward is None
    assert row.selected is False
    assert row.provenance == {}

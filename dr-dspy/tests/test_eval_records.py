from __future__ import annotations

from dr_dspy import eval_records
from dr_dspy.experiment_spec import PredictionPayload


def test_schema_statements_cover_all_tables() -> None:
    sql = "\n".join(eval_records.eval_schema_statements())
    assert "CREATE TABLE IF NOT EXISTS dr_dspy_experiments" in sql
    assert "CREATE TABLE IF NOT EXISTS dr_dspy_predictions" in sql
    assert "dr_dspy_batch_operations" in sql
    assert "dr_dspy_predictions_identity_key" in sql


def test_prediction_indexes_present() -> None:
    assert len(eval_records.PREDICTION_INDEX_SQL) == 4
    joined = "\n".join(eval_records.PREDICTION_INDEX_SQL)
    assert "dimensions_digest" in joined
    assert "generation_status" in joined


def _row() -> eval_records.PredictionRow:
    payload = PredictionPayload(
        pipeline="enc-dec",
        dimensions={"graph": {"nodes": []}},
        task_inputs={"prompt": "p", "test": "t"},
    )
    return eval_records.PredictionRow(
        prediction_id="abc123",
        experiment_name="exp",
        script_kind="humaneval_eval_dbos_v1",
        submission_id="sub",
        task_id="HumanEval/0",
        sample_index=0,
        repetition_seed=0,
        dimensions_digest="d33d",
        payload=payload,
    )


def test_creation_values_match_columns() -> None:
    values = _row().creation_values()
    assert set(values) == set(eval_records.CREATION_COLUMNS)
    assert values["pipeline"] == "enc-dec"
    assert values["schema_version"] == 1
    assert values["dimensions"] == {"graph": {"nodes": []}}
    assert values["task_inputs"]["prompt"] == "p"


def test_parse_round_trips_from_db_columns() -> None:
    payload = _row().payload
    columns = {
        "pipeline": payload.pipeline,
        "schema_version": payload.schema_version,
        "dimensions": payload.dimensions,
        "artifacts": {},
        "metrics": {},
        "errors": {},
        "task_inputs": payload.task_inputs,
    }
    parsed = eval_records.parse_prediction_payload(columns)
    assert parsed == payload

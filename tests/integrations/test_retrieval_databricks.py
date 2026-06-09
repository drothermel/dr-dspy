from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

import dspy.integrations.retrieval.databricks as databricks_rm


def test_databricks_rm_sdk_unavailable_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dspy.integrations.retrieval.databricks.is_available", lambda _: False)
    with pytest.raises(ValueError, match="databricks-sdk"):
        databricks_rm.DatabricksRM(databricks_index_name="index")


def test_databricks_rm_direct_call_returns_retrieved_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    response = {
        "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}, {"name": "source"}]},
        "result": {
            "data_array": [
                ["low", "Low score", 0.1, "a"],
                ["high", "High score", 0.9, "b"],
                ["mid", "Middle score", 0.5, "c"],
            ]
        },
    }

    def fake_query_via_requests(**kwargs: object) -> dict[str, Any]:
        assert kwargs["query_text"] == "example query"
        assert kwargs["k"] == 2
        return response

    monkeypatch.setattr("dspy.integrations.retrieval.databricks.is_available", lambda _: False)
    monkeypatch.setattr(databricks_rm.DatabricksRM, "_query_via_requests", staticmethod(fake_query_via_requests))
    auth_value = "not-a-secret"
    retriever = databricks_rm.DatabricksRM(
        databricks_index_name="index",
        databricks_endpoint="https://example.databricks.com",
        databricks_token=auth_value,
        docs_id_column_name="id",
        text_column_name="text",
        k=2,
    )
    from dspy.retrievers.types import RetrievedPassage

    result = cast("Any", asyncio.run(retriever("example query")))
    assert len(result) == 2
    assert all(isinstance(p, RetrievedPassage) for p in result)
    assert [p.long_text for p in result] == ["High score", "Middle score"]
    assert [p.pid for p in result] == ["high", "mid"]
    assert result[0].score == 0.9
    assert result[0].metadata is not None
    assert result[0].metadata.get("source") == "b"

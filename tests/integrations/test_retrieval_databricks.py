from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import dspy.integrations.retrieval.databricks as databricks_rm

if TYPE_CHECKING:
    import pytest


def test_databricks_rm_import_handles_missing_sdk_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_parent_package(name: str) -> object:
        if name == "databricks.sdk":
            raise ModuleNotFoundError("No module named 'databricks'")
        raise AssertionError(f"unexpected module lookup: {name}")

    monkeypatch.setattr(databricks_rm, "find_spec", missing_parent_package)
    assert databricks_rm._is_databricks_sdk_installed() is False


def test_databricks_rm_direct_call_preserves_prediction_shape(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(databricks_rm, "_databricks_sdk_installed", False)
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
    result = cast("Any", asyncio.run(retriever("example query")))
    assert result.docs == ["High score", "Middle score"]
    assert result.doc_ids == ["high", "mid"]
    assert result.extra_columns == [{"score": 0.9, "source": "b"}, {"score": 0.5, "source": "c"}]

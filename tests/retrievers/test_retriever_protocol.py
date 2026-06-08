from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from dspy.retrievers.embeddings import Embeddings

if TYPE_CHECKING:
    from dspy.primitives.prediction import Prediction
    from dspy.retrievers.types import QueryRetriever


def _embedder(texts: list[str]) -> object:
    np = pytest.importorskip("numpy")
    return np.eye(len(texts), 3, dtype=np.float32)


def test_query_retriever_protocol_documents_direct_call_shape() -> None:
    def search(retriever: QueryRetriever[str, Prediction], query: str) -> Prediction:
        return retriever(query)

    retriever = Embeddings(corpus=["alpha", "beta", "gamma"], embedder=_embedder, k=1)

    result = search(retriever, "alpha")

    assert result.passages == ["alpha"]


def test_query_retriever_protocol_is_explicitly_exported() -> None:
    import dspy.retrievers.types as retriever_types

    assert retriever_types.__all__ == ["QueryRetriever"]
    assert retriever_types.QueryRetriever.__name__ == "QueryRetriever"


def test_databricks_rm_import_handles_missing_sdk_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    import dspy.retrievers.databricks_rm as databricks_rm

    def missing_parent_package(name: str) -> object:
        if name == "databricks.sdk":
            raise ModuleNotFoundError("No module named 'databricks'")
        raise AssertionError(f"unexpected module lookup: {name}")

    monkeypatch.setattr(databricks_rm, "find_spec", missing_parent_package)

    assert databricks_rm._is_databricks_sdk_installed() is False


def test_databricks_rm_direct_call_preserves_prediction_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import dspy.retrievers.databricks_rm as databricks_rm

    response = {
        "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}, {"name": "source"}]},
        "result": {
            "data_array": [
                ["low", "Low score", 0.1, "a"],
                ["high", "High score", 0.9, "b"],
                ["mid", "Middle score", 0.5, "c"],
            ],
        },
    }

    def fake_query_via_requests(**kwargs: object) -> dict[str, object]:
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

    result = retriever("example query")

    assert result.docs == ["High score", "Middle score"]
    assert result.doc_ids == ["high", "mid"]
    assert result.extra_columns == [{"score": 0.9, "source": "b"}, {"score": 0.5, "source": "c"}]


def test_weaviate_rm_direct_call_preserves_long_text_shape() -> None:
    pytest.importorskip("weaviate")
    from dspy.retrievers.weaviate_rm import WeaviateRM

    class FakeQuery:
        def __init__(self) -> None:
            self.query_text = None
            self.limit = None

        def hybrid(self, query: str, limit: int, **kwargs: object) -> SimpleNamespace:
            self.query_text = query
            self.limit = limit
            return SimpleNamespace(
                objects=[
                    SimpleNamespace(properties={"content": "First passage"}),
                    SimpleNamespace(properties={"content": "Second passage"}),
                ],
            )

    class FakeCollection:
        def __init__(self) -> None:
            self.query = FakeQuery()

    class FakeCollections:
        def __init__(self, collection: FakeCollection) -> None:
            self.collection = collection

        def get(self, name: str) -> FakeCollection:
            assert name == "collection"
            return self.collection

    class FakeClient:
        def __init__(self, collection: FakeCollection) -> None:
            self.collections = FakeCollections(collection)

    collection = FakeCollection()
    retriever = WeaviateRM("collection", weaviate_client=FakeClient(collection), k=1)

    result = retriever("question")

    assert [passage.long_text for passage in result] == ["First passage", "Second passage"]
    assert collection.query.query_text == "question"
    assert collection.query.limit == 1

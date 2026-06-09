from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dspy.integrations.retrieval.weaviate import WeaviateRM


def test_weaviate_module_does_not_eagerly_import_weaviate() -> None:
    import dspy.integrations.retrieval.weaviate as weaviate_mod

    assert weaviate_mod.WeaviateRM is not None
    assert "weaviate" not in weaviate_mod.__dict__


def test_weaviate_rm_direct_call_preserves_long_text_shape() -> None:
    pytest.importorskip("weaviate")

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
                ]
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
    retriever = WeaviateRM("collection", weaviate_client=cast("Any", FakeClient(collection)), k=1)
    result = asyncio.run(retriever("question"))
    assert [passage.long_text for passage in result] == ["First passage", "Second passage"]
    assert collection.query.query_text == "question"
    assert collection.query.limit == 1

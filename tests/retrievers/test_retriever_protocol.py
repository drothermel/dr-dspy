from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from dspy.retrievers.embeddings import Embeddings
from dspy.retrievers.types import RetrievedPassage

if TYPE_CHECKING:
    from dspy.retrievers.types import QueryRetriever


def _embedder(texts: list[str]) -> object:
    np = pytest.importorskip("numpy")
    return np.eye(len(texts), 3, dtype=np.float32)


def test_query_retriever_protocol_documents_direct_call_shape() -> None:

    async def search(retriever: QueryRetriever[str], query: str) -> list[RetrievedPassage]:
        return await retriever(query)

    retriever = Embeddings(corpus=["alpha", "beta", "gamma"], embedder=_embedder, k=1)
    result = asyncio.run(search(retriever, "alpha"))
    assert len(result) == 1
    assert isinstance(result[0], RetrievedPassage)
    assert result[0].long_text == "alpha"


def test_query_retriever_protocol_is_explicitly_exported() -> None:
    import dspy.retrievers.types as retriever_types

    assert retriever_types.__all__ == ["QueryRetriever", "RetrievedPassage"]
    assert retriever_types.QueryRetriever.__name__ == "QueryRetriever"

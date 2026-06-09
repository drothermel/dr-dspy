import asyncio
import importlib.util
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import numpy as np
except ImportError:
    pytest.skip(reason="numpy is not installed", allow_module_level=True)
if importlib.util.find_spec("litellm") is None:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.clients.embedding import Embedder


class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [{"embedding": emb} for emb in embeddings]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model = "mock_model"
        self.object = "list"


def test_litellm_embedding():
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch("dspy.clients.embedding._get_litellm") as mock_get_litellm:
        mock_litellm = MagicMock()
        mock_get_litellm.return_value = mock_litellm
        mock_litellm.aembedding = AsyncMock(return_value=MockEmbeddingResponse(mock_embeddings))
        embedding = Embedder(model)
        result = asyncio.run(embedding(inputs))
        mock_litellm.aembedding.assert_called_once_with(model=model, input=inputs, caching=False)
        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)
        asyncio.run(embedding(inputs))
        assert mock_litellm.aembedding.call_count == 2


def test_callable_embedding():
    inputs = ["hello", "world", "test"]
    expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    class EmbeddingFn:
        def __init__(self):
            self.call_count = 0

        def __call__(self, texts):
            self.call_count += 1
            return expected_embeddings

    embedding_fn = EmbeddingFn()
    embedding = Embedder(embedding_fn)
    result = asyncio.run(embedding(inputs))
    assert embedding_fn.call_count == 1
    np.testing.assert_allclose(result, expected_embeddings)
    asyncio.run(embedding(inputs))
    assert embedding_fn.call_count == 2


def test_invalid_model_type():
    embedding = cast("Any", Embedder)(123)
    with pytest.raises(ValueError, match=r"must be a string or a callable"):
        asyncio.run(embedding(["test"]))


@pytest.mark.asyncio
async def test_async_embedding():
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch("dspy.clients.embedding._get_litellm") as mock_get_litellm:
        mock_litellm = MagicMock()
        mock_get_litellm.return_value = mock_litellm

        async def aembedding(**kwargs):
            return MockEmbeddingResponse(mock_embeddings)

        mock_litellm.aembedding = aembedding
        embedding = Embedder(model)
        result = await embedding(inputs)
        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)

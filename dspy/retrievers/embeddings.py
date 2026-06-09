"""Embedding-based corpus retriever with optional FAISS index.

Import ``Embeddings`` and ``EmbeddingsWithScores`` from ``dspy.retrievers.embeddings``.
Call ``close()`` when finished to stop the background batch worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType

from typing_extensions import override

from dspy._internal.lazy_import import require
from dspy._internal.unbatchify import Unbatchify
from dspy.persistence.embeddings import load_embeddings_into, save_embeddings
from dspy.retrievers.types import RetrievedPassage

np = require("numpy")

__all__ = ["Embeddings", "EmbeddingsWithScores"]


class FaissIndex(Protocol):
    def search(self, query_embeddings: np.ndarray, num_candidates: int) -> tuple[np.ndarray, np.ndarray]: ...


Embedder = Callable[[list[str]], np.ndarray]
SearchResult = tuple[list[str], list[int], list[float]]


def _search_result_to_passages(
    passages: list[str],
    indices: list[int],
    scores: list[float] | None = None,
) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            long_text=text,
            pid=pid,
            score=scores[i] if scores is not None else None,
        )
        for i, (text, pid) in enumerate(zip(passages, indices, strict=True))
    ]


class Embeddings:
    """Embedding retriever with optional FAISS index.

    Prefer ``with Embeddings(...) as retriever:`` or call ``close()`` when the
    retriever is no longer needed to stop the background ``Unbatchify`` worker thread.
    """

    def __init__(
        self,
        corpus: list[str],
        embedder: Embedder,
        k: int = 5,
        brute_force_threshold: int = 20000,
        normalize: bool = True,
    ) -> None:
        self.embedder = embedder
        self.k = k
        self.corpus = corpus
        self.normalize = normalize
        self.corpus_embeddings = self.embedder(self.corpus)
        self.corpus_embeddings = self._normalize(self.corpus_embeddings) if self.normalize else self.corpus_embeddings
        self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None
        self.search_fn = Unbatchify(self._batch_forward)

    async def __call__(self, query: str) -> list[RetrievedPassage]:
        return await self.aforward(query)

    async def aforward(self, query: str) -> list[RetrievedPassage]:
        passages, indices, _scores = await asyncio.to_thread(self.search_fn, query)
        return _search_result_to_passages(passages, indices)

    def close(self) -> None:
        self.search_fn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _batch_forward(self, queries: list[str]) -> list[SearchResult]:
        q_embeds = self.embedder(queries)
        q_embeds = self._normalize(q_embeds) if self.normalize else q_embeds
        pids = self._faiss_search(query_embeddings=q_embeds, num_candidates=self.k * 10) if self.index else None
        pids = np.tile(np.arange(len(self.corpus)), (len(queries), 1)) if pids is None else pids
        return self._rerank_and_predict(q_embeds=q_embeds, candidate_indices=pids)

    def _build_faiss(self) -> FaissIndex:
        nbytes = 32
        partitions = int(2 * np.sqrt(len(self.corpus)))
        dim = self.corpus_embeddings.shape[1]
        try:
            import faiss
        except ImportError:
            raise ImportError("Please `pip install faiss-cpu` or increase `brute_force_threshold` to avoid FAISS.")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, partitions, nbytes, 8)
        index.train(self.corpus_embeddings)
        index.add(self.corpus_embeddings)
        index.nprobe = min(16, partitions)
        return index

    def _faiss_search(self, query_embeddings: np.ndarray, num_candidates: int) -> np.ndarray:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")
        return self.index.search(query_embeddings, num_candidates)[1]

    def _rerank_and_predict(self, q_embeds: np.ndarray, candidate_indices: np.ndarray) -> list[SearchResult]:
        candidate_embeddings = self.corpus_embeddings[candidate_indices]
        scores = np.einsum("qd,qkd->qk", q_embeds, candidate_embeddings)
        top_k_indices = np.argsort(-scores, axis=1)[:, : self.k]
        top_indices = candidate_indices[np.arange(len(q_embeds))[:, None], top_k_indices]
        top_scores = scores[np.arange(len(q_embeds))[:, None], top_k_indices]
        results = []
        for indices, query_scores in zip(top_indices, top_scores, strict=True):
            passages = [self.corpus[idx] for idx in indices]
            results.append((passages, indices.tolist(), query_scores.tolist()))
        return results

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)

    def save(self, path: str) -> None:
        save_embeddings(self, path)

    def load(self, path: str, embedder: Embedder) -> Embeddings:
        return load_embeddings_into(self, path, embedder=embedder)


class EmbeddingsWithScores(Embeddings):
    @override
    async def aforward(self, query: str) -> list[RetrievedPassage]:
        passages, indices, scores = await asyncio.to_thread(self.search_fn, query)
        return _search_result_to_passages(passages, indices, scores)

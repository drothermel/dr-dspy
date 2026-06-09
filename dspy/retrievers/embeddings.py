from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import srsly
from typing_extensions import override

from dspy.primitives.prediction import Prediction
from dspy.utils.lazy_import import require
from dspy.utils.unbatchify import Unbatchify

np = require("numpy")


class FaissIndex(Protocol):
    def search(self, query_embeddings: np.ndarray, num_candidates: int) -> tuple[np.ndarray, np.ndarray]: ...


Embedder = Callable[[list[str]], np.ndarray]
SearchResult = tuple[list[str], list[int], list[float]]


class Embeddings:
    """Embedding retriever with optional FAISS index.

    Call ``close()`` when the retriever is no longer needed to stop the
    background ``Unbatchify`` worker thread.
    """

    def __init__(
        self,
        corpus: list[str],
        embedder: Embedder,
        k: int = 5,
        callbacks: list[object] | None = None,
        brute_force_threshold: int = 20000,
        normalize: bool = True,
    ) -> None:
        self.callbacks = callbacks or []
        self.embedder = embedder
        self.k = k
        self.corpus = corpus
        self.normalize = normalize
        self.corpus_embeddings = self.embedder(self.corpus)
        self.corpus_embeddings = self._normalize(self.corpus_embeddings) if self.normalize else self.corpus_embeddings
        self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None
        self.search_fn = Unbatchify(self._batch_forward)

    async def __call__(self, query: str) -> Prediction:
        return await self.aforward(query)

    async def aforward(self, query: str) -> Prediction:
        passages, indices, _scores = await asyncio.to_thread(self.search_fn, query)
        return Prediction(passages=passages, indices=indices)

    def close(self) -> None:
        self.search_fn.close()

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
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        config = {
            "k": self.k,
            "normalize": self.normalize,
            "corpus": self.corpus,
            "has_faiss_index": self.index is not None,
        }
        srsly.write_json(save_path / "config.json", config)
        np.save(save_path / "corpus_embeddings.npy", self.corpus_embeddings)
        if self.index is not None:
            try:
                import faiss

                faiss.write_index(self.index, str(save_path / "faiss_index.bin"))
            except ImportError:
                pass

    def load(self, path: str, embedder: Embedder) -> Embeddings:
        save_path = Path(path)
        if not save_path.exists():
            raise FileNotFoundError(f"Save directory not found: {path}")
        config_path = save_path / "config.json"
        embeddings_path = save_path / "corpus_embeddings.npy"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        config = cast("dict[str, Any]", srsly.read_json(config_path))
        required_fields = ["k", "normalize", "corpus", "has_faiss_index"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Invalid config: missing required field '{field}'")
        self.k = config["k"]
        self.normalize = config["normalize"]
        self.corpus = config["corpus"]
        self.embedder = embedder
        self.corpus_embeddings = np.load(embeddings_path)
        faiss_index_path = save_path / "faiss_index.bin"
        if config["has_faiss_index"] and faiss_index_path.exists():
            try:
                import faiss

                self.index = faiss.read_index(str(faiss_index_path))
            except ImportError:
                self.index = None
        else:
            self.index = None
        return self

    @classmethod
    def from_saved(cls, path: str, embedder: Embedder) -> Embeddings:
        instance = cls.__new__(cls)
        instance.search_fn = Unbatchify(instance._batch_forward)
        instance.load(path=path, embedder=embedder)
        return instance


class EmbeddingsWithScores(Embeddings):
    @override
    async def aforward(self, query: str) -> Prediction:
        passages, indices, scores = await asyncio.to_thread(self.search_fn, query)
        return Prediction(passages=passages, indices=indices, scores=scores)

"""Retriever artifact directory persistence for embedding-based search."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import srsly

from dspy._internal.lazy_import import require
from dspy._internal.unbatchify import Unbatchify

if TYPE_CHECKING:
    from dspy.retrievers.embeddings import Embedder, Embeddings

np = require("numpy")

CONFIG_FILENAME = "config.json"
CORPUS_EMBEDDINGS_FILENAME = "corpus_embeddings.npy"
FAISS_INDEX_FILENAME = "faiss_index.bin"

REQUIRED_CONFIG_FIELDS = ("k", "normalize", "corpus", "has_faiss_index")


def save_embeddings(retriever: Embeddings, path: str | Path) -> None:
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    config = {
        "k": retriever.k,
        "normalize": retriever.normalize,
        "corpus": retriever.corpus,
        "has_faiss_index": retriever.index is not None,
    }
    srsly.write_json(save_path / CONFIG_FILENAME, config)
    np.save(save_path / CORPUS_EMBEDDINGS_FILENAME, retriever.corpus_embeddings)
    if retriever.index is not None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS index is configured but `faiss-cpu` is not installed. "
                "Install faiss-cpu or rebuild without a FAISS index."
            ) from exc
        faiss.write_index(retriever.index, str(save_path / FAISS_INDEX_FILENAME))


def load_embeddings_into(
    retriever: Embeddings,
    path: str | Path,
    *,
    embedder: Embedder,
) -> Embeddings:
    if hasattr(retriever, "search_fn"):
        retriever.search_fn.close()

    save_path = Path(path)
    if not save_path.exists():
        raise FileNotFoundError(f"Save directory not found: {path}")
    config_path = save_path / CONFIG_FILENAME
    embeddings_path = save_path / CORPUS_EMBEDDINGS_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    config = cast("dict[str, Any]", srsly.read_json(config_path))
    for field in REQUIRED_CONFIG_FIELDS:
        if field not in config:
            raise ValueError(f"Invalid config: missing required field '{field}'")
    retriever.k = config["k"]
    retriever.normalize = config["normalize"]
    retriever.corpus = config["corpus"]
    retriever.embedder = embedder
    retriever.corpus_embeddings = np.load(embeddings_path)
    faiss_index_path = save_path / FAISS_INDEX_FILENAME
    if config["has_faiss_index"]:
        if not faiss_index_path.exists():
            raise FileNotFoundError(f"Saved config expects a FAISS index but file not found: {faiss_index_path}")
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "Saved embeddings require FAISS but `faiss-cpu` is not installed. Install faiss-cpu to load this index."
            ) from exc
        retriever.index = faiss.read_index(str(faiss_index_path))
    else:
        retriever.index = None
    retriever.search_fn = Unbatchify(retriever._batch_forward)
    return retriever


def load_embeddings(
    path: str | Path,
    *,
    embedder: Embedder,
    retriever_cls: type[Embeddings] | None = None,
) -> Embeddings:
    from dspy.retrievers.embeddings import Embeddings as EmbeddingsCls

    cls = retriever_cls or EmbeddingsCls
    instance = cls.__new__(cls)
    instance.search_fn = Unbatchify(instance._batch_forward)
    return load_embeddings_into(instance, path, embedder=embedder)

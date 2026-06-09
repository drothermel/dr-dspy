import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import srsly

try:
    import numpy as np
except ImportError:
    pytest.skip(reason="numpy is not installed", allow_module_level=True)
from dspy.persistence import load_embeddings
from dspy.retrievers.embeddings import Embeddings, EmbeddingsWithScores
from dspy.retrievers.types import RetrievedPassage


def dummy_corpus():
    return ["The cat sat on the mat.", "The dog barked at the mailman.", "Birds fly in the sky."]


def dummy_embedder(texts):
    embeddings = []
    for text in texts:
        if "cat" in text:
            embeddings.append(np.array([1, 0, 0], dtype=np.float32))
        elif "dog" in text:
            embeddings.append(np.array([0, 1, 0], dtype=np.float32))
        else:
            embeddings.append(np.array([0, 0, 1], dtype=np.float32))
    return np.stack(embeddings)


def test_embeddings_basic_search():
    corpus = dummy_corpus()
    embedder = dummy_embedder
    with Embeddings(corpus=corpus, embedder=embedder, k=1) as retriever:
        query = "I saw a dog running."
        result = asyncio.run(retriever(query))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], RetrievedPassage)
        assert result[0].long_text == "The dog barked at the mailman."
        assert result[0].pid == 1


def test_embeddings_multithreaded_search():
    corpus = dummy_corpus()
    embedder = dummy_embedder
    with Embeddings(corpus=corpus, embedder=embedder, k=1) as retriever:
        queries = [
            ("A cat is sitting on the mat.", "The cat sat on the mat."),
            ("My dog is awesome!", "The dog barked at the mailman."),
            ("Birds flying high.", "Birds fly in the sky."),
        ] * 10

        def worker(query_text, expected_passage):
            result = asyncio.run(retriever(query_text))
            assert result[0].long_text == expected_passage
            return result[0].long_text

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, q, expected) for q, expected in queries]
            results = [f.result() for f in futures]
            assert results[0] == "The cat sat on the mat."
            assert results[1] == "The dog barked at the mailman."
            assert results[2] == "Birds fly in the sky."


def test_embeddings_save_load():
    corpus = dummy_corpus()
    embedder = dummy_embedder
    with (
        Embeddings(
            corpus=corpus, embedder=embedder, k=2, normalize=False, brute_force_threshold=1000
        ) as original_retriever,
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        save_path = os.path.join(temp_dir, "test_embeddings")
        original_retriever.save(save_path)
        assert os.path.exists(os.path.join(save_path, "config.json"))
        assert os.path.exists(os.path.join(save_path, "corpus_embeddings.npy"))
        assert not os.path.exists(os.path.join(save_path, "faiss_index.bin"))
        with Embeddings(
            corpus=["dummy"], embedder=embedder, k=1, normalize=True, brute_force_threshold=500
        ) as new_retriever:
            new_retriever.load(save_path, embedder)
            assert new_retriever.corpus == corpus
            assert new_retriever.k == 2
            assert new_retriever.normalize is False
            assert new_retriever.embedder == embedder
            assert new_retriever.index is None
            query = "cat sitting"
            original_result = asyncio.run(original_retriever(query))
            loaded_result = asyncio.run(new_retriever(query))
            assert [p.long_text for p in loaded_result] == [p.long_text for p in original_result]
            assert [p.pid for p in loaded_result] == [p.pid for p in original_result]


def test_embeddings_from_saved():
    corpus = dummy_corpus()
    embedder = dummy_embedder
    with (
        Embeddings(
            corpus=corpus, embedder=embedder, k=3, normalize=True, brute_force_threshold=1000
        ) as original_retriever,
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        save_path = os.path.join(temp_dir, "test_embeddings")
        original_retriever.save(save_path)
        with load_embeddings(save_path, embedder=embedder) as loaded_retriever:
            assert loaded_retriever.k == original_retriever.k
            assert loaded_retriever.normalize == original_retriever.normalize
            assert loaded_retriever.corpus == original_retriever.corpus


def test_embeddings_load_nonexistent_path():
    with pytest.raises((FileNotFoundError, OSError)):
        load_embeddings("/nonexistent/path", embedder=dummy_embedder)


def test_embeddings_with_scores_basic_search():
    corpus = dummy_corpus()
    with EmbeddingsWithScores(corpus=corpus, embedder=dummy_embedder, k=2) as retriever:
        result = asyncio.run(retriever("A dog is barking."))
        assert [p.long_text for p in result] == ["The dog barked at the mailman.", "The cat sat on the mat."]
        assert [p.pid for p in result] == [1, 0]
        assert [p.score for p in result] == pytest.approx([1.0, 0.0])


def test_embeddings_with_scores_save_load():
    corpus = dummy_corpus()
    with (
        EmbeddingsWithScores(
            corpus=corpus, embedder=dummy_embedder, k=2, normalize=False, brute_force_threshold=1000
        ) as original_retriever,
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        save_path = os.path.join(temp_dir, "test_embeddings_with_scores")
        original_retriever.save(save_path)
        with load_embeddings(
            save_path, embedder=dummy_embedder, retriever_cls=EmbeddingsWithScores
        ) as loaded_retriever:
            original_result = asyncio.run(original_retriever("cat sitting"))
            loaded_result = asyncio.run(loaded_retriever("cat sitting"))
            assert [p.long_text for p in loaded_result] == [p.long_text for p in original_result]
            assert [p.pid for p in loaded_result] == [p.pid for p in original_result]
            assert [p.score for p in loaded_result] == pytest.approx([p.score for p in original_result])


def test_embeddings_load_closes_existing_search_fn():
    corpus = dummy_corpus()
    with Embeddings(corpus=corpus, embedder=dummy_embedder, k=1, brute_force_threshold=1000) as retriever:
        old_search_fn = retriever.search_fn
        with patch.object(old_search_fn, "close") as mock_close, tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "embeddings")
            retriever.save(save_path)
            retriever.load(save_path, dummy_embedder)
            mock_close.assert_called_once()


def test_embeddings_load_raises_when_faiss_index_file_missing(tmp_path):
    corpus = dummy_corpus()
    save_path = tmp_path / "embeddings"
    save_path.mkdir()
    config = {
        "k": 1,
        "normalize": True,
        "corpus": corpus,
        "has_faiss_index": True,
    }
    srsly.write_json(save_path / "config.json", config)
    np.save(save_path / "corpus_embeddings.npy", dummy_embedder(corpus))
    with pytest.raises(FileNotFoundError, match="FAISS index"):
        load_embeddings(str(save_path), embedder=dummy_embedder)


def test_embeddings_load_raises_when_faiss_not_installed(tmp_path, monkeypatch):
    import builtins

    corpus = dummy_corpus()
    save_path = tmp_path / "embeddings"
    save_path.mkdir()
    config = {
        "k": 1,
        "normalize": True,
        "corpus": corpus,
        "has_faiss_index": True,
    }
    srsly.write_json(save_path / "config.json", config)
    np.save(save_path / "corpus_embeddings.npy", dummy_embedder(corpus))
    (save_path / "faiss_index.bin").write_bytes(b"not-a-real-index")

    real_import = builtins.__import__

    def _raise_import_error(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "faiss":
            raise ImportError("no faiss")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_import_error)
    with pytest.raises(ImportError, match="faiss-cpu"):
        load_embeddings(str(save_path), embedder=dummy_embedder)

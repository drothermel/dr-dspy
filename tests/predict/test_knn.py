import pytest

from dspy.clients.embedding import Embedder
from dspy.predict.knn import KNN
from dspy.primitives import Example
from dspy.testing import DummyVectorizer

pytestmark = [pytest.mark.extra, pytest.mark.asyncio]


def mock_example(question: str, answer: str) -> Example:
    return Example.from_record({"question": question, "answer": answer}, input_keys=("question",))


@pytest.fixture
async def setup_knn() -> KNN:
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    knn = KNN(k=2, trainset=trainset, vectorizer=Embedder(DummyVectorizer()))
    await knn._ensure_train_vectors()
    return knn


async def test_knn_initialization(setup_knn):
    import numpy as np

    knn = setup_knn
    assert knn.k == 2, "Incorrect k value"
    assert len(knn.trainset_vectors) == 3, "Incorrect size of trainset vectors"
    assert isinstance(knn.trainset_vectors, np.ndarray), "Trainset vectors should be a NumPy array"


async def test_knn_query(setup_knn):
    knn = setup_knn
    query = {"question": "What is 3+3?"}
    nearest_samples = await knn(inputs=query)
    assert len(nearest_samples) == 2, "Incorrect number of nearest samples returned"
    assert nearest_samples[0].answer == "4", "Incorrect nearest sample returned"


async def test_knn_query_specificity(setup_knn):
    knn = setup_knn
    query = {"question": "What is the capital of Germany?"}
    nearest_samples = await knn(inputs=query)
    assert len(nearest_samples) == 2, "Incorrect number of nearest samples returned"
    assert "Paris" in [sample.answer for sample in nearest_samples], "Expected Paris to be a nearest sample answer"

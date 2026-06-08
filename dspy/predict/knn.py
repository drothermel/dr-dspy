from dspy.clients.embedding import Embedder
from dspy.primitives.example import Example
from dspy.utils.lazy_import import require

np = require("numpy")


def _embed_vectors(vectorizer: Embedder, texts: list[str]) -> np.ndarray:
    import asyncio

    return asyncio.run(vectorizer(texts))


class KNN:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder) -> None:
        self.k = k
        self.trainset = trainset
        self.embedding = vectorizer
        trainset_casted_to_vectorize = []
        for example in self.trainset:
            input_keys = set(example._input_keys or [])
            trainset_casted_to_vectorize.append(
                " | ".join([f"{key}: {value}" for key, value in example.items() if key in input_keys])
            )
        self.trainset_vectors = _embed_vectors(self.embedding, trainset_casted_to_vectorize).astype(np.float32)

    def __call__(self, **kwargs) -> list:
        input_example_vector = _embed_vectors(
            self.embedding, [" | ".join([f"{key}: {val}" for key, val in kwargs.items()])]
        )
        scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
        nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
        return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]

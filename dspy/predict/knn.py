from collections.abc import Mapping
from typing import Any

from dspy._internal.lazy_import import require
from dspy.clients.embedding import Embedder
from dspy.primitives import Example

np = require("numpy")


def _format_input_text(inputs: Mapping[str, Any], input_keys: frozenset[str]) -> str:
    ordered_keys = [key for key in inputs if key in input_keys] if input_keys else list(inputs)
    return " | ".join(f"{key}: {inputs[key]}" for key in ordered_keys)


class KNN:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder) -> None:
        self.k = k
        self.trainset = trainset
        self.embedding = vectorizer
        trainset_casted_to_vectorize = []
        for example in self.trainset:
            input_keys = example.input_keys
            trainset_casted_to_vectorize.append(_format_input_text(dict(example.items()), input_keys))
        self._train_vectors = trainset_casted_to_vectorize

    async def _ensure_train_vectors(self) -> np.ndarray:
        if not hasattr(self, "trainset_vectors"):
            vectors = await self.embedding(self._train_vectors)
            self.trainset_vectors = np.asarray(vectors, dtype=np.float32)
        return self.trainset_vectors

    async def __call__(self, *, inputs: Mapping[str, Any]) -> list[Example]:
        trainset_vectors = await self._ensure_train_vectors()
        input_example_vector = np.asarray(
            await self.embedding([_format_input_text(inputs, frozenset(inputs))]), dtype=np.float32
        )
        scores = np.dot(trainset_vectors, input_example_vector.T).squeeze()
        nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
        return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]

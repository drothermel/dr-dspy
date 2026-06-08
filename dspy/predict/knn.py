from dspy.clients.embedding import Embedder
from dspy.primitives.example import Example
from dspy.utils.lazy_import import require

np = require("numpy")


class KNN:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder) -> None:
        """
        A k-nearest neighbors retriever that finds similar examples from a training set.

        Args:
            k: Number of nearest neighbors to retrieve
            trainset: List of training examples to search through
            vectorizer: The `Embedder` to use for vectorization

        Examples:
            ```python
            from dspy.clients.embedding import Embedder
            from dspy.primitives.example import Example
            from sentence_transformers import SentenceTransformer

            # Create a training dataset with examples
            trainset = [
                Example(input="hello", output="world"),
                # ... more examples ...
            ]

            # Initialize KNN with a sentence transformer model
            knn = KNN(
                k=3,
                trainset=trainset,
                vectorizer=Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
            )

            # Find similar examples
            similar_examples = knn(input="hello")
            ```
        """
        self.k = k
        self.trainset = trainset
        self.embedding = vectorizer
        trainset_casted_to_vectorize = []
        for example in self.trainset:
            input_keys = set(example._input_keys or [])
            trainset_casted_to_vectorize.append(
                " | ".join([f"{key}: {value}" for key, value in example.items() if key in input_keys])
            )
        self.trainset_vectors = self.embedding(trainset_casted_to_vectorize).astype(np.float32)

    def __call__(self, **kwargs) -> list:
        input_example_vector = self.embedding([" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
        scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
        nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
        return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]

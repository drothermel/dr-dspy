"""Test-only retrieval helpers."""

from __future__ import annotations

import random

from dspy.utils.lazy_import import require

np = require("numpy")


class DummyVectorizer:
    def __init__(self, max_length=100, n_gram=2) -> None:
        self.max_length = max_length
        self.n_gram = n_gram
        self.P = 10**9 + 7
        random.seed(123)
        self.coeffs = [random.randrange(1, self.P) for _ in range(n_gram)]

    def _hash(self, gram):
        h = 1
        for coeff, c in zip(self.coeffs, gram, strict=False):
            h = h * coeff + ord(c)
            h %= self.P
        return h % self.max_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            grams = [text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)]
            vec = [0] * self.max_length
            for gram in grams:
                vec[self._hash(gram)] += 1
            vecs.append(vec)
        vecs = np.array(vecs, dtype=np.float32)
        vecs -= np.mean(vecs, axis=1, keepdims=True)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs

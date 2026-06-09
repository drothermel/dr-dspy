import random

from dspy.testing import DummyVectorizer


def test_dummy_vectorizer_does_not_mutate_global_rng():
    state_before = random.getstate()
    DummyVectorizer()
    state_after = random.getstate()
    assert state_before == state_after

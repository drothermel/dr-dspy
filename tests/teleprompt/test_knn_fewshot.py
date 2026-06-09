import asyncio
from typing import Any, cast

import pytest
from typing_extensions import override

from dspy.clients.embedding import Embedder
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.compile_params import KNNFewShotCompileParams
from dspy.teleprompt.knn_fewshot import KNNFewShot
from dspy.testing import DummyLM, DummyVectorizer
from tests.task_spec.helpers import ts


def mock_example(question: str, answer: str) -> Example:
    return Example.from_record({"question": question, "answer": answer}, input_keys=("question",))


@pytest.fixture
def setup_knn_few_shot() -> KNNFewShot:
    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    return KNNFewShot(k=2, trainset=trainset, vectorizer=Embedder(DummyVectorizer()))


def test_knn_few_shot_initialization(setup_knn_few_shot, make_run):
    knn_few_shot = setup_knn_few_shot
    assert knn_few_shot.knn.k == 2, "Incorrect k value for KNN"
    assert len(knn_few_shot.knn.trainset) == 3, "Incorrect trainset size for KNN"


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, *args: object, **kwargs: object):
        return self.predictor(**kwargs)

    @override
    def reset_copy(self):
        return SimpleModule(self.predictor.task_spec)


def _test_knn_few_shot_compile(setup_knn_few_shot, make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM(cast("Any", ["Madrid", "10"]))
    run = make_run(lm=lm)
    knn_few_shot = setup_knn_few_shot
    compile_result = asyncio.run(
        knn_few_shot.compile(student, params=KNNFewShotCompileParams(teacher=teacher), run=run)
    )
    compiled_student = compile_result.program
    output = asyncio.run(compiled_student(input="What is the capital of Spain?", run=run))
    assert output.output in ["Madrid", "10"], (
        "The compiled student did not return the correct output based on the query"
    )

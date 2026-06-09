import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from typing_extensions import override

from dspy.clients.embedding import Embedder
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.bootstrap import BootstrapFewShot
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


def test_knn_few_shot_initialization(setup_knn_few_shot):
    knn_few_shot = setup_knn_few_shot
    assert knn_few_shot.k == 2
    assert knn_few_shot.knn.k == 2, "Incorrect k value for KNN"
    assert len(knn_few_shot.knn.trainset) == 3, "Incorrect trainset size for KNN"


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)

    @override
    def reset_copy(self):
        copied = SimpleModule(self.predictor.task_spec)
        if self.predictor.lm is not None:
            copied.predictor.set_lm(self.predictor.lm)
        return copied


def test_knn_few_shot_forward_uses_neighbors(setup_knn_few_shot, make_run):
    student = SimpleModule(ts("question -> answer"))
    lm = DummyLM([{"answer": "Madrid"}, {"answer": "10"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    knn_few_shot = setup_knn_few_shot
    compile_result = asyncio.run(knn_few_shot.compile(student, params=KNNFewShotCompileParams(), run=run))
    compiled_student = compile_result.program
    asyncio.run(compiled_student(question="What is the capital of Spain?", run=run))
    assert len(compiled_student.predictor.demos) == 2
    assert all(isinstance(demo, Example) for demo in compiled_student.predictor.demos)


def test_knn_does_not_bootstrap_on_forward(setup_knn_few_shot, make_run):
    student = SimpleModule(ts("question -> answer"))
    lm = DummyLM([{"answer": "Madrid"}, {"answer": "Paris"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    knn_few_shot = setup_knn_few_shot
    with patch.object(BootstrapFewShot, "compile", new_callable=AsyncMock) as mock_bootstrap:
        compile_result = asyncio.run(knn_few_shot.compile(student, params=KNNFewShotCompileParams(), run=run))
        compiled_student = compile_result.program
        asyncio.run(compiled_student(question="What is the capital of Spain?", run=run))
        asyncio.run(compiled_student(question="What is 2+2?", run=run))
        mock_bootstrap.assert_not_called()

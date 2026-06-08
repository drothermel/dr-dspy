import types
from typing import Any

from typing_extensions import override

from dspy.clients.embedding import Embedder
from dspy.predict.knn import KNN
from dspy.primitives.example import Example
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.teleprompt import Teleprompter


class KNNFewShot(Teleprompter):
    def __init__(
        self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any]
    ) -> None:
        self.KNN = KNN(k, trainset, vectorizer=vectorizer)
        self.few_shot_bootstrap_args = few_shot_bootstrap_args

    @override
    async def compile(self, student, *, teacher=None):
        student_copy = student.reset_copy()
        knn_few_shot = self

        async def aforward_pass(_, **kwargs):
            knn_trainset = knn_few_shot.KNN(**kwargs)
            few_shot_bootstrap = BootstrapFewShot(**knn_few_shot.few_shot_bootstrap_args)
            compiled_program = await few_shot_bootstrap.compile(student, teacher=teacher, trainset=knn_trainset)
            return await compiled_program(**kwargs)

        student_copy.aforward = types.MethodType(aforward_pass, student_copy)
        return student_copy

import types
from typing import Any

from typing_extensions import override

from dspy.clients.embedding import Embedder
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.knn import KNN
from dspy.primitives.example import Example
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.teleprompt import Teleprompter


class KNNFewShot(Teleprompter):
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: Any) -> None:
        self.knn = KNN(k, trainset, vectorizer=vectorizer)
        self.few_shot_bootstrap_args = few_shot_bootstrap_args

    @override
    async def compile(self, student, *, teacher=None, run: RunContext):
        student_copy = student.reset_copy()
        knn_few_shot = self

        async def aforward_pass(
            _,
            *,
            run: RunContext,
            options: ModuleCallOptions | None = None,
            **kwargs,
        ):
            knn_trainset = await knn_few_shot.knn(inputs=kwargs)
            bootstrap_args = dict(knn_few_shot.few_shot_bootstrap_args)
            bootstrap_args.pop("run", None)
            few_shot_bootstrap = BootstrapFewShot(**bootstrap_args)
            compiled_program = await few_shot_bootstrap.compile(
                student, teacher=teacher, trainset=knn_trainset, run=run
            )
            return await compiled_program(run=run, options=options, **kwargs)

        student_copy.aforward = types.MethodType(aforward_pass, student_copy)
        return student_copy

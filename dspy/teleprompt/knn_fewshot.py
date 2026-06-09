import types
from typing import Any

from pydantic import BaseModel

from dspy.clients.embedding import Embedder
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.knn import KNN
from dspy.primitives import Example, Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, KNNFewShotCompileParams
from dspy.teleprompt.registry import register_teleprompter


@register_teleprompter(params=KNNFewShotCompileParams)
class KNNFewShot:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: Any) -> None:
        self.knn = KNN(k, trainset, vectorizer=vectorizer)
        self.few_shot_bootstrap_args = few_shot_bootstrap_args

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = KNNFewShotCompileParams.model_validate(params)
        teacher = params.teacher
        student_copy = student.reset_copy()
        knn_few_shot = self

        async def _aforward_impl_pass(
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
            compile_result = await few_shot_bootstrap.compile(
                student,
                params=BootstrapFewShotCompileParams(trainset=knn_trainset, teacher=teacher),
                run=run,
            )
            return await compile_result.program(run=run, options=options, **kwargs)

        student_copy._aforward_impl = types.MethodType(_aforward_impl_pass, student_copy)
        return CompileResult(program=student_copy)

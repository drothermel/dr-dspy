import types

from pydantic import BaseModel

from dspy.clients.embedding import Embedder
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.knn import KNN
from dspy.primitives import Example, Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import KNNFewShotCompileParams
from dspy.teleprompt.registry import register_teleprompter


@register_teleprompter(params=KNNFewShotCompileParams)
class KNNFewShot:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder) -> None:
        self.k = k
        self.knn = KNN(k, trainset, vectorizer=vectorizer)

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = KNNFewShotCompileParams.model_validate(params)
        student_copy = student.reset_copy()
        knn_few_shot = self
        original_forward = student_copy._aforward_impl

        async def _aforward_impl_pass(
            _,
            *,
            run: RunContext,
            options: ModuleCallOptions | None = None,
            **kwargs,
        ):
            knn_trainset = await knn_few_shot.knn(inputs=kwargs)
            for predictor in student_copy.predictors():
                predictor.demos = knn_trainset[: knn_few_shot.k]
            return await original_forward(run=run, options=options, **kwargs)

        student_copy._aforward_impl = types.MethodType(_aforward_impl_pass, student_copy)
        return CompileResult(program=student_copy)

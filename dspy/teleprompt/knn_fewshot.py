from pydantic import BaseModel

from dspy.clients.embedding import Embedder
from dspy.predict.knn import KNN
from dspy.primitives import Example, Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import KNNFewShotCompileParams
from dspy.teleprompt.knn_module import KNNFewShotModule
from dspy.teleprompt.registry import register_teleprompter


@register_teleprompter(params=KNNFewShotCompileParams)
class KNNFewShot:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder) -> None:
        self.k = k
        self.knn = KNN(k, trainset, vectorizer=vectorizer)

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        _ = KNNFewShotCompileParams.model_validate(params)
        student_copy = student.reset_copy()
        wrapped = KNNFewShotModule(student_copy, knn=self.knn, k=self.k)
        return CompileResult(program=wrapped)

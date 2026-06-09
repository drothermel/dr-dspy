import random

from pydantic import BaseModel

from dspy.primitives import Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import LabeledFewShotCompileParams
from dspy.teleprompt.registry import register_teleprompter


@register_teleprompter(params=LabeledFewShotCompileParams)
class LabeledFewShot:
    def __init__(self, k=16) -> None:
        self.k = k

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = LabeledFewShotCompileParams.model_validate(params)
        self.student = student.reset_copy()
        self.trainset = params.trainset
        if len(self.trainset) == 0:
            return CompileResult(program=self.student)
        rng = random.Random(0)
        for predictor in self.student.predictors():
            if params.sample:
                predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
            else:
                predictor.demos = self.trainset[: min(self.k, len(self.trainset))]
        return CompileResult(program=self.student)

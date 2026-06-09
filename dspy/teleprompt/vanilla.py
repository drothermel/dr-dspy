import random

from typing_extensions import override

from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compile_params import LabeledFewShotCompileParams
from dspy.teleprompt.teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    def __init__(self, k=16) -> None:
        self.k = k

    @override
    async def compile(self, student, *, params: LabeledFewShotCompileParams, run: RunContext):
        self.student = student.reset_copy()
        self.trainset = params.trainset
        if len(self.trainset) == 0:
            return self.student
        rng = random.Random(0)
        for predictor in self.student.predictors():
            if params.sample:
                predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
            else:
                predictor.demos = self.trainset[: min(self.k, len(self.trainset))]
        return self.student

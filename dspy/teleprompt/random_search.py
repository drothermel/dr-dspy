import random
from typing import Any, cast

from pydantic import BaseModel
from typing_extensions import override

from dspy.primitives.module import Module
from dspy.runtime.async_parallel import resolve_max_errors
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compile_params import (
    BootstrapFewShotCompileParams,
    LabeledFewShotCompileParams,
    RandomSearchCompileParams,
)
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import make_optimizer_evaluator

from .bootstrap import BootstrapFewShot
from .vanilla import LabeledFewShot


class BootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_run: RunContext | None = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        max_concurrency=None,
        max_errors=None,
        stop_at_score=None,
        metric_threshold=None,
    ) -> None:
        self.metric = metric
        self.teacher_run = teacher_run
        self.max_rounds = max_rounds
        self.max_concurrency = max_concurrency
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

    @override
    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> Module:
        params = RandomSearchCompileParams.model_validate(params)
        self.trainset = params.trainset
        self.valset = params.valset or params.trainset
        teacher = params.teacher
        restrict = params.restrict
        labeled_sample = params.labeled_sample
        effective_max_errors = resolve_max_errors(self.max_errors, run)
        scores = []
        all_subscores = []
        score_data = []
        best_program = student.reset_copy()
        for seed in range(-3, self.num_candidate_sets):
            if restrict is not None and seed not in restrict:
                continue
            trainset_copy = list(self.trainset)
            if seed == -3:
                program = student.reset_copy()
            elif seed == -2:
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = await teleprompter.compile(
                    student,
                    params=LabeledFewShotCompileParams(trainset=trainset_copy, sample=labeled_sample),
                    run=run,
                )
            elif seed == -1:
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_run=self.teacher_run,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = await optimizer.compile(
                    student,
                    params=BootstrapFewShotCompileParams(trainset=trainset_copy, teacher=teacher),
                    run=run,
                )
            else:
                assert seed >= 0, seed
                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_run=self.teacher_run,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = await optimizer.compile(
                    student,
                    params=BootstrapFewShotCompileParams(trainset=trainset_copy, teacher=teacher),
                    run=run,
                )
            evaluate = make_optimizer_evaluator(
                run,
                devset=self.valset,
                metric=self.metric,
                max_concurrency=self.max_concurrency,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
            )
            result = await evaluate(program, run=run)
            score, subscores = (result.score, [output[2] for output in result.results])
            all_subscores.append(subscores)
            if len(scores) == 0 or score > max(scores):
                best_program = program
            scores.append(score)
            score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})
            if self.stop_at_score is not None and score >= self.stop_at_score:
                break
        compiled = cast("Any", best_program)
        compiled.candidate_programs = score_data
        compiled.candidate_programs = sorted(compiled.candidate_programs, key=lambda x: x["score"], reverse=True)
        return best_program

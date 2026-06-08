import random

from typing_extensions import override

from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot
from .vanilla import LabeledFewShot


class BootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
        max_errors=None,
        stop_at_score=None,
        metric_threshold=None,
    ) -> None:
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

    @override
    async def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        self.trainset = trainset
        self.valset = valset or trainset
        effective_max_errors = self.max_errors if self.max_errors is not None else settings.max_errors
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
                program = await teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)
            elif seed == -1:
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = await optimizer.compile(student, teacher=teacher, trainset=trainset_copy)
            else:
                assert seed >= 0, seed
                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = await optimizer.compile(student, teacher=teacher, trainset=trainset_copy)
            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=effective_max_errors,
                display_table=False,
                display_progress=True,
            )
            result = await evaluate(program)
            score, subscores = (result.score, [output[2] for output in result.results])
            all_subscores.append(subscores)
            if len(scores) == 0 or score > max(scores):
                best_program = program
            scores.append(score)
            score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})
            if self.stop_at_score is not None and score >= self.stop_at_score:
                break
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(
            best_program.candidate_programs, key=lambda x: x["score"], reverse=True
        )
        return best_program

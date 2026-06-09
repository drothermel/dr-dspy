from pydantic import BaseModel

from dspy.primitives import Module
from dspy.runtime.async_parallel import resolve_max_errors
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.candidate_ladder import (
    CandidateLadderConfig,
    compile_candidate_program,
    iter_candidate_seeds,
)
from dspy.teleprompt.compile_params import RandomSearchCompileParams
from dspy.teleprompt.utils import make_optimizer_evaluator


class BootstrapFewShotWithRandomSearch:
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
        self.num_random_candidates = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> Module:
        params = RandomSearchCompileParams.model_validate(params)
        self.trainset = params.trainset
        self.valset = params.valset or params.trainset
        teacher = params.teacher
        restrict = params.restrict
        labeled_sample = params.labeled_sample
        effective_max_errors = resolve_max_errors(self.max_errors, run)
        ladder_config = CandidateLadderConfig(
            num_random=self.num_random_candidates,
            include_baseline=params.include_baselines,
            include_labeled_fewshot=True,
            include_bootstrap=True,
            max_labeled_demos=self.max_labeled_demos,
            max_bootstrapped_demos=self.max_num_samples,
            min_bootstrapped_demos=self.min_num_samples,
        )
        scores = []
        all_subscores = []
        score_data = []
        best_program = student.reset_copy()
        for seed_index, seed in enumerate(iter_candidate_seeds(ladder_config)):
            if restrict is not None and seed_index not in restrict:
                continue
            program = await compile_candidate_program(
                seed=seed,
                student=student,
                trainset=self.trainset,
                run=run,
                metric=self.metric,
                teacher=teacher,
                teacher_run=self.teacher_run,
                max_labeled_demos=self.max_labeled_demos,
                max_bootstrapped_demos=self.max_num_samples,
                min_bootstrapped_demos=self.min_num_samples,
                max_rounds=self.max_rounds,
                max_errors=effective_max_errors,
                metric_threshold=self.metric_threshold,
                labeled_sample=labeled_sample,
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
            score_data.append({"score": score, "subscores": subscores, "seed": seed_index, "program": program})
            if self.stop_at_score is not None and score >= self.stop_at_score:
                break
        compiled = best_program
        compiled.candidate_programs = score_data
        compiled.candidate_programs = sorted(compiled.candidate_programs, key=lambda x: x["score"], reverse=True)
        return best_program

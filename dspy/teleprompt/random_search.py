from pydantic import BaseModel

from dspy.primitives import Module
from dspy.runtime.async_parallel import resolve_max_errors
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.candidate_ladder import (
    CandidateLadderConfig,
    compile_candidate_program,
    iter_candidate_seeds,
)
from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate
from dspy.teleprompt.compile_params import RandomSearchCompileParams
from dspy.teleprompt.core.evaluator import make_optimizer_evaluator
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.registry import register_teleprompter


@register_teleprompter(params=RandomSearchCompileParams)
class BootstrapFewShotWithRandomSearch:
    def __init__(
        self,
        metric: OptimizerMetric,
        teacher_run: RunContext | None = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_random_candidates=16,
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
        self.num_random_candidates = num_random_candidates
        self.max_labeled_demos = max_labeled_demos

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = RandomSearchCompileParams.model_validate(params)
        trainset = params.trainset
        valset = params.valset or params.trainset
        teacher = params.teacher
        restrict = params.restrict
        labeled_sample = params.labeled_sample
        effective_max_errors = resolve_max_errors(self.max_errors, run)
        evaluate = make_optimizer_evaluator(
            run,
            devset=valset,
            metric=self.metric,
            max_concurrency=self.max_concurrency,
            max_errors=self.max_errors,
            display_table=False,
            display_progress=True,
        )
        ladder_config = CandidateLadderConfig(
            num_random=self.num_random_candidates,
            include_baseline=params.include_baselines,
            include_labeled_fewshot=True,
            include_bootstrap=True,
            max_labeled_demos=self.max_labeled_demos,
            max_bootstrapped_demos=self.max_num_samples,
            min_bootstrapped_demos=self.min_num_samples,
        )
        candidates: list[ProgramCandidate] = []
        for seed_index, seed in enumerate(iter_candidate_seeds(ladder_config)):
            if restrict is not None and seed_index not in restrict:
                continue
            program = await compile_candidate_program(
                seed=seed,
                student=student,
                trainset=trainset,
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
            result = await evaluate(program, run=run)
            score, subscores = (result.score, [output[2] for output in result.results])
            candidates.append(
                ProgramCandidate(score=score, program=program, subscores=subscores, seed=seed, label=str(seed_index))
            )
            if self.stop_at_score is not None and score >= self.stop_at_score:
                break
        candidates.sort(
            key=lambda candidate: candidate.score if candidate.score is not None else float("-inf"), reverse=True
        )
        best_program = candidates[0].program if candidates else student.reset_copy()
        best_score = candidates[0].score if candidates else None
        return CompileResult(
            program=best_program,
            candidates=candidates,
            stats=CompileStats(best_score=best_score),
        )

from typing import Any, cast

from pydantic import BaseModel

from dspy.integrations.optimizers.optuna.study import create_maximize_study, run_ask_tell_loop
from dspy.primitives import Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, BootstrapOptunaCompileParams
from dspy.teleprompt.core.evaluator import make_optimizer_evaluator
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.registry import register_teleprompter

from .bootstrap import BootstrapFewShot


@register_teleprompter(params=BootstrapOptunaCompileParams)
class BootstrapFewShotWithOptuna:
    def __init__(
        self,
        metric: OptimizerMetric,
        teacher_run: RunContext | None = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_random_candidates=16,
        max_concurrency=None,
    ) -> None:
        self.metric = metric
        self.teacher_run = teacher_run
        self.max_rounds = max_rounds
        self.max_concurrency = max_concurrency
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_random_candidates
        self.max_labeled_demos = max_labeled_demos

    async def _evaluate_program(self, program, *, run: RunContext):
        evaluate = make_optimizer_evaluator(
            run,
            devset=self.valset,
            metric=self.metric,
            max_concurrency=self.max_concurrency,
            max_errors=None,
            display_table=False,
            display_progress=True,
        )
        return await evaluate(program, run=run)

    async def _run_trial(self, trial) -> float:
        program2 = self.student.reset_copy()
        for (name, compiled_predictor), (_, program2_predictor) in zip(
            self.compiled_teleprompter.named_predictors(), program2.named_predictors(), strict=True
        ):
            all_demos = compiled_predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            selected_demo = dict(all_demos[demo_index])
            program2_predictor.demos = [selected_demo]
        result = await self._evaluate_program(program2, run=self.run)
        trial.set_user_attr("program", program2)
        return cast("Any", result).score

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = BootstrapOptunaCompileParams.model_validate(params)
        self.trainset = params.trainset
        self.valset = params.valset or params.trainset
        self.run = run
        self.student = student.reset_copy()
        teacher = params.teacher
        if teacher is None:
            self.teacher = student.reset_copy()
        elif isinstance(teacher, list):
            raise ValueError(
                "BootstrapFewShotWithOptuna accepts a single teacher Module, not a list. Pass one teacher or None."
            )
        else:
            self.teacher = teacher.deepcopy()
        max_demos = params.max_demos
        teleprompter_optimize = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=self.max_labeled_demos,
            teacher_run=self.teacher_run,
            max_rounds=self.max_rounds,
        )
        bootstrap_result = await teleprompter_optimize.compile(
            self.student,
            params=BootstrapFewShotCompileParams(trainset=self.trainset, teacher=self.teacher),
            run=run,
        )
        self.compiled_teleprompter = bootstrap_result.program
        study = create_maximize_study(feature="BootstrapFewShotWithOptuna")
        await run_ask_tell_loop(study, self.num_candidate_sets, self._run_trial)
        best_program = study.trials[study.best_trial.number].user_attrs["program"]
        return CompileResult.with_compiled_program(best_program)

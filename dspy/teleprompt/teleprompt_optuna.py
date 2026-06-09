from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel

from dspy.evaluate.evaluator import Evaluate
from dspy.integrations.optimizers.optuna.study import create_maximize_study, run_ask_tell_loop
from dspy.primitives import Example, Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, BootstrapOptunaCompileParams
from dspy.teleprompt.core.evaluator import make_optimizer_evaluator
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.registry import register_teleprompter

from .bootstrap import BootstrapFewShot


@dataclass
class OptunaCompileSession:
    trainset: list[Example]
    valset: list[Example]
    run: RunContext
    student: Module
    teacher: Module
    compiled_teleprompter: Module
    evaluator: Evaluate


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

    async def _evaluate_program(self, program, *, evaluator: Evaluate, run: RunContext):
        return await evaluator(program, run=run)

    async def _run_trial(self, trial, *, session: OptunaCompileSession) -> float:
        program2 = session.student.reset_copy()
        for (name, compiled_predictor), (_, program2_predictor) in zip(
            session.compiled_teleprompter.named_predictors(), program2.named_predictors(), strict=True
        ):
            all_demos = compiled_predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            selected_demo = dict(all_demos[demo_index])
            program2_predictor.demos = [selected_demo]
        result = await self._evaluate_program(program2, evaluator=session.evaluator, run=session.run)
        trial.set_user_attr("program", program2)
        return cast("Any", result).score

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = BootstrapOptunaCompileParams.model_validate(params)
        trainset = params.trainset
        valset = params.valset or params.trainset
        student_copy = student.reset_copy()
        teacher = params.teacher
        if teacher is None:
            teacher_copy = student.reset_copy()
        elif isinstance(teacher, list):
            raise ValueError(
                "BootstrapFewShotWithOptuna accepts a single teacher Module, not a list. Pass one teacher or None."
            )
        else:
            teacher_copy = teacher.deepcopy()
        max_demos = params.max_demos
        teleprompter_optimize = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=self.max_labeled_demos,
            teacher_run=self.teacher_run,
            max_rounds=self.max_rounds,
        )
        bootstrap_result = await teleprompter_optimize.compile(
            student_copy,
            params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher_copy),
            run=run,
        )
        evaluator = make_optimizer_evaluator(
            run,
            devset=valset,
            metric=self.metric,
            max_concurrency=self.max_concurrency,
            max_errors=None,
            display_table=False,
            display_progress=True,
        )
        session = OptunaCompileSession(
            trainset=trainset,
            valset=valset,
            run=run,
            student=student_copy,
            teacher=teacher_copy,
            compiled_teleprompter=bootstrap_result.program,
            evaluator=evaluator,
        )
        study = create_maximize_study(feature="BootstrapFewShotWithOptuna")

        async def _trial_fn(trial):
            return await self._run_trial(trial, session=session)

        await run_ask_tell_loop(study, self.num_candidate_sets, _trial_fn)
        best_program = study.trials[study.best_trial.number].user_attrs["program"]
        return CompileResult.with_compiled_program(best_program)

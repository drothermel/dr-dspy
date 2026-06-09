import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from pydantic import BaseModel
from typing_extensions import override

from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.module import Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, BootstrapOptunaCompileParams
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot

_optuna_executor = ThreadPoolExecutor(max_workers=1)


def _import_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        if exc.name == "optuna":
            raise ImportError(
                "BootstrapFewShotWithOptuna requires optional dependency 'optuna'. Install it with `pip install dspy[optuna]`."
            ) from exc
        raise
    return optuna


class BootstrapFewShotWithOptuna(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_run: RunContext | None = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        max_concurrency=None,
    ) -> None:
        self.metric = metric
        self.teacher_run = teacher_run
        self.max_rounds = max_rounds
        self.max_concurrency = max_concurrency
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

    async def _evaluate_program(self, program, *, run: RunContext):
        evaluate = Evaluate(
            devset=self.valset,
            metric=self.metric,
            max_concurrency=self.max_concurrency,
            display_table=False,
            display_progress=True,
        )
        return await evaluate(program, run=run)

    def objective(self, trial):
        program2 = self.student.reset_copy()
        for (name, compiled_predictor), (_, program2_predictor) in zip(
            self.compiled_teleprompter.named_predictors(), program2.named_predictors(), strict=False
        ):
            all_demos = compiled_predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            selected_demo = dict(all_demos[demo_index])
            program2_predictor.demos = [selected_demo]
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self._evaluate_program(program2, run=self.run))
        else:
            result = _optuna_executor.submit(asyncio.run, self._evaluate_program(program2, run=self.run)).result()
        trial.set_user_attr("program", program2)
        return cast("Any", result).score

    @override
    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> Module:
        params = BootstrapOptunaCompileParams.model_validate(params)
        optuna = _import_optuna()
        self.trainset = params.trainset
        self.valset = params.valset or params.trainset
        self.run = run
        self.student = student.reset_copy()
        teacher = params.teacher
        if teacher is None:
            self.teacher = student.reset_copy()
        elif isinstance(teacher, list):
            self.teacher = cast("Module", teacher[0]).deepcopy()
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
        self.compiled_teleprompter = await teleprompter_optimize.compile(
            self.student,
            params=BootstrapFewShotCompileParams(trainset=self.trainset, teacher=self.teacher),
            run=run,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.num_candidate_sets)
        return study.trials[study.best_trial.number].user_attrs["program"]

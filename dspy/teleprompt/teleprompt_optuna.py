import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from typing_extensions import override

from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot

_optuna_executor = ThreadPoolExecutor(max_workers=1)


def _import_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        if exc.name == "optuna":
            raise ImportError(
                "BootstrapFewShotWithOptuna requires optional dependency 'optuna'. "
                "Install it with `pip install dspy[optuna]`."
            ) from exc
        raise
    return optuna


class BootstrapFewShotWithOptuna(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
    ) -> None:
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        # self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        # Semi-hacky way to get the parent class's _bootstrap function to stop early.
        # self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        # print("Going to sample", self.max_num_traces, "traces in total.")

    async def _evaluate_program(self, program):
        evaluate = Evaluate(
            devset=self.valset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_table=False,
            display_progress=True,
        )
        return await evaluate(program)

    def objective(self, trial):
        program2 = self.student.reset_copy()
        for (name, compiled_predictor), (_, program2_predictor) in zip(
            self.compiled_teleprompter.named_predictors(),
            program2.named_predictors(),
            strict=False,
        ):
            all_demos = compiled_predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            selected_demo = dict(all_demos[demo_index])
            program2_predictor.demos = [selected_demo]

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self._evaluate_program(program2))
        else:
            result = _optuna_executor.submit(asyncio.run, self._evaluate_program(program2)).result()

        trial.set_user_attr("program", program2)
        return cast("Any", result).score

    @override
    async def compile(self, student, *, teacher=None, max_demos, trainset, valset=None):
        optuna = _import_optuna()
        self.trainset = trainset
        self.valset = valset or trainset
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.reset_copy()
        teleprompter_optimize = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=self.max_labeled_demos,
            teacher_settings=self.teacher_settings,
            max_rounds=self.max_rounds,
        )
        self.compiled_teleprompter = await teleprompter_optimize.compile(
            self.student,
            teacher=self.teacher,
            trainset=self.trainset,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.num_candidate_sets)
        return study.trials[study.best_trial.number].user_attrs["program"]

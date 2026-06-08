from typing import Any, Callable, Literal

from typing_extensions import override

from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.mipro.bootstrap import bootstrap_fewshot_examples
from dspy.teleprompt.mipro.propose import propose_instructions
from dspy.teleprompt.mipro.search import optimize_prompt_parameters
from dspy.teleprompt.mipro.settings import (
    print_auto_run_settings,
    set_and_validate_datasets,
    set_hyperparams_from_run_mode,
    set_num_trials_from_num_candidates,
    set_random_seeds,
)
from dspy.teleprompt.optimizer_context import optimizer_lm_context
from dspy.teleprompt.teleprompt import Teleprompter


class MIPROv2(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        prompt_model: Any | None = None,
        task_model: Any | None = None,
        teacher_settings: dict | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: int | None = None,
        num_threads: int | None = None,
        max_errors: int | None = None,
        seed: int = 9,
        init_temperature: float = 1.0,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: str | None = None,
        metric_threshold: float | None = None,
    ) -> None:
        allowed_modes = {None, "light", "medium", "heavy"}
        if auto not in allowed_modes:
            raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
        self.auto = auto
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.num_candidates = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.task_model = task_model if task_model else settings.lm
        self.prompt_model = prompt_model if prompt_model else settings.lm
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.teacher_settings = teacher_settings or {}
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.seed = seed
        self.rng = None
        if not self.prompt_model or not self.task_model:
            raise ValueError(
                "Either provide both prompt_model and task_model or set a default LM through settings.configure(lm=...) from dspy.dsp.utils.settings."
            )

    @override
    async def compile(
        self,
        student: Any,
        *,
        trainset: list,
        teacher: Any = None,
        valset: list | None = None,
        num_trials: int | None = None,
        max_bootstrapped_demos: int | None = None,
        max_labeled_demos: int | None = None,
        seed: int | None = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        provide_traceback: bool | None = None,
    ) -> Any:
        effective_max_errors = self.max_errors if self.max_errors is not None else settings.max_errors
        effective_max_bootstrapped_demos = (
            max_bootstrapped_demos if max_bootstrapped_demos is not None else self.max_bootstrapped_demos
        )
        effective_max_labeled_demos = max_labeled_demos if max_labeled_demos is not None else self.max_labeled_demos
        zeroshot_opt = effective_max_bootstrapped_demos == 0 and effective_max_labeled_demos == 0
        if self.auto is None and (self.num_candidates is not None and num_trials is None):
            raise ValueError(
                f"If auto is None, num_trials must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting num_trials to ~{set_num_trials_from_num_candidates(optimizer=self, program=student, zeroshot_opt=zeroshot_opt, num_candidates=self.num_candidates)}."
            )
        if self.auto is None and (self.num_candidates is None or num_trials is None):
            raise ValueError("If auto is None, num_candidates must also be provided.")
        if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
            raise ValueError(
                "If auto is not None, num_candidates and num_trials cannot be set, since they would be overridden by the auto settings. Please either set auto to None, or do not specify num_candidates and num_trials."
            )
        seed = seed or self.seed
        set_random_seeds(self, seed)
        trainset, valset = set_and_validate_datasets(trainset, valset)
        num_instruct_candidates = (
            self.num_instruct_candidates if self.num_instruct_candidates is not None else self.num_candidates
        )
        num_fewshot_candidates = (
            self.num_fewshot_candidates if self.num_fewshot_candidates is not None else self.num_candidates
        )
        num_trials, valset, minibatch, num_instruct_candidates, num_fewshot_candidates = set_hyperparams_from_run_mode(
            self, student, num_trials, minibatch, zeroshot_opt, valset, num_instruct_candidates, num_fewshot_candidates
        )
        if self.auto:
            print_auto_run_settings(
                self, num_trials, minibatch, valset, num_fewshot_candidates, num_instruct_candidates
            )
        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")
        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
            provide_traceback=provide_traceback,
        )
        with optimizer_lm_context(lm=self.task_model, phase="mipro.bootstrap", lm_role="task_model"):
            demo_candidates = await bootstrap_fewshot_examples(
                self,
                program,
                trainset,
                seed,
                teacher,
                num_fewshot_candidates=num_fewshot_candidates,
                max_bootstrapped_demos=effective_max_bootstrapped_demos,
                max_labeled_demos=effective_max_labeled_demos,
                max_errors=effective_max_errors,
                metric_threshold=self.metric_threshold,
            )
        instruction_candidates = await propose_instructions(
            self,
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
            num_instruct_candidates=num_instruct_candidates,
        )
        if zeroshot_opt:
            demo_candidates = None
        with optimizer_lm_context(lm=self.task_model, phase="mipro.optimize", lm_role="task_model"):
            return await optimize_prompt_parameters(
                self,
                program,
                instruction_candidates,
                demo_candidates,
                evaluate,
                valset,
                num_trials,
                minibatch,
                minibatch_size,
                minibatch_full_eval_steps,
                seed,
            )

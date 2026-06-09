import logging
import random
from typing import Callable, cast

from pydantic import BaseModel, ConfigDict

from dspy.primitives import Example, Module
from dspy.runtime.async_parallel import resolve_max_errors
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.bootstrap_finetune import (
    BootstrapFinetune,
    all_predictors_have_lms,
    kill_lms,
    launch_lms,
    prepare_student,
    prepare_teacher,
)
from dspy.teleprompt.compile_params import (
    BetterTogetherCompileParams,
    BootstrapFewShotCompileParams,
    BootstrapFinetuneCompileParams,
    GEPACompileParams,
    RandomSearchCompileParams,
)
from dspy.teleprompt.eval_batch import eval_candidate_program
from dspy.teleprompt.protocol import Teleprompter
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.utils import make_optimizer_evaluator

logger = logging.getLogger(__name__)
YELLOW = "\x1b[93m"
GREEN = "\x1b[92m"
BLUE = "\x1b[94m"
BOLD = "\x1b[1m"
ENDC = "\x1b[0m"


class PassthroughCompileParams(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    trainset: list[Example] | None = None
    teacher: Module | list[Module] | None = None
    valset: list[Example] | None = None


_OPTIMIZER_PARAMS_TYPES: dict[str, type[BaseModel]] = {
    "BootstrapFewShotWithRandomSearch": RandomSearchCompileParams,
    "BootstrapFinetune": BootstrapFinetuneCompileParams,
    "BootstrapFewShot": BootstrapFewShotCompileParams,
    "GEPA": GEPACompileParams,
}


def _normalize_teacher(teacher: list[Module] | None) -> Module | list[Module] | None:
    if teacher is None:
        return None
    if len(teacher) == 1:
        return teacher[0]
    return teacher


def _default_compile_params(
    optimizer: Teleprompter,
    *,
    trainset: list[Example],
    teacher: list[Module] | None,
    valset: list[Example] | None,
) -> BaseModel:
    teacher_arg = _normalize_teacher(teacher)
    name = optimizer.__class__.__name__
    if name == "BootstrapFewShotWithRandomSearch":
        return RandomSearchCompileParams(trainset=trainset, teacher=teacher_arg, valset=valset)
    if name == "BootstrapFinetune":
        return BootstrapFinetuneCompileParams(trainset=trainset, teacher=teacher_arg)
    if name == "BootstrapFewShot":
        return BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher_arg)
    if name == "GEPA":
        return GEPACompileParams(trainset=trainset, teacher=teacher_arg, valset=valset)
    return PassthroughCompileParams(trainset=trainset, teacher=teacher_arg, valset=valset)


class BetterTogether:
    STRAT_SEP = " -> "

    def __init__(self, metric: Callable, **optimizers: Teleprompter) -> None:
        self.metric = metric
        if not optimizers:
            logger.info(
                "No optimizers provided. Using defaults: BootstrapFewShotWithRandomSearch (p) and BootstrapFinetune (w). You can use the letters p and w to specify the compile strategy. For example, to run weight optimization after prompt optimization, use strategy='p -> w'."
            )
            optimizers = {"p": BootstrapFewShotWithRandomSearch(metric=metric), "w": BootstrapFinetune(metric=metric)}
        for key, optimizer in optimizers.items():
            if not isinstance(optimizer, Teleprompter):
                raise TypeError(f"Optimizer '{key}' must be a Teleprompter, got {type(optimizer).__name__}")
        self.optimizers: dict[str, Teleprompter] = optimizers

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> Module:
        params = BetterTogetherCompileParams.model_validate(params)
        logger.info(f"\n{BOLD}==> BETTERTOGETHER COMPILATION STARTED <=={ENDC}")
        logger.info(f"{BLUE}Strategy:{ENDC} {params.strategy}")
        logger.info(f"{BLUE}Trainset size:{ENDC} {len(params.trainset)}")
        logger.info(
            f"{BLUE}Validation ratio:{ENDC} {(params.valset_ratio if params.valset is None else 'using provided valset')}"
        )
        student, teacher = self._prepare_student_and_teacher(student=student, teacher=params.teacher)
        trainset, valset = self._prepare_trainset_and_valset(
            trainset=params.trainset, valset=params.valset, valset_ratio=params.valset_ratio
        )
        effective_max_errors = resolve_max_errors(params.max_errors, run)
        parsed_strategy = self._prepare_strategy(params.strategy)
        optimizer_compile_args = self._prepare_optimizer_compile_args(params.optimizer_compile_args, teacher)
        student = await self._run_strategies(
            student,
            trainset,
            teacher,
            valset,
            params.max_concurrency,
            effective_max_errors,
            params.provide_traceback,
            params.seed,
            parsed_strategy,
            params.shuffle_trainset_between_steps,
            optimizer_compile_args,
            run,
        )
        logger.info(f"\n{BOLD}{GREEN}==> BETTERTOGETHER COMPILATION COMPLETE <=={ENDC}")
        logger.info(f"{GREEN}Best score achieved:{ENDC} {student.candidate_programs[0]['score']}")
        logger.info(
            f"{GREEN}Best strategy:{ENDC} {student.candidate_programs[0]['strategy'] or 'original (no optimization)'}"
        )
        student._compiled = True
        return student

    def _prepare_student_and_teacher(
        self, student: Module, teacher: Module | list[Module] | None
    ) -> tuple[Module, list[Module] | None]:
        student = prepare_student(student)
        all_predictors_have_lms(student)
        if not teacher:
            return (student, None)
        teacher = [teacher] if not isinstance(teacher, list) else teacher
        teacher = [prepare_teacher(student=student, teacher=cast("Module", t)) for t in teacher]
        return (student, teacher)

    def _prepare_trainset_and_valset(
        self, trainset: list[Example], valset: list[Example] | None, valset_ratio: float
    ) -> tuple[list[Example], list[Example] | None]:
        if not trainset:
            raise ValueError("trainset cannot be empty")
        if valset_ratio < 0 or valset_ratio >= 1:
            raise ValueError(f"valset_ratio must be in range [0, 1), got {valset_ratio}")
        trainset = trainset[:]
        if valset:
            logger.info(f"{BLUE}Using provided validation set ({len(valset)} examples). Ignoring valset_ratio.{ENDC}")
            return (trainset, valset)
        if valset_ratio == 0:
            logger.info(f"{YELLOW}No validation set provided and valset_ratio=0. No validation set created.{ENDC}")
            return (trainset, None)
        logger.info(f"{BLUE}Sampling {valset_ratio:.1%} of trainset as validation set.{ENDC}")
        num_val_examples = int(valset_ratio * len(trainset))
        valset = trainset[:num_val_examples]
        trainset = trainset[num_val_examples:]
        logger.info(
            f"{BLUE}Created validation set: {len(valset)} examples. Training set: {len(trainset)} examples.{ENDC}"
        )
        return (trainset, valset)

    def _prepare_strategy(self, strategy: str) -> list[str]:
        if not strategy or not strategy.strip():
            raise ValueError("strategy cannot be empty")
        parsed_strategy = strategy.split(self.STRAT_SEP)
        invalid_steps = [s for s in parsed_strategy if s not in self.optimizers]
        if invalid_steps:
            raise ValueError(
                f"Strategy contains invalid optimizer keys: {invalid_steps}. Valid keys are: {list(self.optimizers.keys())}"
            )
        return parsed_strategy

    def _prepare_optimizer_compile_args(
        self, optimizer_compile_args: dict[str, BaseModel] | None, teacher: list[Module] | None
    ) -> dict[str, BaseModel]:
        logger.info(f"{BLUE}Validating optimizer compile arguments...{ENDC}")
        if not optimizer_compile_args:
            return {}
        for optimizer_key, compile_args in optimizer_compile_args.items():
            if optimizer_key not in self.optimizers:
                raise ValueError(
                    f"Invalid optimizer key '{optimizer_key}'. Valid keys are: {list(self.optimizers.keys())}"
                )
            optimizer = self.optimizers[optimizer_key]
            self._validate_compile_args(optimizer=optimizer, optimizer_key=optimizer_key, compile_args=compile_args)
            if optimizer.__class__.__name__ == "GEPA":
                if teacher is not None:
                    raise ValueError("GEPA does not accept a teacher argument. Please remove the teacher argument.")
        return optimizer_compile_args

    def _validate_compile_args(self, optimizer: Teleprompter, optimizer_key: str, compile_args: BaseModel) -> None:
        expected = _OPTIMIZER_PARAMS_TYPES.get(optimizer.__class__.__name__)
        if expected is None:
            return
        if not isinstance(compile_args, expected):
            raise TypeError(
                f"optimizer_compile_args['{optimizer_key}'] must be {expected.__name__}, "
                f"got {type(compile_args).__name__}"
            )

    async def _run_strategies(
        self,
        student: Module,
        trainset: list[Example],
        teacher: list[Module] | None,
        valset: list[Example] | None,
        max_concurrency: int | None,
        effective_max_errors: int | None,
        provide_traceback: bool | None,
        seed: int | None,
        parsed_strategy: list[str],
        shuffle_trainset_between_steps: bool,
        optimizer_args: dict[str, BaseModel],
        run: RunContext,
    ) -> Module:
        rng = random.Random(seed)
        candidate_programs = []
        flag_lms_launched = False
        flag_compilation_error_occurred = False
        logger.info(f"\n{BOLD}==> BASELINE EVALUATION <=={ENDC}")
        logger.info("Evaluating original program (no optimization applied)")
        launch_lms(student)
        flag_lms_launched = True
        score = await self._evaluate_on_valset(
            student, valset, rng, max_concurrency, effective_max_errors, provide_traceback, run
        )
        self._add_candidate(candidate_programs=candidate_programs, student=student, strategy="", score=score)
        logger.info(f"{YELLOW}Baseline score:{ENDC} {score}")
        for ind, step_code in enumerate(parsed_strategy):
            current_strategy = self.STRAT_SEP.join(parsed_strategy[: ind + 1])
            optimizer = self.optimizers[step_code]
            logger.info(
                f"\n{BOLD}==> STEP {ind + 1}/{len(parsed_strategy)}: {optimizer.__class__.__name__.upper()} <=={ENDC}"
            )
            logger.info(f"{BLUE}Current strategy:{ENDC} '{current_strategy}'")
            logger.info(f"{BLUE}Optimizer:{ENDC} {optimizer.__class__.__name__}")
            try:
                if shuffle_trainset_between_steps:
                    logger.info(f"{BLUE}Shuffling trainset...{ENDC}")
                    rng.shuffle(trainset)
                compile_params = optimizer_args.get(step_code)
                if compile_params is None:
                    compile_params = _default_compile_params(
                        optimizer, trainset=trainset, teacher=teacher, valset=valset
                    )
                student, score, is_new_best, lms_relaunched = await self._run_and_evaluate_step(
                    optimizer,
                    student,
                    compile_params,
                    valset,
                    candidate_programs,
                    current_strategy,
                    rng,
                    max_concurrency,
                    effective_max_errors,
                    provide_traceback,
                    run,
                )
                if lms_relaunched:
                    flag_lms_launched = True
                if is_new_best:
                    logger.info(f"{GREEN}New best score!{ENDC} {score} (strategy: '{current_strategy}')")
                else:
                    logger.info(f"{YELLOW}Score after optimization:{ENDC} {score}")
            except Exception as e:
                flag_compilation_error_occurred = True
                logger.exception(
                    f"{YELLOW}Step {ind + 1}/{len(parsed_strategy)} failed with error: {type(e).__name__}: {e}{ENDC}"
                )
                logger.exception(
                    f"{YELLOW}Stopping optimization early. Returning best program found so far from {len(candidate_programs)} candidate(s).{ENDC}"
                )
                logger.error(f"{YELLOW}Traceback:{ENDC}", exc_info=True)
                break
        if flag_lms_launched:
            kill_lms(student)
        candidate_programs_with_idx = [(i, cp) for i, cp in enumerate(candidate_programs)]
        candidate_programs_with_idx.sort(
            key=lambda x: (x[1]["score"] if x[1]["score"] is not None else float("-inf"), -x[0]), reverse=True
        )
        candidate_programs = [cp for _, cp in candidate_programs_with_idx]
        if valset is None or len(valset) == 0:
            best_program = candidate_programs_with_idx[-1][1]
        else:
            best_program = candidate_programs[0]
        best_student = best_program["program"]
        best_student.candidate_programs = candidate_programs
        best_student.flag_compilation_error_occurred = flag_compilation_error_occurred
        logger.info(f"\n{BOLD}==> OPTIMIZATION SUMMARY <=={ENDC}")
        logger.info(f"{GREEN}Best score:{ENDC} {best_program['score']}")
        strategy_display = best_program["strategy"] if best_program["strategy"] else "original (no optimization)"
        logger.info(f"{GREEN}Best strategy:{ENDC} {strategy_display}")
        logger.info(f"{BLUE}Total candidates evaluated:{ENDC} {len(candidate_programs)}")
        return best_student

    async def _run_and_evaluate_step(
        self,
        optimizer: Teleprompter,
        student: Module,
        compile_params: BaseModel,
        valset: list[Example] | None,
        candidate_programs: list,
        current_strategy: str,
        rng: random.Random,
        max_concurrency: int | None,
        effective_max_errors: int | None,
        provide_traceback: bool | None,
        run: RunContext,
    ) -> tuple[Module, float | None, bool, bool]:
        pred_lms_before = [pred.lm for pred in student.predictors()]
        student._compiled = False
        logger.info(f"{BLUE}Running {optimizer.__class__.__name__}...{ENDC}")
        student = await optimizer.compile(student, params=compile_params, run=run)
        if not all_predictors_have_lms(student):
            logger.warning(
                f"{YELLOW}Warning: {optimizer.__class__.__name__} incorrectly reset predictor LMs. Restoring to original LMs.{ENDC}"
            )
            for pred, lm in zip(student.predictors(), pred_lms_before, strict=False):
                pred.lm = lm
        lms_relaunched = False
        if self._models_changed(student=student, pred_lms_before=pred_lms_before):
            launch_lms(student)
            lms_relaunched = True
        score = await self._evaluate_on_valset(
            student, valset, rng, max_concurrency, effective_max_errors, provide_traceback, run
        )
        self._add_candidate(
            candidate_programs=candidate_programs, student=student, strategy=current_strategy, score=score
        )
        valid_scores = [cp["score"] for cp in candidate_programs if cp["score"] is not None]
        best_score_so_far = max(valid_scores) if valid_scores else float("-inf")
        is_new_best = score is not None and score >= best_score_so_far
        return (student, score, is_new_best, lms_relaunched)

    def _models_changed(self, student: Module, pred_lms_before: list) -> bool:
        pred_lms_after = [pred.lm for pred in student.predictors()]
        model_names_before = [lm.model if lm else None for lm in pred_lms_before]
        model_names_after = [lm.model if lm else None for lm in pred_lms_after]
        return model_names_before != model_names_after

    def _add_candidate(self, candidate_programs: list, student: Module, strategy: str, score: float | None) -> None:
        candidate_programs.append({"score": score, "program": student.deepcopy(), "strategy": strategy})

    async def _evaluate_on_valset(
        self,
        program: Module,
        valset: list[Example] | None,
        rng: random.Random,
        max_concurrency: int | None,
        effective_max_errors: int | None,
        provide_traceback: bool | None,
        run: RunContext,
    ) -> float | None:
        if valset is None or len(valset) == 0:
            logger.info(f"{YELLOW}No validation set provided. Skipping evaluation.{ENDC}")
            return None
        logger.info(f"{BLUE}Evaluating on {len(valset)} validation examples...{ENDC}")
        evaluate = make_optimizer_evaluator(
            run,
            devset=valset,
            metric=self.metric,
            max_concurrency=max_concurrency,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
            provide_traceback=provide_traceback,
        )
        eval_result = await eval_candidate_program(
            batch_size=len(valset), trainset=valset, candidate_program=program, evaluate=evaluate, run=run, rng=rng
        )
        return eval_result.score

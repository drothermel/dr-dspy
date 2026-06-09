from __future__ import annotations

import functools
import logging
import operator
import random
from typing import Any, Literal, cast

from pydantic import BaseModel  # noqa: TC002 — compile params validation at runtime

from dspy.adapters.base import Adapter  # noqa: TC001 — GRPO adapter map
from dspy.clients.finetune import FinetuneService, GRPORolloutGroup
from dspy.clients.lm import LM  # noqa: TC001 — job keys and adapter map
from dspy.primitives import Module  # noqa: TC001 — student/teacher programs
from dspy.runtime.run_context import RunContext  # noqa: TC001 — compile run context
from dspy.task_spec.predictor_context import get_task_spec
from dspy.teleprompt.bootstrap_finetune import (
    FinetuneTeleprompter,
    all_predictors_have_lms,
    assert_structural_equivalency,
)
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import GRPOCompileParams
from dspy.teleprompt.grpo.batch_dispatch import dispatch_training_step
from dspy.teleprompt.grpo.rollout_groups import (
    build_rollout_batches,
    log_rollout_batch_warnings,
    validate_trace_data_and_log_issues,
)
from dspy.teleprompt.grpo.sampling import select_training_sample
from dspy.teleprompt.grpo.session import GRPOCompileSession
from dspy.teleprompt.grpo.trace_grid import collect_teacher_trace_grid
from dspy.teleprompt.grpo.validation_metrics import report_validation_metrics
from dspy.teleprompt.grpo.wait import wait_until
from dspy.teleprompt.metrics import OptimizerMetric  # noqa: TC001 — metric callback
from dspy.teleprompt.registry import register_teleprompter

logger = logging.getLogger(__name__)


def _validate_grpo_config(
    *,
    failure_score: float,
    format_failure_score: float,
    use_train_as_val: bool,
    report_train_scores: bool,
    exclude_demos: bool,
    multitask: bool,
    variably_invoked_predictor_grouping_mode: Literal["truncate", "fill", "ragged"],
    variably_invoked_predictor_fill_strategy: Literal["randint", "max"] | None,
) -> None:
    if failure_score <= format_failure_score:
        raise ValueError(
            "failure_score must be greater than format_failure_score since the range "
            "[format_failure_score, failure_score] is used to provide dspy formatting rewards"
        )
    if use_train_as_val and not report_train_scores:
        raise ValueError("If use_train_as_val is True, report_train_scores must be True.")
    if not exclude_demos:
        raise ValueError("exclude_demos==False is not supported yet. Please set it to True.")
    if not multitask:
        raise ValueError(
            "independent GRPO training jobs for each predictor in the student program is not supported yet. "
            "Please set multitask=True."
        )
    if variably_invoked_predictor_grouping_mode == "fill":
        if variably_invoked_predictor_fill_strategy is None:
            raise ValueError(
                "variably_invoked_predictor_fill_strategy must be set when "
                "variably_invoked_predictor_grouping_mode is 'fill'"
            )
        if variably_invoked_predictor_fill_strategy not in ("randint", "max"):
            raise ValueError("variably_invoked_predictor_fill_strategy must be either 'randint' or 'max'")


@register_teleprompter(params=GRPOCompileParams)
class GRPO(FinetuneTeleprompter):
    def __init__(
        self,
        metric: OptimizerMetric | None = None,
        multitask: bool = True,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
        adapter: Adapter | dict[LM, Adapter] | None = None,
        exclude_demos: bool = False,
        max_concurrency: int = 6,
        num_train_steps: int = 100,
        seed: int = 0,
        num_dspy_examples_per_grpo_step: int = 1,
        num_rollouts_per_grpo_step: int = 1,
        use_train_as_val: bool = False,
        num_steps_for_val: int = 5,
        report_train_scores: bool = False,
        failure_score: float = 0,
        format_failure_score: float = -1,
        variably_invoked_predictor_grouping_mode: Literal["truncate"]
        | Literal["fill"]
        | Literal["ragged"] = "truncate",
        variably_invoked_predictor_fill_strategy: Literal["randint"] | Literal["max"] | None = None,
    ) -> None:
        super().__init__(train_kwargs=train_kwargs)
        _validate_grpo_config(
            failure_score=failure_score,
            format_failure_score=format_failure_score,
            use_train_as_val=use_train_as_val,
            report_train_scores=report_train_scores,
            exclude_demos=exclude_demos,
            multitask=multitask,
            variably_invoked_predictor_grouping_mode=variably_invoked_predictor_grouping_mode,
            variably_invoked_predictor_fill_strategy=variably_invoked_predictor_fill_strategy,
        )
        self.metric = metric
        self.multitask = multitask
        self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.max_concurrency = max_concurrency
        self.num_train_steps = num_train_steps
        self.rng = random.Random(seed)
        self.num_dspy_examples_per_grpo_step = num_dspy_examples_per_grpo_step
        self.num_rollouts_per_grpo_step = num_rollouts_per_grpo_step
        self.use_train_as_val = use_train_as_val
        self.num_steps_for_val = num_steps_for_val
        self.report_train_scores = report_train_scores
        self.failure_score = failure_score
        self.format_failure_score = format_failure_score
        self.variably_invoked_predictor_grouping_mode = variably_invoked_predictor_grouping_mode
        self.variably_invoked_predictor_fill_strategy = variably_invoked_predictor_fill_strategy

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = GRPOCompileParams.model_validate(params)
        trainset = params.trainset
        teacher = params.teacher
        valset = params.valset
        session = GRPOCompileSession()

        logger.info(
            "Starting the GRPO compilation process... The LM(s) for the student program will be updated in place "
            "at the end of the training."
        )
        logger.info("Validating the inputs...")
        if len(trainset) == 0:
            raise ValueError("Training set is empty. Please provide a non-empty training set.")
        if len(trainset) < self.num_dspy_examples_per_grpo_step:
            logger.warning(
                f"Number of training examples {len(trainset)} is less than the number of examples per GRPO step "
                f"{self.num_dspy_examples_per_grpo_step}. Repeating the training set to fill the GRPO step. "
                "This could lead to overfitting and training instability."
            )
            multiplier = (self.num_dspy_examples_per_grpo_step + len(trainset) - 1) // len(trainset)
            if multiplier > 1:
                logger.warning(
                    f"Repeating the training set {multiplier} times to fill the GRPO step. "
                    "This could lead to overfitting and training instability."
                )
                trainset = trainset * multiplier
        if not self.multitask:
            raise ValueError(
                "Independent GRPO training jobs for each predictor in the student program are not supported yet. "
                "Please set multitask=True."
            )
        student_lms = {id(pred.lm) for pred in student.predictors()}
        if len(student_lms) != 1:
            raise ValueError(
                f"Student program has multiple LMs: {student_lms}. GRPO only supports student programs with a single "
                "LM. You can set the LM for a program with `program.set_lm(...)`"
            )
        if self.use_train_as_val and valset is not None:
            raise ValueError("If use_train_as_val is True, valset must be None.")

        logger.info("Preparing the student program...")
        all_predictors_have_lms(student)
        pred_signature_hash_to_ind = {
            get_task_spec(pred).fingerprint(): ind for ind, pred in enumerate(student.predictors())
        }
        logging.info(
            "Preparing the teacher program(s)... We will ensure that the provided programs have the same program "
            "structure as the student program."
        )
        if (isinstance(teacher, list) and len(teacher) == 0) or teacher is None:
            teacher = student
        teachers = cast("list[Module]", teacher) if isinstance(teacher, list) else [cast("Module", teacher)]
        for t in teachers:
            assert_structural_equivalency(program1=student, program2=t)
            all_predictors_have_lms(t)
        if student not in teachers:
            raise ValueError(
                f"Student program {student} is not in the list of teachers {teachers}. Please provide the student "
                "program as one of the teachers. Alternatively, you can leave the teacher argument as None, and the "
                "student program will be used as the teacher program."
            )
        if self.num_rollouts_per_grpo_step % len(teachers) != 0:
            raise ValueError(
                f"The GRPO group size (num_rollouts_per_grpo_step) {self.num_rollouts_per_grpo_step} is not divisible "
                f"by the number of teachers {len(teachers)}. This is required to ensure that each teacher gets the "
                "same number of examples. Please provide a number of examples that is divisible by the number of "
                "teachers."
            )
        num_samples_per_input = self.num_rollouts_per_grpo_step // len(teachers)

        for pred in student.predictors():
            train_kwargs = self.train_kwargs[pred.lm]
            train_kwargs = {} if train_kwargs is None else train_kwargs
            train_kwargs["num_generations"] = self.num_rollouts_per_grpo_step
            self.train_kwargs[pred.lm] = train_kwargs

        logger.info("Preparing the GRPO training job(s)...")
        grpo_training_jobs: dict[tuple[LM, Any], Any] = {}
        for pred_ind, pred in enumerate(student.predictors()):
            data_key = None if self.multitask else pred_ind
            job_key = (pred.lm, data_key)
            if job_key not in grpo_training_jobs:
                train_kwargs = self.train_kwargs[pred.lm]
                job = FinetuneService(pred.lm, train_kwargs=train_kwargs).reinforce(train_kwargs=train_kwargs)
                grpo_training_jobs[job_key] = job

        await report_validation_metrics(
            student=student,
            trainset=trainset,
            valset=valset,
            step_idx=-1,
            num_train_steps=self.num_train_steps,
            num_steps_for_val=self.num_steps_for_val,
            max_concurrency=self.max_concurrency,
            failure_score=self.failure_score,
            use_train_as_val=self.use_train_as_val,
            report_train_scores=self.report_train_scores,
            metric=self.metric,
            run=run,
        )

        logger.info("Starting the GRPO training loop...")
        for train_step_idx in range(self.num_train_steps):
            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps}...")
            subsample_training_dataset = select_training_sample(
                session,
                original_trainset=trainset,
                train_step_idx=train_step_idx,
                num_dspy_examples_per_grpo_step=self.num_dspy_examples_per_grpo_step,
                rng=self.rng,
            )

            def _any_available_for_step() -> bool:
                for job in grpo_training_jobs.values():
                    pending_batch_ids = job.get_status().pending_batch_ids
                    available = set(pending_batch_ids) - set(session.fulfilled_batch_ids)
                    if available:
                        return True
                return False

            await wait_until(_any_available_for_step)
            logger.info("Bootstrapping data...")
            trace_data = await collect_teacher_trace_grid(
                teachers=teachers,
                subsample=subsample_training_dataset,
                num_samples_per_input=num_samples_per_input,
                run=run,
                metric=self.metric,
                max_concurrency=self.max_concurrency,
                failure_score=self.failure_score,
                format_failure_score=self.format_failure_score,
            )
            validate_trace_data_and_log_issues(
                trace_data,
                subsample_training_dataset=subsample_training_dataset,
                num_teachers=len(teachers),
                num_samples_per_input=num_samples_per_input,
                pred_signature_hash_to_ind=pred_signature_hash_to_ind,
            )
            logger.info("Preparing the training data batch from bootstrapped examples for GRPO...")
            train_batch_per_predictor = build_rollout_batches(
                trace_data,
                student=student,
                pred_signature_hash_to_ind=pred_signature_hash_to_ind,
                num_rollouts_per_grpo_step=self.num_rollouts_per_grpo_step,
                adapter=self.adapter,
                run=run,
                format_failure_score=self.format_failure_score,
                variably_invoked_predictor_grouping_mode=self.variably_invoked_predictor_grouping_mode,
                variably_invoked_predictor_fill_strategy=self.variably_invoked_predictor_fill_strategy,
                rng=self.rng,
            )
            if not any(train_batch_per_predictor):
                logger.warning(
                    "No training data found for this training step. This means that the model did not generate "
                    "valid formatted responses for any of the examples in the training set. This is a critical error. "
                    "Please check the model and the training set."
                )
                continue
            log_rollout_batch_warnings(
                train_batch_per_predictor,
                num_rollouts_per_grpo_step=self.num_rollouts_per_grpo_step,
            )
            logger.info("Invoking GRPO training step...")
            for job_key, job in grpo_training_jobs.items():
                _lm_for_job, data_key = job_key
                train_data: list[GRPORolloutGroup] = (
                    functools.reduce(operator.iadd, train_batch_per_predictor, [])
                    if data_key is None
                    else train_batch_per_predictor[data_key]
                )
                dispatch_training_step(
                    session,
                    job_key=job_key,
                    job=job,
                    train_data=train_data,
                    train_batch_per_predictor=train_batch_per_predictor,
                    num_rollouts_per_grpo_step=self.num_rollouts_per_grpo_step,
                    rng=self.rng,
                )
            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps} completed.")
            await report_validation_metrics(
                student=student,
                trainset=trainset,
                valset=valset,
                step_idx=train_step_idx,
                num_train_steps=self.num_train_steps,
                num_steps_for_val=self.num_steps_for_val,
                max_concurrency=self.max_concurrency,
                failure_score=self.failure_score,
                use_train_as_val=self.use_train_as_val,
                report_train_scores=self.report_train_scores,
                metric=self.metric,
                run=run,
            )

        logger.info("Done with the iterations! Retrieving the final model(s)...")
        for job in grpo_training_jobs.values():
            job.terminate()
        logger.info("GRPO compiler has finished compiling the student program")
        return CompileResult.with_compiled_program(student)

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dspy.teleprompt.core.evaluator import make_optimizer_evaluator

if TYPE_CHECKING:
    from dspy.primitives import Example, Module
    from dspy.runtime.run_context import RunContext
    from dspy.teleprompt.metrics import OptimizerMetric

logger = logging.getLogger(__name__)


async def report_validation_metrics(
    *,
    student: Module,
    trainset: list[Example],
    valset: list[Example] | None,
    step_idx: int,
    num_train_steps: int,
    num_steps_for_val: int,
    max_concurrency: int,
    failure_score: float,
    use_train_as_val: bool,
    report_train_scores: bool,
    metric: OptimizerMetric | None,
    run: RunContext,
) -> None:
    if step_idx != -1 and step_idx != num_train_steps - 1 and (step_idx + 1) % num_steps_for_val != 0:
        return
    if valset is not None:
        if use_train_as_val:
            raise ValueError("If valset is provided, use_train_as_val must be False.")
        if not isinstance(num_steps_for_val, int) or num_steps_for_val <= 0:
            raise ValueError("num_steps_for_val must be a positive integer.")
        if report_train_scores:
            if step_idx == -1:
                logger.info(
                    "Using user provided validation set and reporting train scores for every validation step "
                    "in addition."
                )
            valset_evaluator = make_optimizer_evaluator(
                run,
                devset=valset + trainset,
                metric=metric,
                max_concurrency=max_concurrency,
                max_errors=len(valset) * 10,
                display_progress=True,
                provide_traceback=False,
                failure_score=failure_score,
            )
            if step_idx == -1:
                logger.info("Evaluating the student program on the train+validation set before training loop...")
            else:
                logger.info(
                    f"Evaluating the student program on the validation set after training step "
                    f"{step_idx + 1}/{num_train_steps}"
                )
            valset_evaluation = await valset_evaluator(student, run=run, metric=metric)
            trainset_scores = [r[-1] for r in valset_evaluation.results[len(valset) :]]
            valset_scores = [r[-1] for r in valset_evaluation.results[: len(valset)]]
            trainset_agg = sum(trainset_scores) / len(trainset_scores)
            valset_agg = sum(valset_scores) / len(valset_scores)
            if step_idx == -1:
                logger.info(f"Student program training set score before training loop: {trainset_agg}")
                logger.info(f"Student program validation set score before training loop: {valset_agg}")
            else:
                logger.info(
                    f"Student program training set score after training step {step_idx + 1}/{num_train_steps}: "
                    f"{trainset_agg}"
                )
                logger.info(
                    f"Student program validation set score after training step {step_idx + 1}/{num_train_steps}: "
                    f"{valset_agg}"
                )
        else:
            if step_idx == -1:
                logger.info("Using user provided validation set and not reporting train scores.")
            valset_evaluator = make_optimizer_evaluator(
                run,
                devset=valset,
                metric=metric,
                max_concurrency=max_concurrency,
                max_errors=len(valset) * 10,
                display_progress=True,
                provide_traceback=False,
                failure_score=failure_score,
            )
            if step_idx == -1:
                logger.info("Evaluating the student program on the validation set before training loop...")
            else:
                logger.info(
                    f"Evaluating the student program on the validation set after training step "
                    f"{step_idx + 1}/{num_train_steps}"
                )
            valset_evaluation = await valset_evaluator(student, run=run, metric=metric)
            if step_idx == -1:
                logger.info(f"Student program validation set score before training loop: {valset_evaluation.score}")
            else:
                logger.info(
                    f"Student program validation set score after training step {step_idx + 1}/{num_train_steps}: "
                    f"{valset_evaluation.score}"
                )
    elif report_train_scores:
        if not use_train_as_val:
            raise ValueError(
                "If report_train_scores is True, use_train_as_val must be True when valset is not provided explicitly."
            )
        if not isinstance(num_steps_for_val, int) or num_steps_for_val <= 0:
            raise ValueError("num_steps_for_val must be a positive integer.")
        if step_idx == -1:
            logger.info("Using trainset as validation set.")
        valset_evaluator = make_optimizer_evaluator(
            run,
            devset=trainset,
            metric=metric,
            max_concurrency=max_concurrency,
            max_errors=len(trainset) * 10,
            display_progress=True,
            provide_traceback=False,
            failure_score=failure_score,
        )
        if step_idx == -1:
            logger.info("Evaluating the student program on the validation set before training loop...")
        else:
            logger.info(
                f"Evaluating the student program on the validation set after training step "
                f"{step_idx + 1}/{num_train_steps}"
            )
        valset_evaluation = await valset_evaluator(student, run=run, metric=metric)
        if step_idx == -1:
            logger.info(f"Student program training set score before training loop: {valset_evaluation.score}")
        else:
            logger.info(
                f"Student program training set score after training step {step_idx + 1}/{num_train_steps}: "
                f"{valset_evaluation.score}"
            )
    else:
        if use_train_as_val:
            raise ValueError("If report_train_scores is False, use_train_as_val must be False.")
        if step_idx == -1:
            logger.info("Not using any validation set and not reporting train scores.")

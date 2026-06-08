import logging
from typing import TYPE_CHECKING, Any

from dspy.dsp.utils.settings import settings
from dspy.teleprompt.mipro.settings import (
    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
    LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT,
)
from dspy.teleprompt.utils import create_n_fewshot_demo_sets

if TYPE_CHECKING:
    from dspy.teleprompt.mipro.optimizer import MIPROv2

logger = logging.getLogger(__name__)


async def bootstrap_fewshot_examples(
    optimizer: "MIPROv2",
    program: Any,
    trainset: list,
    seed: int,
    teacher: Any,
    *,
    num_fewshot_candidates: int,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
    max_errors: int | None,
    metric_threshold: float | None,
) -> list | None:
    logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
    if max_bootstrapped_demos > 0:
        logger.info(
            "These will be used as few-shot example candidates for our program and for creating instructions.\n"
        )
    else:
        logger.info("These will be used for informing instruction proposal.\n")

    logger.info(f"Bootstrapping N={num_fewshot_candidates} sets of demonstrations...")

    zeroshot = max_bootstrapped_demos == 0 and max_labeled_demos == 0

    if max_errors is None:
        max_errors = settings.max_errors

    return await create_n_fewshot_demo_sets(
        student=program,
        num_candidate_sets=num_fewshot_candidates,
        trainset=trainset,
        max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_labeled_demos),
        max_bootstrapped_demos=(BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_bootstrapped_demos),
        metric=optimizer.metric,
        max_errors=max_errors,
        teacher=teacher,
        teacher_settings=optimizer.teacher_settings,
        seed=seed,
        metric_threshold=metric_threshold,
        rng=optimizer.rng,
    )
    # Bootstrapping failures intentionally propagate because running MIPRO without few-shot candidates weakens the optimization substantially.

from __future__ import annotations

import random
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dspy.runtime.run_context import RunContext  # noqa: TC001 — used at runtime in async signatures
from dspy.teleprompt.bootstrap import BootstrapFewShot, LabeledFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, LabeledFewShotCompileParams

if TYPE_CHECKING:
    from dspy.teleprompt.metrics import OptimizerMetric


class CandidateSeedKind(StrEnum):
    BASELINE = "baseline"
    LABELED_FEWSHOT = "labeled_fewshot"
    BOOTSTRAP = "bootstrap"
    RANDOMIZED_BOOTSTRAP = "randomized_bootstrap"


class BaselineSeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[CandidateSeedKind.BASELINE] = CandidateSeedKind.BASELINE


class LabeledFewShotSeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[CandidateSeedKind.LABELED_FEWSHOT] = CandidateSeedKind.LABELED_FEWSHOT


class BootstrapSeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[CandidateSeedKind.BOOTSTRAP] = CandidateSeedKind.BOOTSTRAP


class RandomizedBootstrapSeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[CandidateSeedKind.RANDOMIZED_BOOTSTRAP] = CandidateSeedKind.RANDOMIZED_BOOTSTRAP
    index: int


CandidateSeed = Annotated[
    BaselineSeed | LabeledFewShotSeed | BootstrapSeed | RandomizedBootstrapSeed,
    Field(discriminator="kind"),
]


class CandidateLadderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_random: int = 16
    include_baseline: bool = True
    include_labeled_fewshot: bool = True
    include_bootstrap: bool = True
    max_labeled_demos: int
    max_bootstrapped_demos: int
    min_bootstrapped_demos: int = 1


def iter_candidate_seeds(config: CandidateLadderConfig) -> list[CandidateSeed]:
    seeds: list[CandidateSeed] = []
    if config.include_baseline:
        seeds.append(BaselineSeed())
    if config.include_labeled_fewshot and config.max_labeled_demos > 0:
        seeds.append(LabeledFewShotSeed())
    if config.include_bootstrap:
        seeds.append(BootstrapSeed())
    seeds.extend(RandomizedBootstrapSeed(index=i) for i in range(config.num_random))
    return seeds


async def compile_candidate_program(
    *,
    seed: CandidateSeed,
    student: Any,
    trainset: list,
    run: RunContext,
    metric: OptimizerMetric,
    teacher: Any = None,
    teacher_run: RunContext | None = None,
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    min_bootstrapped_demos: int = 1,
    max_rounds: int = 1,
    max_errors: int | None = None,
    metric_threshold: float | None = None,
    labeled_sample: bool = True,
) -> Any:
    trainset_copy = list(trainset)
    if isinstance(seed, BaselineSeed):
        return student.reset_copy()
    if isinstance(seed, LabeledFewShotSeed):
        teleprompter = LabeledFewShot(k=max_labeled_demos)
        result = await teleprompter.compile(
            student,
            params=LabeledFewShotCompileParams(trainset=trainset_copy, sample=labeled_sample),
            run=run,
        )
        return result.program
    if isinstance(seed, BootstrapSeed):
        bootstrapped_demos = max_bootstrapped_demos
    elif isinstance(seed, RandomizedBootstrapSeed):
        rng = random.Random(seed.index)
        rng.shuffle(trainset_copy)
        bootstrapped_demos = rng.randint(min_bootstrapped_demos, max_bootstrapped_demos)
    else:
        raise TypeError(f"Unsupported candidate seed: {seed!r}")
    teleprompter = BootstrapFewShot(
        metric=metric,
        max_errors=max_errors,
        metric_threshold=metric_threshold,
        max_bootstrapped_demos=bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        teacher_run=teacher_run,
        max_rounds=max_rounds,
    )
    result = await teleprompter.compile(
        student,
        params=BootstrapFewShotCompileParams(trainset=trainset_copy, teacher=teacher),
        run=run,
    )
    return result.program


async def generate_demo_candidate_sets(
    *,
    student: Any,
    config: CandidateLadderConfig,
    trainset: list,
    run: RunContext,
    metric: OptimizerMetric,
    teacher: Any = None,
    teacher_run: RunContext | None = None,
    max_errors: int | None = None,
    max_rounds: int = 1,
    metric_threshold: float | None = None,
    labeled_sample: bool = True,
) -> dict[int, list]:
    if max_errors is None:
        max_errors = run.execution.max_errors
    demo_candidates: dict[int, list] = {}
    for i, _ in enumerate(student.predictors()):
        demo_candidates[i] = []
    for seed in iter_candidate_seeds(config):
        program = await compile_candidate_program(
            seed=seed,
            student=student,
            trainset=trainset,
            run=run,
            metric=metric,
            teacher=teacher,
            teacher_run=teacher_run,
            max_labeled_demos=config.max_labeled_demos,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            min_bootstrapped_demos=config.min_bootstrapped_demos,
            max_rounds=max_rounds,
            max_errors=max_errors,
            metric_threshold=metric_threshold,
            labeled_sample=labeled_sample,
        )
        for i, _ in enumerate(student.predictors()):
            demo_candidates[i].append(program.predictors()[i].demos)
    return demo_candidates

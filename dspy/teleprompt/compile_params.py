from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dspy.primitives import Example, Module  # noqa: TC001 — pydantic field types
from dspy.teleprompt.bettertogether_types import DEFAULT_BETTER_TOGETHER_STRATEGY


class EvaluateCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int | None = None
    display_progress: bool = False
    display_table: bool | int = False
    max_errors: int | None = None
    provide_traceback: bool | None = None
    failure_score: float = 0.0
    save_as_csv: str | None = None
    save_as_json: str | None = None


class LabeledFewShotCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]
    sample: bool = True


class BootstrapFewShotCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]
    teacher: Module | list[Module] | None = None


class BootstrapOptunaCompileParams(BootstrapFewShotCompileParams):
    max_demos: int
    valset: list[Example] | None = None


class RandomSearchCompileParams(BootstrapFewShotCompileParams):
    valset: list[Example] | None = None
    restrict: list[int] | None = None
    labeled_sample: bool = True
    include_baselines: bool = True


class COPROCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]
    evaluate: EvaluateCompileParams = Field(default_factory=EvaluateCompileParams)


class MIPROv2CompileParams(BootstrapFewShotCompileParams):
    valset: list[Example] | None = None
    num_trials: int | None = None
    max_bootstrapped_demos: int | None = None
    max_labeled_demos: int | None = None
    seed: int | None = None
    minibatch: bool = True
    minibatch_size: int = 35
    minibatch_full_eval_steps: int = 5
    program_aware_proposer: bool = True
    data_aware_proposer: bool = True
    view_data_batch_size: int = 10
    tip_aware_proposer: bool = True
    fewshot_aware_proposer: bool = True
    provide_traceback: bool | None = None


class SIMBACompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]
    seed: int = 0


class GEPACompileParams(BootstrapFewShotCompileParams):
    valset: list[Example] | None = None


class AvatarOptimizerCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]


class EnsembleCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    programs: list[Module]


class KNNFewShotCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    teacher: Module | list[Module] | None = None


class InferRulesCompileParams(BootstrapFewShotCompileParams):
    valset: list[Example] | None = None
    split_seed: int = 0


class GRPOCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]
    teacher: Module | list[Module] | None = None
    valset: list[Example] | None = None


class BetterTogetherCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example]
    teacher: Module | list[Module] | None = None
    valset: list[Example] | None = None
    max_concurrency: int | None = None
    max_errors: int | None = None
    provide_traceback: bool | None = None
    seed: int | None = None
    valset_ratio: float = 0.1
    shuffle_trainset_between_steps: bool = True
    strategy: list[str] = Field(default_factory=lambda: list(DEFAULT_BETTER_TOGETHER_STRATEGY))
    optimizer_compile_args: dict[str, BaseModel] | None = None

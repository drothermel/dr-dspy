"""Curated public surface for DSPy teleprompters (optimizers)."""

from dspy.runtime.async_parallel import resolve_max_errors
from dspy.teleprompt.avatar_optimizer import AvatarOptimizer
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.candidate_ladder import (
    CandidateLadderConfig,
    CandidateSeed,
    CandidateSeedKind,
    compile_candidate_program,
    generate_demo_candidate_sets,
)
from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate
from dspy.teleprompt.compile_params import (
    AvatarOptimizerCompileParams,
    BetterTogetherCompileParams,
    BootstrapFewShotCompileParams,
    BootstrapOptunaCompileParams,
    COPROCompileParams,
    EnsembleCompileParams,
    EvaluateCompileParams,
    GEPACompileParams,
    GRPOCompileParams,
    InferRulesCompileParams,
    KNNFewShotCompileParams,
    LabeledFewShotCompileParams,
    MIPROv2CompileParams,
    RandomSearchCompileParams,
    SIMBACompileParams,
)
from dspy.teleprompt.copro_optimizer import COPRO
from dspy.teleprompt.ensemble import Ensemble
from dspy.teleprompt.gepa.gepa import GEPA
from dspy.teleprompt.grpo import GRPO
from dspy.teleprompt.infer_rules import InferRules
from dspy.teleprompt.knn_fewshot import KNNFewShot
from dspy.teleprompt.mipro.optimizer import MIPROv2
from dspy.teleprompt.protocol import Teleprompter
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.registry import compile_params_type, register_teleprompter
from dspy.teleprompt.simba import SIMBA
from dspy.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna
from dspy.teleprompt.trace_helpers import run_program_with_trace, trace_to_demos
from dspy.teleprompt.utils import make_optimizer_evaluator
from dspy.teleprompt.vanilla import LabeledFewShot

__all__ = [
    "AvatarOptimizer",
    "AvatarOptimizerCompileParams",
    "BetterTogether",
    "BetterTogetherCompileParams",
    "BootstrapFewShot",
    "BootstrapFewShotCompileParams",
    "BootstrapFewShotWithOptuna",
    "BootstrapFewShotWithRandomSearch",
    "BootstrapFinetune",
    "BootstrapOptunaCompileParams",
    "COPRO",
    "COPROCompileParams",
    "CandidateLadderConfig",
    "CandidateSeed",
    "CandidateSeedKind",
    "CompileResult",
    "CompileStats",
    "Ensemble",
    "EnsembleCompileParams",
    "EvaluateCompileParams",
    "GEPA",
    "GEPACompileParams",
    "GRPO",
    "GRPOCompileParams",
    "InferRules",
    "InferRulesCompileParams",
    "KNNFewShot",
    "KNNFewShotCompileParams",
    "LabeledFewShot",
    "LabeledFewShotCompileParams",
    "MIPROv2",
    "MIPROv2CompileParams",
    "ProgramCandidate",
    "RandomSearchCompileParams",
    "SIMBA",
    "SIMBACompileParams",
    "Teleprompter",
    "compile_candidate_program",
    "compile_params_type",
    "generate_demo_candidate_sets",
    "make_optimizer_evaluator",
    "register_teleprompter",
    "resolve_max_errors",
    "run_program_with_trace",
    "trace_to_demos",
]

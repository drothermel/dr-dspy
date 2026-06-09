"""Curated public surface for DSPy teleprompters (optimizers)."""

from __future__ import annotations

import importlib
from typing import Any

from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate
from dspy.teleprompt.protocol import Teleprompter
from dspy.teleprompt.registry import compile_params_type, register_teleprompter

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AvatarOptimizer": ("dspy.teleprompt.avatar_optimizer", "AvatarOptimizer"),
    "AvatarOptimizerCompileParams": ("dspy.teleprompt.compile_params", "AvatarOptimizerCompileParams"),
    "BetterTogether": ("dspy.teleprompt.bettertogether", "BetterTogether"),
    "BetterTogetherCompileParams": ("dspy.teleprompt.compile_params", "BetterTogetherCompileParams"),
    "BetterTogetherBuiltinKey": ("dspy.teleprompt.bettertogether_types", "BetterTogetherBuiltinKey"),
    "BootstrapFewShot": ("dspy.teleprompt.bootstrap", "BootstrapFewShot"),
    "BootstrapFewShotCompileParams": ("dspy.teleprompt.compile_params", "BootstrapFewShotCompileParams"),
    "BootstrapFewShotWithOptuna": ("dspy.teleprompt.teleprompt_optuna", "BootstrapFewShotWithOptuna"),
    "BootstrapFewShotWithRandomSearch": ("dspy.teleprompt.random_search", "BootstrapFewShotWithRandomSearch"),
    "BootstrapFinetune": ("dspy.teleprompt.bootstrap_finetune", "BootstrapFinetune"),
    "BootstrapOptunaCompileParams": ("dspy.teleprompt.compile_params", "BootstrapOptunaCompileParams"),
    "COPRO": ("dspy.teleprompt.copro_optimizer", "COPRO"),
    "COPROCompileParams": ("dspy.teleprompt.compile_params", "COPROCompileParams"),
    "CandidateLadderConfig": ("dspy.teleprompt.candidate_ladder", "CandidateLadderConfig"),
    "CandidateSeedKind": ("dspy.teleprompt.candidate_ladder", "CandidateSeedKind"),
    "Ensemble": ("dspy.teleprompt.ensemble", "Ensemble"),
    "EnsembleCompileParams": ("dspy.teleprompt.compile_params", "EnsembleCompileParams"),
    "EvaluateCompileParams": ("dspy.teleprompt.compile_params", "EvaluateCompileParams"),
    "GEPA": ("dspy.teleprompt.gepa.gepa", "GEPA"),
    "GEPACompileParams": ("dspy.teleprompt.compile_params", "GEPACompileParams"),
    "GRPO": ("dspy.teleprompt.grpo.optimizer", "GRPO"),
    "GRPOCompileParams": ("dspy.teleprompt.compile_params", "GRPOCompileParams"),
    "InferRules": ("dspy.teleprompt.infer_rules", "InferRules"),
    "InferRulesCompileParams": ("dspy.teleprompt.compile_params", "InferRulesCompileParams"),
    "KNNFewShot": ("dspy.teleprompt.knn_fewshot", "KNNFewShot"),
    "KNNFewShotCompileParams": ("dspy.teleprompt.compile_params", "KNNFewShotCompileParams"),
    "LabeledFewShot": ("dspy.teleprompt.vanilla", "LabeledFewShot"),
    "LabeledFewShotCompileParams": ("dspy.teleprompt.compile_params", "LabeledFewShotCompileParams"),
    "MIPROv2": ("dspy.teleprompt.mipro.optimizer", "MIPROv2"),
    "MIPROv2CompileParams": ("dspy.teleprompt.compile_params", "MIPROv2CompileParams"),
    "OptimizerMetric": ("dspy.teleprompt.metrics", "OptimizerMetric"),
    "RandomSearchCompileParams": ("dspy.teleprompt.compile_params", "RandomSearchCompileParams"),
    "SIMBA": ("dspy.teleprompt.simba", "SIMBA"),
    "SIMBACompileParams": ("dspy.teleprompt.compile_params", "SIMBACompileParams"),
    "compile_candidate_program": ("dspy.teleprompt.candidate_ladder", "compile_candidate_program"),
    "collect_trace_data": ("dspy.teleprompt.core.trace_collection", "collect_trace_data"),
    "generate_demo_candidate_sets": ("dspy.teleprompt.candidate_ladder", "generate_demo_candidate_sets"),
    "make_optimizer_evaluator": ("dspy.teleprompt.core.evaluator", "make_optimizer_evaluator"),
    "resolve_max_errors": ("dspy.runtime.async_parallel", "resolve_max_errors"),
    "trace_to_demos": ("dspy.teleprompt.core.demos", "trace_to_demos"),
}

# Type alias exported from candidate_ladder (not a module attribute).
_LAZY_EXPORTS["CandidateSeed"] = ("dspy.teleprompt.candidate_ladder", "CandidateSeed")

__all__ = [
    "AvatarOptimizer",
    "AvatarOptimizerCompileParams",
    "BetterTogether",
    "BetterTogetherBuiltinKey",
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
    "OptimizerMetric",
    "ProgramCandidate",
    "RandomSearchCompileParams",
    "SIMBA",
    "SIMBACompileParams",
    "Teleprompter",
    "compile_candidate_program",
    "collect_trace_data",
    "compile_params_type",
    "generate_demo_candidate_sets",
    "make_optimizer_evaluator",
    "register_teleprompter",
    "resolve_max_errors",
    "trace_to_demos",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        obj = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

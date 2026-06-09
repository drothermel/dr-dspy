from __future__ import annotations

import inspect
import logging
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from typing_extensions import override

from dspy.primitives.prediction import Prediction
from dspy.teleprompt.compile_params import GEPACompileParams
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

if TYPE_CHECKING:
    from gepa import GEPAResult
    from gepa.core.adapter import ProposalFn
    from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector
    from pydantic import BaseModel

    from dspy.clients.lm import LM
    from dspy.primitives.example import Example
    from dspy.primitives.module import Module
    from dspy.runtime.run_context import RunContext
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, DSPyTrace, PredictorFeedbackFn, ScoreWithFeedback
logger = logging.getLogger(__name__)
AUTO_RUN_SETTINGS = {"light": {"n": 6}, "medium": {"n": 12}, "heavy": {"n": 18}}


@experimental(version="3.0.0")
class GEPAFeedbackMetric(Protocol):
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: DSPyTrace | None,
        pred_name: str | None,
        pred_trace: DSPyTrace | None,
    ) -> float | ScoreWithFeedback: ...


@experimental(version="3.0.0")
@dataclass(frozen=True)
class DspyGEPAResult:
    candidates: list[Module]
    parents: list[list[int | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[Any, float]]
    per_val_instance_best_candidates: dict[Any, set[int]]
    discovery_eval_counts: list[int]
    best_outputs_valset: dict[Any, list[tuple[int, Prediction]]] | None = None
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    log_dir: str | None = None
    seed: int | None = None

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> Module:
        return self.candidates[self.best_idx]

    @property
    def highest_score_achieved_per_val_task(self) -> dict[Any, float]:
        return {
            val_id: self.val_subscores[next(iter(self.per_val_instance_best_candidates[val_id]))][val_id]
            for val_id in self.per_val_instance_best_candidates
        }

    def to_dict(self) -> dict[str, Any]:
        cands = [
            {name: pred.task_spec.instructions for name, pred in cand.named_predictors()} for cand in self.candidates
        ]
        return {
            "candidates": cands,
            "parents": self.parents,
            "val_aggregate_scores": self.val_aggregate_scores,
            "best_outputs_valset": self.best_outputs_valset,
            "val_subscores": self.val_subscores,
            "per_val_instance_best_candidates": {
                val_id: list(s) for val_id, s in self.per_val_instance_best_candidates.items()
            },
            "discovery_eval_counts": self.discovery_eval_counts,
            "total_metric_calls": self.total_metric_calls,
            "num_full_val_evals": self.num_full_val_evals,
            "log_dir": self.log_dir,
            "seed": self.seed,
            "best_idx": self.best_idx,
        }

    @staticmethod
    def from_gepa_result(gepa_result: GEPAResult, adapter: DspyAdapter) -> DspyGEPAResult:
        return DspyGEPAResult(
            candidates=[adapter.build_program(c) for c in gepa_result.candidates],
            parents=gepa_result.parents,
            val_aggregate_scores=gepa_result.val_aggregate_scores,
            best_outputs_valset=gepa_result.best_outputs_valset,
            val_subscores=gepa_result.val_subscores,
            per_val_instance_best_candidates=gepa_result.per_val_instance_best_candidates,
            discovery_eval_counts=gepa_result.discovery_eval_counts,
            total_metric_calls=gepa_result.total_metric_calls,
            num_full_val_evals=gepa_result.num_full_val_evals,
            log_dir=gepa_result.run_dir,
            seed=gepa_result.seed,
        )


@experimental(version="3.0.0")
class GEPA(Teleprompter):
    def __init__(
        self,
        metric: GEPAFeedbackMetric,
        *,
        auto: Literal["light", "medium", "heavy"] | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        reflection_lm: LM | None = None,
        skip_perfect_score: bool = True,
        add_format_failure_as_feedback: bool = False,
        instruction_proposer: ProposalFn | None = None,
        component_selector: ReflectionComponentSelector | str = "round_robin",
        use_merge: bool = True,
        max_merge_invocations: int | None = 5,
        max_concurrency: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        log_dir: str | None = None,
        track_stats: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        warn_on_score_mismatch: bool = True,
        use_mlflow: bool = False,
        seed: int | None = 0,
        gepa_kwargs: dict | None = None,
    ) -> None:
        try:
            inspect.signature(metric).bind(None, None, None, None, None)
        except TypeError as e:
            raise TypeError(
                "GEPA metric must accept five arguments: (gold, pred, trace, pred_name, pred_trace). See https://dspy.ai/api/optimizers/GEPA for details."
            ) from e
        self.metric_fn = metric
        assert (max_metric_calls is not None) + (max_full_evals is not None) + (auto is not None) == 1, (
            f"Exactly one of max_metric_calls, max_full_evals, auto must be set. You set max_metric_calls={max_metric_calls}, max_full_evals={max_full_evals}, auto={auto}."
        )
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        assert reflection_lm is not None or instruction_proposer is not None, (
            "GEPA requires a reflection language model, or custom instruction proposer to be provided. Typically, you can use `from dspy.clients.lm import LM; LM(model='gpt-5', temperature=1.0, max_tokens=32000)` to get a good reflection model. Reflection LM is used by GEPA to reflect on the behavior of the program and propose new instructions, and will benefit from a strong model. "
        )
        self.reflection_lm = reflection_lm
        self.skip_perfect_score = skip_perfect_score
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.max_concurrency = max_concurrency
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.use_mlflow = use_mlflow
        if track_best_outputs:
            assert track_stats, "track_stats must be True if track_best_outputs is True."
        self.track_best_outputs = track_best_outputs
        self.seed = seed
        self.custom_instruction_proposer = instruction_proposer
        self.component_selector = component_selector
        self.gepa_kwargs = gepa_kwargs or {}
        if "reflection_prompt_template" in self.gepa_kwargs:
            raise ValueError(
                "reflection_prompt_template cannot be passed via gepa_kwargs when using dspy.GEPA. DspyAdapter implements its own propose_new_texts, so reflection_prompt_template is unused. To customize reflection behavior, pass a custom ProposalFn via the instruction_proposer parameter instead."
            )

    def auto_budget(
        self, num_preds, num_candidates, valset_size: int, minibatch_size: int = 35, full_eval_steps: int = 5
    ) -> int:
        num_trials = int(max(2 * (num_preds * 2) * math.log2(num_candidates), 1.5 * num_candidates))
        if num_trials < 0 or valset_size < 0 or minibatch_size < 0:
            raise ValueError("num_trials, valset_size, and minibatch_size must be >= 0.")
        if full_eval_steps < 1:
            raise ValueError("full_eval_steps must be >= 1.")
        V = valset_size
        N = num_trials
        M = minibatch_size
        m = full_eval_steps
        total = V
        total += num_candidates * 5
        total += N * M
        if N == 0:
            return total
        periodic_fulls = (N + 1) // m + 1
        extra_final = 1 if m > N else 0
        total += (periodic_fulls + extra_final) * V
        return total

    @override
    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> Module:
        params = GEPACompileParams.model_validate(params)
        trainset = params.trainset
        teacher = params.teacher
        valset = params.valset
        from gepa import optimize

        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, LoggerAdapter, ScoreWithFeedback

        assert trainset is not None and len(trainset) > 0, "Trainset must be provided and non-empty"
        assert teacher is None, "Teacher is not supported in DspyGEPA yet."
        if self.auto is not None:
            self.max_metric_calls = self.auto_budget(
                num_preds=len(student.predictors()),
                num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
                valset_size=len(valset) if valset is not None else len(trainset),
            )
        elif self.max_full_evals is not None:
            self.max_metric_calls = self.max_full_evals * (len(trainset) + (len(valset) if valset is not None else 0))
        else:
            assert self.max_metric_calls is not None, "Either auto, max_full_evals, or max_metric_calls must be set."
        logger.info(
            f"Running GEPA for approx {self.max_metric_calls} metric calls of the program. This amounts to {(self.max_metric_calls / len(trainset) if valset is None else self.max_metric_calls / (len(trainset) + len(valset))):.2f} full evals on the {('train' if valset is None else 'train+val')} set."
        )
        if valset is None:
            logger.warning(
                "No valset provided; Using trainset as valset. This is useful as an inference-time scaling strategy where you want GEPA to find the best solutions for the provided tasks in the trainset, as it makes GEPA overfit prompts to the provided trainset. In order to ensure generalization and perform well on unseen tasks, please provide separate trainset and valset. Provide the smallest valset that is just large enough to match the downstream task distribution, while keeping trainset as large as possible."
            )
        valset = valset or trainset
        if len(valset) > 35:
            logger.info(
                f"Using {len(valset)} examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget. GEPA requires you to provide the smallest valset that is just large enough to match your downstream task distribution, while providing as large trainset as possible."
            )
        else:
            logger.info(f"Using {len(valset)} examples for tracking Pareto scores.")
        rng = random.Random(self.seed)

        def feedback_fn_creator(pred_name: str, predictor) -> PredictorFeedbackFn:

            def feedback_fn(
                predictor_output: dict[str, Any],
                predictor_inputs: dict[str, Any],
                module_inputs: Example,
                module_outputs: Prediction,
                captured_trace: DSPyTrace,
            ) -> ScoreWithFeedback:
                pred_output = Prediction(**predictor_output) if isinstance(predictor_output, dict) else predictor_output
                trace_for_pred: DSPyTrace = [(predictor, predictor_inputs, pred_output)]
                o = self.metric_fn(module_inputs, module_outputs, captured_trace, pred_name, trace_for_pred)
                if isinstance(o, ScoreWithFeedback):
                    if o.feedback is None:
                        o.feedback = f"This trajectory got a score of {o.score}."
                    return o
                if isinstance(o, (int, float)):
                    return ScoreWithFeedback(score=float(o), feedback=f"This trajectory got a score of {o}.")
                raise TypeError(f"Unexpected metric return type: {type(o).__name__}")

            return feedback_fn

        feedback_map = {k: feedback_fn_creator(k, v) for k, v in student.named_predictors()}
        adapter = DspyAdapter(
            student_module=student,
            metric_fn=self.metric_fn,
            feedback_map=feedback_map,
            failure_score=self.failure_score,
            max_concurrency=self.max_concurrency,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=rng,
            reflection_lm=self.reflection_lm,
            custom_instruction_proposer=self.custom_instruction_proposer,
            warn_on_score_mismatch=self.warn_on_score_mismatch,
            reflection_minibatch_size=self.reflection_minibatch_size,
            run=run,
        )
        seed_candidate = {name: pred.task_spec.instructions for name, pred in student.named_predictors()}
        gepa_result: GEPAResult = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=None,
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            module_selector=self.component_selector,
            perfect_score=self.perfect_score,
            use_merge=self.use_merge,
            max_merge_invocations=cast("Any", self.max_merge_invocations),
            max_metric_calls=self.max_metric_calls,
            logger=cast("Any", LoggerAdapter(logger)),
            run_dir=self.log_dir,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            wandb_init_kwargs=self.wandb_init_kwargs,
            use_mlflow=self.use_mlflow,
            track_best_outputs=self.track_best_outputs,
            display_progress_bar=True,
            raise_on_exception=True,
            seed=cast("Any", self.seed),
            **self.gepa_kwargs,
        )
        new_prog = adapter.build_program(cast("dict[str, str]", gepa_result.best_candidate))
        if self.track_stats:
            dspy_gepa_result = DspyGEPAResult.from_gepa_result(gepa_result, adapter)
            new_prog.detailed_results = dspy_gepa_result
        return new_prog

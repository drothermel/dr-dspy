from copy import deepcopy
from random import sample
from typing import Any, cast

from pydantic import BaseModel

from dspy.evaluate.metric_invoke import invoke_metric
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.runtime import run_with_trace
from dspy.runtime.call_options import ModuleCallOptions
from dspy.runtime.run_context import RunContext
from dspy.task_spec.predictor_context import get_task_spec, set_task_spec
from dspy.teleprompt.avatar.task_specs import (
    ComparatorTaskSpec,
    EvalResult,
    FeedbackBasedInstructionTaskSpec,
)
from dspy.teleprompt.compilation import CompileResult, CompileStats
from dspy.teleprompt.compile_params import AvatarOptimizerCompileParams
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.registry import register_teleprompter

DEFAULT_MAX_EXAMPLES = 10


class _AvatarEvalModule(Module):
    def __init__(self, optimizer: "AvatarOptimizer", actor: Module, return_outputs: bool) -> None:
        super().__init__()
        self._optimizer = optimizer
        self._actor = actor
        self._return_outputs = return_outputs

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        example = Example.from_record(inputs)
        return await self._optimizer.process_example(
            actor=self._actor, example=example, return_outputs=self._return_outputs, run=run
        )


@register_teleprompter(params=AvatarOptimizerCompileParams)
class AvatarOptimizer:
    def __init__(
        self,
        metric: OptimizerMetric,
        max_iters: int = 10,
        lower_bound: int = 0,
        upper_bound: int = 1,
        max_positive_inputs: int | None = None,
        max_negative_inputs: int | None = None,
        optimize_for: str = "max",
    ) -> None:
        assert metric is not None, "`metric` argument cannot be None. Please provide a metric function."
        self.metric = metric
        self.optimize_for = optimize_for
        self.max_iters = max_iters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_positive_inputs = max_positive_inputs or DEFAULT_MAX_EXAMPLES
        self.max_negative_inputs = max_negative_inputs or DEFAULT_MAX_EXAMPLES
        self.comparator = Predict(ComparatorTaskSpec())
        self.feedback_instruction = Predict(FeedbackBasedInstructionTaskSpec())

    async def process_example(self, actor, example, return_outputs, *, run: RunContext):
        actor = deepcopy(actor)
        try:
            prediction, trace = await run_with_trace(actor, example, run)
            score = await invoke_metric(
                self.metric,
                example=example,
                prediction=prediction,
                trace=trace,
                run=run,
            )
            if return_outputs:
                return (example, prediction, score)
            return score
        except Exception:
            if return_outputs:
                return (example, None, 0)
            return 0

    async def thread_safe_evaluator(
        self, devset, actor, return_outputs=False, max_concurrency=None, *, run: RunContext
    ):
        total_score = 0
        total_examples = len(devset)
        max_concurrency = max_concurrency or run.execution.max_concurrency
        eval_module = _AvatarEvalModule(self, actor, return_outputs)
        run_parallel = Parallel(run=run, max_concurrency=max_concurrency, disable_progress_bar=False)
        exec_pairs = [(eval_module, example) for example in devset]
        parallel_results = (await run_parallel(exec_pairs)).results
        if return_outputs:
            results = []
            for result in parallel_results:
                example, prediction, score = cast("tuple[Any, Any, float]", result)
                total_score += score
                results.append((example, prediction, score))
            avg_metric = total_score / total_examples
            return (avg_metric, results)
        total_score = sum(cast("float", score) for score in parallel_results)
        return total_score / total_examples

    async def _get_pos_neg_results(
        self, actor: Module, trainset: list[Example], *, run: RunContext
    ) -> tuple[float, list[EvalResult], list[EvalResult]]:
        pos_inputs = []
        neg_inputs = []
        avg_score, results = await self.thread_safe_evaluator(
            devset=trainset, actor=actor, return_outputs=True, run=run
        )
        for example, prediction, score in results:
            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example.as_inputs(),
                        score=score,
                        actions=prediction.actions if prediction else None,
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example.as_inputs(),
                        score=score,
                        actions=prediction.actions if prediction else None,
                    )
                )
        if len(pos_inputs) == 0:
            raise ValueError("No positive examples found, try lowering the upper_bound or providing more training data")
        if len(neg_inputs) == 0:
            raise ValueError("No negative examples found, try raising the lower_bound or providing more training data")
        return (avg_score, pos_inputs, neg_inputs)

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = AvatarOptimizerCompileParams.model_validate(params)
        trainset = params.trainset
        best_actor = deepcopy(student)
        best_score = -999 if self.optimize_for == "max" else 999
        for _i in range(self.max_iters):
            score, pos_inputs, neg_inputs = await self._get_pos_neg_results(best_actor, trainset, run=run)
            if self.max_positive_inputs and len(pos_inputs) > self.max_positive_inputs:
                pos_inputs = sample(pos_inputs, self.max_positive_inputs)
            if self.max_negative_inputs and len(neg_inputs) > self.max_negative_inputs:
                neg_inputs = sample(neg_inputs, self.max_negative_inputs)
            actor_task_spec = get_task_spec(best_actor.actor)
            feedback = (
                await self.comparator(
                    instruction=actor_task_spec.instructions,
                    actions=[str(tool) for tool in best_actor.tools],
                    pos_input_with_metrics=pos_inputs,
                    neg_input_with_metrics=neg_inputs,
                    run=run,
                )
            ).feedback
            new_instruction = (
                await self.feedback_instruction(
                    previous_instruction=actor_task_spec.instructions, feedback=feedback, run=run
                )
            ).new_instruction
            if (self.optimize_for == "max" and best_score < score) or (
                self.optimize_for == "min" and best_score > score
            ):
                set_task_spec(predictor=best_actor.actor, task_spec=actor_task_spec.with_instructions(new_instruction))
                best_actor.actor_clone = deepcopy(best_actor.actor)
                best_score = score
        return CompileResult(program=best_actor, stats=CompileStats(best_score=best_score))

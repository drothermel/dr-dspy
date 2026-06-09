import json
import logging
import random
from typing import Any, Callable, Protocol, TypedDict, cast

from typing_extensions import override

from dspy._internal.lazy_import import _detect_dspy_dist
from dspy.adapters.types.field_type import is_field_type
from dspy.history import TurnLog
from dspy.integrations.optimizers.gepa.sync_bridge import run_gepa_sync
from dspy.integrations.optimizers.gepa.task_specs import FrameworkGepaInstructionProposalTaskSpec
from dspy.predict.predict import Predict
from dspy.primitives import Example, Prediction
from dspy.runtime.optimization_trace import FailedPrediction, TraceData
from dspy.runtime.run_context import RunContext
from dspy.runtime.transparency.resolve import require_adapter
from dspy.serialization.json import to_jsonable
from dspy.task_spec.predictor_context import get_task_spec, set_task_spec
from dspy.teleprompt.core.evaluator import make_optimizer_evaluator, optimizer_lm_context
from dspy.teleprompt.core.trace_collection import collect_trace_data

try:
    from gepa import EvaluationBatch, GEPAAdapter
except ImportError as err:
    raise ImportError(
        f"The 'gepa' extra is required to use GEPA integrations. "
        f"Install it with `pip install {_detect_dspy_dist()}[gepa]`."
    ) from err

logger = logging.getLogger(__name__)


class AsyncProposalFn(Protocol):
    async def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]: ...


class LoggerAdapter:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def log(self, x: str) -> None:
        self.logger.info(x)


DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]
ReflectiveExample = TypedDict(
    "ReflectiveExample", {"Inputs": dict[str, Any], "Generated Outputs": dict[str, Any] | str, "Feedback": str}
)


class ScoreWithFeedback(Prediction):
    """Prediction carrying optimizer ``score`` and ``feedback`` fields."""


class PredictorFeedbackFn(Protocol):
    def __call__(
        self,
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Example,
        module_outputs: Prediction,
        captured_trace: DSPyTrace,
    ) -> ScoreWithFeedback: ...


class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: dict[str, Callable],
        failure_score=0.0,
        max_concurrency: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
        reflection_lm=None,
        custom_instruction_proposer: AsyncProposalFn | None = None,
        warn_on_score_mismatch: bool = True,
        reflection_minibatch_size: int | None = None,
        run: RunContext | None = None,
    ) -> None:
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.max_concurrency = max_concurrency
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)
        self.reflection_lm = reflection_lm
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.reflection_minibatch_size = reflection_minibatch_size
        self.run = run

    @override
    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        return run_gepa_sync(
            self._apropose_new_texts(
                candidate=candidate,
                reflective_dataset=reflective_dataset,
                components_to_update=components_to_update,
            )
        )

    async def _apropose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.run is None:
            raise ValueError("DspyAdapter requires a RunContext.")
        reflection_lm = self.reflection_lm or self.run.lm
        if self.custom_instruction_proposer:
            with optimizer_lm_context(
                self.run, lm=reflection_lm, phase="gepa.reflection", lm_role="reflection_lm"
            ) as opt_run:
                return await self.custom_instruction_proposer(
                    candidate=candidate,
                    reflective_dataset=reflective_dataset,
                    components_to_update=components_to_update,
                )
        results: dict[str, str] = {}
        proposer = Predict(FrameworkGepaInstructionProposalTaskSpec())
        with optimizer_lm_context(
            self.run, lm=reflection_lm, phase="gepa.reflection", lm_role="reflection_lm"
        ) as opt_run:
            for name in components_to_update:
                base_instruction = candidate[name]
                dataset_with_feedback = reflective_dataset[name]
                prediction = await proposer(
                    current_instruction_doc=base_instruction,
                    dataset_with_feedback=json.dumps(to_jsonable(dataset_with_feedback), indent=2),
                    run=opt_run,
                )
                results[name] = prediction.new_instruction
        return results

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                set_task_spec(predictor=pred, task_spec=get_task_spec(pred).with_instructions(candidate[name]))
        return new_prog

    @override
    def evaluate(self, batch, candidate, capture_traces=False):
        return run_gepa_sync(self._aevaluate(batch=batch, candidate=candidate, capture_traces=capture_traces))

    async def _aevaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )
        if capture_traces:
            if self.run is None:
                raise ValueError("DspyAdapter requires a RunContext.")
            trajs = await collect_trace_data(
                program=program,
                dataset=batch,
                run=self.run,
                metric=self.metric_fn,
                max_concurrency=self.max_concurrency,
                raise_on_error=False,
                capture_parse_failures=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
                callback_metadata=callback_metadata,
            )
            scores = []
            outputs = []
            for t in trajs:
                outputs.append(t["prediction"])
                if isinstance(t["prediction"], FailedPrediction) or t.get("score") is None:
                    scores.append(self.failure_score)
                else:
                    score = t["score"]
                    if isinstance(score, ScoreWithFeedback):
                        score = score.score
                    elif score is None:
                        score = self.failure_score
                    scores.append(score)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)
        if self.run is None:
            raise ValueError("DspyAdapter requires a RunContext.")
        evaluator = make_optimizer_evaluator(
            self.run,
            devset=batch,
            metric=self.metric_fn,
            max_concurrency=self.max_concurrency,
            max_errors=len(batch) * 100,
            failure_score=self.failure_score,
            provide_traceback=True,
        )
        res = await evaluator(program, run=self.run, callback_metadata=callback_metadata)
        outputs = [r[1] for r in res.results]
        scores = [r[2] for r in res.results]
        scores = [s["score"] if hasattr(s, "score") else s for s in scores]
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    @override
    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ) -> dict[str, list[ReflectiveExample]]:
        program = self.build_program(candidate)
        ret_d: dict[str, list[ReflectiveExample]] = {}
        for pred_name in components_to_update:
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None, f"Predictor not found: {pred_name}"
            items: list[ReflectiveExample] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score = data["score"]
                if isinstance(module_score, ScoreWithFeedback):
                    module_score = module_score.score
                trace_instances = [t for t in trace if get_task_spec(t[0]) == get_task_spec(module)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue
                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break
                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)
                inputs = selected[1]
                outputs = selected[2]
                new_inputs = {}
                new_outputs = {}
                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, TurnLog):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key
                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].turns):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s
                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue
                    if is_field_type(input_val) and self.custom_instruction_proposer is not None:
                        new_inputs[input_key] = input_val
                    else:
                        new_inputs[input_key] = str(input_val)
                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    for output_key, output_val in outputs.items():
                        new_outputs[output_key] = str(output_val)
                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    if self.run is None:
                        raise ValueError("DspyAdapter requires a RunContext.")
                    adapter = require_adapter(self.run.adapter)
                    structure_instruction = ""
                    for message in adapter.format(task_spec=get_task_spec(module), demos=[], inputs={}):
                        structure_instruction += message.role + ": " + (message.text or "") + "\n"
                    d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                else:
                    feedback_fn = self.feedback_map[pred_name]
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    d["Feedback"] = fb["feedback"]
                    if fb["score"] != module_score:
                        if self.warn_on_score_mismatch:
                            logger.warning(
                                "The score returned by the metric with pred_name is different from the overall metric score. This can indicate 2 things: Either the metric is non-deterministic (e.g., LLM-as-judge, Semantic score, etc.) or the metric returned a score specific to pred_name that differs from the module level score. Currently, GEPA does not support predictor level scoring (support coming soon), and only requires a feedback text to be provided, which can be specific to the predictor or program level. GEPA will ignore the differing score returned, and instead use module level score. You can safely ignore this warning if using a semantic metric, however, if this mismatch is caused due to predictor scoring, please return module-level scores. To disable this warning, set warn_on_score_mismatch=False."
                            )
                            self.warn_on_score_mismatch = False
                        fb["score"] = module_score
                items.append(cast("ReflectiveExample", d))
            if len(items) == 0:
                logger.warning(f"  No valid reflective examples found for {pred_name}")
                continue
            ret_d[pred_name] = items
        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")
        return ret_d

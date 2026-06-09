import logging
import textwrap

import orjson

from dspy.adapters.prompt_format import get_field_spec_description_string
from dspy.evaluate.metric_invoke import call_metric, normalize_metric_score
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module, Prediction
from dspy.propose.source_format import get_formatted_source
from dspy.runtime.run_context import RunContext
from dspy.task_spec.predictor_context import get_task_spec, resolve_optimizer_lm, set_task_spec
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.simba_specs import SimbaOfferFeedbackTaskSpec
from dspy.teleprompt.trace_helpers import run_program_with_trace
from dspy.teleprompt.utils import optimizer_lm_context

logger = logging.getLogger(__name__)


def prepare_models_for_resampling(*, program: Module, n: int, run: RunContext, teacher_run: RunContext | None = None):
    lm = program.optional_lm() or run.lm
    models = []
    if teacher_run:
        teacher_lm = (teacher_run.lm or lm).copy(temperature=1.0)
        models.append(teacher_lm)
        remaining = n - 1
    else:
        remaining = n
    models.extend([lm.copy(temperature=1.0) for _ in range(remaining)])
    return models


def wrap_program(*, program: Module, metric: OptimizerMetric, run: RunContext):

    async def wrapped_program(example):
        prediction, trace, score = (None, None, 0.0)
        try:
            prediction, trace = await run_program_with_trace(program, example, run)
        except Exception as e:
            logger.warning(e)
            trace = []
        output = None
        score = 0.0
        output_metadata = {}
        try:
            output = await call_metric(
                metric,
                example=example,
                prediction=prediction,
                trace=trace,
                run=run,
            )
            score = normalize_metric_score(output)
            if isinstance(output, Prediction):
                output_metadata = {k: v for k, v in output.items() if k != "score"}
        except Exception as e:
            logger.warning(e)
        return {
            "prediction": prediction,
            "trace": trace,
            "score": score,
            "example": example,
            "output_metadata": output_metadata,
        }

    return wrapped_program


def append_a_demo(demo_input_field_maxlen):

    def append_a_demo_(bucket, system, **kwargs) -> bool:
        predictor2name, name2predictor = (kwargs["predictor2name"], kwargs["name2predictor"])
        batch_10p_score = kwargs["batch_10p_score"]
        good = bucket[0]
        trace = good["trace"]
        name2demo = {}
        if good["score"] <= batch_10p_score:
            logger.info(f"Skipping appending a demo as good score {good['score']} is at or below the 10th percentile.")
            return False
        for step in trace:
            predictor, _inputs, _outputs = step
            for k, v in _inputs.items():
                if demo_input_field_maxlen and len(str(v)) > demo_input_field_maxlen:
                    _inputs[k] = f"{str(v)[:demo_input_field_maxlen]}\n\t\t... <TRUNCATED FOR BREVITY>"
            demo = Example.from_record({"augmented": True, **_inputs, **_outputs})
            name = predictor2name[id(predictor)]
            name2demo[name] = demo
        for name, demo in name2demo.items():
            predictor = name2predictor[name]
            predictor.demos.append(demo)
        logger.info(f"Added {len(name2demo)} demos (one each) across all predictors.")
        return True

    return append_a_demo_


async def append_a_rule(bucket, system, *, run: RunContext, **kwargs) -> bool:
    predictor2name = kwargs["predictor2name"]
    batch_10p_score, batch_90p_score = (kwargs["batch_10p_score"], kwargs["batch_90p_score"])
    prompt_model = resolve_optimizer_lm(kwargs["prompt_model"], run=run)
    module_names = [name for name, _ in system.named_predictors()]
    good, bad = (bucket[0], bucket[-1])
    example = good["example"]
    if good["score"] <= batch_10p_score or bad["score"] >= batch_90p_score:
        logger.info(
            f"Skipping rule generation as good score {good['score']} is at or below the 10th percentile *or* bad score {bad['score']} is at or above the 90th percentile."
        )
        return False
    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p_score:
            bad["trace"] = []
            bad["score"] = "N/A"
            bad["prediction"] = {"N/A": "Prediction not available"}
        else:
            good["trace"] = []
            good["score"] = "N/A"
            good["prediction"] = {"N/A": "Prediction not available"}
    better_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)} for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)} for p, i, o in bad["trace"]
    ]
    kwargs = {
        "program_code": get_formatted_source(system.__class__),
        "modules_defn": inspect_modules(system),
        "program_inputs": {**example.as_inputs()},
        "oracle_metadata": {**example.as_labels()},
        "better_program_trajectory": better_trajectory,
        "better_program_outputs": dict(good["prediction"]),
        "worse_program_trajectory": worse_trajectory,
        "worse_program_outputs": dict(bad["prediction"] or {}),
        "worse_reward_value": bad["score"],
        "better_reward_value": good["score"],
        "worse_reward_info": bad["output_metadata"],
        "better_reward_info": good["output_metadata"],
        "module_names": module_names,
    }
    kwargs = {
        k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
        for k, v in kwargs.items()
    }
    with optimizer_lm_context(
        run, lm=prompt_model, phase="simba.offer_feedback", lm_role="prompt_model", optimization_trace=[]
    ) as opt_run:
        advice_program = Predict(SimbaOfferFeedbackTaskSpec())
        advice = (await advice_program(**kwargs, run=opt_run)).module_advice
    for name, predictor in system.named_predictors():
        if name in advice:
            logger.info(f"Advice for {name}: {advice[name]}")
            task_spec = get_task_spec(predictor)
            instructions = task_spec.instructions + "\n\n" + advice[name]
            set_task_spec(predictor=predictor, task_spec=task_spec.with_instructions(instructions))
    return True


def inspect_modules(program):
    separator = "-" * 80
    output = [separator]
    for name, predictor in program.named_predictors():
        task_spec = get_task_spec(predictor)
        instructions = textwrap.dedent(task_spec.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())
        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(
            ("\n" + "\t" * 2).join([""] + get_field_spec_description_string(task_spec.input_fields).splitlines())
        )
        output.append("\tOutput Fields:")
        output.append(
            ("\n" + "\t" * 2).join([""] + get_field_spec_description_string(task_spec.output_fields).splitlines())
        )
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)
    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    try:
        orjson.dumps(o)
        return o
    except (TypeError, orjson.JSONEncodeError):
        pass
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    if isinstance(o, list):
        return [recursive_mask(v) for v in o]
    if isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    return f"<non-serializable: {type(o).__name__}>"

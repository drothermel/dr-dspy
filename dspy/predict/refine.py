import textwrap
from collections.abc import Callable

import orjson

from dspy.adapters.call.wrappers import HintInjectingAdapter
from dspy.adapters.prompt_format import get_field_spec_description_string
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.predict import Predict
from dspy.predict.sampling import SamplingAttempt, sample_with_reward
from dspy.primitives import Module, Prediction
from dspy.propose.source_format import get_formatted_source
from dspy.runtime import run_with_trace
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.runtime.transparency.resolve import require_adapter
from dspy.task_spec.framework.refine import OfferFeedbackTaskSpec


class Refine(Module):
    def __init__(
        self,
        module: Module,
        num_samples: int,
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)
        self.threshold = threshold
        self.num_samples = num_samples
        self.fail_count = fail_count or num_samples
        self.module_code = get_formatted_source(module.__class__)
        try:
            self.reward_fn_code = get_formatted_source(reward_fn)
        except TypeError:
            self.reward_fn_code = get_formatted_source(reward_fn.__class__)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        run = resolve_run(run=run, bound_run=self.run)
        adapter = require_adapter(run.adapter)
        advice: dict[str, str] | None = None

        async def execute_with_advice(attempt: SamplingAttempt) -> tuple[Prediction, list]:
            lm_copy = attempt.lm.copy(temperature=1.0)
            mod = attempt.module.deepcopy()
            mod.set_lm(lm_copy)
            if not advice:
                return await run_with_trace(mod, attempt.inputs, attempt.run, options=attempt.options)
            task_spec2name = {predictor.task_spec: name for name, predictor in mod.named_predictors()}
            hint_adapter = HintInjectingAdapter(
                inner=adapter,
                hint_map=advice,
                task_spec_to_name=task_spec2name,
            )
            hint_run = attempt.run.fork(adapter=hint_adapter)
            return await run_with_trace(mod, attempt.inputs, hint_run, options=attempt.options)

        async def build_advice(
            attempt: SamplingAttempt,
            _state,
            outputs: Prediction,
            trace: list,
            reward: float,
        ) -> None:
            nonlocal advice
            mod = attempt.module.deepcopy()
            mod.set_lm(attempt.lm.copy(temperature=1.0))
            task_spec2name = {predictor.task_spec: name for name, predictor in mod.named_predictors()}
            module_names = [name for name, _ in mod.named_predictors()]
            modules = {"program_code": self.module_code, "modules_defn": inspect_modules(mod)}
            trajectory = [
                {"module_name": task_spec2name[p.task_spec], "inputs": i, "outputs": dict(o)} for p, i, o in trace
            ]
            trajectory_payload = {
                "program_inputs": attempt.inputs,
                "program_trajectory": trajectory,
                "program_outputs": dict(outputs),
            }
            reward_payload = {
                "reward_code": self.reward_fn_code,
                "target_threshold": self.threshold,
                "reward_value": reward,
            }
            advise_kwargs = dict(**modules, **reward_payload, module_names=module_names)
            for key in ("program_inputs", "program_trajectory", "program_outputs"):
                advise_kwargs[key] = orjson.dumps(
                    recursive_mask(trajectory_payload[key]),
                    option=orjson.OPT_INDENT_2,
                ).decode()
            advice = (await Predict(OfferFeedbackTaskSpec())(**advise_kwargs, run=attempt.run)).advice

        return await sample_with_reward(
            module=self.module,
            num_samples=self.num_samples,
            fail_count=self.fail_count,
            reward_fn=self.reward_fn,
            threshold=self.threshold,
            run=run,
            options=options,
            inputs=inputs,
            should_stop=lambda attempt, reward, _state: (
                (self.threshold is not None and reward >= self.threshold) or attempt.idx == self.num_samples - 1
            ),
            execute_attempt=execute_with_advice,
            after_attempt=build_advice,
        )


def inspect_modules(program):
    separator = "-" * 80
    output = [separator]
    for _, (name, predictor) in enumerate(program.named_predictors()):
        task_spec = predictor.task_spec
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
    except TypeError:
        pass
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    if isinstance(o, list):
        return [recursive_mask(v) for v in o]
    if isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    return f"<non-serializable: {type(o).__name__}>"

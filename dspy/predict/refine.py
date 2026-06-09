import textwrap
from collections.abc import Callable

import orjson

from dspy.adapters.call.wrappers import HintInjectingAdapter
from dspy.adapters.utils import get_field_description_string
from dspy.compile.resolve import resolve_adapter
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.predict import Predict, Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos, task_spec_output_field_infos
from dspy.utils.source_format import get_formatted_source

from .predict import Module


class OfferFeedbackTaskSpec(TaskSpec):
    name: str = "framework.refine.offer_feedback"
    instructions: str = "In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A. The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you to avoid the same mistake on the same or similar inputs."
    inputs: tuple[FieldSpec, ...] = (
        input_field("program_code", str, desc="The code of the program that we are analyzing"),
        input_field("modules_defn", str, desc="The definition of each module in the program, including its I/O"),
        input_field("program_inputs", str, desc="The inputs to the program that we are analyzing"),
        input_field(
            "program_trajectory", str, desc="The trajectory of the program's execution, showing each module's I/O"
        ),
        input_field("program_outputs", str, desc="The outputs of the program that we are analyzing"),
        input_field("reward_code", str, desc="The code of the reward function that we are analyzing"),
        input_field("target_threshold", float, desc="The target threshold for the reward function"),
        input_field("reward_value", float, desc="The reward value assigned to the program's outputs"),
        input_field(
            "module_names", list[str], desc="The names of the modules in the program, for which we seek advice"
        ),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("discussion", str, desc="Discussing blame of where each module went wrong, if it did"),
        output_field(
            "advice",
            dict[str, str],
            desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made mistakes in the past and what each mistake was, followed by what it should do differently in that kind ofscenario in the future. If the module is not to blame, write N/A.",
        ),
    )


class Refine(Module):
    def __init__(
        self,
        module: Module,
        N: int,
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ) -> None:
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N
        self.module_code = get_formatted_source(module.__class__)
        try:
            self.reward_fn_code = get_formatted_source(reward_fn)
        except TypeError:
            self.reward_fn_code = get_formatted_source(reward_fn.__class__)

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        run = resolve_run(run=run, bound_run=self.run)
        lm = self.module.get_lm() or run.lm
        best_pred, best_trace, best_reward = (None, None, -float("inf"))
        advice = None
        adapter, _ = resolve_adapter(run.adapter)
        for idx in range(self.N):
            lm_ = lm.copy(temperature=1.0)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)
            predictor2name = {predictor: name for name, predictor in mod.named_predictors()}
            task_spec2name = {predictor.task_spec: name for name, predictor in mod.named_predictors()}
            module_names = [name for name, _ in mod.named_predictors()]
            try:
                item_run = run.fork(optimization_trace=[], call_log=[])
                if not advice:
                    outputs = await mod(**inputs, run=item_run, options=options)
                else:
                    hint_adapter = HintInjectingAdapter(
                        inner=adapter,
                        hint_map=advice,
                        task_spec_to_name=task_spec2name,
                    )
                    hint_run = item_run.fork(adapter=hint_adapter)
                    outputs = await mod(**inputs, run=hint_run, options=options)
                trace = list(item_run.optimization_trace)
                reward = self.reward_fn(inputs, outputs)
                if reward > best_reward:
                    best_reward, best_pred, best_trace = (reward, outputs, trace)
                if self.threshold is not None and reward >= self.threshold:
                    break
                if idx == self.N - 1:
                    break
                modules = {"program_code": self.module_code, "modules_defn": inspect_modules(mod)}
                trajectory = [{"module_name": predictor2name[p], "inputs": i, "outputs": dict(o)} for p, i, o in trace]
                trajectory = {
                    "program_inputs": inputs,
                    "program_trajectory": trajectory,
                    "program_outputs": dict(outputs),
                }
                reward = {
                    "reward_code": self.reward_fn_code,
                    "target_threshold": self.threshold,
                    "reward_value": reward,
                }
                advise_kwargs = dict(**modules, **trajectory, **reward, module_names=module_names)
                advise_kwargs = {
                    k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
                    for k, v in advise_kwargs.items()
                }
                advice = (await Predict(OfferFeedbackTaskSpec())(**advise_kwargs, run=run)).advice
            except Exception:
                if idx > self.fail_count:
                    raise
                self.fail_count -= 1
        if best_trace:
            run.optimization_trace.extend(best_trace)
        return best_pred


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
            ("\n" + "\t" * 2).join(
                [""] + get_field_description_string(task_spec_input_field_infos(task_spec)).splitlines()
            )
        )
        output.append("\tOutput Fields:")
        output.append(
            ("\n" + "\t" * 2).join(
                [""] + get_field_description_string(task_spec_output_field_infos(task_spec)).splitlines()
            )
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

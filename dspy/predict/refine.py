from collections.abc import Callable

import orjson

from dspy.adapters.call.wrappers import HintInjectingAdapter
from dspy.predict.predict import Predict
from dspy.predict.sampling import SamplingAttempt, sample_with_reward
from dspy.primitives import Module, Prediction
from dspy.propose.source_format import get_formatted_source
from dspy.runtime import run_with_trace
from dspy.runtime.call_options import ModuleCallOptions
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.runtime.transparency.resolve import require_adapter
from dspy.serialization.json import to_jsonable
from dspy.task_spec.framework.refine import OfferFeedbackTaskSpec
from dspy.teleprompt.core.inspect_modules import inspect_modules


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

        def stamp_predictor_names(module: Module) -> dict[int, str]:
            predictor_id_to_name = {id(predictor): name for name, predictor in module.named_predictors()}
            for name, predictor in module.named_predictors():
                object.__setattr__(predictor, "_dspy_predictor_name", name)
            return predictor_id_to_name

        async def execute_with_advice(attempt: SamplingAttempt) -> tuple[Prediction, list]:
            lm_copy = attempt.lm.copy(temperature=1.0)
            mod = attempt.module.deepcopy()
            mod.set_lm(lm_copy)
            stamp_predictor_names(mod)
            if not advice:
                return await run_with_trace(mod, attempt.inputs, attempt.run, options=attempt.options)
            hint_adapter = HintInjectingAdapter(
                inner=adapter,
                hint_map=advice,
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
            predictor_id_to_name = stamp_predictor_names(mod)
            module_names = [name for name, _ in mod.named_predictors()]
            modules = {"program_code": self.module_code, "modules_defn": inspect_modules(mod)}
            trajectory = [
                {"module_name": predictor_id_to_name[id(p)], "inputs": i, "outputs": dict(o)} for p, i, o in trace
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
                    to_jsonable(trajectory_payload[key]),
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

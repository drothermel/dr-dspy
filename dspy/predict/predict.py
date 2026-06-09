import logging
import random
from typing import Any

from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMConfig, coerce_lm_config, merge_lm_config
from dspy.core.types.call_options import ModuleCallOptions, PredictOptions
from dspy.predict.call_validation import resolve_predict_options
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.runtime.callback import Callback
from dspy.runtime.run_context import RunContext
from dspy.runtime.transparency import resolve_call_site, resolve_lm_config
from dspy.serialization.json import to_jsonable
from dspy.task_spec.task_spec import TaskSpec

logger = logging.getLogger(__name__)
UNSAFE_LM_STATE_KEYS = {"api_base", "base_url", "model_list"}


def _sanitize_lm_state(lm_state: dict, allow_unsafe_lm_state: bool) -> dict:
    if allow_unsafe_lm_state:
        return lm_state
    unsafe_keys = sorted(UNSAFE_LM_STATE_KEYS.intersection(lm_state))
    if not unsafe_keys:
        return lm_state
    sanitized_lm_state = {k: v for k, v in lm_state.items() if k not in UNSAFE_LM_STATE_KEYS}
    logger.warning(
        "Ignoring unsafe LM config key(s) during state load: %s. Pass allow_unsafe_lm_state=True to preserve these keys for trusted files.",
        unsafe_keys,
    )
    return sanitized_lm_state


class Predict(Module, Parameter):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        config: LMConfig | None = None,
        callbacks: list[Callback] | None = None,
        run: RunContext | None = None,
    ) -> None:
        if isinstance(task_spec, str):
            raise TypeError(
                "Predict requires a TaskSpec instance, not a string. Use a TaskSpec subclass or make_task_spec(...) to create one."
            )
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"Predict requires a TaskSpec instance, got {type(task_spec).__name__}.")
        super().__init__(callbacks=callbacks, run=run)
        self.stage = random.randbytes(8).hex()
        self.task_spec: TaskSpec = task_spec
        self.config = config or LMConfig()
        self.reset()

    def reset(self) -> None:
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    @override
    def dump_state(self, json_mode=True):
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}
        state["demos"] = []
        for demo in self.demos:
            demo = demo.fork()
            for field in demo:
                demo[field] = to_jsonable(demo[field])
            if json_mode and (not isinstance(demo, dict)):
                state["demos"].append(demo.to_dict())
            else:
                state["demos"].append(demo)
        state["task_spec"] = self.task_spec.to_dict()
        state["config"] = self.config.model_dump(mode="json")
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    @override
    def load_state(
        self, state: dict, *, allow_unsafe_lm_state: bool = False, custom_types: dict[str, type] | None = None
    ) -> "Predict":
        excluded_keys = ["task_spec", "lm", "config"]
        for name, value in state.items():
            if name not in excluded_keys:
                setattr(self, name, value)
        if "task_spec" not in state:
            if "signature" in state:
                raise ValueError(
                    "Saved state uses legacy 'signature' format. Re-save the program with the current DSPy version."
                )
            raise ValueError("Missing required 'task_spec' key in saved Predict state.")
        self.task_spec = TaskSpec.from_dict(state["task_spec"], custom_types=custom_types)
        config_data = state.get("config")
        self.config = coerce_lm_config(config_data) if config_data is not None else LMConfig()
        sanitized_lm_state = _sanitize_lm_state(state["lm"], allow_unsafe_lm_state) if state["lm"] else None
        self.lm = (
            BaseLM.load_state(sanitized_lm_state, allow_custom_lm_class=allow_unsafe_lm_state)
            if sanitized_lm_state
            else None
        )
        return self

    def _forward_preprocess(
        self,
        *,
        inputs: dict[str, Any],
        options: PredictOptions,
        run: RunContext,
    ):
        task_spec = options.task_spec or self.task_spec
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"Predict expected a TaskSpec, got {type(task_spec).__name__}.")
        demos = options.demos if options.demos is not None else self.demos
        base_config = self.config
        if options.config is not None:
            config = merge_lm_config(base_config, options.config) or options.config
        else:
            config = base_config
        lm = options.lm or self.lm or run.lm
        if lm is None:
            raise ValueError(
                "No LM is loaded. Pass run=RunContext.create(lm=LM(...), adapter=...) to the call, "
                "or bind run at Module/Predict construction."
            )
        if isinstance(lm, str):
            raise ValueError(
                f"LM must be an instance of `dspy.clients.base_lm.BaseLM`, not a string. "
                f"Create a RunContext with `RunContext.create(lm=LM('{lm}'), adapter=...)` instead."
            )
        if not isinstance(lm, BaseLM):
            raise ValueError(
                f"LM must be an instance of `dspy.clients.base_lm.BaseLM`, not {type(lm)}. Received `lm={lm}`."
            )
        prediction = options.prediction
        if (
            prediction is not None
            and isinstance(prediction, dict)
            and prediction.get("type") == "content"
            and "content" in prediction
        ):
            extensions = dict(config.extensions)
            extensions["prediction"] = prediction
            config = config.model_copy(update={"extensions": extensions})
        return lm, config, task_spec, demos, inputs, run, options.trace

    def _forward_postprocess(self, completions, task_spec, run, inputs, *, trace: bool):
        pred = Prediction.from_completions(completions, task_spec=task_spec)
        if trace and run.optimization_trace is not None and run.telemetry.max_optimization_trace_entries > 0:
            trace_list = run.optimization_trace
            if len(trace_list) >= run.telemetry.max_optimization_trace_entries:
                trace_list.pop(0)
            trace_list.append((self, dict(inputs), pred))
        return pred

    @override
    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        predict_options = resolve_predict_options(options if isinstance(options, PredictOptions) else None)
        lm, config, task_spec, demos, inputs, run, trace = self._forward_preprocess(
            inputs=inputs,
            options=predict_options,
            run=run,
        )
        config, _provenance = resolve_lm_config(lm, config)
        call_site = resolve_call_site(
            run=run,
            default_module=type(self).__name__,
            default_phase="predict",
        )
        completions = await run.adapter(
            lm=lm,
            config=config,
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            run=run,
            call_site=call_site,
        )
        return self._forward_postprocess(completions, task_spec, run, inputs, trace=trace)

    def update_config(self, config: LMConfig) -> None:
        merged = merge_lm_config(self.config, config)
        self.config = merged if merged is not None else config

    def get_config(self) -> LMConfig:
        return self.config

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.task_spec})"

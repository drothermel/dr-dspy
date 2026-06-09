import logging
from typing import Any, TextIO

from typing_extensions import override

from dspy.core.types import LMForward
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.parallel import Parallel
from dspy.primitives.base_module import BaseModule
from dspy.primitives.batch_result import BatchResult
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.runtime import RunContext, pretty_print_call_log, resolve_run, track_usage, with_callbacks
from dspy.runtime.callback import ACTIVE_RUN

logger = logging.getLogger(__name__)

_DIRECT_AFORWARD_WARNED: set[type] = set()


def _warn_direct_aforward_once(cls: type) -> None:
    if cls in _DIRECT_AFORWARD_WARNED:
        return
    _DIRECT_AFORWARD_WARNED.add(cls)
    logger.warning(
        "Calling module.aforward(...) on %s directly is discouraged. Please use await module(...) instead.",
        cls.__name__,
    )


class Module(BaseModule):
    def __init__(self, callbacks: list[Any] | None = None, run: RunContext | None = None) -> None:
        self.callbacks = callbacks or []
        self.run = run
        self._compiled = False
        self.call_log = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("call_log", None)
        state.pop("callbacks", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "call_log"):
            self.call_log = []
        if not hasattr(self, "callbacks"):
            self.callbacks = []
        if not hasattr(self, "_compiled"):
            self._compiled = False
        if not hasattr(self, "run"):
            self.run = None

    @with_callbacks(kind="module")
    async def __call__(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        run = resolve_run(run=run, bound_run=self.run)
        run_token = ACTIVE_RUN.set(run)
        run.caller_modules.append(self)
        try:
            if run.telemetry.track_usage and run.usage_tracker is None:
                with track_usage(run) as usage_tracker:
                    output = await self._aforward_impl(run=run, options=options, **inputs)
                tokens = usage_tracker.get_total_tokens()
            else:
                output = await self._aforward_impl(run=run, options=options, **inputs)
                tokens = (
                    run.usage_tracker.get_total_tokens() if run.telemetry.track_usage and run.usage_tracker else None
                )
            if tokens:
                self._set_lm_usage(tokens, output)
            return output
        finally:
            run.caller_modules.pop()
            ACTIVE_RUN.reset(run_token)

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        _warn_direct_aforward_once(type(self))
        return await self._aforward_impl(run=run, options=options, **inputs)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        raise NotImplementedError(f"{type(self).__name__} must implement _aforward_impl().")

    def set_lm(self, lm: LMForward | None) -> None:
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self) -> LMForward:
        all_used_lms = [param.lm for _, param in self.named_predictors()]
        if len(set(all_used_lms)) == 1:
            lm = all_used_lms[0]
            if lm is None:
                raise ValueError("No LM is configured on this module's predictors.")
            return lm
        raise ValueError(
            "Multiple LMs are configured on this module. Inspect per-predictor LMs via "
            "named_predictors() and read param.lm on each predictor."
        )

    @override
    def __repr__(self) -> str:
        s = []
        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")
        return "\n".join(s)

    def inspect_call_log(self, n: int = 1, file: "TextIO | None" = None) -> None:
        pretty_print_call_log(call_log=self.call_log, n=n, file=file)

    async def batch(
        self,
        examples: list[Example],
        run: RunContext,
        max_concurrency: int | None = None,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
        timeout: int = 120,
        straggler_limit: int = 3,
    ) -> BatchResult:
        exec_pairs = [(self, example.as_inputs()) for example in examples]
        parallel_executor = Parallel(
            run=run,
            max_concurrency=max_concurrency,
            max_errors=max_errors,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
            timeout=timeout,
            straggler_limit=straggler_limit,
        )
        return await parallel_executor(exec_pairs)

    def _set_lm_usage(self, tokens: dict[str, Any], output: Any) -> None:
        prediction_in_output = None
        if isinstance(output, Prediction):
            prediction_in_output = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Prediction):
            prediction_in_output = output[0]
        if prediction_in_output:
            prediction_in_output.set_lm_usage(tokens)
        else:
            logger.warning(
                "Failed to set LM usage. Please return `dspy.primitives.prediction.Prediction` object from Module to enable usage tracking."
            )

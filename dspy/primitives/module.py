import logging
from typing import Any, TextIO

from typing_extensions import Self, override

from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.parallel import Parallel
from dspy.predict.protocol import Predictor
from dspy.primitives.base_module import BaseModule
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


class ProgramMeta(type):
    @override
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            cls.__init__(obj, *args, **kwargs)
        return obj


class Module(BaseModule, metaclass=ProgramMeta):
    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        instance = super().__new__(cls)
        instance._compiled = False
        instance.callbacks = []
        instance.call_log = []
        instance.run = None
        return instance

    def __init__(self, callbacks=None, run: RunContext | None = None) -> None:
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

    def named_predictors(self):
        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predictor)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm) -> None:
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        all_used_lms = [param.lm for _, param in self.named_predictors()]
        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]
        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

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
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
        timeout: int = 120,
        straggler_limit: int = 3,
    ) -> list[Any] | tuple[list[Any], list[Any], list[BaseException]]:
        exec_pairs = [(self, example.as_inputs()) for example in examples]
        parallel_executor = Parallel(
            run=run,
            max_concurrency=max_concurrency,
            max_errors=max_errors,
            return_failed_examples=return_failed_examples,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
            timeout=timeout,
            straggler_limit=straggler_limit,
        )
        if return_failed_examples:
            results, failed_examples, exceptions = await parallel_executor(exec_pairs)
            return (results, failed_examples, exceptions)
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

    @override
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

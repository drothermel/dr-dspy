from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Self, override

from dspy.persistence.program import save_program as persist_program
from dspy.persistence.state import apply_module_state, dump_module_state, save_state
from dspy.persistence.state import load_state as load_state_file
from dspy.primitives import module_copy, module_execution, module_graph, module_lm
from dspy.runtime import Callback, RunContext, with_callbacks
from dspy.runtime.batch import Parallel

if TYPE_CHECKING:
    from pathlib import Path

    from dspy.clients.base_lm import BaseLM
    from dspy.primitives.batch_result import BatchResult
    from dspy.primitives.example import Example
    from dspy.primitives.prediction import Prediction
    from dspy.runtime.call_options import ModuleCallOptions


class Module:
    def __init__(self, callbacks: list[Callback] | None = None, run: RunContext | None = None) -> None:
        self.callbacks = callbacks or []
        self.run = run
        self._compiled = False
        self.call_log = []

    @property
    def run(self) -> RunContext | None:
        return vars(self).get("run")

    @run.setter
    def run(self, value: RunContext | None) -> None:
        vars(self)["run"] = value

    def __getstate__(self) -> dict[str, Any]:
        return module_copy.module_getstate(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        module_copy.module_setstate(self, state)

    def named_predictors(self):
        return module_graph.named_predictors(self)

    def named_sub_modules(self, type_: type | None = None):
        return module_graph.named_sub_modules(self, type_=type_)

    def predictors(self):
        return module_graph.predictors(self)

    def deepcopy(self) -> Self:
        return cast("Self", module_copy.deepcopy_module(self))

    def reset_copy(self) -> Self:
        return cast("Self", module_copy.reset_copy(self))

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        return dump_module_state(self, json_mode=json_mode)

    def load_state(
        self,
        state: dict[str, Any],
        *,
        allow_unsafe_lm_state: bool = False,
        custom_types: dict[str, type] | None = None,
    ) -> Self:
        apply_module_state(
            self,
            state,
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            custom_types=custom_types,
        )
        return self

    def save(
        self,
        path: str | Path,
        save_program: bool = False,
        modules_to_serialize: list[object] | None = None,
    ) -> None:
        if save_program:
            persist_program(self, path, modules_to_serialize=modules_to_serialize)
            return
        save_state(self, path)

    def load(
        self,
        path: str | Path,
        allow_pickle: bool = False,
        allow_unsafe_lm_state: bool = False,
        custom_types: dict[str, type] | None = None,
    ) -> Self:
        load_state_file(
            self,
            path,
            allow_pickle=allow_pickle,
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            custom_types=custom_types,
        )
        return self

    @with_callbacks(kind="module")
    async def __call__(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        return await module_execution.invoke_module(self, run=run, options=options, **inputs)

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        module_execution.warn_direct_aforward_once(type(self))
        return await self._aforward_impl(run=run, options=options, **inputs)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs: Any,
    ) -> Prediction:
        raise NotImplementedError(f"{type(self).__name__} must implement _aforward_impl().")

    def set_lm(self, lm: BaseLM | None) -> None:
        module_lm.set_lm(self, lm)

    def get_lm(self) -> BaseLM:
        return module_lm.get_lm(self)

    def optional_lm(self) -> BaseLM | None:
        return module_lm.optional_lm(self)

    @override
    def __repr__(self) -> str:
        s = []
        for name, predictor in self.named_predictors():
            s.append(f"{name} = {predictor}")
        return "\n".join(s)

    async def batch(
        self,
        examples: list[Example],
        run: RunContext,
        max_concurrency: int | None = None,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
        timeout: int = 120,
    ) -> BatchResult:
        exec_pairs = [(self, example.as_inputs()) for example in examples]
        parallel_executor = Parallel(
            run=run,
            max_concurrency=max_concurrency,
            max_errors=max_errors,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
            timeout=timeout,
        )
        return await parallel_executor(exec_pairs)

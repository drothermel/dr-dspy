from __future__ import annotations

import copy
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Self, override

from dspy.persistence.program import save_program as persist_program
from dspy.persistence.state import apply_module_state, dump_module_state, save_state
from dspy.persistence.state import load_state as load_state_file
from dspy.predict.protocol import Predictor
from dspy.primitives.prediction import Prediction
from dspy.runtime import Callback, RunContext, resolve_run, track_usage, with_callbacks
from dspy.runtime.active_run import call_scope, get_active_usage_tracker
from dspy.runtime.batch import Parallel

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from dspy.clients.base_lm import BaseLM
    from dspy.primitives.batch_result import BatchResult
    from dspy.primitives.example import Example
    from dspy.runtime.call_options import ModuleCallOptions

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
        state = self.__dict__.copy()
        state.pop("call_log", None)
        state.pop("callbacks", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "call_log"):
            self.call_log = []
        if not hasattr(self, "callbacks"):
            self.callbacks = []
        if not hasattr(self, "_compiled"):
            self._compiled = False
        if not hasattr(self, "run"):
            self.run = None

    def _enqueue_graph_children(
        self,
        name: str,
        item: object,
        queue: deque[tuple[str, object]],
        seen: set[int],
    ) -> None:
        def enqueue(child_name: str, child: object) -> None:
            child_id = id(child)
            if child_id not in seen:
                seen.add(child_id)
                queue.append((child_name, child))

        if isinstance(item, Module):
            if name == "self" or not getattr(item, "_compiled", False):
                for sub_name, sub_item in item.__dict__.items():
                    enqueue(f"{name}.{sub_name}", sub_item)
            return
        if isinstance(item, (list, tuple)):
            for idx, sub_item in enumerate(item):
                enqueue(f"{name}[{idx}]", sub_item)
            return
        if isinstance(item, dict):
            for key, sub_item in item.items():
                enqueue(f"{name}[{key}]", sub_item)

    def _walk_module_graph(self) -> Generator[tuple[str, object], None, None]:
        """Breadth-first traversal of module-owned object graph.

        Compiled subgraphs (``_compiled=True``) are opaque: their children are not
        enqueued. The root module is always expanded via the ``self`` entry.
        """
        queue: deque[tuple[str, object]] = deque([("self", self)])
        seen = {id(self)}
        while queue:
            name, item = queue.popleft()
            yield name, item
            self._enqueue_graph_children(name=name, item=item, queue=queue, seen=seen)

    def named_predictors(self) -> list[tuple[str, Predictor]]:
        """Return ``(name, Predictor)`` pairs. Skips predictors inside compiled subgraphs.

        When the same ``Predictor`` instance is reachable via multiple paths, only the
        first name encountered during breadth-first traversal is returned.
        """
        named_predictors: list[tuple[str, Predictor]] = []
        visited_predictors: set[int] = set()
        for name, item in self._walk_module_graph():
            if not isinstance(item, Predictor):
                continue
            predictor_id = id(item)
            if predictor_id in visited_predictors:
                continue
            visited_predictors.add(predictor_id)
            named_predictors.append((name, item))
        return named_predictors

    def named_sub_modules(self, type_: type | None = None) -> Generator[tuple[str, Module], None, None]:
        """Yield ``(name, module)`` pairs for modules of ``type_``.

        Compiled subgraphs are opaque by default (same policy as ``named_predictors``).
        """
        if type_ is None:
            type_ = Module
        for name, item in self._walk_module_graph():
            if isinstance(item, type_):
                yield name, cast("Module", item)

    def predictors(self) -> list[Predictor]:
        return [predictor for _, predictor in self.named_predictors()]

    def deepcopy(self) -> Self:
        try:
            return copy.deepcopy(self)
        except Exception:
            logger.debug(
                "copy.deepcopy failed for %s; falling back to manual deepcopy",
                self.__class__.__name__,
                exc_info=True,
            )
        new_instance = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            if isinstance(value, Module):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logger.warning(
                        "Failed to deep copy attribute '%s' of %s, falling back to shallow copy or reference copy.",
                        attr,
                        self.__class__.__name__,
                    )
                    try:
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        setattr(new_instance, attr, value)
        return new_instance

    def reset_copy(self) -> Self:
        new_instance = self.deepcopy()
        for predictor in new_instance.predictors():
            predictor.reset()
        return new_instance

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
        run = resolve_run(run=run, bound_run=self.run)
        async with call_scope(run=run, caller=self):
            if run.telemetry.track_usage and run.usage_tracker is None:
                with track_usage(run) as usage_tracker:
                    output = await self._aforward_impl(run=run, options=options, **inputs)
                tokens = usage_tracker.get_total_tokens()
            else:
                output = await self._aforward_impl(run=run, options=options, **inputs)
                usage_tracker = get_active_usage_tracker(run) if run.telemetry.track_usage else None
                tokens = usage_tracker.get_total_tokens() if usage_tracker else None
            if tokens:
                self._set_lm_usage(tokens, output)
            return output

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

    def set_lm(self, lm: BaseLM | None) -> None:
        for _, predictor in self.named_predictors():
            predictor.lm = lm

    def get_lm(self) -> BaseLM:
        lm = self.optional_lm()
        if lm is None:
            raise ValueError("No LM is configured on this module's predictors.")
        return lm

    def optional_lm(self) -> BaseLM | None:
        """Return the module's LM when all predictors share one; otherwise ``None`` or raise."""
        all_used_lms = [predictor.lm for _, predictor in self.named_predictors()]
        if not all_used_lms:
            return None
        if len(set(all_used_lms)) != 1:
            raise ValueError(
                "Multiple LMs are configured on this module. Inspect per-predictor LMs via "
                "named_predictors() and read predictor.lm on each predictor."
            )
        return all_used_lms[0]

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

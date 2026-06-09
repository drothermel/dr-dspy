from __future__ import annotations

import inspect
import logging
import uuid
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from dspy.runtime.active_run import get_active_run

ACTIVE_CALL_ID: ContextVar[str | None] = ContextVar("active_call_id", default=None)

logger = logging.getLogger(__name__)

R = TypeVar("R")


class CallbackKind(StrEnum):
    MODULE = "module"
    LM = "lm"
    ADAPTER = "adapter"
    TOOL = "tool"
    EVALUATE = "evaluate"


class CallbackPhase(StrEnum):
    START = "start"
    END = "end"


def _callable_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


_START_HANDLERS: dict[CallbackKind, Callable[[Any], Callable[..., Any]]] = {
    CallbackKind.MODULE: lambda callback: callback.on_module_start,
    CallbackKind.LM: lambda callback: callback.on_lm_start,
    CallbackKind.TOOL: lambda callback: callback.on_tool_start,
    CallbackKind.EVALUATE: lambda callback: callback.on_evaluate_start,
}

_END_HANDLERS: dict[CallbackKind, Callable[[Any], Callable[..., Any]]] = {
    CallbackKind.MODULE: lambda callback: callback.on_module_end,
    CallbackKind.LM: lambda callback: callback.on_lm_end,
    CallbackKind.TOOL: lambda callback: callback.on_tool_end,
    CallbackKind.EVALUATE: lambda callback: callback.on_evaluate_end,
}

_ADAPTER_START_HANDLERS: dict[str, Callable[[Any], Callable[..., Any]]] = {
    "format": lambda callback: callback.on_adapter_format_start,
    "parse": lambda callback: callback.on_adapter_parse_start,
}

_ADAPTER_END_HANDLERS: dict[str, Callable[[Any], Callable[..., Any]]] = {
    "format": lambda callback: callback.on_adapter_format_end,
    "parse": lambda callback: callback.on_adapter_parse_end,
}


def _resolve_handler(
    *,
    callback: Any,
    kind: CallbackKind,
    phase: CallbackPhase,
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    if kind == CallbackKind.ADAPTER:
        fn_name = _callable_name(fn)
        table = _ADAPTER_START_HANDLERS if phase == CallbackPhase.START else _ADAPTER_END_HANDLERS
        try:
            return table[fn_name](callback)
        except KeyError as exc:
            raise ValueError(f"Unsupported adapter method for using callback: {fn_name}.") from exc
    table = _START_HANDLERS if phase == CallbackPhase.START else _END_HANDLERS
    return table[kind](callback)


def _bind_callback_inputs(
    fn: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(fn)
    bound_arguments = signature.bind_partial(instance, *args, **kwargs)
    bound_arguments.apply_defaults()
    inputs = dict(bound_arguments.arguments)
    if "self" in inputs:
        inputs.pop("self")
    elif "instance" in inputs:
        inputs.pop("instance")
    return inputs


def _dispatch_callbacks(
    *,
    phase: CallbackPhase,
    instance: Any,
    fn: Callable[..., Any],
    kind: CallbackKind,
    call_id: str,
    callbacks: list[Any],
    inputs: dict[str, Any] | None = None,
    results: Any = None,
    exception: Exception | None = None,
) -> None:
    for callback in callbacks:
        try:
            handler = _resolve_handler(callback=callback, kind=kind, phase=phase, fn=fn)
            if phase == CallbackPhase.START:
                handler(call_id=call_id, instance=instance, inputs=inputs or {})
            else:
                handler(call_id=call_id, outputs=results, exception=exception)
        except Exception as e:
            if phase == CallbackPhase.START:
                logger.warning(f"Error when calling callback {callback}: {e}")
            else:
                logger.warning(
                    f"Error when applying callback {callback}'s end handler on function {_callable_name(fn)}: {e}."
                )


class _CallbackScope:
    def __init__(
        self,
        *,
        instance: Any,
        fn: Callable[..., Any],
        kind: CallbackKind,
        callbacks: list[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.instance = instance
        self.fn = fn
        self.kind = kind
        self.callbacks = callbacks
        self.args = args
        self.kwargs = kwargs
        self.call_id = uuid.uuid4().hex
        self.parent_call_id: str | None = None
        self.results: Any = None
        self.exception: Exception | None = None

    def __enter__(self) -> _CallbackScope:
        inputs = _bind_callback_inputs(self.fn, self.instance, self.args, self.kwargs)
        _dispatch_callbacks(
            phase=CallbackPhase.START,
            instance=self.instance,
            fn=self.fn,
            kind=self.kind,
            call_id=self.call_id,
            callbacks=self.callbacks,
            inputs=inputs,
        )
        self.parent_call_id = ACTIVE_CALL_ID.get()
        ACTIVE_CALL_ID.set(self.call_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if exc_value is not None and isinstance(exc_value, Exception):
            self.exception = exc_value
        ACTIVE_CALL_ID.set(self.parent_call_id)
        _dispatch_callbacks(
            phase=CallbackPhase.END,
            instance=self.instance,
            fn=self.fn,
            kind=self.kind,
            call_id=self.call_id,
            callbacks=self.callbacks,
            results=self.results,
            exception=self.exception,
        )


def get_active_callbacks(instance: Any, run: Any | None = None, *, kind: CallbackKind) -> list[Any]:
    effective_run = run if run is not None else (get_active_run() if kind == CallbackKind.TOOL else None)
    callbacks = list(effective_run.callbacks) if effective_run else []
    return callbacks + getattr(instance, "callbacks", [])


def invoke_with_callbacks(
    *,
    instance: Any,
    fn: Callable[..., R],
    kind: CallbackKind,
    callbacks: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    with _CallbackScope(
        instance=instance,
        fn=fn,
        kind=kind,
        callbacks=callbacks,
        args=args,
        kwargs=kwargs,
    ) as scope:
        scope.results = fn(instance, *args, **kwargs)
        return scope.results


async def ainvoke_with_callbacks(
    *,
    instance: Any,
    fn: Callable[..., Awaitable[R]],
    kind: CallbackKind,
    callbacks: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    with _CallbackScope(
        instance=instance,
        fn=fn,
        kind=kind,
        callbacks=callbacks,
        args=args,
        kwargs=kwargs,
    ) as scope:
        scope.results = await fn(instance, *args, **kwargs)
        return scope.results

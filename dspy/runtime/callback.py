from __future__ import annotations

import functools
import inspect
import logging
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext

ACTIVE_CALL_ID: ContextVar[str | None] = ContextVar("active_call_id", default=None)
logger = logging.getLogger(__name__)


def _callable_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


CallbackKind = Literal["module", "lm", "adapter", "tool", "evaluate"]
F = TypeVar("F", bound=Callable[..., Any])


class Callback(Protocol):
    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None: ...

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None) -> None: ...

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...

    def on_adapter_format_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None: ...

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...

    def on_adapter_parse_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None: ...

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...

    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None) -> None: ...

    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...

    def on_evaluate_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None: ...


class NoOpCallback:
    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        pass

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        pass

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        pass

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None) -> None:
        pass

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        pass

    def on_adapter_format_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None:
        pass

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        pass

    def on_adapter_parse_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None:
        pass

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        pass

    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None) -> None:
        pass

    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        pass

    def on_evaluate_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None) -> None:
        pass


def _begin_callback_scope(
    *,
    instance: Any,
    fn: Callable[..., Any],
    kind: CallbackKind,
    callbacks: list[Callback],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[str, str | None]:
    call_id = uuid.uuid4().hex
    _execute_start_callbacks(
        instance=instance,
        fn=fn,
        kind=kind,
        call_id=call_id,
        callbacks=callbacks,
        args=args,
        kwargs=kwargs,
    )
    parent_call_id = ACTIVE_CALL_ID.get()
    ACTIVE_CALL_ID.set(call_id)
    return call_id, parent_call_id


def _end_callback_scope(
    *,
    instance: Any,
    fn: Callable[..., Any],
    kind: CallbackKind,
    call_id: str,
    parent_call_id: str | None,
    results: Any,
    exception: Exception | None,
    callbacks: list[Callback],
) -> None:
    ACTIVE_CALL_ID.set(parent_call_id)
    _execute_end_callbacks(
        instance=instance,
        fn=fn,
        kind=kind,
        call_id=call_id,
        results=results,
        exception=exception,
        callbacks=callbacks,
    )


def with_callbacks(fn: F | None = None, *, kind: CallbackKind = "module") -> F | Callable[[F], F]:
    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(instance, *args, **kwargs):
                run = kwargs.get("run")
                callbacks = _get_active_callbacks(instance, run)
                if not callbacks:
                    return await fn(instance, *args, **kwargs)
                call_id, parent_call_id = _begin_callback_scope(
                    instance=instance,
                    fn=fn,
                    kind=kind,
                    callbacks=callbacks,
                    args=args,
                    kwargs=kwargs,
                )
                results = None
                exception = None
                try:
                    results = await fn(instance, *args, **kwargs)
                except Exception as e:
                    exception = e
                    raise exception
                else:
                    return results
                finally:
                    _end_callback_scope(
                        instance=instance,
                        fn=fn,
                        kind=kind,
                        call_id=call_id,
                        parent_call_id=parent_call_id,
                        results=results,
                        exception=exception,
                        callbacks=callbacks,
                    )

            return async_wrapper  # ty:ignore[invalid-return-type]

        @functools.wraps(fn)
        def sync_wrapper(instance, *args, **kwargs):
            run = kwargs.get("run")
            callbacks = _get_active_callbacks(instance, run)
            if not callbacks:
                return fn(instance, *args, **kwargs)
            call_id, parent_call_id = _begin_callback_scope(
                instance=instance,
                fn=fn,
                kind=kind,
                callbacks=callbacks,
                args=args,
                kwargs=kwargs,
            )
            results = None
            exception = None
            try:
                results = fn(instance, *args, **kwargs)
            except Exception as e:
                exception = e
                raise exception
            else:
                return results
            finally:
                _end_callback_scope(
                    instance=instance,
                    fn=fn,
                    kind=kind,
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                    results=results,
                    exception=exception,
                    callbacks=callbacks,
                )

        return sync_wrapper  # ty:ignore[invalid-return-type]

    if fn is not None:
        return decorator(fn)
    return decorator


def _get_active_callbacks(instance: Any, run: RunContext | None = None) -> list[Callback]:
    callbacks = list(run.callbacks) if run else []
    return callbacks + getattr(instance, "callbacks", [])


def _execute_start_callbacks(
    *,
    instance: Any,
    fn: Callable[..., Any],
    kind: CallbackKind,
    call_id: str,
    callbacks: list[Callback],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    signature = inspect.signature(fn)
    bound_arguments = signature.bind_partial(instance, *args, **kwargs)
    bound_arguments.apply_defaults()
    inputs = dict(bound_arguments.arguments)
    if "self" in inputs:
        inputs.pop("self")
    elif "instance" in inputs:
        inputs.pop("instance")
    for callback in callbacks:
        try:
            _get_on_start_handler(callback=callback, kind=kind, fn=fn)(
                call_id=call_id, instance=instance, inputs=inputs
            )
        except Exception as e:
            logger.warning(f"Error when calling callback {callback}: {e}")


def _execute_end_callbacks(
    *,
    instance: Any,
    fn: Callable[..., Any],
    kind: CallbackKind,
    call_id: str,
    results: Any,
    exception: Exception | None,
    callbacks: list[Callback],
) -> None:
    for callback in callbacks:
        try:
            _get_on_end_handler(callback=callback, kind=kind, fn=fn)(
                call_id=call_id, outputs=results, exception=exception
            )
        except Exception as e:
            logger.warning(
                f"Error when applying callback {callback}'s end handler on function {_callable_name(fn)}: {e}."
            )


def _get_on_start_handler(*, callback: Callback, kind: CallbackKind, fn: Callable[..., Any]) -> Callable[..., Any]:
    if kind == "lm":
        return callback.on_lm_start
    if kind == "evaluate":
        return callback.on_evaluate_start
    if kind == "adapter":
        fn_name = _callable_name(fn)
        if fn_name == "format":
            return callback.on_adapter_format_start
        if fn_name == "parse":
            return callback.on_adapter_parse_start
        raise ValueError(f"Unsupported adapter method for using callback: {fn_name}.")
    if kind == "tool":
        return callback.on_tool_start
    return callback.on_module_start


def _get_on_end_handler(*, callback: Callback, kind: CallbackKind, fn: Callable[..., Any]) -> Callable[..., Any]:
    if kind == "lm":
        return callback.on_lm_end
    if kind == "evaluate":
        return callback.on_evaluate_end
    if kind == "adapter":
        fn_name = _callable_name(fn)
        if fn_name == "format":
            return callback.on_adapter_format_end
        if fn_name == "parse":
            return callback.on_adapter_parse_end
        raise ValueError(f"Unsupported adapter method for using callback: {fn_name}.")
    if kind == "tool":
        return callback.on_tool_end
    return callback.on_module_end

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Protocol, TypeVar

from dspy.runtime.callback_dispatch import (
    ACTIVE_CALL_ID,
    CallbackKind,
    ainvoke_with_callbacks,
    get_active_callbacks,
    invoke_with_callbacks,
)

__all__ = [
    "ACTIVE_CALL_ID",
    "Callback",
    "CallbackKind",
    "NoOpCallback",
    "with_callbacks",
]

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


def with_callbacks(fn: F | None = None, *, kind: CallbackKind | str = CallbackKind.MODULE) -> F | Callable[[F], F]:
    callback_kind = CallbackKind(kind)

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(instance, *args, **kwargs):
                run = kwargs.get("run")
                callbacks = get_active_callbacks(instance, run, kind=callback_kind)
                if not callbacks:
                    return await fn(instance, *args, **kwargs)
                return await ainvoke_with_callbacks(
                    instance=instance,
                    fn=fn,
                    kind=callback_kind,
                    callbacks=callbacks,
                    args=args,
                    kwargs=kwargs,
                )

            return async_wrapper  # ty:ignore[invalid-return-type]

        @functools.wraps(fn)
        def sync_wrapper(instance, *args, **kwargs):
            run = kwargs.get("run")
            callbacks = get_active_callbacks(instance, run, kind=callback_kind)
            if not callbacks:
                return fn(instance, *args, **kwargs)
            return invoke_with_callbacks(
                instance=instance,
                fn=fn,
                kind=callback_kind,
                callbacks=callbacks,
                args=args,
                kwargs=kwargs,
            )

        return sync_wrapper  # ty:ignore[invalid-return-type]

    if fn is not None:
        return decorator(fn)
    return decorator

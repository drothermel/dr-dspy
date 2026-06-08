import functools
import inspect
import logging
import uuid
from contextvars import ContextVar
from typing import Any, Callable

from dspy.dsp.utils.settings import settings

ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)
logger = logging.getLogger(__name__)


def _is_lm(instance: Any) -> bool:
    from dspy.clients.base_lm import BaseLM

    return isinstance(instance, BaseLM)


def _is_evaluate(instance: Any) -> bool:
    from dspy.evaluate.evaluate import Evaluate

    return isinstance(instance, Evaluate)


def _is_adapter(instance: Any) -> bool:
    from dspy.adapters.base import Adapter

    return isinstance(instance, Adapter)


def _is_tool(instance: Any) -> bool:
    from dspy.adapters.types.tool import Tool

    return isinstance(instance, Tool)


class BaseCallback:
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


def with_callbacks(fn):

    def _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs) -> None:
        inputs = inspect.getcallargs(fn, instance, *args, **kwargs)
        if "self" in inputs:
            inputs.pop("self")
        elif "instance" in inputs:
            inputs.pop("instance")
        for callback in callbacks:
            try:
                _get_on_start_handler(callback=callback, instance=instance, fn=fn)(
                    call_id=call_id, instance=instance, inputs=inputs
                )
            except Exception as e:
                logger.warning(f"Error when calling callback {callback}: {e}")

    def _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks) -> None:
        for callback in callbacks:
            try:
                _get_on_end_handler(callback=callback, instance=instance, fn=fn)(
                    call_id=call_id, outputs=results, exception=exception
                )
            except Exception as e:
                logger.warning(f"Error when applying callback {callback}'s end handler on function {fn.__name__}: {e}.")

    def _get_active_callbacks(instance):
        return settings.get("callbacks", []) + getattr(instance, "callbacks", [])

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return await fn(instance, *args, **kwargs)
            call_id = uuid.uuid4().hex
            _execute_start_callbacks(
                instance=instance, fn=fn, call_id=call_id, callbacks=callbacks, args=args, kwargs=kwargs
            )
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)
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
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(
                    instance=instance, fn=fn, call_id=call_id, results=results, exception=exception, callbacks=callbacks
                )

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(instance, *args, **kwargs):
        callbacks = _get_active_callbacks(instance)
        if not callbacks:
            return fn(instance, *args, **kwargs)
        call_id = uuid.uuid4().hex
        _execute_start_callbacks(
            instance=instance, fn=fn, call_id=call_id, callbacks=callbacks, args=args, kwargs=kwargs
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        ACTIVE_CALL_ID.set(call_id)
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
            ACTIVE_CALL_ID.set(parent_call_id)
            _execute_end_callbacks(
                instance=instance, fn=fn, call_id=call_id, results=results, exception=exception, callbacks=callbacks
            )

    return sync_wrapper


def _get_on_start_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    if _is_lm(instance):
        return callback.on_lm_start
    if _is_evaluate(instance):
        return callback.on_evaluate_start
    if _is_adapter(instance):
        if fn.__name__ == "format":
            return callback.on_adapter_format_start
        if fn.__name__ == "parse":
            return callback.on_adapter_parse_start
        raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")
    if _is_tool(instance):
        return callback.on_tool_start
    return callback.on_module_start


def _get_on_end_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    if _is_lm(instance):
        return callback.on_lm_end
    if _is_evaluate(instance):
        return callback.on_evaluate_end
    if _is_adapter(instance):
        if fn.__name__ == "format":
            return callback.on_adapter_format_end
        if fn.__name__ == "parse":
            return callback.on_adapter_parse_end
        raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")
    if _is_tool(instance):
        return callback.on_tool_end
    return callback.on_module_end

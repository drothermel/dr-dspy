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
    """A base class for defining callback handlers for DSPy components.

    To use a callback, subclass this class and implement the desired handlers. Each handler
    will be called at the appropriate time before/after the execution of the corresponding component.  For example, if
    you want to print a message before and after an LM is called, implement `the on_llm_start` and `on_lm_end` handler.
    Users can set the callback globally using `dspy.configure` or locally by passing it to the component
    constructor.


    Example 1: Set a global callback using `dspy.configure`.

    ```
    from dspy.dsp.utils.settings import settings
    from dspy.predict.chain_of_thought import ChainOfThought
    from dspy.utils.callback import BaseCallback

    class LoggingCallback(BaseCallback):

        def on_lm_start(self, call_id, instance, inputs):
            print(f"LM is called with inputs: {inputs}")

        def on_lm_end(self, call_id, outputs, exception):
            print(f"LM is finished with outputs: {outputs}")

    settings.configure(
        callbacks=[LoggingCallback()]
    )

    cot = ChainOfThought("question -> answer")
    cot(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}
    ```

    Example 2: Set a local callback by passing it to the component constructor.

    ```
    from dspy.clients.lm import LM

    lm_1 = LM("gpt-3.5-turbo", callbacks=[LoggingCallback()])
    lm_1(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}

    lm_2 = LM("gpt-3.5-turbo")
    lm_2(question="What is the meaning of life?")
    # No logging here because only `lm_1` has the callback set.
    ```
    """

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """A handler triggered when forward() method of a module (subclass of Module) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Module instance.
            inputs: The inputs to the module's forward() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        """A handler triggered after forward() method of a module (subclass of Module) is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the module's forward() method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """A handler triggered when __call__ method of LM instance is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The LM instance.
            inputs: The inputs to the LM's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """A handler triggered after __call__ method of LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the LM's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """A handler triggered when format() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's format() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """A handler triggered after format() method of an adapter (subclass of dspy.Adapter) is called..

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's format() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """A handler triggered when parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's parse() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """A handler triggered after parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's parse() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """A handler triggered when a tool is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Tool instance.
            inputs: The inputs to the Tool's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """A handler triggered after a tool is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Tool's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """A handler triggered when evaluation is started.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Evaluate instance.
            inputs: The inputs to the Evaluate's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        """A handler triggered after evaluation is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Evaluate's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """


def with_callbacks(fn):
    """Decorator to add callback functionality to instance methods."""

    def _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs) -> None:
        """Execute all start callbacks for a function call."""
        inputs = inspect.getcallargs(fn, instance, *args, **kwargs)  # ty:ignore[deprecated]
        if "self" in inputs:
            inputs.pop("self")
        elif "instance" in inputs:
            inputs.pop("instance")
        for callback in callbacks:
            try:
                _get_on_start_handler(callback, instance, fn)(call_id=call_id, instance=instance, inputs=inputs)
            except Exception as e:
                logger.warning(f"Error when calling callback {callback}: {e}")

    def _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks) -> None:
        """Execute all end callbacks for a function call."""
        for callback in callbacks:
            try:
                _get_on_end_handler(callback, instance, fn)(
                    call_id=call_id,
                    outputs=results,
                    exception=exception,
                )
            except Exception as e:
                logger.warning(f"Error when applying callback {callback}'s end handler on function {fn.__name__}: {e}.")

    def _get_active_callbacks(instance):
        """Get combined global and instance-level callbacks."""
        return settings.get("callbacks", []) + getattr(instance, "callbacks", [])

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return await fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)  # ty:ignore[invalid-argument-type]

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
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(instance, *args, **kwargs):
        callbacks = _get_active_callbacks(instance)
        if not callbacks:
            return fn(instance, *args, **kwargs)

        call_id = uuid.uuid4().hex

        _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

        # Active ID must be set right before the function is called, not before calling the callbacks.
        parent_call_id = ACTIVE_CALL_ID.get()
        ACTIVE_CALL_ID.set(call_id)  # ty:ignore[invalid-argument-type]

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
            _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

    return sync_wrapper


def _get_on_start_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Selects the appropriate on_start handler of the callback based on the instance and function name."""
    if _is_lm(instance):
        return callback.on_lm_start
    if _is_evaluate(instance):
        return callback.on_evaluate_start

    if _is_adapter(instance):
        if fn.__name__ == "format":  # ty:ignore[unresolved-attribute]
            return callback.on_adapter_format_start
        if fn.__name__ == "parse":  # ty:ignore[unresolved-attribute]
            return callback.on_adapter_parse_start
        raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")  # ty:ignore[unresolved-attribute]

    if _is_tool(instance):
        return callback.on_tool_start

    # We treat everything else as a module.
    return callback.on_module_start


def _get_on_end_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Selects the appropriate on_end handler of the callback based on the instance and function name."""
    if _is_lm(instance):
        return callback.on_lm_end
    if _is_evaluate(instance):
        return callback.on_evaluate_end

    if _is_adapter(instance):
        if fn.__name__ == "format":  # ty:ignore[unresolved-attribute]
            return callback.on_adapter_format_end
        if fn.__name__ == "parse":  # ty:ignore[unresolved-attribute]
            return callback.on_adapter_parse_end
        raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")  # ty:ignore[unresolved-attribute]

    if _is_tool(instance):
        return callback.on_tool_end

    # We treat everything else as a module.
    return callback.on_module_end

import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Any

import anyio.from_thread
from typing_extensions import override

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback


@dataclass
class StreamResponse:
    predict_name: str
    signature_field_name: str
    chunk: str
    is_last_chunk: bool


@dataclass
class StatusMessage:
    """Dataclass that wraps a status message for status streaming."""

    message: str


def sync_send_to_stream(stream, message):
    """Send message to stream in a sync context, regardless of event loop state."""

    async def _send() -> None:
        await stream.send(message)

    try:
        asyncio.get_running_loop()

        # If we're in an event loop, offload to a new thread with its own event loop
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_send())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except RuntimeError:
        # Not in an event loop, safe to use a new event loop in this thread
        return anyio.from_thread.run(_send)


class StatusMessageProvider:
    """Provides customizable status message streaming for DSPy programs.

    This class serves as a base for creating custom status message providers. Users can subclass
    and override its methods to define specific status messages for different stages of program execution,
    each method must return a string.

    Examples:
    ```python
    from dspy.predict.predict import Predict
    from dspy.streaming.streamify import streamify

    class MyStatusMessageProvider(StatusMessageProvider):
        def lm_start_status_message(self, instance, inputs):
            return f"Calling LM with inputs {inputs}..."

        def module_end_status_message(self, outputs):
            return f"Module finished with output: {outputs}!"

    program = streamify(Predict("q->a"), status_message_provider=MyStatusMessageProvider())
    ```
    """

    def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]) -> str:
        """Status message before a `dspy.adapters.types.tool.Tool` is called."""
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs: Any) -> str:
        """Status message after a `dspy.adapters.types.tool.Tool` is called."""
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance: Any, inputs: dict[str, Any]) -> None:
        """Status message before a `dspy.primitives.module.Module` or `dspy.predict.predict.Predict` is called."""

    def module_end_status_message(self, outputs: Any) -> None:
        """Status message after a `dspy.primitives.module.Module` or `dspy.predict.predict.Predict` is called."""

    def lm_start_status_message(self, instance: Any, inputs: dict[str, Any]) -> None:
        """Status message before a `dspy.clients.lm.LM` is called."""

    def lm_end_status_message(self, outputs: Any) -> None:
        """Status message after a `dspy.clients.lm.LM` is called."""


class StatusStreamingCallback(BaseCallback):
    def __init__(self, status_message_provider: StatusMessageProvider | None = None) -> None:
        self.status_message_provider = status_message_provider or StatusMessageProvider()

    @override
    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        stream = settings.send_stream
        if stream is None or instance.name == "finish":
            return

        status_message = self.status_message_provider.tool_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    @override
    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        stream = settings.send_stream
        if stream is None or outputs == "Completed.":
            return

        status_message = self.status_message_provider.tool_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    @override
    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    @override
    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    @override
    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    @override
    def on_module_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

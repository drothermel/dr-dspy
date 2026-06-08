"""Normalized LM types — message builder helpers."""

from __future__ import annotations

from typing import Any

from dspy.core.types.messages import LMMessage
from dspy.core.types.parts import LMToolCallPart, LMToolResultPart, _coerce_part


def System(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:  # noqa: N802 dynamic typing/lint migration for scoped ty adoption
    """Create a system message for a direct LM call.

    A system message gives model-level instructions, such as tone, scope, or
    formatting rules. Pass text or normalized `LMPart` objects; DSPy stores
    them as one `LMMessage` with role `"system"`.

    Args:
        *parts: Text or normalized LM parts to include in the message.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` that can be passed to `dspy.clients.lm.LM` or `LMRequest`.

    Examples:
        System instruction with a user turn:

        ```python
        from dspy.core.types import LMRequest, System, User

        request = LMRequest.from_call(
            model="openai/gpt-4o-mini",
            items=(
                System("You are concise."),
                User("What is DSPy?"),
            ),
        )
        ```

    See Also:
        `User`
        `Assistant`
        `LMRequest`
    """
    return LMMessage(role="system", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


def Developer(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:  # noqa: N802 dynamic typing/lint migration for scoped ty adoption
    """Create a developer message for a direct LM call.

    A developer message carries instructions that sit between system guidance
    and user content. Use it when a provider supports a `"developer"` role and
    you want to keep implementation guidance separate from the user's request.

    Args:
        *parts: Text or normalized LM parts to include in the message.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` with role `"developer"`.

    Examples:
        Add house-style instructions:

        ```python
        from dspy.core.types import Developer, LMRequest, System, User

        request = LMRequest.from_call(
            model="openai/gpt-4o-mini",
            items=(
                System("You are a technical editor."),
                Developer("Prefer short examples."),
                User("Explain callbacks."),
            ),
        )
        ```

    See Also:
        [`dspy.System`][dspy.System]
        [`dspy.User`][dspy.User]
    """
    return LMMessage(role="developer", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


def User(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:  # noqa: N802 dynamic typing/lint migration for scoped ty adoption
    """Create a user message for a direct LM call.

    A user message contains the request or data you want the model to answer.
    Pass plain text for simple prompts, or mix text with normalized image, audio,
    document, binary, and other `LMPart` objects for multimodal calls.

    Args:
        *parts: Text or normalized LM parts to include in the message.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` with role `"user"`.

    Examples:
        Multi-turn LM call:

        ```python
        from dspy.clients.lm import LM
        from dspy.core.types import Assistant, User

        lm = LM("openai/gpt-4o-mini")
        request = LMRequest.from_call(
            model=lm.model,
            items=(
                User("What is DSPy?"),
                Assistant("DSPy is a framework for programming LM pipelines."),
                User("Say that in five words."),
            ),
        )
        response = lm(request)
        ```

        Multi-turn call with media:

        ```python
        from dspy.clients.lm import LM
        from dspy.core.types import LMImagePart, System, User
        from dspy.dsp.utils.settings import settings

        lm = LM("openai/gpt-4o-mini")
        request = LMRequest.from_call(
            model=lm.model,
            messages=[
                System("Answer in one sentence."),
                User(
                    "Describe this image.",
                    LMImagePart(url="https://example.com/dog.png"),
                ),
            ],
        )
        response = lm(request)
        ```

        For a single user turn, build the request explicitly:

        ```python
        request = LMRequest.from_call(
            model=lm.model,
            items=("Describe this image.", LMImagePart(url="https://example.com/dog.png")),
        )
        response = lm(request)
        ```

        Explicit `LMRequest` for custom LM authors and advanced users:

        ```python
        from dspy.clients.lm import LM
        from dspy.core.types import LMConfig, LMImagePart, LMRequest, System, User

        lm = LM("openai/gpt-4o-mini")
        request = LMRequest(
            model="openai/gpt-4o-mini",
            messages=[
                System("You are concise."),
                User(
                    "Describe this image.",
                    LMImagePart(url="https://example.com/dog.png"),
                ),
            ],
            config=LMConfig(temperature=0.2, max_tokens=200),
        )

        response = lm(request)
        ```

    See Also:
        [`dspy.System`][dspy.System]
        [`dspy.Assistant`][dspy.Assistant]
        [`dspy.ToolResult`][dspy.ToolResult]
    """
    return LMMessage(role="user", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


def Assistant(*parts: Any, name: str | None = None, metadata: dict[str, Any] | None = None) -> LMMessage:  # noqa: N802 dynamic typing/lint migration for scoped ty adoption
    """Create an assistant message for a direct LM call.

    An assistant message represents a previous model response. Use it when you
    build a multi-turn request by hand, or when you send tool calls back to the
    model before adding tool results.

    Args:
        *parts: Text, tool calls, citations, media parts, or other normalized LM
            parts from an assistant turn.
        name: Optional sender name for providers that support named messages.
        metadata: Extra information to keep with the message.

    Returns:
        An `LMMessage` with role `"assistant"`.

    Examples:
        Continue a conversation:

        ```python
        from dspy.core.types import Assistant, LMRequest, User

        request = LMRequest.from_call(
            model="openai/gpt-4o-mini",
            items=(
                User("What is DSPy?"),
                Assistant("DSPy is a framework for programming LM pipelines."),
                User("Say that in five words."),
            ),
        )
        ```

    See Also:
        [`dspy.User`][dspy.User]
        [`dspy.ToolCall`][dspy.ToolCall]
        [`dspy.ToolResult`][dspy.ToolResult]
    """
    return LMMessage(role="assistant", parts=[_coerce_part(part) for part in parts], name=name, metadata=metadata or {})


ToolCall = LMToolCallPart
"""Create a tool-call part for an assistant message.

`ToolCall` is an alias for `LMToolCallPart`. Use it inside `Assistant(...)`
when you want to include a model-requested tool call in a normalized
conversation.
"""


def ToolResult(  # noqa: N802
    *parts: Any,
    call_id: str | None = None,
    name: str | None = None,
    content: Any | None = None,
    is_error: bool = False,
) -> LMMessage:
    """Create a tool-result message for a direct LM call.

    A tool-result message sends the output of a tool back to the model. Pass the
    returned text or media as `*parts` or with `content=...`, and include the
    `call_id` from the matching assistant tool call when the provider uses call
    IDs.

    Args:
        *parts: Text, DSPy media objects, or normalized LM parts returned by the
            tool. If you pass one `LMToolResultPart`, DSPy uses it directly.
        call_id: Identifier of the assistant tool call this result answers.
        name: Tool name associated with the result.
        content: Optional tool output passed by keyword. Use this when adapting
            OpenAI-style code that stores the result under `content`.
        is_error: Whether this result represents a failed tool execution.

    Returns:
        An `LMMessage` with role `"tool"` and one `LMToolResultPart`.

    Examples:
        Send a weather result back to the model:

        ```python
        from dspy.core.types import Assistant, ToolCall, ToolResult, User

        messages = [
            User("What is the weather in Paris?"),
            Assistant(
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    args={"location": "Paris"},
                )
            ),
            ToolResult(
                '{"temperature": "22", "unit": "celsius"}',
                call_id="call_1",
                name="get_weather",
            ),
            User("Summarize the result."),
        ]
        ```

    See Also:
        [`dspy.ToolCall`][dspy.ToolCall]
        [`dspy.Assistant`][dspy.Assistant]
        `dspy.adapters.types.tool.Tool`
    """
    if content is not None:
        if parts:
            raise TypeError("Pass tool output either as positional parts or as `content=...`, not both.")
        parts = tuple(content if isinstance(content, list) else [content])

    if len(parts) == 1 and isinstance(parts[0], LMToolResultPart):
        result = parts[0]
    else:
        result = LMToolResultPart(
            call_id=call_id,
            name=name,
            content=[_coerce_part(part) for part in parts],
            is_error=is_error,
        )
    return LMMessage(role="tool", parts=[result])

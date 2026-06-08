from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pydantic
from typing_extensions import override

from dspy.adapters.types.base_type import Type

if TYPE_CHECKING:
    from collections.abc import Iterator


class Reasoning(Type):
    """Reasoning type in DSPy.

    This type is useful when you want the DSPy output to include the reasoning of the LM. We build this type so that
    DSPy can support the reasoning model and non-reasoning model with the same code.

    This is a str-like type, you can convert a string directly to a Reasoning object, and from DSPy adapters'
    perspective, `Reasoning` is treated as a string.
    """

    content: str

    @override
    def format(self) -> str:
        return f"{self.content}"

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"content": data}

        if isinstance(data, dict):
            data = cast("dict[str, object]", data)
            if "content" not in data:
                raise ValueError("`content` field is required for `dspy.Reasoning`")
            if not isinstance(data["content"], str):
                raise ValueError(f"`content` field must be a string, but received type: {type(data['content'])}")
            return {"content": data["content"]}

        raise ValueError(f"Received invalid value for `dspy.Reasoning`: {data}")

    @classmethod
    @override
    def parse_lm_output(cls, output: object) -> Reasoning | None:
        """Parse the typed LM output into a Reasoning object."""
        reasoning_content = getattr(output, "reasoning_content", None)
        if reasoning_content:
            return Reasoning(content=reasoning_content)
        return None

    @classmethod
    @override
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Reasoning | None:
        """Parse the LM response into a Reasoning object."""
        if isinstance(response, dict) and "reasoning_content" in response:
            return Reasoning(content=response["reasoning_content"])
        return None

    @classmethod
    @override
    def parse_stream_chunk(cls, chunk: object) -> str | None:
        """
        Parse a stream chunk into reasoning content if available.

        Args:
            chunk: A stream chunk from the LM.

        Returns:
            The reasoning content (str) if available, None otherwise.
        """
        try:
            if choices := getattr(chunk, "choices", None):
                return getattr(choices[0].delta, "reasoning_content", None)
        except Exception:
            return None

    @classmethod
    @override
    def is_streamable(cls) -> bool:
        return True

    @override
    def __repr__(self) -> str:
        return f"{self.content!r}"

    @override
    def __str__(self) -> str:
        return self.content

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Reasoning):
            return self.content == other.content
        if isinstance(other, str):
            return self.content == other
        return False

    @override
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return len(self.content)

    def __getitem__(self, key: int | slice) -> str:
        return self.content[key]

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return item in self.content

    @override
    def __iter__(self) -> Iterator[str]:  # ty: ignore[invalid-method-override]
        return iter(self.content)

    def __add__(self, other: object) -> object:
        if isinstance(other, Reasoning):
            return Reasoning(content=self.content + other.content)
        if isinstance(other, str):
            return self.content + other
        return NotImplemented

    def __radd__(self, other: object) -> object:
        if isinstance(other, str):
            return other + self.content
        if isinstance(other, Reasoning):
            return Reasoning(content=other.content + self.content)
        return NotImplemented

    def __getattr__(self, name: str) -> object:
        """
        Delegate string methods to the underlying content.

        This makes Reasoning fully str-like by forwarding any string method calls
        (like .strip(), .lower(), .split(), etc.) to the content string.

        Note: This is called only when the attribute is not found on the Reasoning instance,
        so it won't interfere with Pydantic fields or existing methods.
        """
        # Check if this is a valid string method/attribute
        if hasattr(str, name):
            # Delegate to the content string
            return getattr(self.content, name)

        # If it's not a string method, provide a helpful error
        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute '{name}'. "
            f"If you are using `dspy.predict.chain_of_thought.ChainOfThought`, note that the 'reasoning' field "
            "in ChainOfThought is now a "
            "`dspy.Reasoning` object (not a plain string). "
            f"You can convert it to a string with str(reasoning) or access the content with reasoning.content."
        )

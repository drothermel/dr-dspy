from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.base.call import AdapterCallMixin
from dspy.adapters.base.conversation import AdapterConversationMixin
from dspy.adapters.base.format import AdapterFormatMixin
from dspy.adapters.base.native import _DEFAULT_NATIVE_RESPONSE_TYPES
from dspy.adapters.types.base_type import Type
from dspy.task_spec import TaskSpec

if TYPE_CHECKING:
    from dspy.utils.callback import BaseCallback


class Adapter(AdapterCallMixin, AdapterFormatMixin, AdapterConversationMixin):
    """Base Adapter class.

    The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs). It handles the
    complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.

    Key responsibilities:
        - Transform user inputs and signatures into properly formatted LM prompts, which also instructs the LM to format
            the response in a specific format.
        - Parse LM outputs into dictionaries matching the signature's output fields.
        - Enable/disable native LM features (function calling, citations, etc.) based on configuration.
        - Handle conversation history, few-shot examples, and custom type processing.

    The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming
    model for users.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> None:
        """
        Args:
            callbacks: List of callback functions to execute during `format()` and `parse()` methods. Callbacks can be
                used for logging, monitoring, or custom processing. Defaults to None (empty list).
            use_native_function_calling: Whether to enable native function calling capabilities when the LM supports it.
                If True, the adapter will automatically configure function calling when input fields contain
                `dspy.adapters.types.tool.Tool` or `list[dspy.adapters.types.tool.Tool]` types. Defaults to False.
            native_response_types: List of output field types that should be handled by native LM features rather than
                adapter parsing. For example, `dspy.adapters.types.citation.Citations` can be populated directly by citation APIs
                (e.g., Anthropic's citation feature). Defaults to `[Citations]`.
            parallel_tool_calls: Whether to request provider-side parallel tool-call generation when native function
                calling is active. If None, the adapter does not set the provider option. Defaults to None.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.parallel_tool_calls = parallel_tool_calls
        self.native_response_types = native_response_types or cast("list[type[Type]]", _DEFAULT_NATIVE_RESPONSE_TYPES)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        from dspy.utils.callback import with_callbacks

        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            task_spec: The DSPy task spec for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError

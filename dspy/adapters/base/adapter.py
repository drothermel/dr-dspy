from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dspy.adapters.base.call import AdapterCallMixin
from dspy.adapters.base.format import AdapterFormatMixin
from dspy.adapters.base.native import _DEFAULT_NATIVE_RESPONSE_TYPES
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.mode import AdapterCallMode
from dspy.adapters.types.base_type import Type
from dspy.runtime.callback import with_callbacks
from dspy.task_spec import TaskSpec

if TYPE_CHECKING:
    from dspy.adapters.call.policies.parse_fallback import ParseFallbackPolicy
    from dspy.adapters.call.policies.response_format import ResponseFormatPolicy
    from dspy.runtime.callback import Callback


class Adapter(AdapterCallMixin, AdapterFormatMixin):
    """Base adapter for formatting task specs into LM requests and parsing responses.

    Native function-calling defaults by adapter subclass:

    | Adapter | ``use_native_function_calling`` default | Rationale |
    |---------|----------------------------------------|-----------|
    | ChatAdapter / XMLAdapter | ``False`` | Text marker parsing; FC optional |
    | JSONAdapter / BAMLAdapter | ``True`` | Structured JSON + tool calls via provider FC |
    | TwoStepAdapter | inherits kwargs | Main call text; extraction uses inner adapter |
    """

    response_format_policy: ResponseFormatPolicy | None = None
    parse_fallback_policy: ParseFallbackPolicy | None = None
    call_mode: AdapterCallMode | None = None
    capabilities: AdapterCapabilities = AdapterCapabilities()

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
        allow_json_repair: bool = False,
    ) -> None:
        """Configure adapter behavior.

        ``parallel_tool_calls`` merges into ``LMToolChoice.parallel`` only when
        ``tool_choice`` is ``None`` or ``tool_choice.parallel`` is ``None``.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.parallel_tool_calls = parallel_tool_calls
        self.allow_json_repair = allow_json_repair
        self.native_response_types = native_response_types or cast("list[type[Type]]", _DEFAULT_NATIVE_RESPONSE_TYPES)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.format = with_callbacks(kind="adapter")(cls.format)
        cls.parse = with_callbacks(kind="adapter")(cls.parse)

    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        """Parse LM text into output fields.

        Contract: return a dict whose keys match ``task_spec.output_fields`` exactly.
        Missing or extra keys raise ``AdapterParseError`` via ``validate_parsed_fields``.
        """
        raise NotImplementedError

    def format_finetune_data(
        self,
        task_spec: TaskSpec,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support finetune data formatting. "
            "Use an adapter with capabilities.supports_finetune=True."
        )

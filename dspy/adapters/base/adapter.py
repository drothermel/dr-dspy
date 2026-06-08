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
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> None:
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
        raise NotImplementedError

from typing import Any

from dspy.core.types.adaptation import NativeAdaptationMode
from dspy.core.types.builders import Assistant, Developer, System, ToolCall, ToolResult, User
from dspy.core.types.call_record import CallRecord
from dspy.core.types.embedding_options import EmbedderOptions
from dspy.core.types.lm import LMForward
from dspy.core.types.lm_config import (
    LMConfig,
    LMPromptCacheConfig,
    LMReasoningConfig,
    ReasoningEffort,
    coerce_lm_config,
    lm_defaults_config,
    merge_lm_config,
    merge_lm_request_config,
)
from dspy.core.types.lm_output import LMOutput
from dspy.core.types.lm_provider import LMProviderOptions, merge_provider_options
from dspy.core.types.lm_response import LMResponse
from dspy.core.types.messages import LMMessage, LMMessageRole
from dspy.core.types.parts import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMDocumentPart,
    LMImagePart,
    LMOpaquePart,
    LMPart,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMToolResultPart,
    LMVideoPart,
)
from dspy.core.types.request import LMRequest
from dspy.core.types.stream import AsyncLMStream, LMStream
from dspy.core.types.stream_builder import LMOutputBuilder
from dspy.core.types.stream_events import (
    LMAnyDelta,
    LMAudioDelta,
    LMCitationDelta,
    LMImageDelta,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamErrorEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMToolCallDelta,
)
from dspy.core.types.tool_spec import LMToolChoice, LMToolSpec, coerce_tool_spec
from dspy.core.types.usage import LMUsage

UserMessageContent = str | list[dict[str, Any]]

__all__ = [
    "Assistant",
    "AsyncLMStream",
    "Developer",
    "LMAnyDelta",
    "LMAudioDelta",
    "LMAudioPart",
    "LMBinaryPart",
    "LMCitationDelta",
    "LMCitationPart",
    "EmbedderOptions",
    "LMConfig",
    "LMProviderOptions",
    "coerce_lm_config",
    "coerce_tool_spec",
    "lm_defaults_config",
    "merge_lm_config",
    "merge_lm_request_config",
    "merge_provider_options",
    "LMDocumentPart",
    "CallRecord",
    "LMImageDelta",
    "LMImagePart",
    "LMForward",
    "LMMessage",
    "LMMessageRole",
    "LMOpaquePart",
    "LMOutput",
    "LMOutputBuilder",
    "LMPart",
    "LMPromptCacheConfig",
    "LMReasoningConfig",
    "NativeAdaptationMode",
    "ReasoningEffort",
    "UserMessageContent",
    "LMRefusalPart",
    "LMRequest",
    "LMResponse",
    "LMStream",
    "LMStreamDeltaEvent",
    "LMStreamEndEvent",
    "LMStreamErrorEvent",
    "LMStreamEvent",
    "LMStreamOutputEndEvent",
    "LMStreamStartEvent",
    "LMTextDelta",
    "LMTextPart",
    "LMThinkingDelta",
    "LMThinkingPart",
    "LMToolCallDelta",
    "LMToolCallPart",
    "LMToolChoice",
    "LMToolResultPart",
    "LMToolSpec",
    "LMUsage",
    "LMVideoPart",
    "System",
    "ToolCall",
    "ToolResult",
    "User",
]

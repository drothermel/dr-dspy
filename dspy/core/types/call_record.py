from __future__ import annotations

from pprint import pformat
from typing import Any

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from dspy.core.types.lm_response import LMResponse
from dspy.core.types.messages import LMMessage
from dspy.core.types.request import LMRequest
from dspy.core.types.request_views import request_kwargs, request_prompt
from dspy.serialization.json import to_jsonable


class CallRecord(BaseModel):
    request: LMRequest
    response: LMResponse
    timestamp: str
    uuid: str
    model_type: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @property
    def outputs(self) -> list[Any]:
        return self.response.to_outputs()

    @property
    def usage(self) -> dict[str, Any]:
        return self.response.usage_as_dict()

    @property
    def cost(self) -> float | None:
        return self.response.cost

    @property
    def model(self) -> str:
        return self.request.model

    @property
    def prompt(self) -> str | None:
        return request_prompt(self.request)

    @property
    def messages(self) -> list[LMMessage]:
        return self.request.messages

    @property
    def kwargs(self) -> dict[str, Any]:
        return request_kwargs(self.request)

    @property
    def response_model(self) -> str | None:
        return self.response.model

    @override
    def __repr__(self) -> str:
        formatted = pformat(self.model_dump(mode="python", exclude_none=True), width=100, sort_dicts=False)
        return f"CallRecord(\n{formatted}\n)"

    @override
    def __str__(self) -> str:
        return repr(self)

    def to_dict(self, *, mode: str = "python", exclude_none: bool = False, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            return self.model_dump(mode=mode, exclude_none=exclude_none, **kwargs)
        data = {
            **self.model_dump(mode="python", exclude_none=True),
            "outputs": self.outputs,
            "usage": self.usage,
            "cost": self.cost,
            "model": self.model,
            "prompt": self.prompt,
            "kwargs": self.kwargs,
            "response_model": self.response_model,
        }
        if mode != "python":
            data = to_jsonable(data)
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data

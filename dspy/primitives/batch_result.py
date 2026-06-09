from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BatchFailure(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    input: Any
    exception: BaseException


class BatchResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    results: list[Any]
    failures: tuple[BatchFailure, ...] = Field(default_factory=tuple)

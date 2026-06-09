from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict


class BatchFailure(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    input: Any
    exception: BaseException


class BatchResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    results: Sequence[Any]
    failures: Sequence[BatchFailure] = ()

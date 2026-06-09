from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class CoproEvaluatedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    score: float
    program: Any
    instruction: str
    prefix: str
    depth: int

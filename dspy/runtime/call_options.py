from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModuleCallOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

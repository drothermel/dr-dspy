from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AdapterCapabilities:
    supports_finetune: bool = False
    field_value_role: Literal["none", "user", "assistant"] = "none"
    default_native_fc: bool = False
    supports_structured_output: bool = False

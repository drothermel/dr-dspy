from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy.primitives import Example, Module


@dataclass
class BootstrapCompileSession:
    """Per-compile mutable state for a single BootstrapFewShot ``compile()`` invocation."""

    student: Module
    teacher: Module
    trainset: list[Example]
    name2predictor: dict[str, Any] = field(default_factory=dict)
    predictor2name: dict[int, str] = field(default_factory=dict)
    name2traces: dict[str, list] = field(default_factory=dict)
    validation: list[Example] = field(default_factory=list)
    error_count: int = 0

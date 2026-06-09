from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dspy.adapters.call.preprocessors.context import PreprocessState


class CallPreprocessor(Protocol):
    def run(self, state: PreprocessState) -> PreprocessState: ...

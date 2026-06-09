from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.core.types import LMConfig, coerce_lm_config

if TYPE_CHECKING:
    from dspy.adapters.call.preprocessors.context import PreprocessState


class CoerceConfigPreprocessor:
    def run(self, state: PreprocessState) -> PreprocessState:
        if not isinstance(state.config, LMConfig):
            state.config = coerce_lm_config(state.config)
        return state

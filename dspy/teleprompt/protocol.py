from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from dspy.primitives import Module
    from dspy.runtime.run_context import RunContext
    from dspy.teleprompt.compilation import CompileResult


@runtime_checkable
class Teleprompter(Protocol):
    async def compile(
        self,
        student: Module,
        *,
        params: BaseModel,
        run: RunContext,
    ) -> CompileResult: ...

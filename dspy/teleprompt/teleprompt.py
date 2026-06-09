from abc import ABC, abstractmethod
from typing import Any

from dspy.primitives.module import Module
from dspy.runtime.run_context import RunContext


class Teleprompter(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def compile(self, student: Module, *, run: RunContext) -> Module:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return self.__dict__

from abc import ABC, abstractmethod
from typing import Any


class Proposer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def propose_instructions_for_program(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def propose_instruction_for_predictor(self, *args: Any, **kwargs: Any) -> Any:
        pass

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from typing_extensions import override

SIMPLE_TYPES = (str, int, float, bool, list, dict, type(None))


class CodeInterpreterError(RuntimeError):
    pass


class FinalOutput:
    def __init__(self, output: Any) -> None:
        self.output = output

    @override
    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalOutput):
            return NotImplemented
        return self.output == other.output


@runtime_checkable
class CodeInterpreter(Protocol):
    @property
    def tools(self) -> dict[str, Callable[..., str]]: ...

    def start(self) -> None: ...

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any: ...

    def shutdown(self) -> None: ...

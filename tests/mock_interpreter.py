from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

__all__ = ["MockInterpreter"]


class MockInterpreter:
    def __init__(
        self,
        responses: list[str | FinalOutput | Exception] | None = None,
        execute_fn: Callable[[str, dict[str, Any]], Any] | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
    ):
        self.responses = list(responses) if responses else []
        self.execute_fn = execute_fn
        self.tools = tools or {}
        self.call_count = 0
        self.call_history: list[tuple[str, dict[str, Any]]] = []
        self._shutdown = False

    def start(self) -> None:
        pass

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        if self._shutdown:
            raise CodeInterpreterError("MockInterpreter has been shutdown")
        variables = variables or {}
        self.call_history.append((code, variables))
        self.call_count += 1
        if self.execute_fn is not None:
            return self.execute_fn(code, variables)
        if not self.responses:
            return ""
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def shutdown(self) -> None:
        self._shutdown = True

    def reset(self) -> None:
        self.call_count = 0
        self.call_history = []
        self._shutdown = False

    def __enter__(self) -> "MockInterpreter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

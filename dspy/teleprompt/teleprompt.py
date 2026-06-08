from typing import Any

from dspy.runtime.run_context import RunContext


class Teleprompter:
    def __init__(self) -> None:
        pass

    async def compile(self, *args: Any, run: RunContext, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return self.__dict__

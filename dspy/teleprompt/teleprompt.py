from typing import Any


class Teleprompter:
    def __init__(self) -> None:
        pass

    async def compile(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return self.__dict__

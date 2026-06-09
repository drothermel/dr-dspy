from pydantic import BaseModel, ConfigDict

from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.protocol import Teleprompter


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ok:
    async def compile(self, student, *, params, run) -> CompileResult:
        return CompileResult.with_compiled_program(student)


def test_teleprompter_protocol_isinstance():
    assert isinstance(Ok(), Teleprompter)
    assert not isinstance(object(), Teleprompter)

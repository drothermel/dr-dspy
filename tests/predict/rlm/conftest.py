from contextlib import contextmanager

from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMRequest, LMResponse
from dspy.core.types.lm_provider import LMProviderOptions
from dspy.primitives.prediction import Prediction
from dspy.primitives.sandbox_serializable import SandboxSerializable


class FailingSubLM(BaseLM):
    def __init__(self) -> None:
        super().__init__(
            "fail-lm",
            "chat",
            temperature=0.0,
            max_tokens=1000,
            provider_options=LMProviderOptions(cache=False),
        )

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        raise RuntimeError("LM failed")


def make_mock_predictor(responses: list[dict]):

    class MockPredictor:
        def __init__(self):
            self.idx = 0

        def _next_response(self):
            result = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction.from_record(result)

        async def __call__(self, **kwargs: object):
            return self._next_response()

    return MockPredictor()


@contextmanager
def dummy_lm_context(responses: list[dict], make_run):
    from dspy.testing import DummyLM

    lm = DummyLM(responses)
    make_run(lm=lm)
    yield lm


def echo_tool(text: str = "") -> str:
    return f"Echo: {text}"


def add_tool(a: int = 0, b: int = 0) -> str:
    return str(a + b)


def multiply_tool(a: int = 0, b: int = 0) -> str:
    return str(a * b)


class _StubSerializable(SandboxSerializable):
    def __init__(self, data: str = "stub_data"):
        self.data = data

    @override
    def sandbox_setup(self) -> str:
        return "import json"

    @override
    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    @override
    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    @override
    def rlm_preview(self, max_chars: int = 500) -> str:
        return f"StubData({self.data})"


class _BinarySerializable(SandboxSerializable):
    @override
    def sandbox_setup(self) -> str:
        return ""

    @override
    def to_sandbox(self) -> bytes:
        return b"\xff\xfe\xfd"

    @override
    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    @override
    def rlm_preview(self, max_chars: int = 500) -> str:
        return "BinaryPayload"

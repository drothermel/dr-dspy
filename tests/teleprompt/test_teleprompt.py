from typing_extensions import override

from dspy.teleprompt.teleprompt import Teleprompter


class DummyTeleprompter(Teleprompter):
    def __init__(self, param1: int, param2: str):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    @override
    def compile(self, student, *, trainset, teacher=None, valset=None, **kwargs: object):
        return student


def test_get_params():
    teleprompter = DummyTeleprompter(param1=1, param2="test")
    params = teleprompter.get_params()
    assert params == {"param1": 1, "param2": "test"}

from dspy.teleprompt.protocol import Teleprompter


def test_teleprompter_protocol_isinstance():
    class Ok:
        async def compile(self, student, *, params, run):
            return student

    assert isinstance(Ok(), Teleprompter)
    assert not isinstance(object(), Teleprompter)

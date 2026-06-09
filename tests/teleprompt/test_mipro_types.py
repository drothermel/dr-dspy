import inspect

from dspy.predict.predict import Predict


def test_mipro_compile_matches_teleprompter_protocol():
    from dspy.teleprompt.mipro.optimizer import MIPROv2

    _ = Predict

    signature = inspect.signature(MIPROv2.compile)
    assert signature.parameters["student"].annotation == "Module"
    assert signature.return_annotation.__name__ == "CompileResult"
    assert signature.parameters["run"].annotation.__name__ == "RunContext"
    assert inspect.iscoroutinefunction(MIPROv2.compile)

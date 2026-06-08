import pytest

from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.propose.utils import get_dspy_source_code


def test_get_dspy_source_code_from_py_module():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict("question -> answer")

    source = get_dspy_source_code(MyProgram())
    assert "class MyProgram(Module):" in source
    assert "Predict" in source


def test_get_dspy_source_code_skips_builtin_predict():
    program = Predict("question -> answer")
    source = get_dspy_source_code(program)
    assert "class Predict" not in source


def test_get_dspy_source_code_raises_for_unsourcable_class():
    UnsourcableProgram = type("UnsourcableProgram", (Module,), {})

    with pytest.raises(OSError, match="source code not available"):
        get_dspy_source_code(UnsourcableProgram())

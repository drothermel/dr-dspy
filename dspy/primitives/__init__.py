import importlib

from dspy.primitives.base_module import BaseModule
from dspy.primitives.code_interpreter import (
    CodeInterpreter,
    CodeInterpreterError,
    FinalOutput,
)
from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions, Prediction
from dspy.primitives.record_store import RecordStore
from dspy.primitives.sandbox_protocol import SandboxSerializable, build_repl_variable

__all__ = [
    "BaseModule",
    "Module",
    "Example",
    "Prediction",
    "Completions",
    "RecordStore",
    "CodeInterpreter",
    "CodeInterpreterError",
    "PythonInterpreter",
    "FinalOutput",
    "SandboxSerializable",
    "build_repl_variable",
]

_LAZY_EXPORTS = {
    "Module": ("dspy.primitives.module", "Module"),
    "PythonInterpreter": ("dspy.primitives.python_interpreter.interpreter", "PythonInterpreter"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

"""
Local interpreter for secure Python code execution using Deno/Pyodide.

This package provides PythonInterpreter, which runs Python code in a sandboxed
WASM environment using Deno and Pyodide. It implements the Interpreter
protocol defined in code_interpreter.py.
"""

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter.interpreter import PythonInterpreter
from dspy.primitives.python_interpreter.serialize import LARGE_VAR_THRESHOLD

__all__ = ["PythonInterpreter", "FinalOutput", "CodeInterpreterError", "LARGE_VAR_THRESHOLD"]

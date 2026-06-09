from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from dspy.primitives import FinalOutput, Prediction
from dspy.primitives.python_interpreter import PythonInterpreter  # noqa: TC001 — runtime execute_generated_code param

_PYTHON_FENCE_LANGS = frozenset({"python", "py", "python3", "py3", ""})

_PYTHON_FENCE_PATTERN = re.compile(r"```python[ \n](.*?)[ \n]```?", re.DOTALL)
_LAST_LINE_ASSIGN_PATTERN = re.compile(r"^(\w+)\s*=")


def _generated_code_from_data(code_data: Prediction | Mapping[str, Any]) -> str:
    if isinstance(code_data, Mapping):
        return str(code_data.get("generated_code", ""))
    return str(getattr(code_data, "generated_code", ""))


def parse_generated_code(code_data: Prediction | Mapping[str, Any]) -> tuple[str, str | None]:
    code = _generated_code_from_data(code_data).split("---", 1)[0].split("\n\n\n", 1)[0]
    code_match = _PYTHON_FENCE_PATTERN.search(code)
    code_block = code_match.group(1) if code_match else code
    if not code_block:
        return code, "Error: Empty code after parsing."
    if "\n" not in code_block and code_block.count("=") > 1:
        return code, "Error: Code format is not correct."
    lines = code_block.split("\n")
    last_line_match = _LAST_LINE_ASSIGN_PATTERN.match(lines[-1].strip())
    if last_line_match and len(lines) > 1:
        code_block += "\n" + last_line_match.group(1)
    return code_block, None


def execute_generated_code(*, code: str, interpreter: PythonInterpreter) -> tuple[str | None, str | None]:
    if not code:
        return None, "Error: Empty code before execution."
    try:
        result = interpreter.execute(code)
        if isinstance(result, FinalOutput):
            result = result.output
        output = json.dumps(result)
        return output, None
    except Exception as err:
        return None, str(err)


def strip_python_fences(code: str) -> str:
    code = code.strip()
    if "```" not in code:
        return code
    lines = code.splitlines()
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()
    if "```" not in code:
        return code
    fence_start = code.find("```")
    lang_line, separator, remainder = code[fence_start + 3 :].partition("\n")
    if not separator:
        return code
    lang = (lang_line.strip().split(maxsplit=1)[0] if lang_line.strip() else "").lower()
    if lang not in _PYTHON_FENCE_LANGS:
        raise SyntaxError(f"Expected Python code but got ```{lang} fence. Write Python code, not {lang}.")
    block_end = remainder.find("```")
    if block_end == -1:
        return remainder.strip()
    return remainder[:block_end].strip()

from __future__ import annotations

from typing import Any

_PYTHON_FENCE_LANGS = {"python", "py", "python3", "py3", ""}


def _run_sub_lm_async(coro):
    import asyncio
    import contextvars

    ctx = contextvars.copy_context()

    def _run_in_context() -> Any:
        return ctx.run(asyncio.run, coro)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_in_context()
    raise RuntimeError("RLM sub-LM queries cannot run inside an active asyncio loop from sync REPL tools.")


def _strip_code_fences(code: str) -> str:
    """Extract Python code from markdown fences, or return as-is if no fences."""
    code = code.strip()
    if "```" not in code:
        return code

    # Strip outer decorative fence pairs (e.g. ```\n```python\n...\n```\n```)
    lines = code.splitlines()
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()
    if "```" not in code:
        return code

    # Find the first opening fence (skip any text before it)
    fence_start = code.find("```")
    lang_line, separator, remainder = code[fence_start + 3 :].partition("\n")
    if not separator:
        return code

    # Accept python-labeled fences or bare ``` fences; reject explicit non-Python tags
    lang = (lang_line.strip().split(maxsplit=1)[0] if lang_line.strip() else "").lower()
    if lang not in _PYTHON_FENCE_LANGS:
        raise SyntaxError(f"Expected Python code but got ```{lang} fence. Write Python code, not {lang}.")

    # Find closing fence
    block_end = remainder.find("```")
    if block_end == -1:
        return remainder.strip()

    return remainder[:block_end].strip()

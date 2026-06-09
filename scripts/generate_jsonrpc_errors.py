#!/usr/bin/env python3
"""Generate JSON-RPC application error maps from the canonical JSON file."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "dspy/primitives/jsonrpc_app_errors.json"
PYTHON_TARGET = ROOT / "dspy/primitives/python_interpreter/jsonrpc.py"
JS_TARGET = ROOT / "dspy/primitives/runner.js"

PYTHON_BEGIN = "# JSONRPC_APP_ERRORS_BEGIN"
PYTHON_END = "# JSONRPC_APP_ERRORS_END"
JS_BEGIN = "/* JSONRPC_APP_ERRORS_BEGIN */"
JS_END = "/* JSONRPC_APP_ERRORS_END */"


def load_errors() -> dict[str, int]:
    payload = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {CANONICAL_PATH}")
    return {str(key): int(value) for key, value in payload.items()}


def render_python(errors: dict[str, int]) -> str:
    lines = [f"{PYTHON_BEGIN}", "JSONRPC_APP_ERRORS = {"]
    for key, code in errors.items():
        lines.append(f'    "{key}": {code},')
    lines.append("}")
    lines.append(PYTHON_END)
    return "\n".join(lines) + "\n"


def render_js(errors: dict[str, int]) -> str:
    lines = [JS_BEGIN, "const JSONRPC_APP_ERRORS = {"]
    for key, code in errors.items():
        lines.append(f"  {key}: {code},")
    lines.append("};")
    lines.append(JS_END)
    return "\n".join(lines) + "\n"


def patch_block(path: Path, begin: str, end: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index(begin)
    stop = text.index(end, start) + len(end)
    updated = text[:start] + replacement.rstrip("\n") + text[stop:]
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    errors = load_errors()
    patch_block(PYTHON_TARGET, PYTHON_BEGIN, PYTHON_END, render_python(errors))
    patch_block(JS_TARGET, JS_BEGIN, JS_END, render_js(errors))


if __name__ == "__main__":
    main()

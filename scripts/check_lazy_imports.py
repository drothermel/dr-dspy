#!/usr/bin/env python3
"""Fail on function-body lazy imports of dspy submodules."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSPY_ROOT = ROOT / "dspy"

ALLOWLIST_SUFFIXES = (
    "adapters/base/call.py",
    "clients/lm/errors.py",
    "clients/lm_registry.py",
    "clients/finetune/registry.py",
    "adapters/types/tool/tool.py",
    "adapters/utils/multimodal.py",
    "compile/resolve.py",
    # optional-dependency entrypoints
    "clients/__init__.py",
    "teleprompt/gepa/gepa.py",
    # deferred Pydantic model_rebuild; breaks base_lm ↔ run_context import cycle
    "runtime/run_context_model.py",
    # lazy terminal pretty-print; keeps RunContext spine import light
    "runtime/call_log/inspect.py",
    # breaks openai_format.parse ↔ core.types.message_coercion import cycle
    "core/types/message_coercion.py",
    # breaks runtime.batch ↔ primitives ↔ history import cycle
    "runtime/batch.py",
    # breaks persistence.program ↔ primitives.module import cycle
    "persistence/program.py",
    # breaks persistence.embeddings ↔ retrievers.embeddings import cycle
    "persistence/embeddings.py",
    "evaluate/metrics.py",
)

OPTIONAL_DEP_MODULES = frozenset(
    {
        "gepa",
        "optuna",
        "litellm",
        "mcp",
        "langchain",
        "langchain_core",
    }
)


def _is_allowlisted(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if relative.endswith("lazy_import.py"):
        return True
    return any(relative.endswith(suffix) for suffix in ALLOWLIST_SUFFIXES)


def _lazy_imports_in_file(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if not self.stack or node.module is None:
                return
            root = node.module.split(".", 1)[0]
            if root in OPTIONAL_DEP_MODULES:
                return
            if not node.module.startswith("dspy"):
                return
            location = ".".join(self.stack)
            violations.append((node.lineno, f"{location}: from {node.module} import ..."))

    Visitor().visit(tree)
    return violations


def main() -> int:
    violations: list[str] = []
    for path in sorted(DSPY_ROOT.rglob("*.py")):
        if _is_allowlisted(path):
            continue
        for lineno, message in _lazy_imports_in_file(path):
            relative = path.relative_to(ROOT)
            violations.append(f"{relative}:{lineno}: {message}")

    if violations:
        print("Lazy dspy imports found inside functions/classes:", file=sys.stderr)
        for item in violations:
            print(f"  {item}", file=sys.stderr)
        return 1

    print("No disallowed lazy dspy imports found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

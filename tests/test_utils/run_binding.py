"""Static checks for explicit RunContext wiring in tests (pytest-time, not ruff)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunBindingViolation:
    path: Path
    func_name: str
    line: int

    def format(self) -> str:
        return (
            f"{self.path}:{self.line}: {self.func_name} uses run=run but `run` is not a "
            "parameter and is not assigned earlier in the function (e.g. run = make_run(lm=lm)). "
            "Add the `run` fixture, assign run from make_run, or pass make_run(...) inline."
        )


def _iter_test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                    funcs.append(item)
    return funcs


def _target_binds_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name) and target.id == name:
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(isinstance(elt, ast.Name) and elt.id == name for elt in target.elts)
    return False


def _run_assign_lines(func: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _target_binds_name(target, "run"):
                    lines.append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and _target_binds_name(node.target, "run"):
            lines.append(node.lineno)
    return lines


def find_run_binding_violations(source: str, path: Path | None = None) -> list[RunBindingViolation]:
    path = path or Path("<string>")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    violations: list[RunBindingViolation] = []
    for func in _iter_test_functions(tree):
        param_names = {arg.arg for arg in func.args.args}
        assign_lines = _run_assign_lines(func)
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "run" or not isinstance(kw.value, ast.Name) or kw.value.id != "run":
                    continue
                if "run" in param_names:
                    continue
                if any(line_no < node.lineno for line_no in assign_lines):
                    continue
                violations.append(RunBindingViolation(path=path, func_name=func.name, line=node.lineno))
    return violations


def collect_run_binding_violations(tests_root: Path | None = None) -> list[RunBindingViolation]:
    root = tests_root or Path(__file__).resolve().parents[1]
    violations: list[RunBindingViolation] = []
    for path in sorted(root.rglob("test_*.py")):
        if "reliability" in path.parts:
            continue
        source = path.read_text()
        violations.extend(find_run_binding_violations(source, path=path))
    return violations

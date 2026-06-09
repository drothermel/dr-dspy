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
    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            funcs.extend(
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_")
            )
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
            lines.extend(target.lineno for target in node.targets if _target_binds_name(target, "run"))
        elif isinstance(node, ast.AnnAssign) and _target_binds_name(node.target, "run"):
            lines.append(node.lineno)
    return lines


def _function_contains(func: ast.FunctionDef | ast.AsyncFunctionDef, node: ast.AST) -> bool:
    node_lineno = getattr(node, "lineno", None)
    if node_lineno is None:
        return False
    end = getattr(func, "end_lineno", func.lineno)
    return func.lineno <= node_lineno <= end and node is not func


def _enclosing_functions(node: ast.AST, root: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    scopes = [
        candidate
        for candidate in ast.walk(root)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and _function_contains(candidate, node)
    ]
    scopes.sort(key=lambda fn: fn.lineno, reverse=True)
    return scopes


def _param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {arg.arg for arg in func.args.args} | {arg.arg for arg in func.args.kwonlyargs}


def _run_is_bound(node: ast.Call, root: ast.AST) -> bool:
    for scope in _enclosing_functions(node, root):
        if "run" in _param_names(scope):
            return True
        if any(line_no < node.lineno for line_no in _run_assign_lines(scope)):
            return True
    return False


def find_run_binding_violations(source: str, path: Path | None = None) -> list[RunBindingViolation]:
    path = path or Path("<string>")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    violations: list[RunBindingViolation] = []
    for func in _iter_test_functions(tree):
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "run" or not isinstance(kw.value, ast.Name) or kw.value.id != "run":
                    continue
                if _run_is_bound(node, func):
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

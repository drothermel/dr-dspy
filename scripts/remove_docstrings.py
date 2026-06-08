#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from pathlib import Path


class DocstringRemover(ast.NodeTransformer):
    def _drop_docstring(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if not body:
            return [ast.Pass()]
        return body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._drop_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._drop_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._drop_docstring(node.body)
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._drop_docstring(node.body)
        node.body = self._drop_orphan_docstring_literals(node.body)
        return node

    def _drop_orphan_docstring_literals(self, body: list[ast.stmt]) -> list[ast.stmt]:
        cleaned: list[ast.stmt] = []
        for stmt in body:
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue
            cleaned.append(stmt)
        return cleaned


def strip_file(path: Path) -> bool:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"skip syntax error: {path}", file=sys.stderr)
        return False
    DocstringRemover().visit(tree)
    updated = ast.unparse(tree) + "\n"
    if updated != source:
        path.write_text(updated)
        return True
    return False


def main(argv: list[str]) -> int:
    roots = [Path(p) for p in argv[1:]] or [Path("dspy"), Path("tests")]
    changed = 0
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if strip_file(path):
                changed += 1
                print(path)
    print(f"updated {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

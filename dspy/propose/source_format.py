from __future__ import annotations

import ast
import inspect
import io
import textwrap
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["get_formatted_source"]


def get_formatted_source(
    obj: Callable[..., object] | type,
    *,
    docstring: str | None = None,
    include_docstring: bool = False,
    include_comments: bool = False,
) -> str:
    source = inspect.getsource(obj)
    if not include_docstring:
        source = _strip_docstrings(source)
    if not include_comments:
        source = _strip_comments(source)
    source = source.strip() + "\n"
    if docstring is not None:
        source = _inject_docstring(source, docstring)
    return source


def _strip_docstrings(source: str) -> str:
    tree = ast.parse(textwrap.dedent(source))
    DocstringStripper().visit(tree)
    return ast.unparse(tree)


class DocstringStripper(ast.NodeTransformer):
    def _drop_docstring(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
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
        return node


def _strip_comments(source: str) -> str:
    tokens: list[tokenize.TokenInfo] = []
    reader = io.StringIO(source).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.COMMENT:
            continue
        tokens.append(token)
    return tokenize.untokenize(tokens)


def _inject_docstring(source: str, docstring: str) -> str:
    tree = ast.parse(textwrap.dedent(source))
    stmt = ast.Expr(value=ast.Constant(value=docstring))
    if isinstance(tree, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        tree.body.insert(0, stmt)
    else:
        raise TypeError(f"Cannot inject docstring into source for {type(tree).__name__}.")
    return ast.unparse(tree).strip() + "\n"

import ast
import sys
from functools import reduce

_AST_TYPES = (ast.Name, ast.Attribute, ast.Subscript, ast.Call)
_STRING_TYPE = str


def get(obj, attr, **kwargs):
    for chunk in _parse(attr):
        try:
            obj = _lookup(obj=obj, node=chunk)
        except Exception:
            if "default" in kwargs:
                return kwargs["default"]
            raise
    return obj


def set(obj, attr, val) -> None:
    obj, attr_or_key, is_subscript = lookup(obj=obj, attr=attr)
    if is_subscript:
        obj[attr_or_key] = val
    else:
        setattr(obj, attr_or_key, val)


def delete(obj, attr) -> None:
    obj, attr_or_key, is_subscript = lookup(obj=obj, attr=attr)
    if is_subscript:
        del obj[attr_or_key]
    else:
        delattr(obj, attr_or_key)


def lookup(obj, attr):
    nodes = tuple(_parse(attr))
    if len(nodes) > 1:
        obj = reduce(_lookup, nodes[:-1], obj)
        node = nodes[-1]
    else:
        node = nodes[0]
    if isinstance(node, ast.Attribute):
        return (obj, node.attr, False)
    if isinstance(node, ast.Subscript):
        return (obj, _lookup_subscript_value(node.slice), True)
    if isinstance(node, ast.Name):
        return (obj, node.id, False)
    raise NotImplementedError("Node is not supported: %s" % node)


def _parse(attr):
    if not isinstance(attr, _STRING_TYPE):
        raise TypeError("Attribute name must be a string")
    nodes = ast.parse(attr).body
    if not nodes or not isinstance(nodes[0], ast.Expr):
        raise ValueError("Invalid expression: %s" % attr)
    return reversed([n for n in ast.walk(nodes[0]) if isinstance(n, _AST_TYPES)])


def _lookup_subscript_value(node):
    if isinstance(node, ast.Index):
        node = node.value
    if isinstance(node, ast.Constant):
        return node.value
    if sys.version_info < (3, 14):
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n
        if hasattr(ast, "Str") and isinstance(node, ast.Str):
            return node.s
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        if isinstance(operand, ast.Constant):
            return -operand.value
        if sys.version_info < (3, 14) and hasattr(ast, "Num") and isinstance(operand, ast.Num):
            return -operand.n
    raise NotImplementedError("Subscript node is not supported: %s" % ast.dump(node))


def _lookup(obj, node):
    if isinstance(node, ast.Attribute):
        return getattr(obj, node.attr)
    if isinstance(node, ast.Subscript):
        return obj[_lookup_subscript_value(node.slice)]
    if isinstance(node, ast.Name):
        return getattr(obj, node.id)
    if isinstance(node, ast.Call):
        raise ValueError("Function calls are not allowed.")
    raise NotImplementedError("Node is not supported: %s" % node)

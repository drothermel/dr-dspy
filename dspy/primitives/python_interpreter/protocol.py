"""Code interpreter protocol and sandbox type helpers.

Sandbox tool registration supports scalar types (``str``, ``int``, ``float``, ``bool``,
``None``) and homogeneous ``list`` / ``dict`` containers. Parameterized annotations
such as ``list[str]`` map to their container origin; unions with multiple non-``None``
members, ``Callable``, and custom classes are omitted from registration metadata.
"""

import types
from collections.abc import Callable, Mapping
from typing import Any, Protocol, Union, get_args, get_origin, runtime_checkable

from typing_extensions import override

SIMPLE_TYPES = (str, int, float, bool, list, dict, type(None))


def annotation_to_sandbox_type(annotation: Any) -> str | None:
    if annotation is type(None):
        return "None"
    if annotation in SIMPLE_TYPES:
        return annotation.__name__
    origin = get_origin(annotation)
    if origin is None:
        return None
    if origin in (types.UnionType, Union):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return annotation_to_sandbox_type(non_none[0])
        return None
    if origin in (list, dict):
        return origin.__name__
    return None


class CodeInterpreterError(RuntimeError):
    pass


class FinalOutput:
    def __init__(self, output: Any) -> None:
        self.output = output

    @override
    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalOutput):
            return NotImplemented
        return self.output == other.output


@runtime_checkable
class CodeInterpreter(Protocol):
    """Sandbox code interpreter contract.

    ``tools`` is read-only at the protocol boundary; implementations store mutable
    tool callables internally and expose a mapping view via the ``tools`` property.
    """

    @property
    def tools(self) -> Mapping[str, Callable[..., str]]: ...

    def start(self) -> None: ...

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any: ...

    def shutdown(self) -> None: ...

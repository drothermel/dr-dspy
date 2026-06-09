from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

from dspy.predict.protocol import Predictor

if TYPE_CHECKING:
    from collections.abc import Generator

    from dspy.primitives.module import Module

_MODULE_CLASS_ID = ("dspy.primitives.module", "Module")


def is_module_instance(item: object) -> bool:
    return any((cls.__module__, cls.__name__) == _MODULE_CLASS_ID for cls in type(item).__mro__)


def enqueue_graph_children(
    *,
    name: str,
    item: object,
    queue: deque[tuple[str, object]],
    seen: set[int],
) -> None:
    def enqueue(child_name: str, child: object) -> None:
        child_id = id(child)
        if child_id not in seen:
            seen.add(child_id)
            queue.append((child_name, child))

    if is_module_instance(item):
        if name == "self" or not getattr(item, "_compiled", False):
            for sub_name, sub_item in item.__dict__.items():
                enqueue(f"{name}.{sub_name}", sub_item)
        return
    if isinstance(item, (list, tuple)):
        for idx, sub_item in enumerate(item):
            enqueue(f"{name}[{idx}]", sub_item)
        return
    if isinstance(item, dict):
        for key, sub_item in item.items():
            enqueue(f"{name}[{key}]", sub_item)


def walk_module_graph(module: Module) -> Generator[tuple[str, object], None, None]:
    """Breadth-first traversal of module-owned object graph.

    Compiled subgraphs (``_compiled=True``) are opaque: their children are not
    enqueued. The root module is always expanded via the ``self`` entry.
    """
    queue: deque[tuple[str, object]] = deque([("self", module)])
    seen = {id(module)}
    while queue:
        name, item = queue.popleft()
        yield name, item
        enqueue_graph_children(name=name, item=item, queue=queue, seen=seen)


def named_predictors(module: Module) -> list[tuple[str, Predictor]]:
    """Return ``(name, Predictor)`` pairs. Skips predictors inside compiled subgraphs.

    When the same ``Predictor`` instance is reachable via multiple paths, only the
    first name encountered during breadth-first traversal is returned.
    """
    named: list[tuple[str, Predictor]] = []
    visited_predictors: set[int] = set()
    for name, item in walk_module_graph(module):
        if not isinstance(item, Predictor):
            continue
        predictor_id = id(item)
        if predictor_id in visited_predictors:
            continue
        visited_predictors.add(predictor_id)
        named.append((name, item))
    return named


def named_sub_modules(module: Module, type_: type | None = None) -> Generator[tuple[str, Module], None, None]:
    """Yield ``(name, module)`` pairs for modules of ``type_``.

    Compiled subgraphs are opaque by default (same policy as ``named_predictors``).
    """
    for name, item in walk_module_graph(module):
        matches = is_module_instance(item) if type_ is None else isinstance(item, type_)
        if matches:
            yield name, cast("Module", item)


def predictors(module: Module) -> list[Predictor]:
    return [predictor for _, predictor in named_predictors(module)]

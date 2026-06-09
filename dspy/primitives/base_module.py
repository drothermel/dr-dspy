from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
import orjson
from typing_extensions import Self

from dspy.persistence import get_dependency_versions, warn_dependency_version_drift
from dspy.persistence import save_program as persist_program
from dspy.predict.protocol import Predictor

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class BaseModule:
    def _enqueue_graph_children(
        self,
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

        if isinstance(item, BaseModule):
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

    def _walk_module_graph(self) -> Generator[tuple[str, object], None, None]:
        """Breadth-first traversal of module-owned object graph.

        Compiled subgraphs (``_compiled=True``) are opaque: their children are not
        enqueued. The root module is always expanded via the ``self`` entry.
        """
        queue: deque[tuple[str, object]] = deque([("self", self)])
        seen = {id(self)}
        while queue:
            name, item = queue.popleft()
            yield name, item
            self._enqueue_graph_children(name=name, item=item, queue=queue, seen=seen)

    def named_parameters(self) -> list[tuple[str, Predictor]]:
        """Return ``(name, Parameter)`` pairs. Skips parameters inside compiled subgraphs."""
        named_parameters: list[tuple[str, Predictor]] = []
        visited_parameters: set[int] = set()
        for name, item in self._walk_module_graph():
            if not isinstance(item, Predictor):
                continue
            param_id = id(item)
            if param_id in visited_parameters:
                continue
            visited_parameters.add(param_id)
            named_parameters.append((name, item))
        return named_parameters

    def named_sub_modules(self, type_: type | None = None) -> Generator[tuple[str, BaseModule], None, None]:
        """Yield ``(name, module)`` pairs for modules of ``type_``.

        Compiled subgraphs are opaque by default (same policy as ``named_parameters``).
        """
        if type_ is None:
            type_ = BaseModule
        for name, item in self._walk_module_graph():
            if isinstance(item, type_):
                yield name, cast("BaseModule", item)

    def parameters(self) -> list[Predictor]:
        return [param for _, param in self.named_parameters()]

    def deepcopy(self) -> Self:
        try:
            return copy.deepcopy(self)
        except Exception:
            logger.debug(
                "copy.deepcopy failed for %s; falling back to manual deepcopy",
                self.__class__.__name__,
                exc_info=True,
            )
        new_instance = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            if isinstance(value, BaseModule):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logger.warning(
                        "Failed to deep copy attribute '%s' of %s, falling back to shallow copy or reference copy.",
                        attr,
                        self.__class__.__name__,
                    )
                    try:
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        setattr(new_instance, attr, value)
        return new_instance

    def reset_copy(self) -> Self:
        new_instance = self.deepcopy()
        for param in new_instance.parameters():
            param.reset()  # ty: ignore[unresolved-attribute]
        return new_instance

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        return {name: param.dump_state(json_mode=json_mode) for name, param in self.named_parameters()}

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> Self:
        def _apply(module: BaseModule) -> None:
            for name, param in module.named_parameters():
                param.load_state(state[name], allow_unsafe_lm_state=allow_unsafe_lm_state)

        _apply(self.deepcopy())
        _apply(self)
        return self

    def save(
        self,
        path: str | Path,
        save_program: bool = False,
        modules_to_serialize: list[object] | None = None,
    ) -> None:
        metadata = {}
        metadata["dependency_versions"] = get_dependency_versions()
        path = Path(path)
        if save_program:
            persist_program(self, path, modules_to_serialize=modules_to_serialize)
            return
        if path.suffix == ".json":
            state = self.dump_state()
            state["metadata"] = metadata
            try:
                with path.open("wb") as f:
                    f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save state to {path} with error: {e}. Your DSPy program may contain non json-serializable objects, please consider saving the state in .pkl by using `path` ending with `.pkl`, or saving the whole program by setting `save_program=True`."
                )
        elif path.suffix == ".pkl":
            logger.warning(
                'Saving state to .pkl uses pickle serialization, which can execute arbitrary code when loaded. Prefer module.save("module.json") for safer state-only saves.'
            )
            state = self.dump_state(json_mode=False)
            state["metadata"] = metadata
            with path.open("wb") as f:
                cloudpickle.dump(state, f)
        else:
            raise ValueError(f"`path` must end with `.json` or `.pkl` when `save_program=False`, but received: {path}")

    def load(self, path: str | Path, allow_pickle: bool = False, allow_unsafe_lm_state: bool = False) -> None:
        path = Path(path)
        if path.suffix == ".json":
            with path.open("rb") as f:
                state = orjson.loads(f.read())
        elif path.suffix == ".pkl":
            if not allow_pickle:
                raise ValueError(
                    "Loading .pkl files can run arbitrary code, which may be dangerous. Prefer saving with .json files if possible. Set `allow_pickle=True` if you are sure about the source of the file and in a trusted environment."
                )
            with path.open("rb") as f:
                state = cloudpickle.load(f)
        else:
            raise ValueError(f"`path` must end with `.json` or `.pkl`, but received: {path}")
        dependency_versions = get_dependency_versions()
        saved_dependency_versions = state["metadata"]["dependency_versions"]
        warn_dependency_version_drift(
            saved=saved_dependency_versions,
            current=dependency_versions,
            log=logger,
        )
        self.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

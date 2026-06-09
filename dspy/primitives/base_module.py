import copy
import logging
from collections import deque
from collections.abc import Generator
from pathlib import Path
from typing import cast

import cloudpickle
import orjson

from dspy.persistence import get_dependency_versions
from dspy.persistence import save_program as persist_program
from dspy.predict.parameter import Parameter
from dspy.predict.protocol import Predictor

logger = logging.getLogger(__name__)


class BaseModule:
    def __init__(self) -> None:
        pass

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

    def named_parameters(self):
        """Yield ``(name, Parameter)`` pairs. Skips parameters inside compiled subgraphs."""
        named_parameters = []
        visited_parameters: set[int] = set()
        for name, item in self._walk_module_graph():
            if not isinstance(item, Parameter):
                continue
            param_id = id(item)
            if param_id in visited_parameters:
                continue
            visited_parameters.add(param_id)
            named_parameters.append((name, item))
        return named_parameters

    def named_sub_modules(self, type_=None) -> Generator[tuple[str, "BaseModule"], None, None]:
        """Yield ``(name, module)`` pairs for modules of ``type_``.

        Compiled subgraphs are opaque by default (same policy as ``named_parameters``).
        """
        if type_ is None:
            type_ = BaseModule
        for name, item in self._walk_module_graph():
            if isinstance(item, type_):
                yield name, cast("BaseModule", item)

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def deepcopy(self):
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
                    logging.warning(
                        f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, falling back to shallow copy or reference copy."
                    )
                    try:
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        setattr(new_instance, attr, value)
        return new_instance

    def reset_copy(self):
        new_instance = self.deepcopy()
        for param in new_instance.parameters():
            param.reset()
        return new_instance

    def dump_state(self, json_mode=True):
        return {name: param.dump_state(json_mode=json_mode) for name, param in self.named_parameters()}

    def load_state(self, state, *, allow_unsafe_lm_state=False) -> "BaseModule":
        def _apply(module) -> None:
            for name, param in module.named_parameters():
                if isinstance(param, Predictor):
                    param.load_state(state[name], allow_unsafe_lm_state=allow_unsafe_lm_state)
                else:
                    param.load_state(state[name])

        _apply(self.deepcopy())
        _apply(self)
        return self

    def save(self, path, save_program=False, modules_to_serialize=None) -> None:
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

    def load(self, path, allow_pickle=False, allow_unsafe_lm_state=False) -> None:
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
        for key, saved_version in saved_dependency_versions.items():
            if dependency_versions[key] != saved_version:
                logger.warning(
                    f"There is a mismatch of {key} version between saved model and current environment. You saved with `{key}=={saved_version}`, but now you have `{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade on the loaded model, please consider loading the model in the same environment as the saving environment."
                )
        self.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

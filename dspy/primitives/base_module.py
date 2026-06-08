import copy
import logging
from collections import deque
from collections.abc import Generator
from pathlib import Path

import cloudpickle
import orjson

from dspy.predict.parameter import Parameter
from dspy.predict.protocol import Predictor
from dspy.utils.saving import get_dependency_versions

logger = logging.getLogger(__name__)


class BaseModule:
    def __init__(self) -> None:
        pass

    def named_parameters(self):
        named_parameters = []
        visited_parameters = set()
        queue: deque[tuple[str, object]] = deque([("self", self)])
        seen = {id(self)}

        def enqueue(name: str, item: object) -> None:
            if id(item) not in seen:
                seen.add(id(item))
                queue.append((name, item))

        while queue:
            name, item = queue.popleft()
            if isinstance(item, Parameter):
                if id(item) not in visited_parameters:
                    visited_parameters.add(id(item))
                    named_parameters.append((name, item))
                continue
            if isinstance(item, BaseModule):
                if name == "self" or not getattr(item, "_compiled", False):
                    for sub_name, sub_item in item.__dict__.items():
                        enqueue(name=f"{name}.{sub_name}", item=sub_item)
                continue
            if isinstance(item, (list, tuple)):
                for idx, sub_item in enumerate(item):
                    enqueue(name=f"{name}[{idx}]", item=sub_item)
                continue
            if isinstance(item, dict):
                for key, sub_item in item.items():
                    enqueue(name=f"{name}[{key}]", item=sub_item)
        return named_parameters

    def named_sub_modules(self, type_=None, skip_compiled=False) -> Generator[tuple[str, "BaseModule"], None, None]:
        if type_ is None:
            type_ = BaseModule
        queue = deque([("self", self)])
        seen = {id(self)}

        def add_to_queue(name, item) -> None:
            if id(item) not in seen:
                seen.add(id(item))
                queue.append((name, item))

        while queue:
            name, item = queue.popleft()
            if isinstance(item, type_):
                yield (name, item)
            if isinstance(item, BaseModule):
                if skip_compiled and getattr(item, "_compiled", False):
                    continue
                for sub_name, sub_item in item.__dict__.items():
                    add_to_queue(name=f"{name}.{sub_name}", item=sub_item)
            elif isinstance(item, (list, tuple)):
                for i, sub_item in enumerate(item):
                    add_to_queue(name=f"{name}[{i}]", item=sub_item)
            elif isinstance(item, dict):
                for key, sub_item in item.items():
                    add_to_queue(name=f"{name}[{key}]", item=sub_item)

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def deepcopy(self):
        try:
            return copy.deepcopy(self)
        except Exception:
            pass
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
            if path.suffix:
                raise ValueError(
                    f"`path` must point to a directory without a suffix when `save_program=True`, but received: {path}"
                )
            if path.exists() and (not path.is_dir()):
                raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")
            if not path.exists():
                path.mkdir(parents=True)
            logger.warning(
                'Loading untrusted .pkl files can run arbitrary code, which may be dangerous. To avoid this, prefer saving using json format using module.save("module.json").'
            )
            try:
                modules_to_serialize = modules_to_serialize or []
                for module in modules_to_serialize:
                    cloudpickle.register_pickle_by_value(module)
                with (path / "program.pkl").open("wb") as f:
                    cloudpickle.dump(self, f)
            except Exception as e:
                raise RuntimeError(
                    f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, or consider using state-only saving by setting `save_program=False`."
                )
            with (path / "metadata.json").open("wb") as f:
                f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
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
                'Loading untrusted .pkl files can run arbitrary code, which may be dangerous. To avoid this, prefer saving using json format using module.save("module.json").'
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

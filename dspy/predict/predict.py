import logging
import random
import types
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.compile.resolve import resolve_lm_config
from dspy.core.types.config import _merge_lm_config, coerce_lm_config
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos
from dspy.task_spec.task_spec import TaskSpec
from dspy.utils.callback import BaseCallback
from dspy.utils.constants import IS_TYPE_UNDEFINED
from dspy.utils.transparency import reset_active_call_metadata, set_active_call_metadata

logger = logging.getLogger(__name__)
UNSAFE_LM_STATE_KEYS = {"api_base", "base_url", "model_list"}


def _sanitize_lm_state(lm_state: dict, allow_unsafe_lm_state: bool) -> dict:
    if allow_unsafe_lm_state:
        return lm_state
    unsafe_keys = sorted(UNSAFE_LM_STATE_KEYS.intersection(lm_state))
    if not unsafe_keys:
        return lm_state
    sanitized_lm_state = {k: v for k, v in lm_state.items() if k not in UNSAFE_LM_STATE_KEYS}
    logger.warning(
        "Ignoring unsafe LM config key(s) during state load: %s. Pass allow_unsafe_lm_state=True to preserve these keys for trusted files.",
        unsafe_keys,
    )
    return sanitized_lm_state


class Predict(Module, Parameter):
    def __init__(
        self,
        task_spec: TaskSpec,
        callbacks: list[BaseCallback] | None = None,
        run: RunContext | None = None,
        **config,
    ) -> None:
        if isinstance(task_spec, str):
            raise TypeError(
                "Predict requires a TaskSpec instance, not a string. Use a TaskSpec subclass or make_task_spec(...) to create one."
            )
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"Predict requires a TaskSpec instance, got {type(task_spec).__name__}.")
        super().__init__(callbacks=callbacks, run=run)
        self.stage = random.randbytes(8).hex()
        self.task_spec: TaskSpec = task_spec
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    @override
    def dump_state(self, json_mode=True):
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}
        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()
            for field in demo:
                demo[field] = serialize_object(demo[field])
            if json_mode and (not isinstance(demo, dict)):
                state["demos"].append(demo.to_dict())
            else:
                state["demos"].append(demo)
        state["task_spec"] = self.task_spec.to_dict()
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    @override
    def load_state(
        self, state: dict, *, allow_unsafe_lm_state: bool = False, custom_types: dict[str, type] | None = None
    ) -> "Predict":
        excluded_keys = ["task_spec", "extended_signature", "lm"]
        for name, value in state.items():
            if name not in excluded_keys:
                setattr(self, name, value)
        if "task_spec" not in state:
            if "signature" in state:
                raise ValueError(
                    "Saved state uses legacy 'signature' format. Re-save the program with the current DSPy version."
                )
            raise ValueError("Missing required 'task_spec' key in saved Predict state.")
        self.task_spec = TaskSpec.from_dict(state["task_spec"], custom_types=custom_types)
        sanitized_lm_state = _sanitize_lm_state(state["lm"], allow_unsafe_lm_state) if state["lm"] else None
        self.lm = (
            BaseLM.load_state(sanitized_lm_state, allow_custom_lm_class=allow_unsafe_lm_state)
            if sanitized_lm_state
            else None
        )
        return self

    def _get_positional_args_error_message(self) -> str:
        input_fields = list(self.task_spec.input_fields.keys())
        return f"Positional arguments are not allowed when calling `dspy.predict.predict.Predict`, must use keyword arguments that match your task spec input fields: '{', '.join(input_fields)}'. For example: `predict({input_fields[0]}=input_value, ...)`."

    @override
    def __call__(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())
        return super().__call__(**kwargs)

    @override
    async def acall(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())
        return await super().acall(**kwargs)

    def _forward_preprocess(self, **kwargs):
        assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."
        if "signature" in kwargs:
            raise TypeError("The 'signature' keyword argument is no longer supported. Pass 'task_spec' instead.")
        run = resolve_run(run=kwargs.pop("run", None), bound_run=self.run)
        task_spec = kwargs.pop("task_spec", self.task_spec)
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"Predict expected a TaskSpec, got {type(task_spec).__name__}.")
        input_fields = task_spec_input_field_infos(task_spec)
        demos = kwargs.pop("demos", self.demos)
        base_config = coerce_lm_config(self.config)
        override = kwargs.pop("config", {})
        if override:
            merged = _merge_lm_config(base_config, coerce_lm_config(override))
            config = merged if merged is not None else coerce_lm_config(override)
        else:
            config = base_config
        lm = kwargs.pop("lm", self.lm) or run.lm
        if lm is None:
            raise ValueError(
                "No LM is loaded. Pass run=RunContext.create(lm=LM(...), adapter=...) to the call, "
                "or bind run at Module/Predict construction."
            )
        if isinstance(lm, str):
            raise ValueError(
                f"LM must be an instance of `dspy.clients.base_lm.BaseLM`, not a string. "
                f"Create a RunContext with `RunContext.create(lm=LM('{lm}'), adapter=...)` instead."
            )
        if not isinstance(lm, BaseLM):
            raise ValueError(
                f"LM must be an instance of `dspy.clients.base_lm.BaseLM`, not {type(lm)}. Received `lm={lm}`."
            )
        if "prediction" in kwargs and (
            isinstance(kwargs["prediction"], dict)
            and kwargs["prediction"].get("type") == "content"
            and ("content" in kwargs["prediction"])
        ):
            extensions = dict(config.extensions)
            extensions["prediction"] = kwargs.pop("prediction")
            config = config.model_copy(update={"extensions": extensions})
        for k, field_info in input_fields.items():
            if k not in kwargs and field_info.default is not PydanticUndefined:
                kwargs[k] = field_info.default
        extra_fields = [k for k in kwargs if k not in input_fields]
        if extra_fields:
            logger.warning(
                "Input contains fields not in task spec. These fields will be ignored: %s. Expected fields: %s.",
                extra_fields,
                list(input_fields.keys()),
            )
        if run.telemetry.warn_on_type_mismatch:
            for field_name, field_info in input_fields.items():
                if field_name in kwargs:
                    value = kwargs[field_name]
                    expected_type = field_info.annotation
                    json_schema_extra = cast("dict[str, Any]", field_info.json_schema_extra or {})
                    if value is None or json_schema_extra.get(IS_TYPE_UNDEFINED, False):
                        continue
                    if not _is_value_compatible_with_type(value, cast("type", expected_type)):
                        logger.warning(
                            "Type mismatch for field '%s': expected %s based on given task spec, but the provided value is incompatible: %s.",
                            field_name,
                            _get_type_name(expected_type),
                            value,
                        )
        missing = [
            k
            for k, field_info in input_fields.items()
            if k not in kwargs and (not _annotation_allows_none(field_info.annotation))
        ]
        if missing:
            present = [k for k in input_fields if k in kwargs]
            logger.warning("Not all input fields were provided to module. Present: %s. Missing: %s.", present, missing)
        return (lm, config, task_spec, demos, kwargs, run)

    def _forward_postprocess(self, completions, task_spec, run, **kwargs):
        pred = Prediction.from_completions(completions, task_spec=task_spec)
        if kwargs.pop("_trace", True) and run.trace is not None and (run.telemetry.max_trace_size > 0):
            trace = run.trace
            if len(trace) >= run.telemetry.max_trace_size:
                trace.pop(0)
            trace.append((self, {**kwargs}, pred))
        return pred

    async def aforward(self, **kwargs):
        lm, config, task_spec, demos, kwargs, run = self._forward_preprocess(**kwargs)
        config, _provenance = resolve_lm_config(lm, config)
        metadata_token = set_active_call_metadata(module=type(self).__name__, phase="predict", lm_role="default")
        try:
            completions = await run.adapter.acall(
                lm=lm,
                config=config,
                task_spec=task_spec,
                demos=demos,
                inputs=kwargs,
                run=run,
            )
        finally:
            reset_active_call_metadata(metadata_token)
        return self._forward_postprocess(completions, task_spec, run, **kwargs)

    def update_config(self, **kwargs) -> None:
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.task_spec})"


def _get_type_name(type_annotation) -> str:
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)
    if origin is None:
        if hasattr(type_annotation, "__name__"):
            return type_annotation.__name__
        return str(type_annotation)
    if origin is Literal:
        literal_values = ", ".join(repr(arg) for arg in args)
        return f"Literal[{literal_values}]"
    if args:
        args_str = ", ".join("..." if arg is ... else _get_type_name(arg) for arg in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"
    return getattr(origin, "__name__", str(origin))


def _annotation_allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is None or annotation is type(None):
        return True
    if origin is Annotated:
        return bool(args) and _annotation_allows_none(args[0])
    if origin is Union or origin is types.UnionType:
        return any(_annotation_allows_none(arg) for arg in args)
    return False


def _is_value_compatible_with_type(value: Any, expected: type) -> bool:
    if expected is str and isinstance(value, list) and all(isinstance(item, str) for item in value):
        return True
    return _check_type(value, expected)


def _check_type(value: Any, expected: type) -> bool:
    if expected is Any:
        return True
    origin = get_origin(expected)
    args = get_args(expected)
    if origin is Union or origin is types.UnionType:
        return any(_check_type(value, arg) for arg in args)
    if origin is Literal:
        return value in args
    if origin is list:
        if not isinstance(value, list):
            return False
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True
    if origin is dict:
        if not isinstance(value, dict):
            return False
        if args:
            key_type, val_type = args
            return all((_check_type(k, key_type) and _check_type(v, val_type) for k, v in value.items()))
        return True
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                return all(_check_type(item, args[0]) for item in value)
            if len(value) != len(args):
                return False
            return all((_check_type(item, arg) for item, arg in zip(value, args, strict=False)))
        return True
    if origin is set or origin is frozenset:
        if not isinstance(value, origin):
            return False
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True
    if isinstance(expected, type):
        return isinstance(value, expected)
    return False


def serialize_object(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(serialize_object(item) for item in obj)
    if isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    return obj

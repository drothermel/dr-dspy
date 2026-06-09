from __future__ import annotations

import copy as copy_module
import datetime
import importlib
import inspect
import uuid
from typing import TYPE_CHECKING, Any

from dspy.clients.lm_registry import BUILTIN_LM_CLASS_PATH, get_lm_class
from dspy.clients.lm_strict import validate_lm_kwargs, validate_lm_state
from dspy.core.types import CallRecord, LMRequest, LMResponse, NativeAdaptationMode
from dspy.core.types.lm_provider import LMProviderOptions, merge_provider_options
from dspy.runtime.active_run import get_active_usage_tracker
from dspy.runtime.call_log.coordinator import append_disk_call, record_call
from dspy.runtime.callback import Callback, with_callbacks
from dspy.runtime.config import disk_call_log_enabled, memory_call_log_enabled

if TYPE_CHECKING:
    from dspy.runtime.run_context import RunContext
    from dspy.runtime.transparency.types import CompiledCall

LM_CLASS_STATE_KEY = "_dspy_lm_class"
PROVIDER_OPTIONS_STATE_KEY = "_dspy_provider_options"


def _import_lm_class(class_path: str) -> type:
    parts = class_path.split(".")
    last_error = None
    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                last_error = exc
                continue
            raise
        try:
            for attr in parts[split_index:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue
        if not isinstance(obj, type):
            raise TypeError(f"Serialized LM class `{class_path}` did not resolve to a class.")
        return obj
    raise ImportError(f"Could not import serialized LM class `{class_path}`.") from last_error


def _provider_options_from_kwargs(kwargs: dict[str, Any]) -> LMProviderOptions:
    provider_fields = set(LMProviderOptions.model_fields)
    data = {key: value for key, value in kwargs.items() if key in provider_fields}
    return LMProviderOptions(**data)


class BaseLM:
    def __init__(
        self,
        model: str,
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: list[Callback] | None = None,
        num_retries: int = 3,
        provider_options: LMProviderOptions | None = None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.callbacks = list(callbacks or [])
        self.num_retries = num_retries
        self.provider_options = provider_options or LMProviderOptions()
        self.kwargs = self._get_initial_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            provider_options=self.provider_options,
        )
        self.call_log: list[CallRecord] = []

    def _get_initial_kwargs(
        self,
        *,
        temperature: float | None,
        max_tokens: int | None,
        provider_options: LMProviderOptions,
    ) -> dict[str, Any]:
        kwargs = provider_options.to_kwargs()
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return validate_lm_kwargs(kwargs)

    @with_callbacks(kind="lm")
    async def __call__(
        self, request: LMRequest, *, run: RunContext, compiled: CompiledCall | None = None
    ) -> LMResponse:
        if not isinstance(request, LMRequest):
            raise TypeError(
                f"{type(self).__name__}.__call__ expects dspy.core.types.LMRequest, not {type(request).__name__}."
            )
        response = await self.aforward(request)
        if not isinstance(response, LMResponse):
            raise TypeError(
                f"{type(self).__name__}.aforward(request) must return dspy.core.types.LMResponse, but got {type(response).__name__}."
            )
        return self._finalize_lm_response(request=request, response=response, run=run, compiled=compiled)

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def reasoning_adaptation_mode(self) -> NativeAdaptationMode:
        return NativeAdaptationMode.ADAPT

    @property
    def citations_adaptation_mode(self) -> NativeAdaptationMode:
        return NativeAdaptationMode.ADAPT

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supported_params(self) -> set[str]:
        return set()

    @property
    def cache(self) -> bool | None:
        return self.provider_options.cache

    def _finalize_lm_response(
        self, request: LMRequest, response: LMResponse, *, run: RunContext, compiled: CompiledCall | None = None
    ) -> LMResponse:
        usage_tracker = get_active_usage_tracker(run)
        if usage_tracker:
            usage = response.usage_as_dict()
            if usage:
                usage_tracker.add_usage(lm=self.model, usage_entry=usage)
        record = None
        memory_enabled = memory_call_log_enabled(run.telemetry)
        disk_enabled = disk_call_log_enabled(run.telemetry)
        if memory_enabled or disk_enabled:
            record = CallRecord(
                request=request,
                response=response,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
        if memory_enabled and record is not None:
            record_call(entry=record, run=run, lm=self)
        if disk_enabled:
            append_disk_call(
                request=request,
                response=response,
                call_record=record,
                run=run,
                lm=self,
                compiled=compiled,
            )
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    def dump_state(self) -> dict[str, Any]:
        filtered_kwargs = {
            key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)
        }
        provider_data = self.provider_options.model_dump(exclude_none=True)
        provider_data.pop("api_key", None)
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "model": self.model,
            "model_type": self.model_type,
            "num_retries": getattr(self, "num_retries", 3),
            PROVIDER_OPTIONS_STATE_KEY: provider_data,
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> BaseLM:
        state = dict(state)
        class_path = state.pop(LM_CLASS_STATE_KEY, None)
        if cls is BaseLM:
            if class_path is None:
                return get_lm_class(BUILTIN_LM_CLASS_PATH).load_state(
                    state, allow_custom_lm_class=allow_custom_lm_class
                )
            if class_path != BUILTIN_LM_CLASS_PATH and (not allow_custom_lm_class):
                raise ValueError(
                    f"Refusing to import custom serialized LM class `{class_path}`. Pass allow_custom_lm_class=True when loading trusted files to enable custom LM classes."
                )
            lm_cls = _import_lm_class(class_path)
            if not issubclass(lm_cls, BaseLM):
                raise TypeError(
                    f"Serialized LM class `{class_path}` must be a subclass of dspy.clients.base_lm.BaseLM."
                )
            if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
                return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
            return lm_cls.load_state(state)
        state = validate_lm_state(state)
        model = state.pop("model")
        model_type = state.pop("model_type", "chat")
        num_retries = state.pop("num_retries", 3)
        provider_data = state.pop(PROVIDER_OPTIONS_STATE_KEY, None)
        temperature = state.pop("temperature", None)
        max_tokens = state.pop("max_tokens", None)
        remaining = dict(state)
        if provider_data:
            remaining = {**provider_data, **remaining}
        provider_options = _provider_options_from_kwargs(remaining)
        return cls(
            model=model,
            model_type=model_type,
            num_retries=num_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_options=provider_options,
        )

    def copy(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider_options: LMProviderOptions | None = None,
    ) -> BaseLM:
        new_instance = copy_module.copy(self)
        new_instance.call_log = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        if model is not None:
            new_instance.model = model
        merged_provider = merge_provider_options(self.provider_options, provider_options)
        new_instance.provider_options = merged_provider or self.provider_options
        new_kwargs = dict(getattr(self, "kwargs", {}) or {})
        if temperature is not None:
            new_kwargs["temperature"] = temperature
        if max_tokens is not None:
            new_kwargs["max_tokens"] = max_tokens
        new_kwargs.update(new_instance.provider_options.to_kwargs())
        new_instance.kwargs = validate_lm_kwargs(new_kwargs)
        return new_instance

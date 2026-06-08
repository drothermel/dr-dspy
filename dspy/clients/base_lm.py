import copy as copy_module
import datetime
import importlib
import inspect
import uuid
from typing import Any, TextIO

from dspy.core.types import LMHistoryEntry, LMRequest, LMResponse
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY: list[LMHistoryEntry] = []
LM_CLASS_STATE_KEY = "_dspy_lm_class"
_BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"


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


class BaseLM:
    """Base class for DSPy language models.

    The only supported runtime boundary is `LMRequest -> LMResponse`.
    Subclasses should implement `aforward(request: LMRequest) -> LMResponse`.
    """

    def __init__(
        self,
        model: str,
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = list(callbacks or [])
        self.num_retries = num_retries
        self.kwargs = self._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history: list[LMHistoryEntry] = []
        self._warned_zero_temp_rollout = False

    def _get_initial_kwargs(
        self, *, temperature: float | None, max_tokens: int | None, **kwargs: Any
    ) -> dict[str, Any]:
        return dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    @with_callbacks
    async def __call__(self, request: LMRequest) -> LMResponse:
        if not isinstance(request, LMRequest):
            raise TypeError(
                f"{type(self).__name__}.__call__ expects dspy.core.types.LMRequest, not {type(request).__name__}."
            )

        response = await self.aforward(request)
        if not isinstance(response, LMResponse):
            raise TypeError(
                f"{type(self).__name__}.aforward(request) must return dspy.core.types.LMResponse, "
                f"but got {type(response).__name__}."
            )
        return self._finalize_lm_response(request, response)

    async def acall(self, request: LMRequest) -> LMResponse:
        """Compatibility alias for ``__call__``; prefer ``await lm(request)``."""
        return await self.__call__(request)

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supported_params(self) -> set[str]:
        return set()

    def _finalize_lm_response(self, request: LMRequest, response: LMResponse) -> LMResponse:
        if not getattr(response, "cache_hit", False) and settings.usage_tracker:
            usage = response.usage_as_dict()
            if usage:
                settings.usage_tracker.add_usage(self.model, usage)

        if not settings.disable_history:
            entry = LMHistoryEntry(
                request=request,
                response=response,
                timestamp=datetime.datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
            self.update_history(entry)
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    def dump_state(self) -> dict[str, Any]:
        filtered_kwargs = {
            key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)
        }
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "model": self.model,
            "model_type": self.model_type,
            "cache": self.cache,
            "num_retries": getattr(self, "num_retries", 3),
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> "BaseLM":
        state = dict(state)
        class_path = state.pop(LM_CLASS_STATE_KEY, None)

        if cls is BaseLM:
            if class_path is None:
                from dspy.clients.lm import LM

                return LM(**state)

            if class_path != _BUILTIN_LM_CLASS_PATH and not allow_custom_lm_class:
                raise ValueError(
                    f"Refusing to import custom serialized LM class `{class_path}`. "
                    "Pass allow_unsafe_lm_state=True when loading trusted files to enable custom LM classes."
                )

            lm_cls = _import_lm_class(class_path)
            if not issubclass(lm_cls, BaseLM):
                raise TypeError(
                    f"Serialized LM class `{class_path}` must be a subclass of dspy.clients.base_lm.BaseLM."
                )
            if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
                return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
            return lm_cls.load_state(state)

        return cls(**state)

    def copy(self, **kwargs: Any):
        new_instance = copy_module.copy(self)
        new_instance.history = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if (key in new_instance.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        return new_instance

    def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry: LMHistoryEntry) -> None:
        if settings.disable_history:
            return

        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)
        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size == 0:
            return

        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)
        self.history.append(entry)

        caller_modules = settings.caller_modules or []
        for module in caller_modules:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)


def inspect_history(n: int = 1, file: "TextIO | None" = None) -> None:
    """Print the global history shared across all LMs."""
    pretty_print_history(GLOBAL_HISTORY, n, file=file)

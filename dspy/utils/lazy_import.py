import functools
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import sys
import threading
import types
from typing import Any

from typing_extensions import override


def _detect_dspy_dist() -> str:
    for dist in ("dspy", "dspy-ai"):
        try:
            importlib.metadata.version(dist)
            return dist
        except importlib.metadata.PackageNotFoundError:
            continue
    return "dspy"


_INSTALL_HINTS: dict[str, str] = {
    "optuna": "optuna",
    "mcp": "mcp",
    "langchain_core": "langchain",
    "weaviate": "weaviate",
    "anthropic": "anthropic",
    "gepa": "gepa",
    "numpy": "numpy",
    "litellm": "litellm",
}
_lazy_module_locks: dict[str, threading.RLock] = {}
_lazy_module_locks_lock = threading.Lock()


def _get_lazy_module_lock(module: str) -> threading.RLock:
    with _lazy_module_locks_lock:
        return _lazy_module_locks.setdefault(module, threading.RLock())


class _MissingModule(types.ModuleType):
    def __init__(self, module: str, message: str, frame_data: dict) -> None:
        super().__init__(module)
        self._message = message
        self._frame_data = frame_data

    @override
    def __getattr__(self, attr: str):
        fd = self._frame_data
        raise ImportError(
            f"{self._message}\n\nThis error is lazily reported, having originally occurred in\n  File {fd['filename']}, line {fd['lineno']}, in {fd['function']}\n\n----> {''.join(fd['code_context'] or '').strip()}"
        )


class _LazyModule(types.ModuleType):
    def __init__(self, module: str, spec: importlib.machinery.ModuleSpec, lock: threading.RLock) -> None:
        super().__init__(module)
        self.__spec__ = spec
        self.__loader__ = spec.loader
        self.__package__ = spec.parent
        if spec.submodule_search_locations is not None:
            self.__path__ = spec.submodule_search_locations
        self._dspy_lazy_spec = spec
        self._dspy_lazy_lock = lock

    def _load(self) -> types.ModuleType:
        module_name = self.__name__
        with self._dspy_lazy_lock:
            loaded = sys.modules.get(module_name)
            if loaded is not None and loaded is not self:
                return loaded
            spec = self._dspy_lazy_spec
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules[module_name] = self
                raise
            return sys.modules.get(module_name, module)

    @override
    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    @override
    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_dspy_lazy_") or attr in {"__spec__", "__loader__", "__package__", "__path__"}:
            super().__setattr__(attr, value)
        else:
            setattr(self._load(), attr, value)

    @override
    def __dir__(self) -> list[str]:
        return dir(self._load())


@functools.cache
def is_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, *, extra: str | None = None, feature: str | None = None) -> Any:
    lock = _get_lazy_module_lock(module)
    with lock:
        if module in sys.modules:
            return sys.modules[module]
        spec = importlib.util.find_spec(module)
    if spec is None or spec.loader is None:
        top = module.split(".", 1)[0]
        feat = feature or "this feature"
        ext = extra or _INSTALL_HINTS.get(top, top)
        dist = _detect_dspy_dist()
        message = f"{top} is required to use {feat}. Install with `pip install {dist}[{ext}]` or `pip install {top}`."
        parent = inspect.stack()[1]
        frame_data = {
            "filename": parent.filename,
            "lineno": parent.lineno,
            "function": parent.function,
            "code_context": parent.code_context,
        }
        del parent
        return _MissingModule(module, message, frame_data)
    with lock:
        if module in sys.modules:
            return sys.modules[module]
        mod = _LazyModule(module, spec, lock)
        sys.modules[module] = mod
        return mod

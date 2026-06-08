import asyncio
import contextvars
import copy
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cloudpickle
from typing_extensions import override

from dspy.dsp.utils.utils import dotdict

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = dotdict(
    lm=None,
    adapter=None,
    rm=None,
    branch_idx=0,
    trace=[],
    callbacks=[],
    async_max_workers=8,
    disable_history=False,
    track_usage=False,
    usage_tracker=None,
    caller_modules=None,
    provide_traceback=False,
    num_threads=8,
    max_errors=10,
    allow_tool_async_sync_conversion=False,
    max_history_size=10000,
    max_trace_size=10000,
    warn_on_type_mismatch=True,
    transparency="strict",
    run_log_enabled=True,
    run_log_dir=None,
)
main_thread_config = copy.deepcopy(DEFAULT_CONFIG)
config_owner_thread_id = None
config_owner_async_task = None
global_lock = threading.Lock()
thread_local_overrides = contextvars.ContextVar("context_overrides", default=dotdict())


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def lock(self):
        return global_lock

    def __getattr__(self, name):
        overrides = thread_local_overrides.get()
        if name in overrides:
            return overrides[name]
        if name in main_thread_config:
            return main_thread_config[name]
        raise AttributeError(f"'Settings' object has no attribute '{name}'")

    @override
    def __setattr__(self, name, value) -> None:
        if name in ("_instance",):
            super().__setattr__(name, value)
        else:
            self.configure(**{name: value})

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value) -> None:
        self.__setattr__(key, value)

    def __contains__(self, key) -> bool:
        overrides = thread_local_overrides.get()
        return key in overrides or key in main_thread_config

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def copy(self):
        overrides = thread_local_overrides.get()
        return dotdict({**main_thread_config, **overrides})

    @property
    def config(self):
        return self.copy()

    def _ensure_configure_allowed(self) -> None:
        global main_thread_config, config_owner_thread_id, config_owner_async_task
        current_thread_id = threading.get_ident()
        if config_owner_thread_id is None:
            config_owner_thread_id = current_thread_id
        if config_owner_thread_id != current_thread_id:
            raise RuntimeError("settings can only be changed by the thread that initially configured it.")
        is_async_task = False
        try:
            if asyncio.current_task() is not None:
                is_async_task = True
        except RuntimeError:
            is_async_task = False
        if not is_async_task:
            return
        if config_owner_async_task is None:
            config_owner_async_task = asyncio.current_task()
            return
        if config_owner_async_task != asyncio.current_task():
            raise RuntimeError(
                "settings.configure(...) can only be called from the same async task that called it first. Please use `settings.context(...)` in other async tasks instead."
            )

    def configure(self, **kwargs) -> None:
        self._ensure_configure_allowed()
        for k, v in kwargs.items():
            main_thread_config[k] = v
        if kwargs:
            from dspy.utils.run_log import init_run_session

            snapshot = {
                key: value
                for key, value in self.copy().items()
                if key not in {"callbacks", "trace", "usage_tracker", "caller_modules"}
            }
            init_run_session(
                run_log_enabled=main_thread_config.get("run_log_enabled", True),
                run_log_dir=main_thread_config.get("run_log_dir"),
                settings_snapshot=snapshot,
            )

    @contextmanager
    def context(self, **kwargs):
        original_overrides = thread_local_overrides.get().copy()
        new_overrides = dotdict({**main_thread_config, **original_overrides, **kwargs})
        token = thread_local_overrides.set(new_overrides)
        try:
            yield
        finally:
            thread_local_overrides.reset(token)

    @override
    def __repr__(self) -> str:
        overrides = thread_local_overrides.get()
        combined_config = {**main_thread_config, **overrides}
        return repr(combined_config)

    def save(
        self, path: str, modules_to_serialize: list[str] | None = None, exclude_keys: list[str] | None = None
    ) -> None:
        logger.warning(
            "`dspy.settings` are serialized using cloudpickle. Because cloudpickle allows for the execution of arbitrary code during deserialization, you should only load files from verified sources within a trusted environment."
        )
        try:
            modules_to_serialize = modules_to_serialize or []
            for module in modules_to_serialize:
                cloudpickle.register_pickle_by_value(module)
            exclude_keys = exclude_keys or []
            data = {key: value for key, value in self.config.items() if key not in exclude_keys}
            with Path(path).open("wb") as f:
                cloudpickle.dump(data, f)
        except Exception as e:
            raise RuntimeError(
                f"Saving failed with error: {e}. Please remove the non-picklable attributes from the values in the `dspy.settings`."
            )

    @classmethod
    def load(cls, path: str, allow_pickle: bool = False) -> dict[str, Any]:
        if not allow_pickle:
            raise ValueError(
                "Loading .pkl files can run arbitrary code, which may be dangerous. Set `allow_pickle=True` if you trust the source of the file."
            )
        with Path(path).open("rb") as f:
            return cloudpickle.load(f)


settings = Settings()

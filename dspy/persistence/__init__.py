"""Program persistence helpers for saving and loading DSPy modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import orjson

from dspy.__metadata__ import __version__

if TYPE_CHECKING:
    from dspy.primitives import BaseModule

logger = logging.getLogger(__name__)

__all__ = ["get_dependency_versions", "load", "save_program", "warn_dependency_version_drift"]


def get_dependency_versions() -> dict[str, str]:
    cloudpickle_version = ".".join(cloudpickle.__version__.split(".")[:2])
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": __version__,
        "cloudpickle": cloudpickle_version,
    }


def warn_dependency_version_drift(
    *,
    saved: dict[str, str],
    current: dict[str, str],
    log: logging.Logger,
) -> None:
    for key in sorted(set(saved) | set(current)):
        saved_version = saved.get(key)
        current_version = current.get(key)
        if saved_version is None and current_version is not None:
            log.warning(
                f"Saved metadata does not include `{key}` version tracking; current environment has `{key}=={current_version}`. "
                "This file may predate dependency version checks — consider re-saving in the current environment."
            )
            continue
        if current_version is None and saved_version is not None:
            log.warning(
                f"Current environment does not track `{key}` version, but saved metadata has `{key}=={saved_version}`."
            )
            continue
        if saved_version != current_version:
            log.warning(
                f"There is a mismatch of {key} version between saved model and current environment. You saved with `{key}=={saved_version}`, but now you have `{key}=={current_version}`. This might cause errors or performance downgrade on the loaded model, please consider loading the model in the same environment as the saving environment."
            )


def save_program(
    module: BaseModule,
    path: str | Path,
    *,
    modules_to_serialize: list[object] | None = None,
) -> None:
    metadata = {"dependency_versions": get_dependency_versions()}
    save_path = Path(path)
    if save_path.suffix:
        raise ValueError(
            f"`path` must point to a directory without a suffix when saving a program, but received: {save_path}"
        )
    if save_path.exists() and (not save_path.is_dir()):
        raise NotADirectoryError(f"The path '{save_path}' exists but is not a directory.")
    if not save_path.exists():
        save_path.mkdir(parents=True)
    logger.warning(
        'Saving the full program to program.pkl uses pickle serialization, which can execute arbitrary code when loaded. Prefer module.save("module.json") for state-only saves.'
    )
    try:
        modules_to_serialize = modules_to_serialize or []
        for extra_module in modules_to_serialize:
            cloudpickle.register_pickle_by_value(extra_module)
        with (save_path / "program.pkl").open("wb") as f:
            cloudpickle.dump(module, f)
    except Exception as e:
        raise RuntimeError(
            f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, or consider using state-only saving by setting `save_program=False`."
        ) from e
    with (save_path / "metadata.json").open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))


def load(path: str | Path, allow_pickle: bool = False) -> Any:
    if not allow_pickle:
        raise ValueError(
            "Loading with pickle is not allowed. Please set `allow_pickle=True` if you are sure you trust the source of the model."
        )
    save_path = Path(path)
    if not save_path.exists():
        raise FileNotFoundError(f"The path '{save_path}' does not exist.")
    with (save_path / "metadata.json").open() as f:
        metadata = orjson.loads(f.read())
    dependency_versions = get_dependency_versions()
    saved_dependency_versions = metadata["dependency_versions"]
    warn_dependency_version_drift(
        saved=saved_dependency_versions,
        current=dependency_versions,
        log=logger,
    )
    with (save_path / "program.pkl").open("rb") as f:
        loaded_program: Any = cloudpickle.load(f)
    return loaded_program

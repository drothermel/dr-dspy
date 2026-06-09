"""Whole-program cloudpickle persistence for DSPy modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import orjson

from dspy.persistence.metadata import (
    build_metadata,
    get_dependency_versions,
    logger,
    parse_metadata,
    warn_dependency_version_drift,
    warn_pickle_save,
)

if TYPE_CHECKING:
    from dspy.primitives.module import Module

PROGRAM_PICKLE_FILENAME = "program.pkl"
PROGRAM_METADATA_FILENAME = "metadata.json"


def save_program(
    module: Module,
    path: str | Path,
    *,
    modules_to_serialize: list[object] | None = None,
) -> None:
    metadata = build_metadata()
    save_path = Path(path)
    if save_path.suffix:
        raise ValueError(
            f"`path` must point to a directory without a suffix when saving a program, but received: {save_path}"
        )
    if save_path.exists() and (not save_path.is_dir()):
        raise NotADirectoryError(f"The path '{save_path}' exists but is not a directory.")
    if not save_path.exists():
        save_path.mkdir(parents=True)
    warn_pickle_save(target="program")
    try:
        modules_to_serialize = modules_to_serialize or []
        for extra_module in modules_to_serialize:
            cloudpickle.register_pickle_by_value(extra_module)
        with (save_path / PROGRAM_PICKLE_FILENAME).open("wb") as f:
            cloudpickle.dump(module, f)
    except Exception as e:
        raise RuntimeError(
            f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, or consider using state-only saving by setting `save_program=False`."
        ) from e
    with (save_path / PROGRAM_METADATA_FILENAME).open("wb") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))


def load_program(path: str | Path, allow_pickle: bool = False) -> Module:
    if not allow_pickle:
        raise ValueError(
            "Loading with pickle is not allowed. Please set `allow_pickle=True` if you are sure you trust the source of the model."
        )
    save_path = Path(path)
    if not save_path.exists():
        raise FileNotFoundError(f"The path '{save_path}' does not exist.")
    with (save_path / PROGRAM_METADATA_FILENAME).open() as f:
        raw_metadata = orjson.loads(f.read())
    parsed = parse_metadata(raw_metadata)
    warn_dependency_version_drift(
        saved=parsed.dependency_versions,
        current=get_dependency_versions(),
        log=logger,
    )
    with (save_path / PROGRAM_PICKLE_FILENAME).open("rb") as f:
        loaded_program = cloudpickle.load(f)
    from dspy.primitives.module import Module

    if not isinstance(loaded_program, Module):
        raise TypeError(
            f"Expected a Module from {save_path / PROGRAM_PICKLE_FILENAME}, got {type(loaded_program).__name__}."
        )
    return loaded_program

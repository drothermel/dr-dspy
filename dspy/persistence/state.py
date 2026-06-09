"""Module state dump/apply and .json/.pkl file persistence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import orjson

from dspy.persistence.metadata import (
    METADATA_KEY,
    build_metadata,
    get_dependency_versions,
    logger,
    parse_metadata,
    warn_dependency_version_drift,
    warn_pickle_save,
)

if TYPE_CHECKING:
    from dspy.primitives.module import Module

STATE_PKL_LOAD_DENIED_MESSAGE = (
    "Loading .pkl files can run arbitrary code, which may be dangerous. Prefer saving with .json files if possible. "
    "Set `allow_pickle=True` if you are sure about the source of the file and in a trusted environment."
)


def dump_module_state(module: Module, *, json_mode: bool = True) -> dict[str, Any]:
    return {name: predictor.dump_state(json_mode=json_mode) for name, predictor in module.named_predictors()}


def _predictor_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != METADATA_KEY}


def apply_module_state(
    module: Module,
    state: dict[str, Any],
    *,
    allow_unsafe_lm_state: bool = False,
    custom_types: dict[str, type] | None = None,
) -> Module:
    predictor_state = _predictor_state(state)

    def _apply(target: Module) -> None:
        for name, predictor in target.named_predictors():
            predictor.load_state(
                predictor_state[name],
                allow_unsafe_lm_state=allow_unsafe_lm_state,
                custom_types=custom_types,
            )

    # Validate keys on a throwaway deep copy first so a missing predictor state
    # raises before mutating ``module`` (all-or-nothing load semantics).
    _apply(module.deepcopy())
    _apply(module)
    return module


def save_state(module: Module, path: str | Path) -> None:
    path = Path(path)
    metadata = build_metadata()
    if path.suffix == ".json":
        state = module.dump_state()
        state[METADATA_KEY] = metadata
        try:
            with path.open("wb") as f:
                f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save state to {path} with error: {e}. Your DSPy program may contain non json-serializable objects, please consider saving the state in .pkl by using `path` ending with `.pkl`, or saving the whole program by setting `save_program=True`."
            ) from e
    elif path.suffix == ".pkl":
        warn_pickle_save(target="state")
        state = module.dump_state(json_mode=False)
        state[METADATA_KEY] = metadata
        with path.open("wb") as f:
            cloudpickle.dump(state, f)
    else:
        raise ValueError(f"`path` must end with `.json` or `.pkl` when `save_program=False`, but received: {path}")


def load_state(
    module: Module,
    path: str | Path,
    *,
    allow_pickle: bool = False,
    allow_unsafe_lm_state: bool = False,
    custom_types: dict[str, type] | None = None,
) -> Module:
    path = Path(path)
    if path.suffix == ".json":
        with path.open("rb") as f:
            raw_state = orjson.loads(f.read())
    elif path.suffix == ".pkl":
        if not allow_pickle:
            raise ValueError(STATE_PKL_LOAD_DENIED_MESSAGE)
        with path.open("rb") as f:
            raw_state = cloudpickle.load(f)
    else:
        raise ValueError(f"`path` must end with `.json` or `.pkl`, but received: {path}")

    if not isinstance(raw_state, dict):
        raise TypeError(f"Expected a state dict from {path}, got {type(raw_state).__name__}.")
    if METADATA_KEY not in raw_state:
        raise KeyError(f"State file {path} is missing required `{METADATA_KEY}`.")

    parsed_metadata = parse_metadata(raw_state[METADATA_KEY])
    warn_dependency_version_drift(
        saved=parsed_metadata.dependency_versions,
        current=get_dependency_versions(),
        log=logger,
    )
    predictor_state = _predictor_state(raw_state)
    module.load_state(
        predictor_state,
        allow_unsafe_lm_state=allow_unsafe_lm_state,
        custom_types=custom_types,
    )
    return module

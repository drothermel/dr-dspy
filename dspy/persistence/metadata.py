"""Dependency metadata and drift warnings for persisted DSPy artifacts."""

from __future__ import annotations

import logging
import sys
from typing import Literal

import cloudpickle
from pydantic import BaseModel, ConfigDict

from dspy.__metadata__ import __version__

METADATA_KEY = "metadata"
DEPENDENCY_VERSIONS_KEY = "dependency_versions"

PICKLE_SAVE_WARNING = 'Pickle serialization can execute arbitrary code when loaded. Prefer module.save("module.json") for state-only saves.'

logger = logging.getLogger("dspy.persistence")


class PersistenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dependency_versions: dict[str, str]


def get_dependency_versions() -> dict[str, str]:
    cloudpickle_version = ".".join(cloudpickle.__version__.split(".")[:2])
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": __version__,
        "cloudpickle": cloudpickle_version,
    }


def build_metadata() -> dict[str, dict[str, str]]:
    return {DEPENDENCY_VERSIONS_KEY: get_dependency_versions()}


def parse_metadata(raw: dict[str, object]) -> PersistenceMetadata:
    return PersistenceMetadata.model_validate(raw)


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


def warn_pickle_save(*, target: Literal["program", "state"]) -> None:
    if target == "program":
        logger.warning(
            f"Saving the full program to program.pkl uses pickle serialization, which can execute arbitrary code when loaded. {PICKLE_SAVE_WARNING}"
        )
    else:
        logger.warning(
            f"Saving state to .pkl uses pickle serialization, which can execute arbitrary code when loaded. {PICKLE_SAVE_WARNING}"
        )

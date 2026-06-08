import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import orjson

from dspy.__metadata__ import __version__

if TYPE_CHECKING:
    from dspy.primitives.module import Module
logger = logging.getLogger(__name__)


def get_dependency_versions() -> dict[str, str]:
    cloudpickle_version = ".".join(cloudpickle.__version__.split(".")[:2])
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": __version__,
        "cloudpickle": cloudpickle_version,
    }


def load(path: str, allow_pickle: bool = False) -> "Module":
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
    for key, saved_version in saved_dependency_versions.items():
        if dependency_versions[key] != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. You saved with `{key}=={saved_version}`, but now you have `{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade on the loaded model, please consider loading the model in the same environment as the saving environment."
            )
    with (save_path / "program.pkl").open("rb") as f:
        loaded_program: Any = cloudpickle.load(f)
    return loaded_program

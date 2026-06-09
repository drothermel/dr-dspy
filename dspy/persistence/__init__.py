"""Persistence helpers for saving and loading DSPy modules and retriever artifacts."""

from dspy.persistence.metadata import (
    DEPENDENCY_VERSIONS_KEY,
    METADATA_KEY,
    PersistenceMetadata,
    build_metadata,
    get_dependency_versions,
    logger,
    warn_dependency_version_drift,
    warn_pickle_save,
)
from dspy.persistence.program import load_program, save_program

__all__ = [
    "DEPENDENCY_VERSIONS_KEY",
    "METADATA_KEY",
    "PersistenceMetadata",
    "build_metadata",
    "get_dependency_versions",
    "load_program",
    "logger",
    "save_program",
    "warn_dependency_version_drift",
    "warn_pickle_save",
]

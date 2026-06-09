import sys
from unittest.mock import MagicMock

import pytest

from dspy.integrations.finetune.databricks import _create_directory_in_databricks_unity_catalog

VALID_VOLUME_PATH = "/Volumes/main/schema/volume/subdir"


class _NotFoundError(Exception):
    pass


class _ResourceDoesNotExistError(_NotFoundError):
    pass


@pytest.fixture
def databricks_error_types(monkeypatch):
    platform_mod = MagicMock()
    platform_mod.NotFound = _NotFoundError
    platform_mod.ResourceDoesNotExist = _ResourceDoesNotExistError
    errors_mod = MagicMock()
    errors_mod.platform = platform_mod
    sdk_mod = MagicMock()
    sdk_mod.errors = errors_mod
    databricks_mod = MagicMock()
    databricks_mod.sdk = sdk_mod
    monkeypatch.setitem(sys.modules, "databricks", databricks_mod)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_mod)
    monkeypatch.setitem(sys.modules, "databricks.sdk.errors", errors_mod)
    monkeypatch.setitem(sys.modules, "databricks.sdk.errors.platform", platform_mod)
    return _NotFoundError, _ResourceDoesNotExistError


def test_missing_volume_raises_value_error(databricks_error_types):
    not_found, _ = databricks_error_types
    workspace = MagicMock()
    workspace.volumes.read.side_effect = not_found("missing volume")
    with pytest.raises(ValueError, match="Databricks Unity Catalog volume does not exist"):
        _create_directory_in_databricks_unity_catalog(workspace, VALID_VOLUME_PATH)


def test_volume_read_unexpected_error_propagates(databricks_error_types):
    workspace = MagicMock()
    workspace.volumes.read.side_effect = RuntimeError("unexpected")
    with pytest.raises(RuntimeError, match="unexpected"):
        _create_directory_in_databricks_unity_catalog(workspace, VALID_VOLUME_PATH)


def test_missing_directory_is_created(databricks_error_types):
    _, resource_does_not_exist = databricks_error_types
    workspace = MagicMock()
    workspace.files.get_directory_metadata.side_effect = resource_does_not_exist("missing directory")
    _create_directory_in_databricks_unity_catalog(workspace, VALID_VOLUME_PATH)
    workspace.files.create_directory.assert_called_once_with(VALID_VOLUME_PATH)


def test_invalid_path_raises_before_volume_lookup():
    workspace = MagicMock()
    with pytest.raises(ValueError, match="Databricks Unity Catalog path must be in the format"):
        _create_directory_in_databricks_unity_catalog(workspace, "/bad/path")

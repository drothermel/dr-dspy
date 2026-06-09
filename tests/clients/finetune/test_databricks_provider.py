import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from dspy.clients.finetune.utils import TrainDataFormat
from dspy.integrations.finetune.databricks import (
    DatabricksProvider,
    TrainingJobDatabricks,
    _create_directory_in_databricks_unity_catalog,
)

VALID_VOLUME_PATH = "/Volumes/main/schema/volume/subdir"
TEST_AUTH = "not-a-secret"
CHAT_EXAMPLE = [
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
]


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


def test_finetune_requires_train_data_path():
    job = TrainingJobDatabricks()
    with pytest.raises(ValueError, match="train_data_path"):
        DatabricksProvider.finetune(
            job=job,
            model="meta-llama/Llama-3.2-1B",
            train_data=CHAT_EXAMPLE,
            train_data_format=TrainDataFormat.CHAT,
            train_kwargs={"register_to": "main.schema.model"},
        )


def test_finetune_requires_register_to(monkeypatch):
    job = TrainingJobDatabricks()
    monkeypatch.setattr(DatabricksProvider, "upload_data", lambda _data, path: path)
    fm = MagicMock()

    def fake_import_optional(top_level, **kwargs):
        if top_level == "databricks.model_training.foundation_model":
            return fm
        raise ImportError(top_level)

    monkeypatch.setattr("dspy.integrations.finetune.databricks.import_optional", fake_import_optional)
    with pytest.raises(ValueError, match="register_to"):
        DatabricksProvider.finetune(
            job=job,
            model="meta-llama/Llama-3.2-1B",
            train_data=CHAT_EXAMPLE,
            train_data_format=TrainDataFormat.CHAT,
            train_kwargs={"train_data_path": VALID_VOLUME_PATH},
        )


def test_finetune_rejects_invalid_train_data_format_string(monkeypatch):
    job = TrainingJobDatabricks()
    monkeypatch.setattr(DatabricksProvider, "upload_data", lambda _data, path: path)
    with pytest.raises(ValueError, match=r"chat.*completion"):
        DatabricksProvider.finetune(
            job=job,
            model="meta-llama/Llama-3.2-1B",
            train_data=CHAT_EXAMPLE,
            train_data_format="invalid",
            train_kwargs={
                "train_data_path": VALID_VOLUME_PATH,
                "register_to": "main.schema.model",
                "skip_deploy": True,
            },
        )


def test_finetune_normalizes_chat_string_format(monkeypatch):
    job = TrainingJobDatabricks()
    uploaded_path = f"{VALID_VOLUME_PATH}/finetuning.jsonl"
    monkeypatch.setattr(DatabricksProvider, "upload_data", lambda _data, _path: uploaded_path)

    class FakeStatus:
        display_name = "Completed"

    class FakeRun:
        status = FakeStatus()

    fm = MagicMock()
    fm.create.return_value = FakeRun()
    fm.get.return_value = FakeRun()

    def fake_import_optional(top_level, **kwargs):
        if top_level == "databricks.model_training.foundation_model":
            return fm
        raise ImportError(top_level)

    monkeypatch.setattr("dspy.integrations.finetune.databricks.import_optional", fake_import_optional)

    result = DatabricksProvider.finetune(
        job=job,
        model="meta-llama/Llama-3.2-1B",
        train_data=CHAT_EXAMPLE,
        train_data_format="chat",
        train_kwargs={
            "train_data_path": VALID_VOLUME_PATH,
            "register_to": "main.schema.model",
            "skip_deploy": True,
        },
    )
    assert result == ""
    fm.create.assert_called_once()
    assert fm.create.call_args.kwargs["train_data_path"] == uploaded_path


def test_upload_data_writes_jsonl_and_uploads(monkeypatch, databricks_error_types):
    workspace = MagicMock()
    monkeypatch.setattr(
        "dspy.integrations.finetune.databricks._get_workspace_client",
        lambda: workspace,
    )

    uploaded_paths: list[str] = []

    def capture_upload(target_path, _file_obj, overwrite=False):
        uploaded_paths.append(target_path)

    workspace.files.upload.side_effect = capture_upload

    local_file = f"{VALID_VOLUME_PATH}/finetuning.jsonl"
    with (
        patch("dspy.integrations.finetune.databricks._save_data_to_local_file", return_value=local_file),
        patch("builtins.open", mock_open(read_data=b"data")),
    ):
        result = DatabricksProvider.upload_data(CHAT_EXAMPLE, VALID_VOLUME_PATH)

    assert result.endswith("finetuning.jsonl")
    assert uploaded_paths == [f"{VALID_VOLUME_PATH}/finetuning.jsonl"]
    workspace.files.upload.assert_called_once()


def test_deploy_finetuned_model_creates_serving_endpoint(monkeypatch):
    workspace = MagicMock()
    model_version = MagicMock()
    model_version.version = "3"
    workspace.model_versions.list.return_value = iter([model_version])
    workspace.config.host = "https://example.databricks.com"
    workspace.config.token = TEST_AUTH

    monkeypatch.setattr(
        "dspy.integrations.finetune.databricks._get_workspace_client",
        lambda: workspace,
    )

    optimization_response = MagicMock()
    optimization_response.json.return_value = {"optimizable": True, "throughput_chunk_size": 100}
    get_endpoint_response = MagicMock(status_code=404)
    post_response = MagicMock(status_code=200)

    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = MagicMock()

    with (
        patch("requests.get", side_effect=[optimization_response, get_endpoint_response]),
        patch("requests.post", return_value=post_response),
        patch("openai.OpenAI", return_value=openai_client),
    ):
        DatabricksProvider.deploy_finetuned_model(
            model="main.schema.model",
            data_format=TrainDataFormat.CHAT,
            databricks_host="https://example.databricks.com",
            databricks_token=TEST_AUTH,
            deploy_timeout=60,
        )

    openai_client.chat.completions.create.assert_called_once()

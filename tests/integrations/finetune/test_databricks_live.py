import pytest

from dspy.clients.finetune import TrainDataFormat
from dspy.clients.lm import LM
from dspy.core.types import LMRequest
from dspy.integrations.finetune.databricks import (
    DatabricksProvider,
    TrainingJobDatabricks,
    _create_directory_in_databricks_unity_catalog,
)


@pytest.fixture(scope="module")
def databricks_workspace():
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        pytest.skip(f"Databricks SDK is not installed: {exc}")
    try:
        return WorkspaceClient()
    except Exception as exc:
        pytest.skip(f"Databricks SDK not configured or credentials not available: {exc}")


@pytest.mark.integration
def test_create_directory_in_databricks_unity_catalog(databricks_workspace):
    with pytest.raises(
        ValueError,
        match=r"Databricks Unity Catalog path must be in the format '/Volumes/<catalog>/<schema>/<volume>/\.\.\.', but received: /badstring/whatever",
    ):
        _create_directory_in_databricks_unity_catalog(databricks_workspace, "/badstring/whatever")
    _create_directory_in_databricks_unity_catalog(databricks_workspace, "/Volumes/main/chenmoney/testing/dspy_testing")
    databricks_workspace.files.get_directory_metadata("/Volumes/main/chenmoney/testing/dspy_testing")


@pytest.mark.integration
def test_create_finetuning_job(databricks_workspace):
    fake_training_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "Berlin!"},
            ]
        },
    ]
    job = TrainingJobDatabricks()
    DatabricksProvider.finetune(
        job=job,
        model="meta-llama/Llama-3.2-1B",
        train_data=fake_training_data,
        train_data_format="chat",
        train_kwargs={
            "train_data_path": "/Volumes/main/chenmoney/testing/dspy_testing",
            "register_to": "main.chenmoney.finetuned_model",
            "task_type": "CHAT_COMPLETION",
            "skip_deploy": True,
        },
    )
    assert job.finetuning_run is not None
    assert job.finetuning_run.status.display_name is not None


@pytest.mark.integration
@pytest.mark.llm_call
def test_deploy_finetuned_model(databricks_workspace):
    model_to_deploy = "main.chenmoney.finetuned_model"
    DatabricksProvider.deploy_finetuned_model(model=model_to_deploy, data_format=TrainDataFormat.CHAT)
    lm = LM(model="databricks/main_chenmoney_finetuned_model")
    lm(LMRequest.from_call(model=lm.model, prompt="what is 2 + 2?"))

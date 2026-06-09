from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

import orjson
from typing_extensions import override

from dspy.clients.finetune.provider import TrainingJob, UnsupportedReinforceJob
from dspy.clients.finetune.utils import TrainDataFormat, get_finetune_directory, validate_data_format

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

    from dspy.clients.finetune.protocol import ReinforceJob as ReinforceJobProtocol
logger = logging.getLogger(__name__)


class TrainingJobDatabricks(TrainingJob):
    def __init__(self, finetuning_run=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finetuning_run = finetuning_run
        self.run: Any = finetuning_run
        self.launch_started = False
        self.launch_completed = False
        self.endpoint_name = None

    @override
    def status(self):
        if not self.finetuning_run:
            return None
        try:
            from databricks.model_training import foundation_model as fm
        except ImportError:
            raise ImportError(
                "To use Databricks finetuning, please install the databricks_genai package via `pip install databricks_genai`."
            )
        run = fm.get(self.finetuning_run)
        return run.status


class DatabricksProvider:
    finetunable = True
    reinforceable = False
    TrainingJob: type[TrainingJob] = TrainingJobDatabricks
    ReinforceJob: type[ReinforceJobProtocol] = UnsupportedReinforceJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        return model.startswith("databricks/")

    @staticmethod
    def launch(_lm: Any, _launch_kwargs: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def kill(_lm: Any, _launch_kwargs: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def deploy_finetuned_model(
        model: str,
        data_format: TrainDataFormat | None = None,
        databricks_host: str | None = None,
        databricks_token: str | None = None,
        deploy_timeout: int = 900,
    ) -> None:
        import requests

        workspace_client = _get_workspace_client()
        model_version = next(workspace_client.model_versions.list(model)).version
        databricks_host = databricks_host or workspace_client.config.host
        databricks_token = databricks_token or workspace_client.config.token
        headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}
        optimizable_info = requests.get(
            url=f"{databricks_host}/api/2.0/serving-endpoints/get-model-optimization-info/{model}/{model_version}",
            headers=headers,
        ).json()
        if "optimizable" not in optimizable_info or not optimizable_info["optimizable"]:
            raise ValueError(f"Model is not eligible for provisioned throughput: {optimizable_info}")
        chunk_size = optimizable_info["throughput_chunk_size"]
        min_provisioned_throughput = 0
        max_provisioned_throughput = chunk_size
        model_name = model.replace(".", "_")
        get_endpoint_response = requests.get(
            url=f"{databricks_host}/api/2.0/serving-endpoints/{model_name}", json={"name": model_name}, headers=headers
        )
        if get_endpoint_response.status_code == 200:
            logger.info(f"Serving endpoint {model_name} already exists, updating it instead of creating a new one.")
            data = {
                "served_entities": [
                    {
                        "name": model_name,
                        "entity_name": model,
                        "entity_version": model_version,
                        "min_provisioned_throughput": min_provisioned_throughput,
                        "max_provisioned_throughput": max_provisioned_throughput,
                    }
                ]
            }
            response = requests.put(
                url=f"{databricks_host}/api/2.0/serving-endpoints/{model_name}/config", json=data, headers=headers
            )
        else:
            logger.info(f"Creating serving endpoint {model_name} on Databricks model serving!")
            data = {
                "name": model_name,
                "config": {
                    "served_entities": [
                        {
                            "name": model_name,
                            "entity_name": model,
                            "entity_version": model_version,
                            "min_provisioned_throughput": min_provisioned_throughput,
                            "max_provisioned_throughput": max_provisioned_throughput,
                        }
                    ]
                },
            }
            response = requests.post(url=f"{databricks_host}/api/2.0/serving-endpoints", json=data, headers=headers)
        if response.status_code == 200:
            logger.info(
                f"Successfully started creating/updating serving endpoint {model_name} on Databricks model serving!"
            )
        else:
            raise ValueError(f"Failed to create serving endpoint: {response.json()}.")
        logger.info(
            f"Waiting for serving endpoint {model_name} to be ready, this might take a few minutes... You can check the status of the endpoint at {databricks_host}/ml/endpoints/{model_name}"
        )
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            NotFoundError,
            OpenAI,
            RateLimitError,
        )

        client = OpenAI(api_key=databricks_token, base_url=f"{databricks_host}/serving-endpoints")
        num_retries = deploy_timeout // 60
        for _ in range(num_retries):
            try:
                if data_format == TrainDataFormat.CHAT:
                    client.chat.completions.create(
                        messages=[{"role": "user", "content": "hi"}], model=model_name, max_tokens=1
                    )
                elif data_format == TrainDataFormat.COMPLETION:
                    client.completions.create(prompt="hi", model=model_name, max_tokens=1)
                logger.info(f"Databricks model serving endpoint {model_name} is ready!")
                return
            except (NotFoundError, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
                if isinstance(exc, APIStatusError) and exc.status_code not in (404, 503):
                    raise
                time.sleep(60)
        raise ValueError(
            f"Failed to create serving endpoint {model_name} on Databricks model serving platform within {deploy_timeout} seconds."
        )

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        train_kwargs = train_kwargs or {}
        if isinstance(train_data_format, str):
            if train_data_format == "chat":
                train_data_format = TrainDataFormat.CHAT
            elif train_data_format == "completion":
                train_data_format = TrainDataFormat.COMPLETION
            else:
                raise ValueError(
                    f"String `train_data_format` must be one of 'chat' or 'completion', but received: {train_data_format}."
                )
        if "train_data_path" not in train_kwargs:
            raise ValueError("The `train_data_path` must be provided to finetune on Databricks.")
        if not isinstance(train_data_format, TrainDataFormat):
            raise TypeError(f"Expected TrainDataFormat after normalization, got {type(train_data_format).__name__}.")
        validate_data_format(train_data, train_data_format)
        train_kwargs["train_data_path"] = DatabricksProvider.upload_data(train_data, train_kwargs["train_data_path"])
        databricks_job = job
        try:
            from databricks.model_training import foundation_model as fm
        except ImportError:
            raise ImportError(
                "To use Databricks finetuning, please install the databricks_genai package via `pip install databricks_genai`."
            )
        if "register_to" not in train_kwargs:
            raise ValueError("The `register_to` must be provided to finetune on Databricks.")
        databricks_host = train_kwargs.pop("databricks_host", None)
        databricks_token = train_kwargs.pop("databricks_token", None)
        skip_deploy = train_kwargs.pop("skip_deploy", False)
        deploy_timeout = train_kwargs.pop("deploy_timeout", 900)
        logger.info("Starting finetuning on Databricks... this might take a few minutes to finish.")
        finetuning_run = fm.create(model=model, **train_kwargs)
        databricks_job.run = finetuning_run
        while True:
            databricks_job.run = fm.get(databricks_job.run)
            if databricks_job.run.status.display_name == "Completed":
                logger.info("Finetuning run completed successfully!")
                break
            if databricks_job.run.status.display_name == "Failed":
                raise ValueError(
                    f"Finetuning run failed with status: {databricks_job.run.status.display_name}. Please check the Databricks workspace for more details. Finetuning job's metadata: {databricks_job.run}."
                )
            time.sleep(60)
        if skip_deploy:
            return ""
        databricks_job.launch_started = True
        model_to_deploy = train_kwargs["register_to"]
        databricks_job.endpoint_name = model_to_deploy.replace(".", "_")
        DatabricksProvider.deploy_finetuned_model(
            model_to_deploy, train_data_format, databricks_host, databricks_token, deploy_timeout
        )
        databricks_job.launch_completed = True
        return f"databricks/{databricks_job.endpoint_name}"

    @staticmethod
    def upload_data(train_data: list[dict[str, Any]], databricks_unity_catalog_path: str):
        logger.info("Uploading finetuning data to Databricks Unity Catalog...")
        file_path = _save_data_to_local_file(train_data=train_data)
        w = _get_workspace_client()
        _create_directory_in_databricks_unity_catalog(w=w, databricks_unity_catalog_path=databricks_unity_catalog_path)
        try:
            with open(file_path, "rb") as f:
                target_path = os.path.join(databricks_unity_catalog_path, os.path.basename(file_path))
                w.files.upload(target_path, f, overwrite=True)
            logger.info("Successfully uploaded finetuning data to Databricks Unity Catalog!")
            return target_path
        except Exception as e:
            raise ValueError(f"Failed to upload finetuning data to Databricks Unity Catalog: {e}")


def _get_workspace_client() -> WorkspaceClient:
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "To use Databricks finetuning, please install the databricks-sdk package via `pip install databricks-sdk`."
        )
    return WorkspaceClient()


def _create_directory_in_databricks_unity_catalog(w: WorkspaceClient, databricks_unity_catalog_path: str) -> None:
    pattern = "^/Volumes/(?P<catalog>[^/]+)/(?P<schema>[^/]+)/(?P<volume>[^/]+)(/[^/]+)+$"
    match = re.match(pattern, databricks_unity_catalog_path)
    if not match:
        raise ValueError(
            f"Databricks Unity Catalog path must be in the format '/Volumes/<catalog>/<schema>/<volume>/...', but received: {databricks_unity_catalog_path}."
        )
    catalog = match.group("catalog")
    schema = match.group("schema")
    volume = match.group("volume")
    volume_path = f"{catalog}.{schema}.{volume}"
    from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist

    try:
        w.volumes.read(volume_path)
    except (NotFound, ResourceDoesNotExist):
        raise ValueError(
            f"Databricks Unity Catalog volume does not exist: {volume_path}, please create it on the Databricks workspace."
        )
    try:
        w.files.get_directory_metadata(databricks_unity_catalog_path)
        logger.info(f"Directory {databricks_unity_catalog_path} already exists, skip creating it.")
    except (NotFound, ResourceDoesNotExist):
        logger.info(f"Creating directory {databricks_unity_catalog_path} in Databricks Unity Catalog...")
        w.files.create_directory(databricks_unity_catalog_path)
        logger.info(f"Successfully created directory {databricks_unity_catalog_path} in Databricks Unity Catalog!")


def _save_data_to_local_file(train_data: list[dict[str, Any]]):
    import uuid

    file_name = f"finetuning_{uuid.uuid4()}.jsonl"
    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "wb") as f:
        for item in train_data:
            f.write(orjson.dumps(item) + b"\n")
    return file_path

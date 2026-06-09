from __future__ import annotations

import logging
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

from dspy.clients.finetune.registry import infer_finetune_provider
from dspy.clients.model_id import split_provider_model
from dspy.errors import LMUnsupportedFeatureError

if TYPE_CHECKING:
    from dspy.clients.finetune.protocol import FinetuneProvider, ReinforceJob
    from dspy.clients.finetune.provider import TrainingJob
    from dspy.clients.finetune.utils import TrainDataFormat
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)


class FinetuneService:
    def __init__(
        self,
        lm: LM,
        *,
        finetune_provider: FinetuneProvider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.lm = lm
        self.provider = finetune_provider or infer_finetune_provider(lm.model)
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}

    @staticmethod
    def infer_provider(model: str) -> FinetuneProvider:
        return infer_finetune_provider(model)

    def launch(self, launch_kwargs: dict[str, Any] | None = None) -> None:
        self.provider.launch(self.lm, launch_kwargs or self.launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None) -> None:
        self.provider.kill(self.lm, launch_kwargs)

    def finetune(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> TrainingJob:
        if not self.provider.finetunable:
            raise LMUnsupportedFeatureError(
                f"Provider {self.provider} does not support fine-tuning. Pass `finetune_provider=` when creating "
                f"`FinetuneService` for finetunable models: "
                f"`databricks/{{endpoint}}` → `from dspy.integrations.finetune.databricks import DatabricksProvider`; "
                f"`local:{{path}}` → `from dspy.integrations.finetune.local import LocalProvider`; "
                f"`openai/{{model}}` or `ft:{{id}}` → `from dspy.integrations.finetune.openai import OpenAIProvider`. "
                f"Example: `FinetuneService(LM('openai/gpt-4.1-mini-2025-04-14'), finetune_provider=OpenAIProvider())`.",
                model=self.lm.model,
                provider=split_provider_model(self.lm.model)[0],
                features=["finetuning"],
            )

        resolved_train_kwargs = train_kwargs or self.train_kwargs
        model_to_finetune = self.finetuning_model or self.lm.model
        job = self.provider.TrainingJob(
            thread=None,
            model=model_to_finetune,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=resolved_train_kwargs,
        )
        thread = threading.Thread(target=partial(self._run_finetune_job, job))
        job.thread = thread
        thread.start()
        return job

    def reinforce(self, train_kwargs: dict[str, Any]) -> ReinforceJob:
        if not self.provider.reinforceable:
            raise LMUnsupportedFeatureError(
                f"Provider {self.provider} does not implement the reinforcement learning interface.",
                model=self.lm.model,
                provider=split_provider_model(self.lm.model)[0],
                features=["reinforce"],
            )
        job = self.provider.ReinforceJob(lm=self.lm, train_kwargs=train_kwargs)
        job.initialize()
        return job

    def _run_finetune_job(self, job: TrainingJob) -> None:
        try:
            if job.model is None or job.train_data is None:
                raise ValueError("TrainingJob requires model and train_data before finetuning.")
            model = self.provider.finetune(
                job=job,
                model=job.model,
                train_data=job.train_data,
                train_data_format=job.train_data_format,
                train_kwargs=job.train_kwargs,
            )
            result_lm = self.lm.copy(model=model)
            job.set_result(result_lm)
        except Exception as err:
            logger.exception("Finetune job failed")
            job.set_result(err)

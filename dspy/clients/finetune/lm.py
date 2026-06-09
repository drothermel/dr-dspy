from __future__ import annotations

import logging
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

from dspy.clients.finetune.provider import DefaultFinetuneProvider, TrainingJob
from dspy.clients.model_id import split_provider_model
from dspy.errors import LMUnsupportedFeatureError
from dspy.integrations.finetune.openai import OpenAIProvider

if TYPE_CHECKING:
    from dspy.clients.finetune.protocol import FinetuneProvider, ReinforceJob
    from dspy.clients.finetune.utils import TrainDataFormat
    from dspy.clients.lm.client import LM

logger = logging.getLogger(__name__)


def infer_provider(model: str) -> FinetuneProvider:
    if OpenAIProvider.is_provider_model(model):
        return OpenAIProvider()
    return DefaultFinetuneProvider()


def launch(lm: LM, launch_kwargs: dict[str, Any] | None = None) -> None:
    lm.provider.launch(lm, launch_kwargs)


def kill(lm: LM, launch_kwargs: dict[str, Any] | None = None) -> None:
    lm.provider.kill(lm, launch_kwargs)


def finetune(
    lm: LM,
    train_data: list[dict[str, Any]],
    train_data_format: TrainDataFormat | None,
    train_kwargs: dict[str, Any] | None = None,
) -> TrainingJob:
    if not lm.provider.finetunable:
        raise LMUnsupportedFeatureError(
            f"Provider {lm.provider} does not support fine-tuning, please specify your provider by explicitly setting `provider` when creating the `dspy.clients.lm.LM` instance. For example, `from dspy.clients.lm import LM; from dspy.integrations.finetune import OpenAIProvider; LM('openai/gpt-4.1-mini-2025-04-14', provider=OpenAIProvider())`.",
            model=lm.model,
            provider=split_provider_model(lm.model)[0],
            features=["finetuning"],
        )

    train_kwargs = train_kwargs or lm.train_kwargs
    model_to_finetune = lm.finetuning_model or lm.model
    job = lm.provider.TrainingJob(
        thread=None,
        model=model_to_finetune,
        train_data=train_data,
        train_data_format=train_data_format,
        train_kwargs=train_kwargs,
    )
    thread = threading.Thread(target=partial(_run_finetune_job, lm, job))
    job.thread = thread
    thread.start()
    return job


def reinforce(lm: LM, train_kwargs: dict[str, Any]) -> ReinforceJob:
    if not lm.provider.reinforceable:
        raise LMUnsupportedFeatureError(
            f"Provider {lm.provider} does not implement the reinforcement learning interface.",
            model=lm.model,
            provider=split_provider_model(lm.model)[0],
            features=["reinforce"],
        )
    job = lm.provider.ReinforceJob(lm=lm, train_kwargs=train_kwargs)
    job.initialize()
    return job


def _run_finetune_job(lm: LM, job: TrainingJob) -> None:
    try:
        if job.model is None or job.train_data is None:
            raise ValueError("TrainingJob requires model and train_data before finetuning.")
        model = lm.provider.finetune(
            job=job,
            model=job.model,
            train_data=job.train_data,
            train_data_format=job.train_data_format,
            train_kwargs=job.train_kwargs,
        )
        result_lm = lm.copy(model=model)
        job.set_result(result_lm)
    except Exception as err:
        logger.exception("Finetune job failed")
        job.set_result(err)

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import random
import string
from typing import TYPE_CHECKING, Any

from dspy._internal.lazy_import import import_optional, is_available
from dspy.clients.finetune.provider import TrainingJob, UnsupportedReinforceJob
from dspy.clients.finetune.utils import TrainDataFormat, save_data, validate_data_format
from dspy.integrations.finetune.local_server import attach_local_server, kill_local_server, launch_local_server

if TYPE_CHECKING:
    from dspy.clients.finetune.protocol import ReinforceJob as ReinforceJobProtocol
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)

_SGLANG_INSTALL_COMMAND = (
    "Navigate to https://docs.sglang.ai/start/install.html for the latest installation instructions."
)
_LOCAL_FINETUNE_INSTALL_COMMAND = "Run `pip install -U torch transformers accelerate trl peft`."


class LocalProvider:
    finetunable = True
    reinforceable = False
    TrainingJob: type[TrainingJob] = TrainingJob
    ReinforceJob: type[ReinforceJobProtocol] = UnsupportedReinforceJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        return model.startswith("local:")

    @staticmethod
    def launch(lm: LM, launch_kwargs: dict[str, Any] | None = None) -> None:
        if not is_available("sglang"):
            import_optional(
                "sglang",
                feature="local model launching",
                install_command=_SGLANG_INSTALL_COMMAND,
            )
        if hasattr(lm, "process"):
            logger.info("Server is already launched.")
            return
        launch_kwargs = launch_kwargs or {}
        model = _normalize_model_name(lm.model)
        logger.info("Grabbing a free port to launch an SGLang server for model %s", model)
        logger.info("We see that CUDA_VISIBLE_DEVICES is %s", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
        timeout = launch_kwargs.get("timeout", 1800)
        handle = launch_local_server(model=model, timeout=timeout)
        attach_local_server(lm, handle)
        logger.info(
            "Server ready on random port %s! Logs are available via lm.get_logs() method on returned lm.",
            handle.port,
        )

    @staticmethod
    def kill(lm: LM, _launch_kwargs: dict[str, Any] | None = None) -> None:
        kill_local_server(lm)

    @staticmethod
    def finetune(
        _job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        model = _normalize_model_name(model)
        if not isinstance(train_data_format, TrainDataFormat):
            raise TypeError(f"Expected TrainDataFormat, got {type(train_data_format).__name__}.")
        if train_data_format != TrainDataFormat.CHAT:
            raise ValueError("Only chat models are supported for local finetuning.")
        validate_data_format(train_data, train_data_format)
        data_path = save_data(train_data)
        logger.info("Train data saved to %s", data_path)
        output_dir = create_output_dir(model_name=model, data_path=data_path)
        default_train_kwargs = {
            "device": None,
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-05,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
        }
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}
        output_dir = train_kwargs["output_dir"]
        logger.info("Starting local training, will save to %s", output_dir)
        train_sft_locally(model_name=model, train_data=train_data, train_kwargs=train_kwargs)
        logger.info("Training complete")
        return f"local:{output_dir}"


def _normalize_model_name(model: str) -> str:
    if model.startswith("openai/"):
        model = model[7:]
    if model.startswith("local:"):
        model = model[6:]
    if model.startswith("huggingface/"):
        model = model[len("huggingface/") :]
    return model


def create_output_dir(model_name, data_path):
    model_str = model_name.replace("/", "-")
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rnd_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    model_identifier = f"{rnd_str}_{model_str}_{time_str}"
    return data_path.replace(".jsonl", "_" + model_identifier)


def train_sft_locally(model_name, train_data, train_kwargs):
    import_optional(
        "torch",
        feature="local finetuning",
        install_command=_LOCAL_FINETUNE_INSTALL_COMMAND,
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer, setup_chat_format

    device = train_kwargs.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Using device: %s", device)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    with contextlib.suppress(Exception):
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    if tokenizer.pad_token_id is None:
        logger.info("Adding pad token to tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})
    logger.info("Creating dataset")
    if "max_seq_length" not in train_kwargs:
        train_kwargs["max_seq_length"] = 4096
        logger.info(
            "The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to %s",
            train_kwargs["max_seq_length"],
        )
    from datasets import Dataset

    hf_dataset = Dataset.from_list(train_data)

    def tokenize_function(example, tok=tokenizer):
        return encode_sft_example(example=example, tokenizer=tok, max_seq_length=train_kwargs["max_seq_length"])

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type="torch")
    tokenized_dataset = tokenized_dataset.filter(lambda example: (example["labels"] != -100).any())
    use_peft = train_kwargs.get("use_peft", False)
    peft_config = None
    if use_peft:
        from peft import LoraConfig

        rank_dimension = 32
        lora_alpha = 64
        lora_dropout = 0.05
        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    sft_config = SFTConfig(
        output_dir=train_kwargs["output_dir"],
        num_train_epochs=train_kwargs["num_train_epochs"],
        per_device_train_batch_size=train_kwargs["per_device_train_batch_size"],
        gradient_accumulation_steps=train_kwargs["gradient_accumulation_steps"],
        learning_rate=train_kwargs["learning_rate"],
        max_grad_norm=2.0,
        logging_steps=20,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_steps=10000,
        bf16=train_kwargs["bf16"],
        max_seq_length=train_kwargs["max_seq_length"],
        packing=train_kwargs["packing"],
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
    )
    logger.info("Starting training")
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=tokenized_dataset, peft_config=peft_config)
    trainer.train()
    trainer.save_model()
    merge = True
    if use_peft and merge:
        from peft import AutoPeftModelForCausalLM

        model_ = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=sft_config.output_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        merged_model = model_.merge_and_unload()
        merged_model.save_pretrained(sft_config.output_dir, safe_serialization=True, max_shard_size="5GB")
    import gc

    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return sft_config.output_dir


def encode_sft_example(example, tokenizer, max_seq_length):
    import torch

    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids.flatten(), "labels": labels.flatten(), "attention_mask": attention_mask.flatten()}

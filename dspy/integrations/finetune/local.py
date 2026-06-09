from __future__ import annotations

import contextlib
import datetime
import importlib.util
import logging
import random
import socket
import string
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, cast

from dspy.clients.finetune.provider import TrainingJob, _UnsupportedReinforceJob
from dspy.clients.finetune.utils import TrainDataFormat, save_data, validate_data_format

if TYPE_CHECKING:
    from dspy.clients.finetune.protocol import ReinforceJob as ReinforceJobProtocol
    from dspy.clients.lm import LM
logger = logging.getLogger(__name__)


def _sglang_available() -> bool:
    import sys

    if "sglang" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError, AttributeError):
        return False


class LocalProvider:
    finetunable = True
    reinforceable = False
    TrainingJob: type[TrainingJob] = TrainingJob
    ReinforceJob: type[ReinforceJobProtocol] = _UnsupportedReinforceJob

    @staticmethod
    def launch(lm: LM, launch_kwargs: dict[str, Any] | None = None) -> None:
        if not _sglang_available():
            raise ImportError(
                "For local model launching, please install sglang.Navigate to https://docs.sglang.ai/start/install.html for the latest installation instructions!"
            )
        if hasattr(lm, "process"):
            logger.info("Server is already launched.")
            return
        launch_kwargs = launch_kwargs or lm.launch_kwargs
        import os

        model = lm.model
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]
        if model.startswith("huggingface/"):
            model = model[len("huggingface/") :]
        logger.info(f"Grabbing a free port to launch an SGLang server for model {model}")
        logger.info(f"We see that CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        port = get_free_port()
        timeout = launch_kwargs.get("timeout", 1800)
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--port",
            str(port),
            "--host",
            "0.0.0.0",  # noqa: S104
        ]
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logger.info(f"SGLang server process started with PID {process.pid}.")
        stop_printing_event = threading.Event()
        logs_buffer = []

        def _tail_process(proc, buffer, stop_event) -> None:
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    buffer.append(line)
                    if not stop_event.is_set():
                        pass

        thread = threading.Thread(target=_tail_process, args=(process, logs_buffer, stop_printing_event), daemon=True)
        thread.start()
        base_url = f"http://localhost:{port}"
        try:
            wait_for_server(base_url, timeout=timeout)
        except TimeoutError:
            process.kill()
            raise
        stop_printing_event.set()

        def get_logs() -> str:
            return "".join(logs_buffer)

        logger.info(f"Server ready on random port {port}! Logs are available via lm.get_logs() method on returned lm.")
        lm.kwargs["api_base"] = f"http://localhost:{port}/v1"
        lm.kwargs["api_key"] = "local"
        lm.provider_options = lm.provider_options.model_copy(
            update={"api_base": f"http://localhost:{port}/v1", "api_key": "local"}
        )
        lm_attrs = cast("Any", lm)
        lm_attrs.get_logs = get_logs
        lm_attrs.process = process
        lm_attrs.thread = thread

    @staticmethod
    def kill(lm: LM, _launch_kwargs: dict[str, Any] | None = None) -> None:
        from sglang.utils import terminate_process

        if not hasattr(lm, "process"):
            logger.info("No running server to kill.")
            return
        terminate_process(lm.process)
        thread = getattr(lm, "thread", None)
        if thread is not None:
            thread.join()
        del lm.process
        if hasattr(lm, "thread"):
            del lm.thread
        if hasattr(lm, "get_logs"):
            del lm.get_logs
        logger.info("Server killed.")

    @staticmethod
    def finetune(
        _job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]
        if not isinstance(train_data_format, TrainDataFormat):
            raise TypeError(f"Expected TrainDataFormat, got {type(train_data_format).__name__}.")
        if train_data_format != TrainDataFormat.CHAT:
            raise ValueError("Only chat models are supported for local finetuning.")
        validate_data_format(train_data, train_data_format)
        data_path = save_data(train_data)
        logger.info(f"Train data saved to {data_path}")
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
        logger.info(f"Starting local training, will save to {output_dir}")
        train_sft_locally(model_name=model, train_data=train_data, train_kwargs=train_kwargs)
        logger.info("Training complete")
        return f"openai/local:{output_dir}"


def create_output_dir(model_name, data_path):
    model_str = model_name.replace("/", "-")
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rnd_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    model_identifier = f"{rnd_str}_{model_str}_{time_str}"
    return data_path.replace(".jsonl", "_" + model_identifier)


def train_sft_locally(model_name, train_data, train_kwargs):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer, setup_chat_format
    except ImportError:
        raise ImportError(
            "For local finetuning, please install torch, transformers, and trl by running `pip install -U torch transformers accelerate trl peft`"
        )
    device = train_kwargs.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
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
            f"The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to {train_kwargs['max_seq_length']}"
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


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_for_server(base_url: str, timeout: int | None = None) -> None:
    import requests

    start_time = time.time()
    while True:
        try:
            response = requests.get(f"{base_url}/v1/models", headers={"Authorization": "Bearer None"})
            if response.status_code == 200:
                time.sleep(5)
                break
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


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

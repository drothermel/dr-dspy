import logging
import os
import shutil

logger = logging.getLogger(__name__)


def save_candidate_program(*, program, log_dir, trial_num, note=None):
    if log_dir is None:
        return None
    eval_programs_dir = os.path.join(log_dir, "evaluated_programs")
    os.makedirs(eval_programs_dir, exist_ok=True)
    if note:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}_{note}.json")
    else:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}.json")
    program.save(save_path)
    return save_path


def save_file_to_log_dir(*, source_file_path, log_dir) -> None:
    if log_dir is None:
        return
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    destination_file_path = os.path.join(log_dir, os.path.basename(source_file_path))
    shutil.copy(source_file_path, destination_file_path)


def get_token_usage(model) -> tuple[int, int]:
    if not hasattr(model, "call_log"):
        return (0, 0)
    input_tokens = []
    output_tokens = []
    for interaction in model.call_log:
        usage = interaction.usage
        _input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        _output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        input_tokens.append(_input_tokens)
        output_tokens.append(_output_tokens)
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    return (total_input_tokens, total_output_tokens)


def log_token_usage(*, trial_logs, trial_num, model_dict) -> None:
    token_usage_dict = {}
    for model_name, model in model_dict.items():
        in_tokens, out_tokens = get_token_usage(model)
        token_usage_dict[model_name] = {"total_input_tokens": in_tokens, "total_output_tokens": out_tokens}
    trial_logs[trial_num]["token_usage"] = token_usage_dict

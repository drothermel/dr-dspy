import json
import re

from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.teleprompt.task_spec_context import get_task_spec
from dspy.utils.source_format import get_formatted_source


def strip_prefix(text):
    pattern = "^[\\*\\s]*(([\\w\\'\\-]+\\s+){0,4}[\\w\\'\\-]+):\\s*"
    modified_text = re.sub(pattern, "", text)
    return modified_text.strip('"')


def create_instruction_set_history_string(base_program, trial_logs, top_n):
    program_history = []
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            trial_program.load(trial["program_path"])
            program_history.append({"program": trial_program, "score": trial["score"]})
    seen_programs = set()
    unique_program_history = []
    for entry in program_history:
        program = entry["program"]
        instruction_set = get_program_instruction_set_string(program)
        if instruction_set not in seen_programs:
            seen_programs.add(instruction_set)
            unique_program_history.append(entry)
    top_n_program_history = sorted(unique_program_history, key=lambda x: x["score"], reverse=True)[:top_n]
    top_n_program_history.reverse()
    instruction_set_history_string = ""
    for entry in top_n_program_history:
        program = entry["program"]
        score = entry["score"]
        instruction_set = get_program_instruction_set_string(program)
        instruction_set_history_string += instruction_set + f" | Score: {score}\n\n"
    return instruction_set_history_string


def parse_list_of_instructions(instruction_string):
    try:
        return json.loads(instruction_string)
    except json.JSONDecodeError:
        pass
    return re.findall('"([^"]*)"', instruction_string)


def get_program_instruction_set_string(program) -> str:
    instruction_list = []
    for _, pred in enumerate(program.predictors()):
        pred_instructions = get_task_spec(pred).instructions
        instruction_list.append(f'"{pred_instructions}"')
    return f"[{', '.join(instruction_list)}]"


def create_predictor_level_history_string(*, base_program, predictor_i, trial_logs, top_n):
    instruction_aggregate = {}
    instruction_history = []
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            trial_program.load(trial["program_path"])
            instruction_history.append({"program": trial_program, "score": trial["score"]})
    for history_item in instruction_history:
        predictor = history_item["program"].predictors()[predictor_i]
        instruction = get_task_spec(predictor).instructions
        score = history_item["score"]
        if instruction in instruction_aggregate:
            instruction_aggregate[instruction]["total_score"] += score
            instruction_aggregate[instruction]["count"] += 1
        else:
            instruction_aggregate[instruction] = {"total_score": score, "count": 1}
    predictor_history = []
    for instruction, data in instruction_aggregate.items():
        average_score = data["total_score"] / data["count"]
        predictor_history.append((instruction, average_score))
    seen_instructions = set()
    unique_predictor_history = []
    for instruction, score in predictor_history:
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_predictor_history.append((instruction, score))
    top_instructions = sorted(unique_predictor_history, key=lambda x: x[1], reverse=True)[:top_n]
    top_instructions.reverse()
    predictor_history_string = ""
    for instruction, score in top_instructions:
        predictor_history_string += instruction + f" | Score: {score}\n\n"
    return predictor_history_string


def create_example_string(*, fields, example):
    output = []
    for field_name, field_values in fields.items():
        name = field_values.prefix
        value = example.get(field_name)
        field_str = f"{name} {value}"
        output.append(field_str)
    return "\n".join(output)


def get_dspy_source_code(module):
    header = []
    base_code = ""
    if type(module).__name__ != "Predict" and type(module).__name__ != "ChainOfThought":
        base_code = get_formatted_source(type(module))
    completed_set = set()
    for attribute in module.__dict__:
        try:
            iterable = iter(getattr(module, attribute))
        except TypeError:
            iterable = [getattr(module, attribute)]
        for item in iterable:
            try:
                hash(item)
            except TypeError:
                continue
            if isinstance(item, Parameter):
                if (
                    hasattr(item, "task_spec")
                    and item.task_spec is not None
                    and (item.task_spec.name + "_sig" not in completed_set)
                ):
                    header.append(item.task_spec.to_debug_string())
                    completed_set.add(item.task_spec.name + "_sig")
            if isinstance(item, Module):
                code = get_dspy_source_code(item).strip()
                if code not in completed_set:
                    header.append(code)
                    completed_set.add(code)
            completed_set.add(item)
    return "\n\n".join(header) + "\n\n" + base_code

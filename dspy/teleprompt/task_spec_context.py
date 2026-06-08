from dspy.runtime.run_context import RunContext


def get_prompt_model(prompt_model, run: RunContext):
    if prompt_model:
        return prompt_model
    return run.lm


def get_task_spec(predictor):
    assert hasattr(predictor, "task_spec")
    return predictor.task_spec


def set_task_spec(*, predictor, task_spec) -> None:
    assert hasattr(predictor, "task_spec")
    predictor.task_spec = task_spec

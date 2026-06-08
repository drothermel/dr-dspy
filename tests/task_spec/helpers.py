from dspy.task_spec import TaskSpec, make_task_spec


def ts(spec: str, instructions: str, **kwargs) -> TaskSpec:
    return make_task_spec(spec, instructions=instructions, **kwargs)

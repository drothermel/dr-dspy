from dspy.task_spec import TaskSpec, default_task_instructions, make_task_spec


def _field_names(spec_part: str) -> tuple[str, ...]:
    names: list[str] = []
    for chunk in spec_part.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        names.append(chunk.split(":")[0].strip())
    return tuple(names)


def ts(spec: str, instructions: str | None = None, **kwargs) -> TaskSpec:
    if instructions is None:
        inputs_str, outputs_str = spec.split("->", 1)
        instructions = default_task_instructions(inputs=_field_names(inputs_str), outputs=_field_names(outputs_str))
    return make_task_spec(spec, instructions=instructions, **kwargs)

def default_task_instructions(*, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> str:
    inputs_ = ", ".join(f"`{field}`" for field in inputs)
    outputs_ = ", ".join(f"`{field}`" for field in outputs)
    return f"Given the fields {inputs_}, produce the fields {outputs_}."

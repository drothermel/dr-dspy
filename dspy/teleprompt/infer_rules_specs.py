from dspy.task_spec import input_field, make_task_spec, output_field


def rules_induction_task_spec(num_rules: int):
    return make_task_spec(
        {
            "examples_text": input_field("examples_text", str, desc="Text containing examples"),
            "natural_language_rules": output_field(
                "natural_language_rules", str, desc="Induced natural language rules"
            ),
        },
        instructions=f"Given a set of examples, extract a list of {num_rules} concise and non-redundant natural language rules that provide clear guidance for performing the task. All rules should be actionable for a well-specified scope of examples of this general kind of task.",
        name="framework.infer_rules.induction",
    )

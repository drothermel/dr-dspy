from dspy.task_spec import input_field


def test_field_spec_with_updates_clears_constraints():
    field = input_field("q", desc="Question.", constraints="must be short")
    updated = field.with_updates(constraints=None)
    assert updated.constraints is None


def test_field_spec_with_updates_omitting_constraints_leaves_unchanged():
    field = input_field("q", desc="Question.", constraints="must be short")
    updated = field.with_updates(desc="Updated question.")
    assert updated.constraints == "must be short"
    assert updated.desc == "Updated question."

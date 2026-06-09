from dspy.history import REPLHistory, TurnLog
from dspy.task_spec import input_field, make_task_spec, output_field, validate_task_inputs


def test_validate_task_inputs_coerces_dict_turn_log():
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", TurnLog, desc="Previous turns."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer the question.",
    )
    validated = validate_task_inputs(
        task_spec,
        {
            "turn_log": {"turns": [{"thought": "prior", "observation": "done"}]},
            "question": "Q?",
        },
    )
    assert isinstance(validated["turn_log"], TurnLog)
    assert validated["turn_log"].turns[0].thought == "prior"


def test_validate_task_inputs_coerces_dict_repl_history():
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", REPLHistory, desc="Previous REPL steps."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer the question.",
    )
    validated = validate_task_inputs(
        task_spec,
        {
            "turn_log": {"entries": [{"reasoning": "think", "code": "x=1", "output": "1"}]},
            "question": "Q?",
        },
    )
    assert isinstance(validated["turn_log"], REPLHistory)
    assert validated["turn_log"].entries[0].code == "x=1"

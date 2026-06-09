from dspy.adapters.chat_adapter import ChatAdapter
from dspy.history import REPLEntry, REPLHistory
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.conftest import adapter_format_as_openai


def test_repl_history_stays_inline_in_user_message():
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=REPLHistory, desc="The REPL history."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `turn_log`, `question`, produce the fields `answer`.",
    )
    history = REPLHistory(entries=[REPLEntry(code="print(1)", output="1")])
    messages = adapter_format_as_openai(
        adapter=ChatAdapter(),
        task_spec=task_spec,
        demos=[],
        inputs={"turn_log": history, "question": "Q"},
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "=== Step 1 ===" in messages[-1]["content"]
    assert "turn_log" in messages[-1]["content"].lower() or "print(1)" in messages[-1]["content"]

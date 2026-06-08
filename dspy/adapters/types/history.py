from typing import Any

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated task spec.
    For example, if you have the following task spec:

    ```
    from dspy.adapters.types.history import History
    from dspy.task_spec import TaskSpec, input_field, output_field

    class HistoryQATaskSpec(TaskSpec):
        name: str = "HistoryQA"
        instructions: str = "Answer using conversation history."
        inputs: tuple = (
            input_field("question"),
            input_field("history", type_=History),
        )
        outputs: tuple = (output_field("answer"),)
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Examples:
        ```
        import asyncio

        from dspy.adapters.types.history import History
        from dspy.clients.lm import LM
        from dspy.dsp.utils.settings import settings
        from dspy.predict.predict import Predict
        from dspy.task_spec import TaskSpec, input_field, output_field

        class HistoryQATaskSpec(TaskSpec):
            name: str = "HistoryQA"
            instructions: str = "Answer using conversation history."
            inputs: tuple = (
                input_field("question"),
                input_field("history", type_=History),
            )
            outputs: tuple = (output_field("answer"),)

        settings.configure(lm=LM("openai/gpt-4o-mini"))

        history = History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = Predict(HistoryQATaskSpec())
        outputs = asyncio.run(predict(question="What is the capital of France?", history=history))
        ```

    Example of capturing the conversation history:
        ```
        import asyncio

        from dspy.adapters.types.history import History
        from dspy.clients.lm import LM
        from dspy.dsp.utils.settings import settings
        from dspy.predict.predict import Predict
        from dspy.task_spec import TaskSpec, input_field, output_field

        class HistoryQATaskSpec(TaskSpec):
            name: str = "HistoryQA"
            instructions: str = "Answer using conversation history."
            inputs: tuple = (
                input_field("question"),
                input_field("history", type_=History),
            )
            outputs: tuple = (output_field("answer"),)

        settings.configure(lm=LM("openai/gpt-4o-mini"))

        predict = Predict(HistoryQATaskSpec())
        outputs = asyncio.run(predict(question="What is the capital of France?"))
        history = History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = asyncio.run(predict(question="Are you sure?", history=history))
        ```
    """

    messages: list[dict[str, Any]]

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

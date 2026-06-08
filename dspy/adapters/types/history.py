from typing import Any

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated task spec.
    For example, if you have the following task spec:

    ```
    from dspy.adapters.types.history import History
    from dspy.task_spec import FieldSpec, make_task_spec

    task_spec = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "history": FieldSpec.input("history", type_=History),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Answer using conversation history.",
    )
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Examples:
        ```
        import asyncio

        from dspy.adapters.types.history import History
        from dspy.clients.lm import LM
        from dspy.dsp.utils.settings import settings
        from dspy.predict.predict import Predict
        from dspy.task_spec import FieldSpec, make_task_spec

        settings.configure(lm=LM("openai/gpt-4o-mini"))

        task_spec = make_task_spec(
            {
                "question": FieldSpec.input("question"),
                "history": FieldSpec.input("history", type_=History),
                "answer": FieldSpec.output("answer"),
            },
            instructions="Answer using conversation history.",
        )

        history = History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = Predict(task_spec)
        outputs = asyncio.run(predict(question="What is the capital of France?", history=history))
        ```

    Example of capturing the conversation history:
        ```
        import asyncio

        from dspy.adapters.types.history import History
        from dspy.clients.lm import LM
        from dspy.dsp.utils.settings import settings
        from dspy.predict.predict import Predict
        from dspy.task_spec import FieldSpec, make_task_spec

        settings.configure(lm=LM("openai/gpt-4o-mini"))

        task_spec = make_task_spec(
            {
                "question": FieldSpec.input("question"),
                "history": FieldSpec.input("history", type_=History),
                "answer": FieldSpec.output("answer"),
            },
            instructions="Answer using conversation history.",
        )

        predict = Predict(task_spec)
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

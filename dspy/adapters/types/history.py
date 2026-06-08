from typing import Any

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    from dspy.adapters.types.history import History
    from dspy.signatures.field import InputField, OutputField
    from dspy.signatures.signature import Signature

    class MySignature(Signature):
        question: str = InputField()
        history: History = InputField()
        answer: str = OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Examples:
        ```
        from dspy.adapters.types.history import History
        from dspy.clients.lm import LM
        from dspy.dsp.utils.settings import settings
        from dspy.predict.predict import Predict
        from dspy.signatures.field import InputField, OutputField
        from dspy.signatures.signature import Signature

        settings.configure(lm=LM("openai/gpt-4o-mini"))

        class MySignature(Signature):
            question: str = InputField()
            history: History = InputField()
            answer: str = OutputField()

        history = History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```
        from dspy.adapters.types.history import History
        from dspy.clients.lm import LM
        from dspy.dsp.utils.settings import settings
        from dspy.predict.predict import Predict
        from dspy.signatures.field import InputField, OutputField
        from dspy.signatures.signature import Signature

        settings.configure(lm=LM("openai/gpt-4o-mini"))

        class MySignature(Signature):
            question: str = InputField()
            history: History = InputField()
            answer: str = OutputField()

        predict = Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

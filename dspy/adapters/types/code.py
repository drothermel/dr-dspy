import re
from typing import ClassVar, cast

import pydantic
from pydantic import create_model
from typing_extensions import override

from dspy.adapters.types.base_type import Type


class Code(Type):
    """Code type in DSPy.

    This type is useful for code generation and code analysis.

    Example 1: `Code` as output type in code generation:

    ```python
    import asyncio

    from dspy.adapters.types.code import Code
    from dspy.clients.lm import LM
    from dspy.dsp.utils.settings import settings
    from dspy.predict.predict import Predict
    from dspy.task_spec import TaskSpec, input_field, output_field

    class CodeGenerationTaskSpec(TaskSpec):
        name: str = "CodeGeneration"
        instructions: str = "Generate python code to answer the question."
        inputs: tuple = (input_field("question", desc="The question to answer"),)
        outputs: tuple = (output_field("code", type_=Code["java"], desc="The code to execute"),)

    settings.configure(lm=LM("openai/gpt-4o-mini"))

    predict = Predict(CodeGenerationTaskSpec())
    result = asyncio.run(predict(question="Given an array, find if any of the two numbers sum up to 10"))
    print(result.code)
    ```

    Example 2: `Code` as input type in code analysis:

    ```python
    import asyncio
    import inspect

    from dspy.adapters.types.code import Code
    from dspy.clients.lm import LM
    from dspy.dsp.utils.settings import settings
    from dspy.predict.predict import Predict
    from dspy.task_spec import TaskSpec, input_field, output_field

    class CodeAnalysisTaskSpec(TaskSpec):
        name: str = "CodeAnalysis"
        instructions: str = "Analyze the time complexity of the function."
        inputs: tuple = (input_field("code", type_=Code["python"], desc="The function to analyze"),)
        outputs: tuple = (output_field("result", desc="The time complexity of the function"),)

    settings.configure(lm=LM("openai/gpt-4o-mini"))

    predict = Predict(CodeAnalysisTaskSpec())

    def sleepsort(x):
        import time

        for i in x:
            time.sleep(i)
            print(i)

    result = asyncio.run(predict(code=inspect.getsource(sleepsort)))
    print(result.result)
    ```
    """

    code: str

    language: ClassVar[str] = "python"

    @override
    def format(self) -> str:
        return f"{self.code}"

    @pydantic.model_serializer()
    @override
    def serialize_model(self) -> str:
        """Serialize code as plain text instead of JSON content blocks."""
        return self.format()

    @classmethod
    @override
    def description(cls) -> str:
        return (
            "Code represented in a string, specified in the `code` field. If this is an output field, the code "
            f"field should follow the markdown code block format, e.g. \n```{cls.language.lower()}\n{{code}}\n```"
            f"\nProgramming language: {cls.language}"
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"code": _filter_code(data)}

        if isinstance(data, dict):
            data = cast("dict[str, object]", data)
            if "code" not in data:
                raise ValueError("`code` field is required for `dspy.adapters.types.code.Code`")
            if not isinstance(data["code"], str):
                raise ValueError(f"`code` field must be a string, but received type: {type(data['code'])}")
            return {"code": _filter_code(data["code"])}

        raise ValueError(f"Received invalid value for `dspy.adapters.types.code.Code`: {data}")


def _filter_code(code: str) -> str:
    """Extract code from markdown code blocks, stripping any language identifier."""
    # Case 1: format like:
    # ```python
    # {code_block}
    # ```
    regex_pattern = r"```(?:[^\n]*)\n(.*?)```"
    match = re.search(regex_pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: ```<code>``` (no language, single-line)
    regex_pattern_simple = r"```(.*?)```"
    match = re.search(regex_pattern_simple, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback case
    return code


# Patch __class_getitem__ directly on the class to support Code["python"] syntax.
def _code_class_getitem(cls: type[Code], language: str) -> type[Code]:
    code_with_language_cls = create_model(f"{cls.__name__}_{language}", __base__=cls)
    code_with_language_cls.language = language
    return code_with_language_cls


Code.__class_getitem__ = classmethod(_code_class_getitem)  # ty: ignore[invalid-assignment]

import re
from typing import ClassVar, cast

import pydantic
from pydantic import create_model
from typing_extensions import override

from dspy.adapters.types.base_type import Type


class Code(Type):
    code: str
    language: ClassVar[str] = "python"

    @override
    def format(self) -> str:
        return f"{self.code}"

    @pydantic.model_serializer()
    @override
    def serialize_model(self) -> str:
        return self.format()

    @classmethod
    @override
    def description(cls) -> str:
        return f"Code represented in a string, specified in the `code` field. If this is an output field, the code field should follow the markdown code block format, e.g. \n```{cls.language.lower()}\n{{code}}\n```\nProgramming language: {cls.language}"

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
    regex_pattern = "```(?:[^\\n]*)\\n(.*?)```"
    match = re.search(regex_pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    regex_pattern_simple = "```(.*?)```"
    match = re.search(regex_pattern_simple, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code


def _code_class_getitem(cls: type[Code], language: str) -> type[Code]:
    code_with_language_cls = create_model(f"{cls.__name__}_{language}", __base__=cls)
    code_with_language_cls.language = language
    return code_with_language_cls


Code.__class_getitem__ = classmethod(_code_class_getitem)

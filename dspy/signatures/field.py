import pydantic

from dspy.utils.constants import IS_TYPE_UNDEFINED

# DSPy-specific field arguments are stored separately from Pydantic Field arguments. If Pydantic adds one of these names, this list will need explicit conflict handling.
DSPY_FIELD_ARG_NAMES = ["desc", "__dspy_field_type", IS_TYPE_UNDEFINED]
_REJECTED_FIELD_ARGS = frozenset({"prefix"})

PYDANTIC_CONSTRAINT_MAP = {
    "gt": "greater than: ",
    "ge": "greater than or equal to: ",
    "lt": "less than: ",
    "le": "less than or equal to: ",
    "min_length": "minimum length: ",
    "max_length": "maximum length: ",
    "multiple_of": "a multiple of the given number: ",
    "allow_inf_nan": "allow 'inf', '-inf', 'nan' values: ",
}


def move_kwargs(**kwargs):
    # Pydantic doesn't allow arbitrary arguments to be given to fields,
    # but asks that
    # > any extra data you want to add to the JSON schema should be passed
    # > as a dictionary to the json_schema_extra keyword argument.
    # See: https://docs.pydantic.dev/2.6/migration/#changes-to-pydanticfield
    pydantic_kwargs = {}
    json_schema_extra = {}
    for k, v in kwargs.items():
        if k in DSPY_FIELD_ARG_NAMES:
            json_schema_extra[k] = v
        else:
            pydantic_kwargs[k] = v
    # Also copy over the pydantic "description" if no dspy "desc" is given.
    if "description" in kwargs and "desc" not in json_schema_extra:
        json_schema_extra["desc"] = kwargs["description"]
    constraints = _translate_pydantic_field_constraints(**kwargs)
    if constraints:
        json_schema_extra["constraints"] = constraints
    pydantic_kwargs["json_schema_extra"] = json_schema_extra
    return pydantic_kwargs


def _translate_pydantic_field_constraints(**kwargs):
    """Extracts Pydantic constraints and translates them into human-readable format."""

    constraints = []
    for key, value in kwargs.items():
        if key in PYDANTIC_CONSTRAINT_MAP:
            constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")

    return ", ".join(constraints)


def _reject_unknown_field_args(field_name: str, **kwargs) -> None:
    for arg in _REJECTED_FIELD_ARGS.intersection(kwargs):
        raise TypeError(f"{field_name}() got an unexpected keyword argument {arg!r}")


def InputField(**kwargs):  # noqa: N802
    _reject_unknown_field_args("InputField", **kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    _reject_unknown_field_args("OutputField", **kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))

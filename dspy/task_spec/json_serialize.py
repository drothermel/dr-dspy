from pydantic import TypeAdapter


def serialize_for_json(value: object) -> object:
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)

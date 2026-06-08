from dspy.adapters.types.tool import convert_input_schema_to_tool_args


def test_tool_convert_input_schema_to_tool_args_no_input_params():
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(schema={"properties": {}})
    assert args == {}
    assert arg_types == {}
    assert arg_desc == {}


def test_tool_convert_input_schema_to_tool_args_lang_chain():
    # Example from langchain docs:
    # https://web.archive.org/web/20250723101359/https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(
        schema={
            "title": "fooSchema",
            "description": "The foo.",
            "type": "object",
            "properties": {
                "bar": {
                    "title": "Bar",
                    "description": "The bar.",
                    "type": "string",
                },
                "baz": {
                    "title": "Baz",
                    "type": "integer",
                },
            },
            "required": [
                "baz",
            ],
        }
    )
    assert args == {
        "bar": {"title": "Bar", "description": "The bar.", "type": "string"},
        "baz": {"title": "Baz", "type": "integer"},
    }
    assert arg_types == {
        "bar": str,
        "baz": int,
    }
    assert arg_desc == {
        "bar": "The bar.",
        "baz": "No description provided. (Required)",
    }

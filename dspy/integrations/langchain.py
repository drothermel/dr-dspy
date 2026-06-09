from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    from langchain.tools import BaseTool


def convert_langchain_tool(tool: "BaseTool") -> Tool:

    async def func(**kwargs):
        try:
            return await tool.ainvoke(kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to call LangChain tool '{tool.name}': {e!s}") from e

    args_schema = tool.args_schema
    args, _, arg_desc, required_names = convert_input_schema_to_tool_args(args_schema.model_json_schema())
    arg_types = {
        key: field.annotation if field.annotation is not None else Any
        for key, field in args_schema.model_fields.items()
    }
    return Tool(
        func,
        description=tool.description,
        name=tool.name,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
        required_names=required_names,
    )

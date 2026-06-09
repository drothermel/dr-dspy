import json
from typing import TYPE_CHECKING, Any

from dspy._internal.lazy_import import import_optional
from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp


def _serialize_mcp_content_for_error(content: Any, *, text_content_type: type[Any]) -> Any:
    if isinstance(content, text_content_type):
        return content.text
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json", by_alias=True, exclude_none=True)
    return repr(content)


def _convert_mcp_tool_result(call_tool_result: "mcp.types.CallToolResult") -> str | list[Any]:
    mcp_types = import_optional("mcp.types", extra="mcp", feature="MCP tool conversion")
    TextContent = mcp_types.TextContent

    tool_content: list[Any] = []
    error_content: list[Any] = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            tool_content.append(content.text)
        else:
            tool_content.append(content)
        error_content.append(_serialize_mcp_content_for_error(content, text_content_type=TextContent))
    if call_tool_result.isError:
        if len(error_content) == 1 and isinstance(error_content[0], str):
            details = error_content[0]
        else:
            details = json.dumps(error_content, ensure_ascii=False, default=str)
        raise RuntimeError(f"Failed to call MCP tool: {details}")
    if len(tool_content) == 1 and isinstance(call_tool_result.content[0], TextContent):
        return tool_content[0]
    return tool_content


def convert_mcp_tool(session: "mcp.ClientSession", tool: "mcp.types.Tool") -> Tool:
    import_optional("mcp", extra="mcp", feature="MCP tool conversion")
    args, arg_types, arg_desc, required_names = convert_input_schema_to_tool_args(tool.inputSchema)

    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result)

    return Tool(
        func,
        description=tool.description or "",
        name=tool.name,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
        required_names=required_names,
    )

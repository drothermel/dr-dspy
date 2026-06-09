"""Optional third-party integrations (MCP, LangChain)."""

from dspy.integrations.langchain import convert_langchain_tool
from dspy.integrations.mcp import convert_mcp_tool

__all__ = ["convert_langchain_tool", "convert_mcp_tool"]

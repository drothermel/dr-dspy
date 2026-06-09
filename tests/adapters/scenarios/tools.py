from dspy.adapters.types.tool import Tool


def search_tool() -> Tool:
    def search(query: str, k: int = 3) -> str:
        return query

    return Tool(search, description="Search for documents.")

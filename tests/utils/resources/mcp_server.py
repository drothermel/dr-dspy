from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("test")


class Profile(BaseModel):
    name: str
    age: int


class Account(BaseModel):
    profile: Profile
    account_id: str


@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


@mcp.tool()
def hello(names: list[str]) -> list[str]:
    return [f"Hello, {name}!" for name in names]


@mcp.tool()
def wrong_tool():
    raise ValueError("error!")


@mcp.tool()
def get_account_name(account: Account):
    return account.profile.name


@mcp.tool()
def current_datetime() -> str:
    return "2025-07-23T09:10:10.0+00:00"


if __name__ == "__main__":
    mcp.run()

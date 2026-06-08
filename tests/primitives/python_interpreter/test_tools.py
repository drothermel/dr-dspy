import asyncio

import pytest

from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


def test_tools_dict_is_copied():
    """Test that tools dict is defensively copied, not stored by reference."""
    tools = {"my_tool": lambda: "result"}
    sandbox = PythonInterpreter(tools=tools)  # ty:ignore[invalid-argument-type]

    # Modify the original dict after construction
    tools["new_tool"] = lambda: "new"

    # The sandbox should not see the new tool
    assert "new_tool" not in sandbox.tools


def test_deno_command_dict_raises_type_error():
    """Test that passing a dict as deno_command raises TypeError."""
    with pytest.raises(TypeError, match="deno_command must be a list"):
        PythonInterpreter(deno_command={"invalid": "dict"})  # ty:ignore[invalid-argument-type]


def test_tool_with_typed_signature():
    """Test that tools get proper typed signatures from inspect."""

    def my_tool(query: str, limit: int = 10) -> str:
        return f"searched '{query}' with limit {limit}"

    with PythonInterpreter(tools={"my_tool": my_tool}) as sandbox:
        # Tool should be callable with typed signature
        result = sandbox.execute('my_tool(query="test", limit=5)')
        assert result == "searched 'test' with limit 5"


def test_tool_positional_args():
    """Test that tools work with positional arguments."""

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    with PythonInterpreter(tools={"search": search}) as sandbox:
        result = sandbox.execute('search("hello")')
        assert result == "query=hello, limit=10"


def test_tool_keyword_args():
    """Test that tools work with keyword arguments."""

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    with PythonInterpreter(tools={"search": search}) as sandbox:
        result = sandbox.execute('search(query="hello", limit=5)')
        assert result == "query=hello, limit=5"


def test_tool_default_args():
    """Test that tool default arguments work correctly."""

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    with PythonInterpreter(tools={"greet": greet}) as sandbox:
        # Without default
        result = sandbox.execute('greet("World")')
        assert result == "Hello, World!"

        # Overriding default
        result = sandbox.execute('greet("World", "Hi")')
        assert result == "Hi, World!"


def test_tools_re_register_after_process_restart():
    """Tools should remain callable after Deno subprocess restart."""

    def echo(message: str = "") -> str:
        return f"Echo: {message}"

    with PythonInterpreter(tools={"echo": echo}) as interpreter:
        first = interpreter.execute('print(echo(message="one"))')
        assert "Echo: one" in first

        first_pid = interpreter.deno_process.pid
        interpreter.deno_process.kill()
        interpreter.deno_process.wait()

        second = interpreter.execute('print(echo(message="two"))')
        assert "Echo: two" in second
        assert interpreter.deno_process.pid != first_pid


def test_mounts_replay_after_process_restart(tmp_path):
    """Mounted files should still be accessible after subprocess restart."""
    host_file = tmp_path / "mount_restart.txt"
    host_file.write_text("restarted-ok")
    virtual_path = f"/sandbox/{host_file.name}"

    with PythonInterpreter(enable_read_paths=[str(host_file)]) as interpreter:
        first = interpreter.execute(f"with open({virtual_path!r}, 'r') as f:\n    data = f.read()\ndata")
        assert first == "restarted-ok"

        first_pid = interpreter.deno_process.pid
        interpreter.deno_process.kill()
        interpreter.deno_process.wait()

        second = interpreter.execute(f"with open({virtual_path!r}, 'r') as f:\n    data = f.read()\ndata")
        assert second == "restarted-ok"
        assert interpreter.deno_process.pid != first_pid


def test_tool_all_positional_args():
    """Test that tools work when all arguments are passed positionally."""

    def add(a: int, b: int, c: int) -> str:
        return f"{a + b + c}"

    with PythonInterpreter(tools={"add": add}) as sandbox:
        result = sandbox.execute("add(1, 2, 3)")
        assert result == "6"

        # Mixed: some positional, some keyword
        result = sandbox.execute("add(10, 20, c=30)")
        assert result == "60"


def test_tool_error_surfaces_as_runtime_error():
    """Test that exceptions raised by a tool surface as RuntimeError in the sandbox."""

    def failing_tool(x: int) -> str:
        raise ValueError(f"bad value: {x}")

    with PythonInterpreter(tools={"failing_tool": failing_tool}) as sandbox:
        result = sandbox.execute(
            "try:\n"
            "    failing_tool(42)\n"
            "    output = 'no error'\n"
            "except RuntimeError as e:\n"
            "    output = str(e)\n"
            "output"
        )
        assert "ValueError" in result
        assert "bad value: 42" in result


def test_tool_async_def_function():
    """async def tools should be awaited so the sandbox sees the resolved value."""

    async def slow_search(query: str) -> str:
        await asyncio.sleep(0)
        return f"answer:{query}"

    with PythonInterpreter(tools={"slow_search": slow_search}) as sandbox:  # ty:ignore[invalid-argument-type]
        result = sandbox.execute("slow_search(query='hello')")
        assert result == "answer:hello"


def test_tool_async_def_raises_propagates():
    """Exceptions raised inside an async tool should surface as RuntimeError in the sandbox."""

    async def failing_async(x: int) -> str:
        await asyncio.sleep(0)
        raise ValueError(f"boom:{x}")

    with PythonInterpreter(tools={"failing_async": failing_async}) as sandbox:  # ty:ignore[invalid-argument-type]
        result = sandbox.execute(
            "try:\n"
            "    failing_async(7)\n"
            "    output = 'no error'\n"
            "except RuntimeError as e:\n"
            "    output = str(e)\n"
            "output"
        )
        assert "ValueError" in result
        assert "boom:7" in result

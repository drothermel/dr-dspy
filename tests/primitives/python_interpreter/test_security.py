import os

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


def test_enable_env_vars_flag():
    os.environ["FOO_TEST_ENV"] = "test_value"

    with PythonInterpreter(enable_env_vars=None) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "", "Environment variables should be inaccessible without allow-env"

    with PythonInterpreter(enable_env_vars=["FOO_TEST_ENV"]) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "test_value", "Environment variables should be accessible with allow-env"


def test_read_file_access_control(tmp_path):
    testfile_path = tmp_path / "test_temp_file.txt"
    virtual_path = f"/sandbox/{testfile_path.name}"
    with open(testfile_path, "w") as f:
        f.write("test content")

    with PythonInterpreter(enable_read_paths=[str(testfile_path)]) as interpreter:
        code = f"with open({virtual_path!r}, 'r') as f:\n    data = f.read()\ndata"
        result = interpreter.execute(code)
        assert result == "test content", "Test file should be accessible with enable_read_paths and specified file"

    with PythonInterpreter(enable_read_paths=None) as interpreter:
        code = (
            f"try:\n"
            f"    with open({virtual_path!r}, 'r') as f:\n"
            f"        data = f.read()\n"
            f"except Exception as e:\n"
            f"    data = str(e)\n"
            f"data"
        )
        result = interpreter.execute(code)
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be accessible without enable_read_paths"
        )


def test_enable_write_flag(tmp_path):
    testfile_path = tmp_path / "test_temp_output.txt"
    virtual_path = f"/sandbox/{testfile_path.name}"

    with PythonInterpreter(enable_write_paths=None) as interpreter:
        code = (
            f"try:\n"
            f"    with open({virtual_path!r}, 'w') as f:\n"
            f"        f.write('blocked')\n"
            f"    result = 'wrote'\n"
            f"except Exception as e:\n"
            f"    result = str(e)\n"
            f"result"
        )
        result = interpreter.execute(code)
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be writable without enable_write_paths"
        )

    with PythonInterpreter(enable_write_paths=[str(testfile_path)]) as interpreter:
        code = f"with open({virtual_path!r}, 'w') as f:\n    f.write('allowed')\n'ok'"
        result = interpreter.execute(code)
        assert result == "ok", "Test file should be writable with enable_write_paths"
    assert testfile_path.exists()
    with open(testfile_path) as f:
        assert f.read() == "allowed", "Test file outputs should match content written during execution"

    with open(testfile_path, "w") as f:
        f.write("original_content")
    with PythonInterpreter(enable_write_paths=[str(testfile_path)], sync_files=False) as interpreter:
        code = f"with open({virtual_path!r}, 'w') as f:\n    f.write('should_not_sync')\n'done_no_sync'"
        result = interpreter.execute(code)
        assert result == "done_no_sync"
    with open(testfile_path) as f:
        assert f.read() == "original_content", "File should not be changed when sync_files is False"


def test_enable_net_flag():
    test_url = "https://example.com"

    with PythonInterpreter(enable_network_access=None) as interpreter:
        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        with pytest.raises(CodeInterpreterError, match="PythonError"):
            interpreter.execute(code)

    with PythonInterpreter(enable_network_access=["example.com"]) as interpreter:
        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        result = interpreter.execute(code)
        assert int(result) == 200, "Network access is permitted with enable_network_access"


def test_interpreter_security_filesystem_access(tmp_path):
    """
    Verify that the interpreter cannot read arbitrary files from the host system
    unless explicitly allowed.
    """
    # 1. Create a "secret" file on the host
    secret_file = tmp_path / "secret.txt"
    secret_content = "This is a secret content"  # noqa: S105
    secret_file.write_text(secret_content)
    secret_path_str = str(secret_file.absolute())

    # 2. Attempt to read the file WITHOUT permission
    malicious_code = f"""
import js
try:
    content = js.Deno.readTextFileSync('{secret_path_str}')
    print(content)
except Exception as e:
    print(f"Error: {{e}}")
"""

    with PythonInterpreter() as interpreter:
        output = interpreter(malicious_code)
        assert "Requires read access" in output
        assert secret_content not in output

    # 3. Attempt to read the file WITH permission
    with PythonInterpreter(enable_read_paths=[secret_path_str]) as interpreter:
        output = interpreter(malicious_code)
        assert secret_content in output


def test_enable_read_paths_symlink(tmp_path):
    """Regression test for #9501: symlinked enable_read_paths must resolve so Deno
    can read through them (denoland/deno#9607 — Deno prefix-matches against the
    realpath of the file being read). The sandbox virtual path keeps the user's
    original basename so user code refers to the file by the name passed in.
    """
    real_file = tmp_path / "real_name.txt"
    real_file.write_text("through symlink")
    link_file = tmp_path / "link_name.txt"
    try:
        link_file.symlink_to(real_file)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")  # ty: ignore[too-many-positional-arguments]

    with PythonInterpreter(enable_read_paths=[str(link_file)]) as interp:
        allow_read_arg = next(a for a in interp.deno_command if a.startswith("--allow-read="))
        allow_read = allow_read_arg[len("--allow-read=") :].split(",")
        assert os.path.realpath(str(real_file)) in allow_read
        assert str(link_file) not in allow_read

        result = interp.execute("with open('/sandbox/link_name.txt') as f:\n    data = f.read()\ndata")
        assert result == "through symlink"


def test_enable_read_paths_multiple_files(tmp_path):
    """Test that enable_read_paths works with multiple files in the same directory.

    Regression test for bug where mounting multiple files to /sandbox/ failed
    because Pyodide's ErrnoError has errno but no message property, causing
    the 'directory exists' check to fail on the second file.
    """
    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"
    file1.write_text("Content 1")
    file2.write_text("Content 2")
    file3.write_text("Content 3")

    with PythonInterpreter(enable_read_paths=[str(file1), str(file2), str(file3)]) as interpreter:
        code = (
            "import os\n"
            "files = sorted(os.listdir('/sandbox'))\n"
            "contents = {}\n"
            "for f in files:\n"
            "    with open(f'/sandbox/{f}') as fh:\n"
            "        contents[f] = fh.read()\n"
            "(files, contents)"
        )
        result = interpreter.execute(code)
        files, contents = result

        assert files == ["test1.txt", "test2.txt", "test3.txt"], "All three files should be mounted"
        assert contents["test1.txt"] == "Content 1"
        assert contents["test2.txt"] == "Content 2"
        assert contents["test3.txt"] == "Content 3"

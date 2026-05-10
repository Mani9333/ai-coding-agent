import pytest

from agent.tools import Toolbox, ToolError


def test_write_then_read_roundtrip(tmp_path):
    box = Toolbox(tmp_path)
    box.write_file("sub/hello.txt", "hi there")
    assert box.read_file("sub/hello.txt") == "hi there"


def test_list_dir_and_grep(tmp_path):
    box = Toolbox(tmp_path)
    box.write_file("a.py", "def foo():\n    return 42\n")
    box.write_file("b.py", "x = 1\n")
    assert "a.py" in box.list_dir(".")
    hits = box.grep(r"def \w+", ".")
    assert "a.py:1" in hits
    assert "b.py" not in hits


def test_path_traversal_is_blocked(tmp_path):
    box = Toolbox(tmp_path)
    with pytest.raises(ToolError):
        box.read_file("../../etc/passwd")
    with pytest.raises(ToolError):
        box.write_file("/tmp/escape.txt", "nope")


def test_run_shell_captures_output_and_exit(tmp_path):
    box = Toolbox(tmp_path)
    out = box.run_shell("echo hello && exit 3")
    assert "hello" in out
    assert "exit=3" in out


def test_unknown_tool_and_bad_args(tmp_path):
    box = Toolbox(tmp_path)
    with pytest.raises(ToolError):
        box.call("does_not_exist", {})
    with pytest.raises(ToolError):
        box.call("read_file", {"wrong": "arg"})

from pathlib import Path

from agent.graph import parse_action, run_agent
from agent.llm.mock import MockChatModel


def test_parse_action_tolerates_code_fences_and_prose():
    fenced = '```json\n{"action": "final", "answer": "ok"}\n```'
    assert parse_action(fenced)["answer"] == "ok"

    prose = 'Sure! {"action": "tool", "tool": "list_dir", "args": {}} done.'
    parsed = parse_action(prose)
    assert parsed["action"] == "tool" and parsed["tool"] == "list_dir"

    assert parse_action("not json at all")["action"] == "final"


def test_agent_creates_and_runs_file(tmp_path):
    result = run_agent(
        "Create a file 'greet.py' with a greet(name) function, add __main__, then run it.",
        workdir=str(tmp_path),
        model=MockChatModel(),
    )
    created = Path(tmp_path) / "greet.py"
    assert created.is_file()
    assert "def greet" in created.read_text()
    assert "greet.py" in result.result
    assert 0 < result.iterations <= 12


def test_agent_generic_task_terminates(tmp_path):
    result = run_agent(
        "Summarize what is in this directory.",
        workdir=str(tmp_path),
        model=MockChatModel(),
    )
    assert result.result
    assert result.iterations <= 12

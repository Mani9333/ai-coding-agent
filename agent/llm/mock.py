"""Deterministic, offline "model" for the coding agent.

This is **not** a real LLM. It is a small scripted policy that emits the same
JSON tool-call protocol a real model would, so the whole agent loop — planning,
tool execution, observation, finishing — runs end-to-end with **zero API keys**
and behaves identically every run (which is what makes the test-suite hermetic).

Set ``LLM_PROVIDER=openai|anthropic|ollama`` to have a real model drive the loop
instead; nothing else in the agent changes.
"""

from __future__ import annotations

import json
import re

from .base import ChatModel, Message

_PY_FILE = re.compile(r"[\w./-]+\.py")


class MockChatModel(ChatModel):
    name = "mock"

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        user_msgs = [m for m in messages if m.role == "user"]
        task = user_msgs[0].content if user_msgs else ""
        # One assistant message is appended per completed "think" step, so the
        # count tells us which step of the scripted plan we are on.
        step = sum(1 for m in messages if m.role == "assistant")
        return json.dumps(self._policy(task, step))

    def _policy(self, task: str, step: int) -> dict:
        lower = task.lower()
        wants_file = "create" in lower or "write" in lower or "file" in lower
        match = _PY_FILE.search(task)

        if wants_file and match:
            path = match.group(0)
            if step == 0:
                return {
                    "thought": f"Create {path} with the requested function and a runnable __main__.",
                    "action": "tool",
                    "tool": "write_file",
                    "args": {"path": path, "content": _greet_module()},
                }
            if step == 1:
                return {
                    "thought": f"Run {path} to confirm it executes cleanly.",
                    "action": "tool",
                    "tool": "run_shell",
                    "args": {"command": f"python {path}"},
                }
            return {
                "thought": "The file was created and ran successfully.",
                "action": "final",
                "answer": f"Created `{path}` and verified it runs (prints a greeting).",
            }

        # Generic fallback for arbitrary tasks: look around once, then report.
        if step == 0:
            return {
                "thought": "Inspect the working directory before answering.",
                "action": "tool",
                "tool": "list_dir",
                "args": {"path": "."},
            }
        return {
            "thought": "I have enough context to answer.",
            "action": "final",
            "answer": (
                "Offline mock run complete. Set LLM_PROVIDER=openai|anthropic|ollama "
                "for a real model to carry out this task."
            ),
        }


def _greet_module() -> str:
    return (
        '"""Tiny module produced by the coding agent."""\n\n\n'
        "def greet(name: str) -> str:\n"
        '    """Return a friendly greeting for ``name``."""\n'
        '    return f"Hello, {name}!"\n\n\n'
        'if __name__ == "__main__":\n'
        '    print(greet("World"))\n'
    )

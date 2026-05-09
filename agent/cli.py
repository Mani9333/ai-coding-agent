"""Command-line entrypoint: ``python -m agent "your task"``.

Streams the think/act loop to the terminal so you can watch the agent reason,
call tools, and observe results in real time.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from .graph import build_agent
from .llm import get_chat_model
from .state import AgentState

DEFAULT_TASK = (
    "Create a file 'greet.py' with a greet(name) function that returns a greeting, "
    "add a __main__ block that prints greet('World'), then run it to verify."
)

DIM, BOLD, CYAN, GREEN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"


def _p(text: str = "") -> None:
    print(text, flush=True)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="A small LangGraph coding agent.")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="what the agent should do")
    parser.add_argument("--workdir", default="./workspace", help="sandbox directory (default: ./workspace)")
    parser.add_argument("--max-iterations", type=int, default=12, help="step budget (default: 12)")
    args = parser.parse_args(argv)

    model = get_chat_model()
    workdir = os.path.abspath(args.workdir)

    _p(f"{BOLD}Coding agent{RESET} {DIM}(model: {model.name}, workspace: {workdir}){RESET}")
    _p(f"{BOLD}Task:{RESET} {args.task}\n")

    app = build_agent(model)
    initial: AgentState = {
        "task": args.task,
        "workdir": workdir,
        "transcript": [],
        "pending": None,
        "iterations": 0,
        "max_iterations": args.max_iterations,
        "result": "",
        "done": False,
    }

    step = 0
    result = ""
    for update in app.stream(initial, config={"recursion_limit": args.max_iterations * 2 + 5}):
        if "think" in update:
            step += 1
            action = update["think"].get("pending") or {}
            thought = action.get("thought", "")
            if thought:
                _p(f"{CYAN}[{step}] think{RESET} {DIM}{thought}{RESET}")
            if action.get("action") == "tool":
                _p(f"    {YELLOW}→ {action.get('tool')}{RESET}({_fmt_args(action.get('args', {}))})")
        elif "act" in update:
            obs = (update["act"].get("transcript") or [{}])[-1].get("content", "")
            _p(f"    {DIM}{_indent(obs)}{RESET}")
        elif "finalize" in update:
            result = update["finalize"].get("result", "")

    _p(f"\n{GREEN}{BOLD}✓ Done{RESET} {result}")
    return 0


def _fmt_args(args: dict) -> str:
    parts = []
    for key, value in args.items():
        text = value if isinstance(value, str) else str(value)
        if len(text) > 48:
            text = text[:48] + "…"
        parts.append(f"{key}={text!r}")
    return ", ".join(parts)


def _indent(text: str, prefix: str = "    ") -> str:
    return ("\n" + prefix).join(text.splitlines())


if __name__ == "__main__":
    sys.exit(main())

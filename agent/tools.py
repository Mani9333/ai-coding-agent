"""The agent's tools, each jailed to a single working directory.

Every path the model supplies is resolved and checked to be *inside* the
workspace root, so a task can never read or clobber files elsewhere on the
machine (``..`` traversal and absolute paths are rejected). ``run_shell`` is the
one genuinely powerful tool; it runs with the workspace as its cwd, a wall-clock
timeout, and truncated output. See the README's "Safety model" section for the
tradeoffs.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 4000
SHELL_TIMEOUT_SECONDS = 30


class ToolError(Exception):
    """Raised for a bad tool call (unknown tool, unsafe path, bad args)."""


@dataclass
class Toolbox:
    """A set of tools bound to one sandboxed ``root`` directory."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path safety -------------------------------------------------------
    def _safe(self, rel: str) -> Path:
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"path {rel!r} escapes the sandbox root")
        return candidate

    # -- tools -------------------------------------------------------------
    def read_file(self, path: str) -> str:
        target = self._safe(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        return _truncate(target.read_text(encoding="utf-8", errors="replace"))

    def write_file(self, path: str, content: str) -> str:
        target = self._safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def list_dir(self, path: str = ".") -> str:
        target = self._safe(path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = sorted(
            (p.name + ("/" if p.is_dir() else "")) for p in target.iterdir()
        )
        return "\n".join(entries) if entries else "(empty)"

    def grep(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:  # noqa: BLE001
            raise ToolError(f"invalid regex: {exc}") from exc
        base = self._safe(path)
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        hits: list[str] = []
        for file in files:
            try:
                for i, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{file.relative_to(self.root)}:{i}: {line.strip()}")
            except OSError:
                continue
        return _truncate("\n".join(hits)) if hits else "(no matches)"

    def run_shell(self, command: str) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=SHELL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"command timed out after {SHELL_TIMEOUT_SECONDS}s"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return _truncate(f"exit={proc.returncode}\n{out}".strip())

    # -- dispatch ----------------------------------------------------------
    def call(self, tool: str, args: dict) -> str:
        handlers = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_dir": self.list_dir,
            "grep": self.grep,
            "run_shell": self.run_shell,
        }
        if tool not in handlers:
            raise ToolError(f"unknown tool: {tool}")
        try:
            return handlers[tool](**args)
        except ToolError:
            raise
        except TypeError as exc:  # wrong/missing args
            raise ToolError(f"bad arguments for {tool}: {exc}") from exc


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


TOOL_SPECS = {
    "read_file": "read_file(path) — return the file's contents",
    "write_file": "write_file(path, content) — create/overwrite a file",
    "list_dir": "list_dir(path='.') — list a directory",
    "grep": "grep(pattern, path='.') — search files for a regex",
    "run_shell": "run_shell(command) — run a shell command in the workspace",
}

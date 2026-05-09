"""The agent as a LangGraph state machine — a classic think → act loop (ReAct).

    START → think ─(final | budget spent)→ finalize → END
              ▲                │
              └──── act ◀──────┘ (tool call)

``think`` asks the model for the next JSON action; ``act`` executes the tool and
feeds the observation back; ``finalize`` returns the answer. Splitting think and
act keeps the control flow explicit and easy to inspect — the graph *is* the
agent's policy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from .llm import Message, get_chat_model
from .llm.base import ChatModel
from .prompts import SYSTEM_PROMPT
from .state import AgentState
from .tools import Toolbox, ToolError

_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(raw: str) -> dict:
    """Best-effort parse of the model's reply into an action dict.

    Tolerates code fences and surrounding prose. Anything unparseable is treated
    as a plain final answer so the loop always terminates gracefully.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^(json)?\s*", "", text, flags=re.IGNORECASE)
    for candidate in (text, (_JSON_SPAN.search(text) or _Empty()).group()):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except (json.JSONDecodeError, AttributeError):
            continue
    return {"action": "final", "answer": raw.strip()}


class _Empty:
    def group(self) -> str:
        return ""


def build_agent(model: ChatModel | None = None):
    """Compile and return the agent graph. Inject a model for tests."""
    model = model or get_chat_model()

    def think(state: AgentState) -> dict:
        messages = [Message("system", SYSTEM_PROMPT), Message("user", state["task"])]
        for turn in state.get("transcript", []):
            messages.append(Message(turn["role"], turn["content"]))
        raw = model.complete(messages)
        action = parse_action(raw)
        return {
            "pending": action,
            "transcript": [{"role": "assistant", "content": raw}],
            "iterations": state.get("iterations", 0) + 1,
        }

    def act(state: AgentState) -> dict:
        action = state["pending"] or {}
        toolbox = Toolbox(state["workdir"])
        tool = action.get("tool", "")
        args = action.get("args", {}) or {}
        try:
            observation = toolbox.call(tool, args)
        except ToolError as exc:
            observation = f"ERROR: {exc}"
        return {"transcript": [{"role": "user", "content": f"OBSERVATION ({tool}):\n{observation}"}]}

    def finalize(state: AgentState) -> dict:
        action = state.get("pending") or {}
        if action.get("action") == "final":
            result = action.get("answer", "(no answer)")
        else:
            result = f"Stopped after reaching the {state['max_iterations']}-step budget."
        return {"result": result, "done": True}

    def route(state: AgentState) -> str:
        action = state.get("pending") or {}
        if action.get("action") == "final":
            return "finalize"
        if state.get("iterations", 0) >= state.get("max_iterations", 12):
            return "finalize"
        return "act"

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", route, {"act": "act", "finalize": "finalize"})
    graph.add_edge("act", "think")
    graph.add_edge("finalize", END)
    return graph.compile()


@dataclass
class AgentResult:
    result: str
    iterations: int
    transcript: list[dict]


def run_agent(
    task: str,
    workdir: str,
    *,
    max_iterations: int = 12,
    model: ChatModel | None = None,
) -> AgentResult:
    """Run the agent to completion and return the outcome."""
    app = build_agent(model)
    initial: AgentState = {
        "task": task,
        "workdir": workdir,
        "transcript": [],
        "pending": None,
        "iterations": 0,
        "max_iterations": max_iterations,
        "result": "",
        "done": False,
    }
    final = app.invoke(initial, config={"recursion_limit": max_iterations * 2 + 5})
    return AgentResult(
        result=final.get("result", ""),
        iterations=final.get("iterations", 0),
        transcript=final.get("transcript", []),
    )

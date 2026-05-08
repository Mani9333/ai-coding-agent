"""Shared state for the agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    task: str
    workdir: str
    # Running transcript of assistant actions and tool observations, fed back
    # to the model each turn. ``operator.add`` makes returns append, not replace.
    transcript: Annotated[list[dict], operator.add]
    pending: dict | None  # the action parsed by the last "think" step
    iterations: int
    max_iterations: int
    result: str
    done: bool

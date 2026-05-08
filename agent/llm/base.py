"""Minimal, provider-agnostic chat interface.

The interface is deliberately tiny — plain text in, plain text out — so that a
deterministic offline model, a local Ollama model, and hosted APIs
(OpenAI / Anthropic) are fully interchangeable. Any structured protocol on top
(e.g. the JSON tool-call format the agent uses) is owned by the caller, not by
the model layer. That keeps providers simple and the swap between them a pure
configuration change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A single chat message. ``role`` is one of system | user | assistant."""

    role: str
    content: str


class ChatModel(ABC):
    """Text-in / text-out chat model."""

    name: str = "chat"

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant's reply as plain text."""
        raise NotImplementedError

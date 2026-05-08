"""Select a chat model from the ``LLM_PROVIDER`` environment variable.

Default is ``mock`` so the agent runs offline with no configuration. This is the
only place that knows about concrete providers.
"""

from __future__ import annotations

import os

from .base import ChatModel
from .mock import MockChatModel
from .providers import AnthropicChatModel, OpenAIChatModel, OllamaChatModel


def get_chat_model() -> ChatModel:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider in ("", "mock", "offline"):
        return MockChatModel()
    if provider == "openai":
        return OpenAIChatModel()
    if provider == "anthropic":
        return AnthropicChatModel()
    if provider == "ollama":
        return OllamaChatModel()
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: mock, openai, anthropic, ollama."
    )

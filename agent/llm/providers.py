"""Hosted / local model adapters.

Each adapter speaks the provider's HTTP API directly via ``httpx`` — no heavy
vendor SDKs, so installs stay small and there is nothing to keep in sync with
fast-moving client libraries. All three implement the same tiny
:class:`~agent.llm.base.ChatModel` contract, so higher layers never branch on
the provider.
"""

from __future__ import annotations

import os

import httpx

from .base import ChatModel, Message


class OpenAIChatModel(ChatModel):
    """OpenAI (or any OpenAI-compatible ``/chat/completions`` endpoint)."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (needed for LLM_PROVIDER=openai).")
        self.timeout = timeout

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicChatModel(ChatModel):
    """Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for LLM_PROVIDER=anthropic).")
        self.timeout = timeout

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        payload: dict = {
            "model": self.model,
            "messages": chat,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return "".join(block.get("text", "") for block in resp.json().get("content", []))


class OllamaChatModel(ChatModel):
    """Local models served by Ollama (``ollama serve``)."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

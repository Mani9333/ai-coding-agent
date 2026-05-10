import pytest

from agent.llm import Message, get_chat_model
from agent.llm.mock import MockChatModel


def test_default_provider_is_mock(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(get_chat_model(), MockChatModel)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    with pytest.raises(ValueError):
        get_chat_model()


def test_openai_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_chat_model()


def test_mock_emits_valid_json_protocol():
    model = MockChatModel()
    reply = model.complete([Message("user", "Create a file foo.py and run it")])
    assert '"action"' in reply

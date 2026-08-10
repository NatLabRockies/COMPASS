"""Tests for LLM configuration helpers"""

from pathlib import Path

import pytest

from compass.llm.config import OpenAIConfig


def test_openai_client_kwargs_loaded_from_env(monkeypatch):
    """OpenAI kwargs can be populated from OPENAI_* env vars"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://litellm.example.gov")

    config = OpenAIConfig(name="gpt-4o-mini", client_type="openai")

    assert config.client_kwargs["api_key"] == "test-openai-key"
    assert config.client_kwargs["base_url"] == "https://litellm.example.gov"


def test_openai_client_kwargs_user_values_take_precedence(monkeypatch):
    """Explicit client kwargs should not be replaced by env vars"""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example")

    config = OpenAIConfig(
        name="gpt-4o-mini",
        client_type="openai",
        client_kwargs={
            "api_key": "user-key",
            "base_url": "https://user.example",
        },
    )

    assert config.client_kwargs["api_key"] == "user-key"
    assert config.client_kwargs["base_url"] == "https://user.example"


def test_azure_client_kwargs_unchanged(monkeypatch):
    """Azure env var mapping remains unchanged"""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")

    config = OpenAIConfig(name="gpt-4o-mini", client_type="azure")

    assert config.client_kwargs["api_key"] == "azure-key"
    assert config.client_kwargs["api_version"] == "2024-02-15-preview"
    assert config.client_kwargs["azure_endpoint"] == "https://azure.example"
    assert "base_url" not in config.client_kwargs


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])

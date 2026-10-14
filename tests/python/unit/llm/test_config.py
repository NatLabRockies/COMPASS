"""Tests for LLM configuration helpers"""

from pathlib import Path

import pytest

from compass.llm.config import OpenAIConfig, LiteLLMConfig
from compass.services.litellm import LiteLLMClient
from compass.services.openai import OpenAIService


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


@pytest.fixture
def no_proxy_env(monkeypatch):
    """Clear proxy env vars so tests control endpoint resolution"""
    for env_var in (
        "LITELLM_API_BASE",
        "LITELLM_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)


def test_litellm_config_reuses_openai_service(no_proxy_env):
    """LiteLLMConfig builds the shared OpenAIService with a litellm client"""
    config = LiteLLMConfig(
        name="openai/gpt-4o-mini",
        client_kwargs={"api_key": "k", "api_base": "https://proxy"},
    )

    assert isinstance(config, OpenAIConfig)
    assert config.client_type == "litellm"
    assert config.client_kwargs == {
        "api_key": "k",
        "api_base": "https://proxy",
    }

    service = config.llm_service
    assert isinstance(service, OpenAIService)
    assert isinstance(service.client, LiteLLMClient)
    assert service.client.client_kwargs == config.client_kwargs
    assert service.model_name == "openai/gpt-4o-mini"
    assert service.rate_limit == config.llm_service_rate_limit


def test_litellm_config_proxy_env_fallbacks(no_proxy_env, monkeypatch):
    """Proxy endpoint and key are read from LITELLM_* then OPENAI_*"""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")

    config = LiteLLMConfig(name="anthropic/claude-sonnet-4-5")

    assert config.client_kwargs == {
        "api_base": "https://openai-proxy",
        "api_key": "litellm-key",
    }


def test_litellm_config_no_proxy_key_without_api_base(
    no_proxy_env, monkeypatch
):
    """Proxy keys are never forwarded when calling a provider directly"""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")

    config = LiteLLMConfig(name="anthropic/claude-sonnet-4-5")

    assert config.client_kwargs == {}


def test_litellm_config_user_kwargs_take_precedence(no_proxy_env, monkeypatch):
    """Explicit client kwargs are not replaced by env vars"""
    monkeypatch.setenv("LITELLM_API_BASE", "https://env-proxy")
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")

    config = LiteLLMConfig(
        name="openai/gpt-5-mini",
        client_kwargs={"api_key": "user-key", "api_base": "https://user"},
    )

    assert config.client_kwargs == {
        "api_key": "user-key",
        "api_base": "https://user",
    }


def test_litellm_config_duplicate_names_get_unique_tags(no_proxy_env):
    """Duplicate model names receive distinct auto-generated tags"""
    first = LiteLLMConfig(name="anthropic/claude-sonnet-4-5")
    second = LiteLLMConfig(name="anthropic/claude-sonnet-4-5")

    assert first.llm_service.name != second.llm_service.name


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])

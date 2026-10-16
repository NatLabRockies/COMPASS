"""Tests for LLM configuration helpers"""

from pathlib import Path

import pytest
import openai

from compass.llm.config import OpenAIConfig, LiteLLMConfig
from compass.services.openai import OpenAIService


def test_openai_client_kwargs_loaded_from_env(monkeypatch):
    """OpenAI kwargs can be populated from OPENAI_* env vars"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://litellm.example.gov")

    config = OpenAIConfig(name="gpt-4o-mini", client_type="openai")

    assert config.client_kwargs["api_key"] == "test-openai-key"
    assert config.client_kwargs["base_url"] == "https://litellm.example.gov"


def test_openai_client_kwargs_ignore_litellm_proxy_env(monkeypatch):
    """OpenAI config only reads OPENAI_* env vars"""
    monkeypatch.setenv("LITELLM_API_KEY", "test-litellm-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm-proxy.example")

    config = OpenAIConfig(name="gpt-4o-mini", client_type="openai")

    assert config.client_kwargs["api_key"] is None
    assert config.client_kwargs["base_url"] is None


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
        "LITELLM_BASE_URL",
        "LITELLM_API_BASE",
        "LITELLM_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)


def test_litellm_config_reuses_openai_service(no_proxy_env):
    """LiteLLMConfig builds the shared OpenAIService with a litellm client"""
    config = LiteLLMConfig(
        name="gpt-4o-mini",
        client_kwargs={"api_key": "k", "base_url": "https://proxy"},
    )

    assert isinstance(config, OpenAIConfig)
    assert config.client_type == "litellm"
    assert config.client_kwargs == {
        "api_key": "k",
        "base_url": "https://proxy",
    }

    service = config.llm_service
    assert isinstance(service, OpenAIService)
    assert isinstance(service.client, openai.AsyncOpenAI)
    assert service.model_name == "gpt-4o-mini"
    assert service.rate_limit == config.llm_service_rate_limit


def test_litellm_config_reads_litellm_proxy_env(no_proxy_env, monkeypatch):
    """LiteLLM proxy env vars are read from the LiteLLM namespace"""
    monkeypatch.setenv("LITELLM_API_BASE", "https://litellm-proxy")
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")

    config = LiteLLMConfig(name="anthropic/claude-sonnet-4-5")

    assert config.client_kwargs == {
        "base_url": "https://litellm-proxy",
        "api_key": "litellm-key",
    }


def test_litellm_config_prefers_litellm_base_url_alias(no_proxy_env, monkeypatch):
    """LiteLLM proxy prefers the base-url alias used in local env files"""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm-proxy")
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")

    config = LiteLLMConfig(name="anthropic/claude-sonnet-4-5")

    assert config.client_kwargs == {
        "base_url": "https://litellm-proxy",
        "api_key": "litellm-key",
    }


def test_litellm_config_requires_complete_litellm_env_pair(
    no_proxy_env, monkeypatch
):
    """LiteLLM proxy env fallback rejects incomplete proxy credentials"""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm-proxy")

    config = LiteLLMConfig(name="claude-haiku-4-5")

    with pytest.raises(
        ValueError, match="require both LITELLM_BASE_URL"
    ):
        _ = config.client_kwargs


def test_litellm_config_falls_back_to_legacy_openai_env(no_proxy_env, monkeypatch):
    """LiteLLM configs remain backward compatible with legacy OPENAI_* envs"""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy-proxy")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    config = LiteLLMConfig(name="proxy/model")

    assert config.client_kwargs == {
        "base_url": "https://legacy-proxy",
        "api_key": "legacy-key",
    }


def test_litellm_config_requires_complete_legacy_openai_pair(
    no_proxy_env, monkeypatch
):
    """Legacy proxy fallback rejects partial OPENAI_* credentials"""
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    config = LiteLLMConfig(name="gpt-5.4-mini")

    with pytest.raises(
        ValueError, match="requires both OPENAI_BASE_URL and OPENAI_API_KEY"
    ):
        _ = config.client_kwargs


def test_litellm_config_user_kwargs_take_precedence(no_proxy_env, monkeypatch):
    """Explicit client kwargs are not replaced by env vars"""
    config = LiteLLMConfig(
        name="proxy/model",
        client_kwargs={"api_key": "user-key", "api_base": "https://user"},
    )

    assert config.client_kwargs == {
        "api_key": "user-key",
        "base_url": "https://user",
    }


def test_litellm_config_requires_key_for_explicit_base_url(
    no_proxy_env, monkeypatch
):
    """Explicit proxy URLs do not silently borrow unrelated env keys"""
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    config = LiteLLMConfig(
        name="gpt-5.4-mini", client_kwargs={"base_url": "https://proxy"}
    )

    with pytest.raises(ValueError, match="explicit base_url must also provide"):
        _ = config.client_kwargs


def test_litellm_config_duplicate_names_get_unique_tags(no_proxy_env):
    """Duplicate model names receive distinct auto-generated tags"""
    kwargs = {"api_key": "test-key", "base_url": "https://proxy"}
    first = LiteLLMConfig(name="gpt-5.4-mini", client_kwargs=kwargs)
    second = LiteLLMConfig(name="gpt-5.4-mini", client_kwargs=kwargs)

    assert first.llm_service.name != second.llm_service.name


def test_litellm_config_accepts_bare_proxy_alias(no_proxy_env):
    """Bare proxy aliases are forwarded unchanged to LiteLLM"""
    config = LiteLLMConfig(name="claude-sonnet-4-5")

    assert config.name == "claude-sonnet-4-5"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])

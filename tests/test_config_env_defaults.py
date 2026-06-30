"""Tests for the latent env-var-default bug in Memento config classes.

`os.environ.get(key, default)` returns `default` ONLY when the key is
missing — when the key is unset, it returns `None` instead of `default`.
This silently overwrites dataclass defaults with None.

These tests verify that all Memento config classes preserve their dataclass
defaults when env vars are unset, and only override when env vars are
explicitly set.
"""

from __future__ import annotations

import os
from unittest.mock import patch


def _clear_memento_env():
    """Return a copy of os.environ with all MEMENTO_* keys removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MEMENTO_")
    }


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------


def test_embedding_config_preserves_dataclass_default_when_env_unset():
    """When MEMENTO_EMBEDDING_PROVIDER is unset, EmbeddingConfig must keep
    its dataclass default of 'auto' — NOT become None (the latent bug)."""
    from memento.config import EmbeddingConfig

    with patch.dict(os.environ, _clear_memento_env(), clear=True):
        cfg = EmbeddingConfig()
        assert cfg.provider == "auto", (
            f"Expected 'auto', got {cfg.provider!r}. "
            f"This indicates the latent env-var-default bug."
        )
        assert cfg.model == "all-MiniLM-L6-v2"
        assert cfg.dimension is None


def test_embedding_config_overrides_when_env_set():
    """When MEMENTO_EMBEDDING_PROVIDER is set, EmbeddingConfig uses it."""
    from memento.config import EmbeddingConfig

    with patch.dict(
        os.environ,
        {"MEMENTO_EMBEDDING_PROVIDER": "openai-compatible"},
        clear=True,
    ):
        cfg = EmbeddingConfig()
        assert cfg.provider == "openai-compatible"


def test_embedding_config_overrides_with_explicit_constructor_value():
    """Explicit kwargs must survive even when env vars are unset."""
    from memento.config import EmbeddingConfig

    with patch.dict(os.environ, _clear_memento_env(), clear=True):
        cfg = EmbeddingConfig(provider="sentence-transformers", dimension=384)
        assert cfg.provider == "sentence-transformers"
        assert cfg.dimension == 384


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


def test_llm_config_preserves_dataclass_default_when_env_unset():
    """LLMConfig defaults to empty-string provider — must not become None."""
    from memento.config import LLMConfig

    with patch.dict(os.environ, _clear_memento_env(), clear=True):
        cfg = LLMConfig()
        assert cfg.provider == "", (
            f"Expected '', got {cfg.provider!r}. "
            f"This indicates the latent env-var-default bug."
        )
        assert cfg.api_key == ""
        assert cfg.base_url == ""
        assert cfg.extraction_model == ""
        assert cfg.tiebreaker_model == ""
        assert cfg.chat_model == ""


def test_llm_config_overrides_when_env_set():
    from memento.config import LLMConfig

    with patch.dict(
        os.environ,
        {
            "MEMENTO_LLM_PROVIDER": "openai",
            "MEMENTO_LLM_API_KEY": "test-key",
            "MEMENTO_LLM_BASE_URL": "https://api.together.xyz/v1",
            "MEMENTO_EXTRACTION_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "MEMENTO_TIEBREAKER_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "MEMENTO_CHAT_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        },
        clear=True,
    ):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.api_key == "test-key"
        assert cfg.base_url == "https://api.together.xyz/v1"
        assert cfg.extraction_model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert cfg.tiebreaker_model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert cfg.chat_model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def test_llm_config_explicit_kwargs_survive_when_env_unset():
    """Explicit LLMConfig(provider='anthropic') must not become None when
    MEMENTO_LLM_PROVIDER is unset."""
    from memento.config import LLMConfig

    with patch.dict(os.environ, _clear_memento_env(), clear=True):
        cfg = LLMConfig(provider="anthropic", api_key="sk-test")
        assert cfg.provider == "anthropic"
        assert cfg.api_key == "sk-test"


# ---------------------------------------------------------------------------
# RerankerConfig
# ---------------------------------------------------------------------------


def test_reranker_config_preserves_dataclass_default_when_env_unset():
    """RerankerConfig defaults to provider='auto' — must not become None."""
    from memento.config import RerankerConfig

    with patch.dict(os.environ, _clear_memento_env(), clear=True):
        cfg = RerankerConfig()
        assert cfg.provider == "auto", (
            f"Expected 'auto', got {cfg.provider!r}. "
            f"This indicates the latent env-var-default bug."
        )
        assert cfg.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.enabled is True


def test_reranker_config_overrides_when_env_set():
    from memento.config import RerankerConfig

    with patch.dict(
        os.environ,
        {
            "MEMENTO_RERANKER_PROVIDER": "cross-encoder",
            "MEMENTO_RERANKER_MODEL": "BAAI/bge-reranker-base",
            "MEMENTO_RERANKER_ENABLED": "false",
        },
        clear=True,
    ):
        cfg = RerankerConfig()
        assert cfg.provider == "cross-encoder"
        assert cfg.model == "BAAI/bge-reranker-base"
        assert cfg.enabled is False


def test_reranker_config_enabled_truthy_values():
    """enabled accepts common truthy/falsy string values."""
    from memento.config import RerankerConfig

    truthy_cases = ["1", "true", "yes", "on", "TRUE", "True"]
    falsy_cases = ["0", "false", "no", "off", "FALSE"]

    for val in truthy_cases:
        with patch.dict(
            os.environ, {"MEMENTO_RERANKER_ENABLED": val}, clear=True
        ):
            cfg = RerankerConfig()
            assert cfg.enabled is True, f"Expected True for {val!r}"

    for val in falsy_cases:
        with patch.dict(
            os.environ, {"MEMENTO_RERANKER_ENABLED": val}, clear=True
        ):
            cfg = RerankerConfig()
            assert cfg.enabled is False, f"Expected False for {val!r}"


# ---------------------------------------------------------------------------
# MementoConfig (full config assembly)
# ---------------------------------------------------------------------------


def test_full_memento_config_preserves_defaults_when_env_unset():
    """The full MementoConfig must assemble correctly even when no env vars
    are set — exercising every config class together."""
    from memento.config import MementoConfig

    with patch.dict(os.environ, _clear_memento_env(), clear=True):
        cfg = MementoConfig()
        assert cfg.embedding.provider == "auto"
        assert cfg.embedding.dimension is None
        assert cfg.llm.provider == ""
        assert cfg.reranker.provider == "auto"
        assert cfg.reranker.enabled is True
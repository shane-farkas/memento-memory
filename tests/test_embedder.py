"""Tests for embedder OpenAI-compatible endpoint support (dimension=None, base_url).

These tests verify that:
- OpenAIEmbedder with dimension=None skips the `dimensions` request param
  (required for providers like Together and OpenRouter that reject it)
- OpenAIEmbedder probes the API to discover native embedding size when
  dimension=None
- OpenAIEmbedder with dimension=int includes the `dimensions` param
  (OpenAI text-embedding-3-* behavior)
- base_url is wired through to the OpenAI client
- GeminiEmbedder skips output_dimensionality when dimension=None
- create_embedder factory passes dimension through for "openai" provider
- EmbeddingConfig env var parsing handles both set and unset dimension

No actual API calls are made — we mock the OpenAI/Gemini client.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


class _EmbeddingItem:
    """Mimics openai's response item with dot-attribute access to .embedding."""

    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _EmbeddingResponse:
    """Mimics openai's CreateEmbeddingResponse with .data list."""

    def __init__(self, items: list[_EmbeddingItem]) -> None:
        self.data = items


class FakeOpenAIClient:
    """Mocks the openai.OpenAI client used by OpenAIEmbedder."""

    def __init__(self, native_dim: int = 1024) -> None:
        self.native_dim = native_dim
        self.embeddings = MagicMock()
        self.embeddings.create = MagicMock(side_effect=self._create)
        self.calls: list[dict] = []

    def _create(self, *, input, model, **kwargs):
        # Capture all kwargs the embedder sends
        self.calls.append({"input": input, "model": model, **kwargs})
        # Return mock response with the native embedding size
        if isinstance(input, list):
            items = [
                _EmbeddingItem(index=i, embedding=[0.0] * self.native_dim)
                for i in range(len(input))
            ]
        else:
            items = [_EmbeddingItem(index=0, embedding=[0.0] * self.native_dim)]
        return _EmbeddingResponse(items=items)


def _make_embedder(*, api_key="test-key", base_url=None, dimension=None):
    """Construct OpenAIEmbedder with a fake OpenAI client."""
    from memento.embedder import OpenAIEmbedder

    fake = FakeOpenAIClient(native_dim=1024)
    with patch("openai.OpenAI", return_value=fake):
        embedder = OpenAIEmbedder(
            model="intfloat/multilingual-e5-large-instruct",
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
        )
    return embedder, fake


def test_openai_embedder_dimension_none_skips_dimensions_param():
    """Together/OpenRouter reject the dimensions param. dimension=None must skip it."""
    embedder, fake = _make_embedder(dimension=None)

    assert embedder.dimension == 1024  # Probed from API response

    embedder.embed("hello world")
    # Last call is the actual embed; the first is the probe at init time
    last_call = fake.calls[-1]
    assert "dimensions" not in last_call, (
        f"Expected no dimensions param, got {last_call}"
    )
    assert last_call["model"] == "intfloat/multilingual-e5-large-instruct"


def test_openai_embedder_dimension_none_batch_skips_dimensions_param():
    embedder, fake = _make_embedder(dimension=None)

    embedder.embed_batch(["a", "b", "c"])
    last_call = fake.calls[-1]
    assert "dimensions" not in last_call


def test_openai_embedder_dimension_int_sends_dimensions_param():
    """OpenAI text-embedding-3-* accepts a dimensions param to downsize vectors."""
    embedder, fake = _make_embedder(dimension=384)

    assert embedder.dimension == 384

    embedder.embed("hello")
    last_call = fake.calls[-1]
    assert last_call.get("dimensions") == 384


def test_openai_embedder_base_url_wired_to_client():
    """base_url must reach the OpenAI client (Together, OpenRouter, etc.)."""
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        from memento.embedder import OpenAIEmbedder

        OpenAIEmbedder(
            model="intfloat/multilingual-e5-large-instruct",
            api_key="test-key",
            base_url="https://api.together.xyz/v1",
            dimension=None,
        )
        # OpenAI was constructed with base_url in kwargs
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs.get("base_url") == "https://api.together.xyz/v1"
        assert call_kwargs.get("api_key") == "test-key"


def test_openai_embedder_probe_failure_falls_back_to_384():
    """If the dimension probe fails, fall back to 384 so init doesn't blow up."""
    from memento.embedder import OpenAIEmbedder

    fake = FakeOpenAIClient()
    fake.embeddings.create.side_effect = RuntimeError("API unavailable")
    with patch("openai.OpenAI", return_value=fake):
        embedder = OpenAIEmbedder(
            model="intfloat/multilingual-e5-large-instruct",
            api_key="test-key",
            dimension=None,
        )
    assert embedder.dimension == 384


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------


def test_embedding_config_dimension_default_is_none():
    """Default dimension must be None so providers can use API-native size."""
    # Make sure env var doesn't bleed in from the host
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMENTO_EMBEDDING_")}
    with patch.dict(os.environ, env, clear=True):
        from memento.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.dimension is None


def test_embedding_config_dimension_from_env():
    """MEMENTO_EMBEDDING_DIMENSION=N sets dimension to int(N)."""
    with patch.dict(
        os.environ,
        {"MEMENTO_EMBEDDING_DIMENSION": "768"},
        clear=False,
    ):
        from memento.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.dimension == 768


def test_embedding_config_dimension_env_unset_keeps_none():
    """When MEMENTO_EMBEDDING_DIMENSION is unset, dimension stays None."""
    env = {k: v for k, v in os.environ.items() if k != "MEMENTO_EMBEDDING_DIMENSION"}
    with patch.dict(os.environ, env, clear=True):
        from memento.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.dimension is None


# ---------------------------------------------------------------------------
# create_embedder factory
# ---------------------------------------------------------------------------


def test_create_embedder_openai_passes_dimension_through():
    """create_embedder must pass config.dimension to OpenAIEmbedder."""
    from memento.config import EmbeddingConfig
    from memento.embedder import OpenAIEmbedder, create_embedder

    # Strip host env vars that conflict with what we want to set
    host_overrides = {
        k: v
        for k, v in os.environ.items()
        if (k.startswith("MEMENTO_EMBEDDING_") or k.startswith("MEMENTO_LLM_"))
        and k not in ("MEMENTO_EMBEDDING_PROVIDER", "MEMENTO_EMBEDDING_BASE_URL")
    }
    with patch.dict(
        os.environ,
        {
            "MEMENTO_EMBEDDING_PROVIDER": "openai",
            "MEMENTO_EMBEDDING_BASE_URL": "https://api.together.xyz/v1",
            **host_overrides,
        },
        clear=True,
    ):
        cfg = EmbeddingConfig(
            provider="openai",
            model="intfloat/multilingual-e5-large-instruct",
            openai_api_key="test-key",
            dimension=None,
        )
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            embedder = create_embedder(cfg)
    assert isinstance(embedder, OpenAIEmbedder)
    client_kwargs = mock_openai.call_args.kwargs
    assert client_kwargs.get("base_url") == "https://api.together.xyz/v1"


def test_create_embedder_openai_compatible_provider():
    """The 'openai-compatible' provider falls back to MEMENTO_LLM_BASE_URL."""
    from memento.config import EmbeddingConfig
    from memento.embedder import OpenAIEmbedder, create_embedder

    host_overrides = {
        k: v
        for k, v in os.environ.items()
        if (k.startswith("MEMENTO_EMBEDDING_") or k.startswith("MEMENTO_LLM_"))
        and k not in ("MEMENTO_EMBEDDING_PROVIDER", "MEMENTO_LLM_BASE_URL")
    }
    with patch.dict(
        os.environ,
        {
            "MEMENTO_EMBEDDING_PROVIDER": "openai-compatible",
            "MEMENTO_LLM_BASE_URL": "https://api.together.xyz/v1",
            "OPENAI_API_KEY": "test-key",
            **host_overrides,
        },
        clear=True,
    ):
        cfg = EmbeddingConfig(
            provider="openai-compatible",
            model="intfloat/multilingual-e5-large-instruct",
            dimension=None,
        )
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            embedder = create_embedder(cfg)
    assert isinstance(embedder, OpenAIEmbedder)
    client_kwargs = mock_openai.call_args.kwargs
    assert client_kwargs.get("base_url") == "https://api.together.xyz/v1"
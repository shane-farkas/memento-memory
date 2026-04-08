"""Embedding protocol and implementations for Memento."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from memento.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for text embedding providers."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers (local, no API calls)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install memento-memory[local-embeddings]"
            )
        logger.info("Loading sentence-transformer model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embeddings[i] for i in range(len(texts))]


class OpenAIEmbedder:
    """Embedder using OpenAI's embedding API."""

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: str | None = None
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for API embeddings. "
                "Install with: pip install openai"
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        # text-embedding-3-small default is 1536, but we request 384 for consistency
        self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            input=text, model=self._model, dimensions=self._dimension
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        response = self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self._dimension
        )
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory function to create the configured embedder."""
    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=config.model)
    elif config.provider == "openai":
        return OpenAIEmbedder(model=config.model, api_key=config.openai_api_key)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")

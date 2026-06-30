"""Cross-encoder reranker for retrieval candidates.

A reranker rescores a small candidate pool against the query with a
cross-encoder (query and document attend to each other jointly), which is far
more precise than the bi-encoder cosine similarity used for first-stage recall.

The reranker is optional: if the model cannot be loaded (sentence-transformers
not installed, no network for the weights, etc.) ``create_reranker`` returns
``None`` and retrieval continues unchanged.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Protocol

from memento.config import RerankerConfig

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Protocol for cross-encoder rerankers."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...


def sigmoid(x: float) -> float:
    """Squash an unbounded relevance logit into (0, 1) for blending."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Cross-encoder weights are expensive to load, so cache one instance per model
# name for the lifetime of the process. The benchmark builds a fresh
# MemoryStore per question (and may run many in parallel); without this cache
# every store would reload the model. The lock guards the load path so parallel
# workers don't all miss the cache at once and load N copies simultaneously.
_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class CrossEncoderReranker:
    """Reranker backed by a sentence-transformers CrossEncoder (local, no API)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        # Double-checked locking: the fast path skips the lock once the model is
        # cached; the slow path serializes the one-time load across threads.
        if model_name not in _MODEL_CACHE:
            with _MODEL_CACHE_LOCK:
                if model_name not in _MODEL_CACHE:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading cross-encoder reranker: %s", model_name)
                    _MODEL_CACHE[model_name] = CrossEncoder(model_name)
        self._model = _MODEL_CACHE[model_name]

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance logit per document (higher = more relevant)."""
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def create_reranker(config: RerankerConfig) -> Reranker | None:
    """Create the configured reranker, or ``None`` if reranking is unavailable.

    Loading never raises: a missing dependency or download failure logs a
    warning and disables reranking rather than breaking retrieval.
    """
    if not config.enabled or config.provider == "none":
        return None

    if config.provider in ("auto", "cross-encoder"):
        try:
            return CrossEncoderReranker(config.model)
        except Exception as e:  # ImportError, download/network errors, etc.
            logger.warning(
                "Reranker unavailable (%s); continuing without reranking", e
            )
            return None

    logger.warning("Unknown reranker provider %r; reranking disabled", config.provider)
    return None

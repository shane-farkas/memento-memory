"""Configuration for Memento memory system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_db_path() -> Path:
    return Path(os.environ.get("MEMENTO_DB_PATH", "~/.memento/memento.db")).expanduser()


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model.

    `dimension` controls storage size AND, when set, is sent to the API as
    the `dimensions` (OpenAI) or `output_dimensionality` (Gemini) request
    param. Set `dimension=None` (default if MEMENTO_EMBEDDING_DIMENSION is
    not set) to use the API's native embedding size — required for providers
    like Together and OpenRouter that don't accept a dimensions param.
    """

    provider: str = "auto"  # "auto", "sentence-transformers", or "openai"
    model: str = "all-MiniLM-L6-v2"
    dimension: int | None = None
    openai_api_key: str | None = None

    def __post_init__(self) -> None:
        # Use `or default` (not get(key, default)) so unset env vars don't
        # silently overwrite dataclass defaults with None.
        self.provider = os.environ.get("MEMENTO_EMBEDDING_PROVIDER") or self.provider
        self.model = os.environ.get("MEMENTO_EMBEDDING_MODEL") or self.model
        dim = os.environ.get("MEMENTO_EMBEDDING_DIMENSION")
        # Only override if explicitly set; preserve dataclass default otherwise
        if dim is not None:
            self.dimension = int(dim)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY") or self.openai_api_key


@dataclass
class LLMConfig:
    """Configuration for LLM calls. Supports Anthropic, OpenAI, Gemini, Ollama."""

    provider: str = ""  # anthropic, openai, gemini, ollama (auto-detected if empty)
    api_key: str = ""  # Provider API key (or use provider-specific env var)
    base_url: str = ""  # For OpenAI-compatible endpoints (Ollama, vLLM)
    extraction_model: str = ""  # Auto-set from provider defaults if empty
    tiebreaker_model: str = ""
    chat_model: str = ""

    def __post_init__(self) -> None:
        # Use `or default` (not get(key, default)) so unset env vars don't
        # silently overwrite dataclass defaults with None. Empty string is
        # also preserved as the dataclass default (falsy but not None).
        self.provider = os.environ.get("MEMENTO_LLM_PROVIDER") or self.provider
        self.api_key = os.environ.get("MEMENTO_LLM_API_KEY") or self.api_key
        self.base_url = os.environ.get("MEMENTO_LLM_BASE_URL") or self.base_url
        self.extraction_model = (
            os.environ.get("MEMENTO_EXTRACTION_MODEL") or self.extraction_model
        )
        self.tiebreaker_model = (
            os.environ.get("MEMENTO_TIEBREAKER_MODEL") or self.tiebreaker_model
        )
        self.chat_model = os.environ.get("MEMENTO_CHAT_MODEL") or self.chat_model


@dataclass
class ResolutionConfig:
    """Configuration for entity resolution thresholds."""

    high_threshold: float = 0.85
    low_threshold: float = 0.40


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval engine."""

    default_token_budget: int = 2000
    max_hop_depth: int = 3
    # 1/e decay time constant for the recency signal (NOT a half-life):
    # the score is exp(-days / recency_decay_days), so at days == this value
    # the score is ~0.37, not 0.5.
    recency_decay_days: float = 30.0
    # Semantic entity recall: seed graph expansion from entities that appear
    # in semantically-retrieved verbatim chunks, not just literal name matches.
    semantic_entity_recall: bool = True
    semantic_entity_top_k: int = 8


@dataclass
class RerankerConfig:
    """Configuration for the cross-encoder reranker.

    When enabled, retrieval candidates (verbatim chunks and graph facts) are
    rescored against the query with a cross-encoder. Falls back gracefully to
    no reranking if the model cannot be loaded.
    """

    enabled: bool = True
    provider: str = "auto"  # "auto", "cross-encoder", or "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        env = os.environ.get("MEMENTO_RERANKER_ENABLED")
        if env is not None:
            self.enabled = env.strip().lower() not in ("0", "false", "no", "off")
        # Use `or default` (not get(key, default)) so unset env vars don't
        # silently overwrite dataclass defaults with None.
        self.provider = os.environ.get("MEMENTO_RERANKER_PROVIDER") or self.provider
        self.model = os.environ.get("MEMENTO_RERANKER_MODEL") or self.model


@dataclass
class ConsolidationConfig:
    """Configuration for the consolidation engine."""

    decay_interval_ingestions: int = 50
    full_interval_ingestions: int = 200
    half_lives: dict[str, float] = field(
        default_factory=lambda: {
            "employment": 180.0,
            "location": 120.0,
            "preference": 90.0,
            "relationship": 365.0,
            "project": 60.0,
            "contact_info": 365.0,
            "default": 180.0,
        }
    )


@dataclass
class IngestConfig:
    """Configuration for the ingest pipeline gate.

    The gate is opt-in. When disabled (default), every call to
    MemoryStore.ingest runs the full extraction pipeline — matching
    historical behavior, keeping benchmarks deterministic.
    """

    gate_enabled: bool = False
    gate_min_chars: int = 20
    gate_store_verbatim_on_skip: bool = True

    def __post_init__(self) -> None:
        env = os.environ.get("MEMENTO_INGEST_GATE_ENABLED")
        if env is not None:
            self.gate_enabled = env.strip().lower() not in ("0", "false", "no", "off")
        min_chars = os.environ.get("MEMENTO_INGEST_GATE_MIN_CHARS")
        if min_chars:
            try:
                self.gate_min_chars = int(min_chars)
            except ValueError:
                pass
        store_verbatim = os.environ.get("MEMENTO_INGEST_GATE_STORE_VERBATIM_ON_SKIP")
        if store_verbatim is not None:
            self.gate_store_verbatim_on_skip = (
                store_verbatim.strip().lower() not in ("0", "false", "no", "off")
            )


@dataclass
class MementoConfig:
    """Top-level configuration for the Memento system."""

    db_path: Path = field(default_factory=_default_db_path)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)

    def __post_init__(self) -> None:
        db = os.environ.get("MEMENTO_DB_PATH")
        if db:
            self.db_path = Path(db).expanduser()

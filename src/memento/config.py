"""Configuration for Memento memory system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_db_path() -> Path:
    return Path(os.environ.get("MEMENTO_DB_PATH", "~/.memento/memento.db")).expanduser()


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""

    provider: str = "auto"  # "auto", "sentence-transformers", or "openai"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    openai_api_key: str | None = None

    def __post_init__(self) -> None:
        self.provider = os.environ.get("MEMENTO_EMBEDDING_PROVIDER", self.provider)
        self.model = os.environ.get("MEMENTO_EMBEDDING_MODEL", self.model)
        dim = os.environ.get("MEMENTO_EMBEDDING_DIMENSION")
        if dim:
            self.dimension = int(dim)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)


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
        self.provider = os.environ.get("MEMENTO_LLM_PROVIDER", self.provider)
        self.api_key = os.environ.get("MEMENTO_LLM_API_KEY", self.api_key)
        self.base_url = os.environ.get("MEMENTO_LLM_BASE_URL", self.base_url)
        self.extraction_model = os.environ.get(
            "MEMENTO_EXTRACTION_MODEL", self.extraction_model
        )
        self.tiebreaker_model = os.environ.get(
            "MEMENTO_TIEBREAKER_MODEL", self.tiebreaker_model
        )
        self.chat_model = os.environ.get(
            "MEMENTO_CHAT_MODEL", self.chat_model
        )


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
    recency_half_life_days: float = 30.0


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


@dataclass
class MementoConfig:
    """Top-level configuration for the Memento system."""

    db_path: Path = field(default_factory=_default_db_path)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)

    def __post_init__(self) -> None:
        db = os.environ.get("MEMENTO_DB_PATH")
        if db:
            self.db_path = Path(db).expanduser()

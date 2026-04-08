"""LLM provider abstraction — supports Anthropic, OpenAI, Gemini, and Ollama.

Mirrors the embedder.py pattern: Protocol + implementations + factory.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from memento.config import LLMConfig

logger = logging.getLogger(__name__)

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": {"extraction": "claude-haiku-4-5-20251001", "chat": "claude-sonnet-4-6"},
    "openai": {"extraction": "gpt-4o-mini", "chat": "gpt-4o"},
    "gemini": {"extraction": "gemini-2.0-flash", "chat": "gemini-2.5-pro"},
    "ollama": {"extraction": "llama3.2", "chat": "llama3.2"},
}


class LLMClient(Protocol):
    """Protocol for LLM providers. All implementations must support this interface."""

    def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Send messages to the LLM and return the text response."""
        ...


class AnthropicLLMClient:
    """Anthropic Claude implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install memento-memory[anthropic]"
            )

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        client = anthropic.Anthropic(api_key=self._api_key or None)
        response = client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""


class OpenAILLMClient:
    """OpenAI implementation. Also covers Ollama, vLLM, and any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url

    def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install memento-memory[openai]"
            )

        # Prepend system message if provided
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        kwargs: dict = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = openai.OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class GeminiLLMClient:
    """Google Gemini implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: pip install memento-memory[gemini]"
            )

        client = genai.Client(api_key=self._api_key or None)

        # Build contents from messages
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"System: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text or ""


def get_default_model(provider: str, purpose: str = "extraction") -> str:
    """Get the default model for a provider and purpose."""
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"]).get(
        purpose, DEFAULT_MODELS["openai"]["extraction"]
    )


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client from configuration.

    Auto-detects provider from environment if not explicitly set.
    """
    provider = config.provider

    # Auto-detect from environment if not configured
    if not provider or provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = "anthropic"  # Default

    api_key = config.api_key

    if provider == "anthropic":
        return AnthropicLLMClient(api_key=api_key)
    elif provider in ("openai", "openai-compatible"):
        return OpenAILLMClient(api_key=api_key, base_url=config.base_url or None)
    elif provider == "ollama":
        base_url = config.base_url or "http://localhost:11434/v1"
        return OpenAILLMClient(api_key="ollama", base_url=base_url)
    elif provider == "gemini":
        return GeminiLLMClient(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: anthropic, openai, gemini, ollama, openai-compatible"
        )

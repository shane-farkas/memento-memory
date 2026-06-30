"""Tests for the encrypted API-key secret store.

The round-trip tests exercise the real OS backend (Windows DPAPI, or the
``keyring`` library elsewhere) against an isolated store under a temporary
``MEMENTO_DB_PATH``, so they never read or mutate the developer's real
``~/.memento/secrets.dat``.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from memento import secret_store

_HAS_BACKEND = sys.platform == "win32" or importlib.util.find_spec("keyring") is not None
needs_backend = pytest.mark.skipif(
    not _HAS_BACKEND, reason="no secure storage backend (Windows DPAPI / keyring)"
)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the store at a temp dir and clear any real key env vars."""
    monkeypatch.setenv("MEMENTO_DB_PATH", str(tmp_path / "memento.db"))
    for env_var in secret_store.KNOWN_KEYS.values():
        monkeypatch.delenv(env_var, raising=False)
    return tmp_path


def test_get_secret_prefers_env(isolated_store, monkeypatch):
    # Resolves from the environment without any on-disk store.
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")
    assert secret_store.get_secret("openai") == "env-key-123"


def test_get_secret_missing_returns_none(isolated_store):
    assert secret_store.get_secret("together") is None


def test_set_secret_rejects_empty(isolated_store):
    with pytest.raises(ValueError):
        secret_store.set_secret("together", "")


def test_mask_hides_middle():
    assert secret_store._mask("sk-1234567890") == "sk-1" + "*" * 5 + "7890"
    assert secret_store._mask("short") == "*****"


@needs_backend
def test_roundtrip_set_get_list_delete(isolated_store):
    assert secret_store.get_secret("together") is None
    secret_store.set_secret("together", "sk-secret-xyz")
    assert secret_store.get_secret("together") == "sk-secret-xyz"
    assert "together" in secret_store.list_secret_names()
    assert secret_store.delete_secret("together") is True
    assert secret_store.get_secret("together") is None
    # Deleting again reports nothing was there.
    assert secret_store.delete_secret("together") is False


@needs_backend
def test_env_overrides_stored_value(isolated_store, monkeypatch):
    secret_store.set_secret("together", "stored-value")
    monkeypatch.setenv("TOGETHER_API_KEY", "env-value")
    # Env var wins over the encrypted store so CI / one-off shells take priority.
    assert secret_store.get_secret("together") == "env-value"

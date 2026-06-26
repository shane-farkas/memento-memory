"""Secure, dependency-free API-key storage.

Keys are encrypted at rest with the OS user-credential system and decryptable
only by the same Windows account:

  * Windows: DPAPI (CryptProtectData / CryptUnprotectData) — the same per-user
    encryption Chrome/Edge use for saved passwords. No third-party dependency.
  * Other platforms: the ``keyring`` library if installed, else a clear error.

Secrets live in ``~/.memento/secrets.dat`` as an encrypted JSON blob. The
plaintext is never written to disk, never passed on a command line, and never
printed (the CLI reads values from a hidden prompt and masks them on display).

CLI:
    python -m memento.secret_store set together     # hidden prompt
    python -m memento.secret_store set openai
    python -m memento.secret_store list
    python -m memento.secret_store get openai        # masked unless --reveal
    python -m memento.secret_store delete together

Programmatic:
    from memento.secret_store import get_secret, load_into_env
    key = get_secret("together")          # env var wins, then encrypted store
    load_into_env(["together", "openai"]) # export TOGETHER_API_KEY, OPENAI_API_KEY
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Canonical short name -> conventional environment variable. Storing under the
# short name keeps the CLI friendly while load_into_env() exports the exact
# variables the benchmark and MemoryStore already look for.
KNOWN_KEYS: dict[str, str] = {
    "together": "TOGETHER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# DPAPI entropy binds the ciphertext to this application alongside the per-user
# key DPAPI already enforces. A per-install random value is generated on first
# write and stored next to the encrypted blob — it is not itself a secret (the
# real protection is the Windows account's DPAPI master key), but keeping it
# out of source avoids a fixed value shared across every clone of the repo.
# LEGACY_ENTROPY is the original fixed value, kept only so a store created
# before per-install entropy can still be decrypted and migrated on next write.
LEGACY_ENTROPY = b"memento-secret-store-v1"

_SERVICE = "memento"  # keyring service name on non-Windows platforms


def _store_path() -> Path:
    base = Path(os.environ.get("MEMENTO_DB_PATH", "~/.memento/memento.db")).expanduser()
    return base.parent / "secrets.dat"


def _entropy_path() -> Path:
    return _store_path().parent / "secrets.entropy"


def _read_entropy() -> bytes | None:
    """The per-install entropy, or None if it hasn't been established yet."""
    p = _entropy_path()
    return p.read_bytes() if p.exists() else None


def _ensure_entropy() -> bytes:
    """Return the per-install entropy, generating + persisting it on first use."""
    ent = _read_entropy()
    if ent is not None:
        return ent
    ent = os.urandom(32)
    p = _entropy_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(ent)
    try:
        os.chmod(p, 0o600)  # best-effort; DPAPI already gates the blob
    except OSError:
        pass
    return ent


# ── Windows DPAPI backend ───────────────────────────────────────────────────

def _dpapi(encrypt: bool, data: bytes, entropy: bytes) -> bytes:
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    def to_blob(b: bytes) -> DATA_BLOB:
        buf = ctypes.create_string_buffer(b, len(b))
        return DATA_BLOB(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))

    def from_blob(blob: DATA_BLOB) -> bytes:
        out = ctypes.create_string_buffer(blob.cbData)
        ctypes.memmove(out, blob.pbData, blob.cbData)
        return out.raw

    blob_in = to_blob(data)
    blob_entropy = to_blob(entropy)
    blob_out = DATA_BLOB()
    CRYPTPROTECT_UI_FORBIDDEN = 0x01

    fn = (
        ctypes.windll.crypt32.CryptProtectData
        if encrypt
        else ctypes.windll.crypt32.CryptUnprotectData
    )
    ok = fn(
        ctypes.byref(blob_in),
        "memento-secrets",
        ctypes.byref(blob_entropy),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        err = ctypes.GetLastError()
        raise OSError(
            f"DPAPI {'encrypt' if encrypt else 'decrypt'} failed (error {err}). "
            "The encrypted store can only be read by the Windows account that "
            "created it."
        )
    try:
        return from_blob(blob_out)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _is_windows() -> bool:
    return sys.platform == "win32"


# ── Encrypted-blob read/write ───────────────────────────────────────────────

def _read_all() -> dict[str, str]:
    """Decrypt and return the full {name: value} map ({} if no store yet)."""
    path = _store_path()
    if not path.exists():
        return {}

    if _is_windows():
        # Per-install entropy if established, else the legacy fixed value so a
        # store created before this change is still readable (it migrates to a
        # per-install value on the next write).
        entropy = _read_entropy() or LEGACY_ENTROPY
        plaintext = _dpapi(False, path.read_bytes(), entropy)
        return json.loads(plaintext.decode("utf-8"))

    # Non-Windows: fall back to keyring (one entry per known name).
    try:
        import keyring
    except ImportError:
        raise RuntimeError(
            "Secure storage on this platform requires the 'keyring' package: "
            "pip install keyring"
        )
    out: dict[str, str] = {}
    for name in KNOWN_KEYS:
        val = keyring.get_password(_SERVICE, name)
        if val:
            out[name] = val
    return out


def _write_all(secrets: dict[str, str]) -> None:
    """Encrypt and persist the full {name: value} map."""
    if _is_windows():
        path = _store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Establish (or reuse) the per-install entropy and encrypt with it. A
        # store previously written under LEGACY_ENTROPY is transparently
        # re-encrypted here, since _read_all already decoded it.
        entropy = _ensure_entropy()
        ciphertext = _dpapi(True, json.dumps(secrets).encode("utf-8"), entropy)
        path.write_bytes(ciphertext)
        try:
            os.chmod(path, 0o600)  # best-effort; DPAPI already gates access
        except OSError:
            pass
        return

    try:
        import keyring
    except ImportError:
        raise RuntimeError(
            "Secure storage on this platform requires the 'keyring' package: "
            "pip install keyring"
        )
    for name, val in secrets.items():
        keyring.set_password(_SERVICE, name, val)


# ── Public API ──────────────────────────────────────────────────────────────

def _canonical(name: str) -> str:
    return name.strip().lower()


def set_secret(name: str, value: str) -> None:
    """Store (or replace) a secret under a canonical short name."""
    name = _canonical(name)
    if not value:
        raise ValueError("Refusing to store an empty value")
    secrets = _read_all()
    secrets[name] = value
    _write_all(secrets)


def get_secret(name: str) -> str | None:
    """Resolve a secret: environment variable first, then encrypted store.

    The env-var override means CI / one-off shells can supply a key without
    touching the on-disk store, and it never silently goes stale.
    """
    name = _canonical(name)
    env_var = KNOWN_KEYS.get(name)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    return _read_all().get(name)


def delete_secret(name: str) -> bool:
    """Remove a secret. Returns True if it existed."""
    name = _canonical(name)
    secrets = _read_all()
    existed = name in secrets
    if existed:
        del secrets[name]
        _write_all(secrets)
        if not _is_windows():
            try:
                import keyring

                keyring.delete_password(_SERVICE, name)
            except Exception:
                pass
    return existed


def list_secret_names() -> list[str]:
    """Names of stored secrets (values never returned)."""
    return sorted(_read_all().keys())


def load_into_env(names: list[str] | None = None, *, override: bool = False) -> list[str]:
    """Export stored secrets as their conventional env vars for child processes.

    Returns the list of env-var names that were set. Existing env vars are left
    untouched unless override=True.
    """
    names = names or list(KNOWN_KEYS.keys())
    exported: list[str] = []
    store = _read_all()
    for name in names:
        name = _canonical(name)
        env_var = KNOWN_KEYS.get(name)
        if not env_var:
            continue
        if not override and os.environ.get(env_var):
            continue
        val = store.get(name)
        if val:
            os.environ[env_var] = val
            exported.append(env_var)
    return exported


# ── CLI ─────────────────────────────────────────────────────────────────────

def _mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        prog="python -m memento.secret_store",
        description="Securely store API keys (encrypted per-user via OS DPAPI/keyring).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", help="Store a key (read from a hidden prompt)")
    p_set.add_argument("name", help=f"Key name, e.g. {', '.join(KNOWN_KEYS)}")
    p_set.add_argument(
        "--value", default=None,
        help="Provide the value inline instead of the hidden prompt "
             "(NOT recommended — leaks into shell history).",
    )

    p_get = sub.add_parser("get", help="Show a stored key (masked by default)")
    p_get.add_argument("name")
    p_get.add_argument("--reveal", action="store_true", help="Print the raw value")

    sub.add_parser("list", help="List stored key names")

    p_del = sub.add_parser("delete", help="Delete a stored key")
    p_del.add_argument("name")

    args = parser.parse_args(argv)

    if args.command == "set":
        name = _canonical(args.name)
        if name not in KNOWN_KEYS:
            print(f"  Note: '{name}' is not a known key. Known: {', '.join(KNOWN_KEYS)}")
            print("  (storing anyway; load_into_env won't auto-export an unknown name)")
        value = args.value
        if value is None:
            value = getpass.getpass(f"Enter value for '{name}' (hidden): ").strip()
        if not value:
            print("  Aborted: empty value.")
            return 1
        set_secret(name, value)
        env_var = KNOWN_KEYS.get(name, "(no env mapping)")
        print(f"  Stored '{name}' ({_mask(value)}) -> exported as {env_var}")
        return 0

    if args.command == "get":
        val = get_secret(args.name)
        if val is None:
            print(f"  No secret stored for '{_canonical(args.name)}'.")
            return 1
        print(f"  {_canonical(args.name)}: {val if args.reveal else _mask(val)}")
        return 0

    if args.command == "list":
        names = list_secret_names()
        if not names:
            print("  No secrets stored.")
            return 0
        print("  Stored keys:")
        for n in names:
            print(f"    - {n}  ->  {KNOWN_KEYS.get(n, '(no env mapping)')}")
        return 0

    if args.command == "delete":
        if delete_secret(args.name):
            print(f"  Deleted '{_canonical(args.name)}'.")
            return 0
        print(f"  Nothing to delete for '{_canonical(args.name)}'.")
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

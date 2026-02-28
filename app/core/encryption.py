"""Fernet encryption helpers for tenant credentials."""

import logging

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings

logger = logging.getLogger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = settings.fernet_key
        if not key:
            raise ValueError("FERNET_KEY is not configured — cannot encrypt/decrypt credentials")
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def encrypt_value(plaintext: str) -> bytes:
    """Encrypt a string value. Returns bytes suitable for BYTEA column."""
    return _get_fernet().encrypt(plaintext.encode("utf-8"))


def decrypt_value(ciphertext: bytes) -> str:
    """Decrypt a BYTEA value back to string. Returns empty string on failure."""
    if not ciphertext:
        return ""
    try:
        return _get_fernet().decrypt(ciphertext).decode("utf-8")
    except InvalidToken:
        logger.error("Failed to decrypt credential — invalid Fernet key or corrupted data")
        return ""

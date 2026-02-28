"""Tests for entity_vectorizer cache-invalidation helpers."""

from unittest.mock import MagicMock

from app.analysis.entity_vectorizer import (
    _embedding_stale,
    _text_hash,
    profile_to_embedding_text,
)


# ---------------------------------------------------------------------------
# _text_hash
# ---------------------------------------------------------------------------


def test_text_hash_deterministic():
    assert _text_hash("hello") == _text_hash("hello")


def test_text_hash_different_inputs():
    assert _text_hash("hello") != _text_hash("world")


def test_text_hash_length():
    h = _text_hash("anything")
    assert len(h) == 16  # SHA-256 hex prefix


# ---------------------------------------------------------------------------
# profile_to_embedding_text
# ---------------------------------------------------------------------------


def _make_profile(**kwargs):
    """Create a minimal mock EntityProfile."""
    p = MagicMock()
    p.canonical_name = kwargs.get("canonical_name", "Acme Corp")
    p.entity_type = kwargs.get("entity_type", None)
    p.industry = kwargs.get("industry", None)
    p.description = kwargs.get("description", None)
    p.embedding = kwargs.get("embedding", None)
    p.embedding_text_hash = kwargs.get("embedding_text_hash", None)
    p.attributes = kwargs.get("attributes", None)
    p.entity_key = kwargs.get("entity_key", "acme corp")
    return p


def test_profile_to_embedding_text_minimal():
    p = _make_profile(canonical_name="Acme Corp")
    assert profile_to_embedding_text(p) == "Acme Corp"


def test_profile_to_embedding_text_full():
    p = _make_profile(
        canonical_name="Acme Corp",
        entity_type="company",
        industry="Cloud services",
        description="A cloud computing provider.",
    )
    text = profile_to_embedding_text(p)
    assert "Acme Corp" in text
    assert "company" in text
    assert "Cloud services" in text
    assert "A cloud computing provider." in text


def test_profile_to_embedding_text_skips_empty():
    p = _make_profile(
        canonical_name="Test",
        entity_type="",
        industry=None,
        description="",
    )
    assert profile_to_embedding_text(p) == "Test"


# ---------------------------------------------------------------------------
# _embedding_stale
# ---------------------------------------------------------------------------


def test_embedding_stale_no_hash():
    """Profiles without a stored hash are always considered stale."""
    p = _make_profile(embedding=[0.1] * 10, embedding_text_hash=None)
    assert _embedding_stale(p) is True


def test_embedding_stale_matching_hash():
    """Profile with matching hash is NOT stale."""
    p = _make_profile(canonical_name="Acme Corp")
    text = profile_to_embedding_text(p)
    p.embedding_text_hash = _text_hash(text)
    p.embedding = [0.1] * 10
    assert _embedding_stale(p) is False


def test_embedding_stale_changed_text():
    """Profile whose text changed (hash mismatch) IS stale."""
    p = _make_profile(canonical_name="Acme Corp")
    p.embedding_text_hash = _text_hash("Old Acme Corp | old description")
    p.embedding = [0.1] * 10
    assert _embedding_stale(p) is True


def test_embedding_stale_empty_hash_string():
    """Empty string hash is treated as no hash (stale)."""
    p = _make_profile(embedding=[0.1] * 10, embedding_text_hash="")
    assert _embedding_stale(p) is True

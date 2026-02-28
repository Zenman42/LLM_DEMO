"""Entity Vectorizer — Phase 3 of new clustering pipeline.

Converts entity profiles into dense vector embeddings using
OpenAI text-embedding-3-small (1536 dimensions) for cosine-similarity
based hierarchical clustering.

Embeddings are cached in the ``entity_profiles.embedding`` column.
Only profiles whose text has changed since last embedding are re-vectorised.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import httpx
import numpy as np
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity_profile import EntityProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_API_URL = "https://api.openai.com/v1/embeddings"
_EMBEDDING_DIMS = 1536
_EMBEDDING_TIMEOUT = 60
_EMBEDDING_BATCH_SIZE = 512  # OpenAI supports up to 2048, but keep requests reasonable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def vectorize_profiles(
    profiles: list[EntityProfile],
    api_key: str,
    db: AsyncSession | None = None,
) -> dict[str, np.ndarray]:
    """Compute embeddings for entity profiles.

    Returns dict mapping entity_key → numpy vector (1536-d).
    If *db* is provided, caches embeddings back to entity_profiles table.

    Profiles that already have an embedding (and whose text hasn't changed)
    are reused from the ``embedding`` column.
    """
    if not profiles:
        return {}

    # Separate profiles into those needing embedding and those with cache.
    # Invalidate cached embeddings when the profile text has changed
    # (hash of embedding input text vs what was embedded last time).
    need_embedding: list[EntityProfile] = []
    result: dict[str, np.ndarray] = {}

    for p in profiles:
        if p.embedding is not None and not _embedding_stale(p):
            # Use cached embedding
            result[p.entity_key] = np.array(p.embedding, dtype=np.float32)
        else:
            need_embedding.append(p)

    if need_embedding:
        # NOTE: on first deploy after adding embedding_text_hash column,
        # ALL existing profiles will appear stale (no stored hash) and
        # be re-embedded in one burst.  This is a one-time cost.
        logger.info(
            "Vectorizer: %d profiles need embedding (%d cached)",
            len(need_embedding),
            len(result),
        )
        new_vectors = await _embed_profiles(need_embedding, api_key)

        for profile, vector in zip(need_embedding, new_vectors):
            if vector is None:
                logger.debug("No embedding for entity_key=%s (API error), skipping", profile.entity_key)
                continue
            result[profile.entity_key] = vector

            # Cache in DB (with text hash for staleness detection)
            if db is not None:
                text_hash = _text_hash(profile_to_embedding_text(profile))
                await db.execute(
                    update(EntityProfile)
                    .where(EntityProfile.id == profile.id)
                    .values(
                        embedding=vector.tolist(),
                        embedding_text_hash=text_hash,
                        updated_at=datetime.now(timezone.utc),
                    )
                )

        if db is not None:
            await db.flush()
    else:
        logger.info("Vectorizer: all %d profiles have cached embeddings", len(result))

    return result


def profile_to_embedding_text(p: EntityProfile) -> str:
    """Convert a profile to a text string suitable for embedding.

    Format: "canonical_name | entity_type | industry | description"
    Only non-empty fields are included.
    """
    parts = [p.canonical_name]
    if p.entity_type:
        parts.append(p.entity_type)
    if p.industry:
        parts.append(p.industry)
    if p.description:
        parts.append(p.description)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# OpenAI Embeddings API
# ---------------------------------------------------------------------------


async def _embed_profiles(
    profiles: list[EntityProfile],
    api_key: str,
) -> list[np.ndarray | None]:
    """Call OpenAI Embeddings API in batches.

    Returns list of vectors aligned with input profiles.
    None for profiles where embedding failed.
    """
    texts = [profile_to_embedding_text(p) for p in profiles]
    all_vectors: list[np.ndarray | None] = [None] * len(texts)

    for batch_start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch_texts = texts[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]
        batch_vectors = await _call_embedding_api(batch_texts, api_key)

        if batch_vectors is not None:
            for i, vec in enumerate(batch_vectors):
                all_vectors[batch_start + i] = vec

    embedded_count = sum(1 for v in all_vectors if v is not None)
    logger.info("Embedded %d/%d profiles", embedded_count, len(profiles))
    return all_vectors


async def _call_embedding_api(
    texts: list[str],
    api_key: str,
) -> list[np.ndarray] | None:
    """Call OpenAI embeddings API for a batch of texts."""
    payload = {
        "model": _EMBEDDING_MODEL,
        "input": texts,
    }

    try:
        async with httpx.AsyncClient(timeout=_EMBEDDING_TIMEOUT) as client:
            resp = await client.post(
                _EMBEDDING_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        embeddings = data.get("data", [])
        # Sort by index to ensure alignment with input
        embeddings.sort(key=lambda x: x["index"])

        return [np.array(e["embedding"], dtype=np.float32) for e in embeddings]

    except Exception as e:
        logger.warning("Embedding API call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Embedding cache staleness detection
# ---------------------------------------------------------------------------


def _text_hash(text: str) -> str:
    """Short hash of the text used to generate the embedding."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _embedding_stale(p: EntityProfile) -> bool:
    """Check if the cached embedding is stale (profile text changed since last embed).

    Compares stored ``embedding_text_hash`` column against current profile text.
    If no hash is stored (legacy data), conservatively treats embedding as stale.
    """
    stored_hash = p.embedding_text_hash
    if not stored_hash:
        return True  # no hash → re-embed to be safe
    current_hash = _text_hash(profile_to_embedding_text(p))
    return stored_hash != current_hash

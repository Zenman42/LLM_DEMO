"""ProLEA-inspired Entity Profile Generator (Phase 2 of new clustering pipeline).

Aggregates entity mentions (with context) from all snapshots, then generates
a structured semantic profile for each unique entity via LLM.  Optionally
enriches sparse profiles through Perplexity Sonar search.

Profiles are cached in the ``entity_profiles`` table and reused across
consolidation runs (TTL: 7 days before re-generation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity_profile import EntityProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EntityMention:
    """Single mention of a named entity extracted from an LLM snapshot."""

    name: str
    context: str = ""
    entity_type: str = "other"
    snapshot_id: int = 0
    position: int = 0


@dataclass
class ProfileData:
    """Intermediate representation before DB upsert."""

    entity_key: str
    canonical_name: str
    entity_type: str = "other"
    industry: str = ""
    description: str = ""
    attributes: dict = field(default_factory=dict)
    related_entities: list[str] = field(default_factory=list)
    profile_source: str = "llm"


# ---------------------------------------------------------------------------
# Profile generation config
# ---------------------------------------------------------------------------

_PROFILE_TTL_DAYS = 7  # reuse cached profiles younger than this
_PROFILE_BATCH_SIZE = 5  # entities per prompt

# Primary: Perplexity Sonar (has web search, knows fresh entities)
_PPLX_API_URL = "https://api.perplexity.ai/chat/completions"
_PPLX_MODEL = "sonar"
_PPLX_TIMEOUT = 60
_PPLX_MAX_TOKENS = 4096
_PPLX_TEMPERATURE = 0.0

# Fallback: GPT-4o-mini (when Perplexity key unavailable or call fails)
_FALLBACK_MODEL = "gpt-4o-mini"
_FALLBACK_API_URL = "https://api.openai.com/v1/chat/completions"
_FALLBACK_MAX_TOKENS = 1024
_FALLBACK_TEMPERATURE = 0.0
_FALLBACK_TIMEOUT = 30

_PROFILE_SYSTEM_PROMPT = """\
You are a knowledge-base builder for AI visibility monitoring.
Given mentions of one or more entities collected from AI-generated responses,
produce a structured semantic profile for each entity.
Use your knowledge and web search to provide accurate, up-to-date information.

Output ONLY valid JSON (no markdown):
{"profiles": [
  {
    "input_name": "<the entity name from input>",
    "canonical_name": "<most proper/official form>",
    "entity_type": "<company|product|person|location|service|technology|other>",
    "industry": "<primary industry/sector or empty string>",
    "description": "<1-2 factual sentences>",
    "related_entities": ["<related brand/product names>"],
    "attributes": {"<key>": "<value>"}
  }
]}

Rules:
- canonical_name: use the most standard, official spelling
- entity_type: classify based on context and your knowledge
- description: brief, factual, no opinions
- attributes: include parent_company, founded, hq, website if known
- related_entities: brands/products mentioned alongside this entity
- If you cannot determine a field, use empty string or empty object
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_entity_profiles(
    mentions: list[EntityMention],
    db: AsyncSession,
    project_id: int,
    tenant_id: UUID,
    openai_api_key: str,
    perplexity_api_key: str | None = None,
    today: date | None = None,
) -> list[EntityProfile]:
    """Build or update semantic profiles for all mentioned entities.

    1. Aggregate mentions by normalised key
    2. Check cache (entity_profiles table) — skip fresh entries
    3. Generate profiles via LLM for new/stale entities
    4. Optionally enrich sparse profiles via Perplexity Sonar
    5. Upsert into DB and return all profiles for today's entities

    Returns list of EntityProfile ORM objects (with id populated).
    """
    if today is None:
        today = date.today()

    # Step 1: Aggregate mentions by normalised key
    aggregated = _aggregate_mentions(mentions)

    if not aggregated:
        return []

    entity_keys = list(aggregated.keys())
    logger.info("Profile builder: %d unique entities to profile", len(entity_keys))

    # Step 2: Load cached profiles
    cached = await _load_cached_profiles(db, project_id, entity_keys)
    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=_PROFILE_TTL_DAYS)

    fresh_keys = set()
    stale_keys = set()
    for key in entity_keys:
        if key in cached and cached[key].updated_at and cached[key].updated_at >= stale_cutoff:
            fresh_keys.add(key)
        else:
            stale_keys.add(key)

    logger.info(
        "Profile cache: %d fresh, %d need (re)generation",
        len(fresh_keys),
        len(stale_keys),
    )

    # Step 3: Generate profiles for new/stale entities
    # Primary: Perplexity Sonar (web search, fresh data)
    # Fallback: GPT-4o-mini (when Perplexity unavailable)
    if stale_keys:
        stale_grouped = {k: aggregated[k] for k in stale_keys}
        if perplexity_api_key:
            new_profiles = await _generate_profiles_perplexity(
                stale_grouped,
                perplexity_api_key,
            )
            # Fallback for any entities that Perplexity missed
            generated_keys = {p.entity_key for p in new_profiles}
            missed_keys = stale_keys - generated_keys
            if missed_keys:
                logger.info(
                    "Perplexity missed %d entities, falling back to GPT-4o-mini",
                    len(missed_keys),
                )
                fallback = await _generate_profiles_openai(
                    {k: aggregated[k] for k in missed_keys},
                    openai_api_key,
                )
                new_profiles.extend(fallback)
        else:
            new_profiles = await _generate_profiles_openai(
                stale_grouped,
                openai_api_key,
            )

        # Step 4: Upsert into DB
        await _upsert_profiles(db, project_id, tenant_id, new_profiles)

    # Return all profiles for today's entities
    all_profiles = await _load_cached_profiles(db, project_id, entity_keys)
    return list(all_profiles.values())


# ---------------------------------------------------------------------------
# Mention aggregation
# ---------------------------------------------------------------------------


def _normalise_key(name: str) -> str:
    """Normalise entity name for deduplication (lowercase, stripped, dots removed).

    Also normalises Unicode dash variants to ASCII hyphen-minus.
    """
    from app.analysis.llm_entity_extractor import _DASH_RE

    s = name.lower().strip().replace(".", "").replace(",", "")
    return _DASH_RE.sub("-", s)


def _aggregate_mentions(mentions: list[EntityMention]) -> dict[str, list[EntityMention]]:
    """Group mentions by normalised entity key."""
    groups: dict[str, list[EntityMention]] = {}
    for m in mentions:
        key = _normalise_key(m.name)
        if not key:
            continue
        groups.setdefault(key, []).append(m)
    return groups


def parse_entity_mentions(
    snapshot_rows: list[tuple[int, list]],
) -> list[EntityMention]:
    """Parse EntityMention list from snapshot raw_entities (enriched format).

    Handles both old format (list[str]) and new format (list[dict]).

    Args:
        snapshot_rows: list of (snapshot_id, raw_entities_jsonb)
    """
    result = []
    for snapshot_id, raw_ents in snapshot_rows:
        if not raw_ents:
            continue
        for i, e in enumerate(raw_ents):
            if isinstance(e, str):
                result.append(EntityMention(name=e, snapshot_id=snapshot_id, position=i))
            elif isinstance(e, dict) and "name" in e:
                result.append(
                    EntityMention(
                        name=e["name"],
                        context=e.get("context", ""),
                        entity_type=e.get("type", "other"),
                        snapshot_id=snapshot_id,
                        position=i,
                    )
                )
    return result


# ---------------------------------------------------------------------------
# Profile cache
# ---------------------------------------------------------------------------


async def _load_cached_profiles(
    db: AsyncSession,
    project_id: int,
    entity_keys: list[str],
) -> dict[str, EntityProfile]:
    """Load existing profiles from DB by entity_key."""
    if not entity_keys:
        return {}
    stmt = select(EntityProfile).where(
        EntityProfile.project_id == project_id,
        EntityProfile.entity_key.in_(entity_keys),
    )
    rows = (await db.execute(stmt)).scalars().all()
    return {p.entity_key: p for p in rows}


# ---------------------------------------------------------------------------
# Perplexity Sonar profile generation (primary)
# ---------------------------------------------------------------------------


async def _generate_profiles_perplexity(
    grouped: dict[str, list[EntityMention]],
    api_key: str,
) -> list[ProfileData]:
    """Generate profiles via Perplexity Sonar (has web search)."""
    keys = list(grouped.keys())
    all_profiles: list[ProfileData] = []

    for batch_start in range(0, len(keys), _PROFILE_BATCH_SIZE):
        batch_keys = keys[batch_start : batch_start + _PROFILE_BATCH_SIZE]
        batch_input = _build_batch_prompt(grouped, batch_keys)

        profiles = await _call_perplexity_profile(batch_input, api_key)
        all_profiles.extend(profiles)

    logger.info("Perplexity generated %d/%d profiles", len(all_profiles), len(keys))
    return all_profiles


async def _call_perplexity_profile(
    user_prompt: str,
    api_key: str,
) -> list[ProfileData]:
    """Call Perplexity Sonar to generate entity profiles."""
    payload = {
        "model": _PPLX_MODEL,
        "messages": [
            {"role": "system", "content": _PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate profiles for these entities:\n\n{user_prompt}"},
        ],
        "temperature": _PPLX_TEMPERATURE,
        "max_tokens": _PPLX_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=_PPLX_TIMEOUT) as client:
            resp = await client.post(
                _PPLX_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return _parse_profile_response(data)

    except Exception as e:
        logger.warning("Perplexity profile call failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# OpenAI profile generation (fallback)
# ---------------------------------------------------------------------------


async def _generate_profiles_openai(
    grouped: dict[str, list[EntityMention]],
    openai_api_key: str,
) -> list[ProfileData]:
    """Generate profiles via GPT-4o-mini (fallback when Perplexity unavailable)."""
    keys = list(grouped.keys())
    all_profiles: list[ProfileData] = []

    for batch_start in range(0, len(keys), _PROFILE_BATCH_SIZE):
        batch_keys = keys[batch_start : batch_start + _PROFILE_BATCH_SIZE]
        batch_input = _build_batch_prompt(grouped, batch_keys)

        profiles = await _call_openai_profile(batch_input, openai_api_key)
        all_profiles.extend(profiles)

    return all_profiles


def _build_batch_prompt(
    grouped: dict[str, list[EntityMention]],
    keys: list[str],
) -> str:
    """Build the user prompt listing entities and their contexts."""
    parts = []
    for key in keys:
        mentions = grouped[key]
        # Use the most common raw name form
        name = mentions[0].name
        contexts = list({m.context for m in mentions if m.context})[:5]  # max 5 contexts
        entity_type = mentions[0].entity_type

        entry = f"Entity: {name}"
        if entity_type and entity_type != "other":
            entry += f" (likely {entity_type})"
        if contexts:
            entry += "\nContexts:\n" + "\n".join(f"  - {c}" for c in contexts)
        else:
            entry += "\n(no context available)"
        parts.append(entry)

    return "\n\n".join(parts)


async def _call_openai_profile(
    user_prompt: str,
    api_key: str,
) -> list[ProfileData]:
    """Call GPT-4o-mini to generate entity profiles (fallback)."""
    payload = {
        "model": _FALLBACK_MODEL,
        "messages": [
            {"role": "system", "content": _PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate profiles for these entities:\n\n{user_prompt}"},
        ],
        "temperature": _FALLBACK_TEMPERATURE,
        "max_tokens": _FALLBACK_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=_FALLBACK_TIMEOUT) as client:
            resp = await client.post(
                _FALLBACK_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return _parse_profile_response(data, source="openai")

    except Exception as e:
        logger.warning("OpenAI profile call failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Shared response parser
# ---------------------------------------------------------------------------


def _parse_profile_response(
    data: dict,
    source: str = "perplexity",
) -> list[ProfileData]:
    """Parse LLM response (OpenAI-compatible format) into ProfileData list."""
    import re

    raw = data["choices"][0]["message"]["content"] or ""

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to fix trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse profile JSON (batch lost), raw[:300]=%s",
                raw[:300],
            )
            return []

    profiles_data = parsed.get("profiles", [])

    result = []
    for p in profiles_data:
        if not isinstance(p, dict) or "input_name" not in p:
            continue
        result.append(
            ProfileData(
                entity_key=_normalise_key(p["input_name"]),
                canonical_name=p.get("canonical_name", p["input_name"]),
                entity_type=p.get("entity_type", "other"),
                industry=p.get("industry", ""),
                description=p.get("description", ""),
                attributes=p.get("attributes", {}),
                related_entities=p.get("related_entities", []),
                profile_source=source,
            )
        )
    return result


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------


async def _upsert_profiles(
    db: AsyncSession,
    project_id: int,
    tenant_id: UUID,
    profiles: list[ProfileData],
) -> None:
    """Upsert generated profiles into entity_profiles table."""
    for p in profiles:
        stmt = (
            pg_insert(EntityProfile)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                entity_key=p.entity_key,
                canonical_name=p.canonical_name,
                entity_type=p.entity_type,
                industry=p.industry or None,
                description=p.description or None,
                attributes=p.attributes or None,
                related_entities=p.related_entities or None,
                profile_source=p.profile_source,
            )
            .on_conflict_do_update(
                constraint="uq_entity_profile_key",
                set_={
                    "canonical_name": p.canonical_name,
                    "entity_type": p.entity_type,
                    "industry": p.industry or None,
                    "description": p.description or None,
                    "attributes": p.attributes or None,
                    "related_entities": p.related_entities or None,
                    "profile_source": p.profile_source,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        )
        await db.execute(stmt)

    await db.flush()
    logger.info("Upserted %d entity profiles", len(profiles))

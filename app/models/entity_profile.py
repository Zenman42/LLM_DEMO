"""Semantic entity profile for ProLEA-based clustering.

Each unique entity (brand, product, person, etc.) gets a rich textual profile
built from LLM-response context and optionally enriched via Perplexity Sonar.
Profiles are embedded via OpenAI text-embedding-3-small (1536 dims) and stored
as TEXT (pgvector-compatible format ``[0.1,0.2,...]``).

All similarity computation happens in Python (numpy/scipy) so native pgvector
DB extension is NOT required.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.db.base import Base


class VectorAsText(TypeDecorator):
    """Store embedding vectors as TEXT in pgvector-compatible format.

    Serialises Python lists/arrays to ``'[0.1,0.2,...,0.3]'`` on write.
    Deserialises back to Python list of floats on read.

    This avoids requiring the pgvector DB extension (which may not be
    available on managed PostgreSQL services like Railway) while keeping
    the same string format so that a future migration to native pgvector
    would be a simple ``ALTER COLUMN ... TYPE vector(1536)`` with no data
    conversion needed.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):  # type: ignore[override]
        if value is None:
            return None
        # Accept list, tuple, or numpy array
        return "[" + ",".join(str(float(v)) for v in value) + "]"

    def process_result_value(self, value, dialect):  # type: ignore[override]
        if value is None:
            return None
        if isinstance(value, str):
            # Parse pgvector-format string: '[0.1,0.2,...,0.3]'
            return [float(v) for v in value.strip("[] ").split(",")]
        # Already a list/array (e.g. from JSONB fallback)
        return list(value)


class EntityProfile(Base):
    """Semantic profile for a named entity discovered in LLM responses."""

    __tablename__ = "entity_profiles"
    __table_args__ = (UniqueConstraint("project_id", "entity_key", name="uq_entity_profile_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    # Normalised lookup key (lower + stripped) for deduplication
    entity_key: Mapped[str] = mapped_column(String(255), nullable=False)

    # Human-readable canonical form chosen by the profiler
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Entity classification
    entity_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # company | product | person | location | service | other

    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Structured metadata extracted by LLM / Perplexity
    attributes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # {"founded": "1841", "hq": "Moscow"}
    related_entities: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # ["Сбер.Онлайн", "СберМаркет"]

    # Embedding from OpenAI text-embedding-3-small (1536 dims).
    # Stored as TEXT in pgvector-compatible format; see VectorAsText above.
    embedding = mapped_column(VectorAsText(), nullable=True)

    # SHA-256 prefix of the text that was embedded; used to detect when
    # the profile text changed and the embedding needs re-generation.
    embedding_text_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Provenance
    profile_source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "llm" | "perplexity" | "manual"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

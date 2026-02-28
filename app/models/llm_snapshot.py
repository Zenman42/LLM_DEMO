import uuid
from datetime import date, datetime, timezone

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class LlmSnapshot(Base):
    """Result of checking a single query against a single LLM provider on a given date."""

    __tablename__ = "llm_snapshots"
    __table_args__ = (
        UniqueConstraint("llm_query_id", "date", "llm_provider", name="uq_llm_snapshot"),
        # Partitioning handled by Alembic migration (PARTITION BY RANGE (date))
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    llm_query_id: Mapped[int] = mapped_column(
        ForeignKey("llm_queries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(20), nullable=False)  # chatgpt | deepseek | perplexity | yandexgpt
    llm_model: Mapped[str] = mapped_column(String(50), nullable=False)  # gpt-4o, gemini-2.0-flash, etc.

    # Analysis results
    brand_mentioned: Mapped[bool] = mapped_column(Boolean, default=False)
    mention_type: Mapped[str] = mapped_column(
        String(20), default="none"
    )  # direct | recommended | compared | negative | none
    mention_context: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    competitor_mentions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # {"SEMrush": true, "Moz": false}
    cited_urls: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # ["https://...", ...]
    raw_entities: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # all extracted entity names

    # GEO metrics (populated by Module 3 pipeline analysis)
    sentiment_label: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # positive | neutral | negative | mixed
    position_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0=not in list, 1+=position
    structure_type: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )  # numbered_list | bulleted_list | narrative | table | mixed
    is_hallucination: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Raw data
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    llm_query: Mapped["LlmQuery"] = relationship("LlmQuery", back_populates="snapshots")  # noqa: F821

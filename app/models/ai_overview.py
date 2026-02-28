import uuid
from datetime import date, datetime, timezone

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, SmallInteger, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AiOverviewSnapshot(Base):
    """AI Overview presence for a keyword in Google SERP."""

    __tablename__ = "ai_overview_snapshots"
    __table_args__ = (
        UniqueConstraint("keyword_id", "date", name="uq_ai_overview_snapshot"),
        # Partitioning handled by Alembic migration (PARTITION BY RANGE (date))
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    keyword_id: Mapped[int] = mapped_column(ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)

    has_ai_overview: Mapped[bool] = mapped_column(Boolean, default=False)
    brand_in_aio: Mapped[bool] = mapped_column(Boolean, default=False)
    aio_position: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)  # position in AIO sources
    aio_sources: Mapped[list | None] = mapped_column(
        JSONB, nullable=True
    )  # [{"url": ..., "domain": ..., "title": ...}]
    aio_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

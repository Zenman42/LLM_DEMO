import uuid
from datetime import date, datetime, timezone

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, SmallInteger, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SerpSnapshot(Base):
    """Position snapshot per keyword×engine×date. Stored in PostgreSQL.

    Note: SerpDetail (top-50 SERP results) is stored in ClickHouse, not SQLAlchemy.
    """

    __tablename__ = "serp_snapshots"
    __table_args__ = (
        UniqueConstraint("keyword_id", "date", "search_engine", name="uq_serp_snapshot"),
        # Partitioning is handled by Alembic migration (PARTITION BY RANGE (date))
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    keyword_id: Mapped[int] = mapped_column(ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    search_engine: Mapped[str] = mapped_column(String(10), nullable=False)
    position: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    found_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    previous_position: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

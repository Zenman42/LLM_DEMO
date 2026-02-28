import uuid
from datetime import date, datetime, timezone

from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class WebmasterData(Base):
    __tablename__ = "webmaster_data"
    __table_args__ = (UniqueConstraint("keyword_id", "date", "search_engine", name="uq_webmaster_data"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    keyword_id: Mapped[int] = mapped_column(ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    search_engine: Mapped[str] = mapped_column(String(10), nullable=False)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    ctr: Mapped[float | None] = mapped_column(Float, nullable=True)
    position: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

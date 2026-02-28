import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class LlmQuery(Base):
    """A prompt/question to monitor across LLM providers."""

    __tablename__ = "llm_queries"
    __table_args__ = (UniqueConstraint("project_id", "query_text", name="uq_project_llm_query"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query_text: Mapped[str] = mapped_column(String(2000), nullable=False)
    query_type: Mapped[str] = mapped_column(
        String(20), default="custom"
    )  # brand_check | comparison | recommendation | custom
    target_brand: Mapped[str | None] = mapped_column(String(255), nullable=True)
    competitors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # ["SEMrush", "Moz"]
    measurement_type: Mapped[str | None] = mapped_column(String(30), nullable=True)  # top_of_mind | comparison | ...

    # V2 classification
    query_class: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )  # thematic | branded
    query_subtype: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
        index=True,
    )  # category | attribute | scenario | event | reputation | comparison | negative | fact_check

    # Scenario/category from project.categories (for filtering dashboard by scenario)
    category: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="llm_queries")  # noqa: F821
    snapshots: Mapped[list["LlmSnapshot"]] = relationship(  # noqa: F821
        "LlmSnapshot", back_populates="llm_query", cascade="all, delete-orphan"
    )
    tag_links: Mapped[list["QueryTagLink"]] = relationship(  # noqa: F821
        "QueryTagLink", back_populates="query", cascade="all, delete-orphan"
    )
